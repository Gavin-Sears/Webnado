import { GUI } from 'lil-gui';

import { device, profiler } from '../renderer';
import { constants, processShaderRaw } from '../shaders/shaders';
import { Fields, PingPong } from '../stage/fields';
import { WeatherData } from '../weather/noaa_loader';
import { DivergenceCheck, type DivergenceReport } from './divergence_check';

import commonSource from './common.wgsl?raw';
import boundarySource from './boundaries.wgsl?raw';
import projectionSource from './projection.wgsl?raw';
import transportSource from './transport.wgsl?raw';
import forceSource from './forces.wgsl?raw';

const kernelBindings = {
    buildOpenBits: [12, 13],
    enforceBoundaries: [1, 3, 4, 5],
    curl: [1, 3, 4, 8],
    forces: [0, 1, 2, 3, 4, 5, 7],
    advect: [0, 1, 2, 3, 4, 5, 6],
    divergence: [1, 3, 4, 11],
    jacobi: [3, 9, 10, 11],
    project: [1, 3, 4, 5, 9]
} as const;

type KernelName = keyof typeof kernelBindings;

interface Kernel {
    pipeline: GPUComputePipeline;
    groups: Map<string, GPUBindGroup>;
}

function createVolume(label: string, format: GPUTextureFormat): GPUTexture {
    return device.createTexture({
        label,
        dimension: '3d',
        size: [constants.gridX, constants.gridY, constants.gridZ],
        format,
        usage: GPUTextureUsage.TEXTURE_BINDING
            | GPUTextureUsage.STORAGE_BINDING
            | GPUTextureUsage.COPY_SRC
    });
}

export class FluidSolver {
    dt = 0.5;
    jacobiIterations = 40;
    buoyancy = 1;
    confinement = 0.1;
    lastDivergence: DivergenceReport | undefined;

    private readonly pressure = new PingPong('pressure', 'r32float');
    private readonly divergence = createVolume('divergence', 'r32float');
    private readonly openBits = createVolume('open neighbor bits', 'r32uint');
    private readonly vorticity = createVolume('vorticity', 'rgba16float');

    private readonly divergenceView = this.divergence.createView();
    private readonly openBitsView = this.openBits.createView();
    private readonly vorticityView = this.vorticity.createView();

    private readonly parameters = device.createBuffer({
        label: 'solver parameters',
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    private readonly kernels = {} as Record<KernelName, Kernel>;
    private obstacleVersion = -1;
    private thermalRequested = false;
    private steps = 0;
    private readonly divergenceReadback = new URLSearchParams(location.search).has('checkDivergence')
        ? device.createBuffer({
            label: 'solver divergence readback',
            size: DivergenceCheck.bufferSize,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        })
        : undefined;

    private readonly divergenceCheck = this.divergenceReadback
        ? new DivergenceCheck(this.divergenceReadback)
        : undefined;

    constructor(readonly fields: Fields, readonly weather: WeatherData) {
        const source = [
            commonSource,
            boundarySource,
            projectionSource,
            transportSource,
            forceSource
        ].join('\n');
        const module = device.createShaderModule({
            label: 'fluid solver',
            code: processShaderRaw(source)
        });

        for (const name of Object.keys(kernelBindings) as KernelName[]) {
            const entries = kernelBindings[name].map(binding => this.layoutEntry(binding));
            const groupLayout = device.createBindGroupLayout({ entries });
            const layout = device.createPipelineLayout({
                bindGroupLayouts: [groupLayout]
            });

            const pipeline = device.createComputePipeline({
                label: name,
                layout,
                compute: {
                    module,
                    entryPoint: name
                }
            });

            this.kernels[name] = {
                pipeline,
                groups: new Map()
            };
        }
    }

    buildGui(gui: GUI) {
        const folder = gui.addFolder('Fluid solver');
        folder.add(this, 'dt', 0.05, 1, 0.05).name('Time step (s)');
        folder.add(this, 'jacobiIterations', 1, 200, 1).name('Pressure iterations');
        folder.add(this, 'buoyancy', 0, 2, 0.1).name('Buoyancy');
        folder.add(this, 'confinement', 0, 0.5, 0.01).name('Vorticity confinement');
        folder.add(this, 'addThermal').name('Add warm thermal');
    }

    // Encode once per frame before raymarching; the renderer submits the encoder.
    step(encoder: GPUCommandEncoder) {
        device.queue.writeBuffer(this.parameters, 0, new Float32Array([
            this.dt,
            this.buoyancy,
            this.confinement,
            Number(this.thermalRequested)
        ]));
        this.thermalRequested = false;

        this.prepareBoundaries(encoder);
        this.dispatch(encoder, 'curl');
        this.dispatch(encoder, 'forces');
        this.fields.velocity.swap();

        this.dispatch(encoder, 'advect');
        this.fields.velocity.swap();
        this.fields.thermo.swap();

        this.projectPressure(encoder);
        this.steps++;

        if (this.steps % 120 === 1) {
            this.checkDivergence(encoder);
        }
    }

    addThermal() {
        this.thermalRequested = true;
    }

    checkDivergence(encoder: GPUCommandEncoder): Promise<DivergenceReport> | undefined {
        const result = this.divergenceCheck?.capture(
            encoder,
            this.fields,
            this.openBits,
            this.weather.atmosphereBuffer
        );

        void result?.then(report => {
            this.lastDivergence = report;

            if (report.rms > 5e-4 || report.nonFinite > 0) {
                console.warn('Solver divergence is above target', report);
            } else {
                console.log('Solver divergence', report);
            }
        }).catch(console.error);

        return result;
    }

    project(encoder: GPUCommandEncoder) {
        this.prepareBoundaries(encoder);
        this.projectPressure(encoder);
    }

    private projectPressure(encoder: GPUCommandEncoder) {
        this.dispatch(encoder, 'divergence');

        const kernel = this.kernels.jacobi;
        const pass = encoder.beginComputePass({
            label: 'jacobi',
            timestampWrites: profiler?.passWrites('jacobi')
        });
        pass.setPipeline(kernel.pipeline);

        for (let iteration = 0; iteration < this.jacobiIterations; iteration++) {
            pass.setBindGroup(0, this.bindGroup('jacobi'));
            this.dispatchGrid(pass);
            this.pressure.swap();
        }

        pass.end();

        this.dispatch(encoder, 'project');
        this.fields.velocity.swap();
    }

    private prepareBoundaries(encoder: GPUCommandEncoder) {
        if (this.obstacleVersion !== this.fields.obstacleVersion) {
            this.dispatch(encoder, 'buildOpenBits');
            this.obstacleVersion = this.fields.obstacleVersion;
        }

        this.dispatch(encoder, 'enforceBoundaries');
        this.fields.velocity.swap();
    }

    private layoutEntry(binding: number): GPUBindGroupLayoutEntry {
        const entry: GPUBindGroupLayoutEntry = {
            binding,
            visibility: GPUShaderStage.COMPUTE
        };

        if (binding === 0) {
            entry.buffer = { type: 'uniform' };
        } else if (binding === 4) {
            entry.buffer = { type: 'read-only-storage' };
        } else if ([5, 6, 8, 11, 13].includes(binding)) {
            let format: GPUTextureFormat = 'rgba16float';

            if (binding === 11) {
                format = 'r32float';
            } else if (binding === 13) {
                format = 'r32uint';
            }

            entry.storageTexture = {
                access: 'write-only',
                viewDimension: '3d',
                format
            };
        } else {
            entry.texture = {
                viewDimension: '3d',
                sampleType: binding === 3 || binding === 12 ? 'uint' : 'unfilterable-float'
            };
        }

        return entry;
    }

    private bindGroup(name: KernelName): GPUBindGroup {
        const kernel = this.kernels[name];
        const velocityIndex = this.fields.velocity.index;
        const thermoIndex = this.fields.thermo.index;
        const key = `${velocityIndex}/${thermoIndex}/${this.pressure.index}`;
        const cached = kernel.groups.get(key);

        if (cached) {
            return cached;
        }

        const resources: Record<number, GPUBindingResource> = {
            0: { buffer: this.parameters },
            1: this.fields.velocity.read,
            2: this.fields.thermo.read,
            3: this.openBitsView,
            4: { buffer: this.weather.atmosphereBuffer },
            5: this.fields.velocity.write,
            6: this.fields.thermo.write,
            7: this.vorticityView,
            8: this.vorticityView,
            9: this.pressure.read,
            10: this.divergenceView,
            11: name === 'divergence' ? this.divergenceView : this.pressure.write,
            12: this.fields.obstacleView,
            13: this.openBitsView
        };

        const entries = kernelBindings[name].map(binding => ({
            binding,
            resource: resources[binding]
        }));

        const group = device.createBindGroup({
            label: name,
            layout: kernel.pipeline.getBindGroupLayout(0),
            entries
        });
        kernel.groups.set(key, group);

        return group;
    }

    private dispatch(encoder: GPUCommandEncoder, name: KernelName) {
        const kernel = this.kernels[name];
        const pass = encoder.beginComputePass({
            label: name,
            timestampWrites: profiler?.passWrites(name)
        });

        pass.setPipeline(kernel.pipeline);
        pass.setBindGroup(0, this.bindGroup(name));
        this.dispatchGrid(pass);
        pass.end();
    }

    private dispatchGrid(pass: GPUComputePassEncoder) {
        pass.dispatchWorkgroups(
            Math.ceil(constants.gridX / 8),
            Math.ceil(constants.gridY / 8),
            Math.ceil(constants.gridZ / 4)
        );
    }

    destroy() {
        this.parameters.destroy();
        this.pressure.textures.forEach(texture => texture.destroy());
        this.divergence.destroy();
        this.openBits.destroy();
        this.vorticity.destroy();
        this.divergenceReadback?.destroy();
    }
}
