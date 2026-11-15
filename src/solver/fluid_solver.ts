import { GUI } from 'lil-gui';

import { device, profiler } from '../renderer';
import { constants, processShaderRaw } from '../shaders/shaders';
import { Fields, PingPong } from '../stage/fields';
import { WeatherData } from '../weather/noaa_loader';
import { DivergenceCheck, type DivergenceReport } from './divergence_check';
import { DustFluid } from './dust_fluid';

import commonSource from './common.wgsl?raw';
import boundarySource from './boundaries.wgsl?raw';
import projectionSource from './projection.wgsl?raw';
import transportSource from './transport.wgsl?raw';
import forceSource from './forces.wgsl?raw';
import tornadoSource from './tornado.wgsl?raw';

export interface Microphysics {
    // Read thermo.read, write thermo.write, then swap thermo once before returning.
    step(encoder: GPUCommandEncoder, dt: number): void;
}

const kernelBindings = {
    buildOpenBits: [12, 13],
    enforceBoundaries: [1, 3, 4, 5],
    curl: [1, 3, 4, 8],
    forces: [0, 1, 3, 4, 5, 7, 16, 18],
    energy: [0, 1, 2, 3, 4, 14, 15],
    atmosphericForces: [0, 1, 2, 3, 4, 14, 17],
    evolveVorticity: [0, 1, 3, 4, 7, 8, 14, 16, 18],
    mixThermo: [0, 2, 3, 6, 14],
    resetFields: [4, 5, 6, 8, 15],
    clearPressure: [11],
    clearForcing: [19],
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
    turbulence = true;
    transportedVorticity = true;
    mixing = 0.1;
    microphysics: Microphysics | undefined;
    resetVersion = 0;
    lastDivergence: DivergenceReport | undefined;
    readonly dust: DustFluid;

    private readonly pressure = new PingPong('pressure', 'r32float');
    private readonly divergence = createVolume('divergence', 'r32float');
    private readonly openBits = createVolume('open neighbor bits', 'r32uint');
    private readonly atmosphericForce = createVolume('atmospheric forces', 'rgba16float');

    private readonly divergenceView = this.divergence.createView();
    private readonly openBitsView = this.openBits.createView();
    private readonly atmosphericForceView = this.atmosphericForce.createView();

    private readonly parameters = device.createBuffer({
        label: 'solver parameters',
        size: 64,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    private readonly kernels = {} as Record<KernelName, Kernel>;
    private obstacleVersion = -1;
    private thermalRequested = false;
    private steps = 0;
    private resetRequested = false;
    private initializeRotation = true;
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
        this.dust = new DustFluid(fields, weather, this.openBits, this.parameters);
        const source = [
            commonSource,
            boundarySource,
            projectionSource,
            transportSource,
            forceSource,
            tornadoSource
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
        folder.add(this, 'turbulence').name('Turbulent mixing');
        folder.add(this, 'mixing', 0, 0.5, 0.01).name('Mixing coefficient');
        folder.add(this, 'transportedVorticity').name('Transport vorticity').onChange(() => {
            this.initializeRotation = true;
        });
        folder.add(this, 'addThermal').name('Add warm thermal');
        folder.add(this, 'startShearExperiment').name('Start shear experiment');
        folder.add(this, 'reset').name('Reset simulation');
        this.dust.buildGui(gui);
    }

    // Encode once per frame before raymarching; the renderer submits the encoder.
    step(encoder: GPUCommandEncoder) {
        if (this.resetRequested) {
            this.dust.reset(encoder);
            this.dispatch(encoder, 'clearForcing');
            this.dispatch(encoder, 'resetFields');
            this.fields.velocity.swap();
            this.fields.thermo.swap();
            this.fields.turbulence.swap();
            this.fields.vorticity.swap();
            this.dispatch(encoder, 'clearPressure');
            this.pressure.swap();
            this.dispatch(encoder, 'clearPressure');
            this.resetRequested = false;
        }
        device.queue.writeBuffer(this.parameters, 0, new Float32Array([
            this.dt,
            this.buoyancy,
            this.confinement,
            Number(this.thermalRequested),
            this.mixing,
            Number(this.turbulence),
            Number(this.transportedVorticity),
            Number(this.initializeRotation)
        ]));
        this.thermalRequested = false;
        this.initializeRotation = false;
        this.dust.writeParameters();

        this.prepareBoundaries(encoder);
        this.dust.predict(encoder);
        this.dispatch(encoder, 'energy');
        this.fields.turbulence.swap();
        this.dispatch(encoder, 'atmosphericForces');
        this.dispatch(encoder, this.transportedVorticity ? 'evolveVorticity' : 'curl');
        this.fields.vorticity.swap();
        this.dispatch(encoder, 'forces');
        this.fields.velocity.swap();

        this.dispatch(encoder, 'advect');
        this.fields.velocity.swap();
        this.fields.thermo.swap();

        this.projectPressure(encoder);
        this.dust.finish(encoder, this.jacobiIterations);
        this.microphysics?.step(encoder, this.dt);
        this.dispatch(encoder, 'mixThermo');
        this.fields.thermo.swap();
        this.steps++;

        if (this.steps % 120 === 1) {
            this.checkDivergence(encoder);
            void this.dust.checkDivergence(encoder)?.catch(console.error);
        }
    }

    addThermal() {
        this.thermalRequested = true;
    }

    reset() {
        this.dust.clear();
        this.resetVersion++;
        this.resetRequested = true;
        this.initializeRotation = true;
        this.thermalRequested = false;
        this.steps = 0;
    }

    startShearExperiment() {
        const values = new Float32Array(constants.atmosphereSlices * 8);
        for (let slice = 0; slice < constants.atmosphereSlices; slice++) {
            const height = (slice + 0.5) / constants.atmosphereSlices * constants.domainY;
            const pressure = 100000 * Math.exp(-height / 8500);
            const theta = 302 + 0.001 * height;
            const temperature = theta * Math.pow(pressure / 100000, 0.2857);
            values.set([
                theta,
                14 * Math.exp(-height / 2500),
                -8 + 0.004 * height,
                6 * Math.tanh(height / 1500),
                pressure,
                temperature,
                pressure / (287.05 * temperature),
                0
            ], slice * 8);
        }
        this.weather.setAtmosphere(values);
        this.reset();
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
        } else if ([5, 6, 8, 11, 13, 15, 17, 19].includes(binding)) {
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
        const key = `${velocityIndex}/${thermoIndex}/${this.pressure.index}/${this.fields.turbulence.index}/${this.fields.vorticity.index}`;
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
            7: this.fields.vorticity.read,
            8: this.fields.vorticity.write,
            9: this.pressure.read,
            10: this.divergenceView,
            11: name === 'divergence' ? this.divergenceView : this.pressure.write,
            12: this.fields.obstacleView,
            13: this.openBitsView,
            14: this.fields.turbulence.read,
            15: this.fields.turbulence.write,
            16: this.atmosphericForceView,
            17: this.atmosphericForceView,
            18: this.fields.windForcingView,
            19: this.fields.windForcingView
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
        this.dust.destroy();
        this.parameters.destroy();
        this.pressure.textures.forEach(texture => texture.destroy());
        this.divergence.destroy();
        this.openBits.destroy();
        this.atmosphericForce.destroy();
        this.divergenceReadback?.destroy();
    }
}
