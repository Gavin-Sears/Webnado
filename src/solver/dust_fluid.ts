import { GUI } from 'lil-gui';

import { device, profiler } from '../renderer';
import { constants, processShaderRaw } from '../shaders/shaders';
import { Fields, PingPong } from '../stage/fields';
import { WeatherData } from '../weather/noaa_loader';
import { DivergenceCheck, type DivergenceReport } from './divergence_check';
import { MultigridPressure, type PressureMethod } from './multigrid_pressure';
import commonSource from './common.wgsl?raw';
import transportSource from './transport.wgsl?raw';
import projectionSource from './projection.wgsl?raw';
import dustSource from './dust.wgsl?raw';

const bindings = {
    buildDustViscosity: [0, 1, 3, 4, 11, 24, 26, 28],
    buildDustForce: [1, 3, 4, 10, 17, 23, 25, 27],
    predictDust: [0, 1, 3, 4, 5, 16],
    buildDragTime: [0, 4, 11, 22],
    coupleDust: [0, 1, 3, 4, 5, 10, 19, 20, 22],
    transportDustDensity: [0, 1, 3, 4, 20, 21],
    divergence: [1, 3, 4, 11],
    jacobi: [3, 9, 10, 11],
    project: [1, 3, 4, 5, 9],
    resetDust: [5, 21],
    clearDustPressure: [11],
    clearDustFeedback: [19]
} as const;

type KernelName = keyof typeof bindings;
interface DustKernel {
    pipeline: GPUComputePipeline;
    groups: Map<string, GPUBindGroup>;
}

const pipelineCache = new WeakMap<GPUDevice, {
    module: GPUShaderModule;
    pipelines: Partial<Record<KernelName, GPUComputePipeline>>;
}>();

function createDustStorage() {
    const pressure = new PingPong('dust pressure', 'r32float');
    const textures = ['r32float', 'rgba16float', 'rgba16float', 'rgba16float', 'rgba16float']
        .map((format, index) => device.createTexture({
            label: `dust scratch ${index}`,
            dimension: '3d',
            size: [constants.gridX, constants.gridY, constants.gridZ],
            format: format as GPUTextureFormat,
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
        }));
    const views = textures.map(texture => texture.createView());
    return { pressure, textures, divergenceView: views[0], forceView: views[1], gradientViews: views.slice(2) };
}

export class DustFluid {
    enabled = true;
    grainRadiusMicrons = 71;
    grainDensity = 490;
    dynamicViscosity = 1.285;
    loading = 0.02;
    lastDivergence: DivergenceReport | undefined;

    private allocation: ReturnType<typeof createDustStorage> | undefined;

    private get storage() {
        return this.allocation ??= createDustStorage();
    }
    private readonly kernels: Partial<Record<KernelName, DustKernel>> = {};
    private readonly readback = new URLSearchParams(location.search).has('checkDivergence')
        ? device.createBuffer({
            label: 'dust divergence readback',
            size: DivergenceCheck.bufferSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        }) : undefined;
    private readonly checker = this.readback ? new DivergenceCheck(this.readback, true) : undefined;
    private seedRequested = false;
    private resetRequested = false;
    private multigrid: MultigridPressure | undefined;

    constructor(
        readonly fields: Fields,
        private readonly weather: WeatherData,
        private readonly openBits: GPUTexture,
        private readonly parameters: GPUBuffer
    ) {}

    private kernel(name: KernelName): DustKernel {
        const existing = this.kernels[name];
        if (existing) {
            return existing;
        }
        let cached = pipelineCache.get(device);
        if (!cached) {
            cached = {
                module: device.createShaderModule({
                    label: 'two-fluid dust dynamics',
                    code: processShaderRaw([commonSource, transportSource, projectionSource, dustSource].join('\n'))
                }),
                pipelines: {}
            };
            pipelineCache.set(device, cached);
        }
        const pipeline = cached.pipelines[name] ??= device.createComputePipeline({
            label: `dust ${name}`,
            layout: 'auto',
            compute: { module: cached.module, entryPoint: name, constants: { dustPhase: 1 } }
        });
        return this.kernels[name] = { pipeline, groups: new Map() };
    }

    buildGui(gui: GUI) {
        const folder = gui.addFolder('Dust fluid');
        folder.add(this, 'enabled').name('Enabled').onChange(() => this.clear());
        folder.add(this, 'loading', 0.001, 0.2, 0.001).name('Seed density (kg/m3)');
        folder.add(this, 'grainRadiusMicrons', 10, 1000, 1).name('Grain radius (um)');
        folder.add(this, 'grainDensity', 100, 3000, 10).name('Material density (kg/m3)');
        folder.add(this, 'dynamicViscosity', 0.00001, 2, 0.00001).name('Viscosity (Pa s)');
        folder.add(this, 'seed').name('Seed ground dust');
        folder.add(this, 'clear').name('Clear dust');
    }

    seed() {
        this.enabled = true;
        this.seedRequested = true;
    }

    clear() {
        this.resetRequested = true;
        this.seedRequested = false;
    }

    writeParameters() {
        device.queue.writeBuffer(this.parameters, 32, new Float32Array([
            Number(this.enabled), this.grainRadiusMicrons * 1e-6, this.grainDensity,
            this.dynamicViscosity, Number(this.seedRequested), this.loading, 0, 0
        ]));
        this.seedRequested = false;
    }

    predict(encoder: GPUCommandEncoder) {
        if (this.enabled) {
            this.fields.enableDustFields();
        }
        if (this.resetRequested) {
            this.reset(encoder);
        }
        if (!this.enabled) {
            return;
        }
        // Reuse the divergence scratch for viscosity before the pressure solve.
        this.dispatch(encoder, 'buildDustViscosity');
        this.dispatch(encoder, 'buildDustForce');
        this.dispatch(encoder, 'predictDust');
        this.fields.dustVelocity.swap();
        this.couple(encoder);
    }

    couple(encoder: GPUCommandEncoder) {
        this.fields.enableDustFields();
        this.dispatch(encoder, 'buildDragTime');
        this.dispatch(encoder, 'coupleDust');
        this.fields.dustVelocity.swap();
    }

    finish(encoder: GPUCommandEncoder, iterations: number, method: PressureMethod = 'multigrid', cycles = 6) {
        if (!this.enabled) {
            return;
        }
        if (this.enabled) {
            this.dispatch(encoder, 'divergence');
            if (method === 'multigrid' && iterations > 0) {
                this.multigrid ??= new MultigridPressure('dust multigrid', this.storage.pressure,
                    this.openBits.createView(), this.storage.divergenceView, true);
                this.multigrid.solve(encoder, this.fields.obstacleVersion, cycles);
            } else {
                this.solveJacobi(encoder, iterations);
            }
            this.dispatch(encoder, 'project');
            this.fields.dustVelocity.swap();
        }
        this.dispatch(encoder, 'transportDustDensity');
        this.fields.dustDensity.swap();
    }

    private solveJacobi(encoder: GPUCommandEncoder, iterations: number) {
        const kernel = this.kernel('jacobi');
        const pass = encoder.beginComputePass({
            label: 'dust pressure',
            timestampWrites: profiler?.passWrites('dust pressure')
        });
        pass.setPipeline(kernel.pipeline);
        for (let iteration = 0; iteration < iterations; iteration++) {
            pass.setBindGroup(0, this.bindGroup('jacobi'));
            this.dispatchGrid(pass);
            this.storage.pressure.swap();
        }
        pass.end();
    }

    reset(encoder: GPUCommandEncoder) {
        if (this.enabled) {
            this.fields.enableDustFields();
        }
        this.dispatch(encoder, 'clearDustFeedback');
        this.dispatch(encoder, 'resetDust');
        this.fields.dustVelocity.swap();
        this.fields.dustDensity.swap();
        if (this.allocation) {
            this.dispatch(encoder, 'clearDustPressure');
            this.allocation.pressure.swap();
            this.dispatch(encoder, 'clearDustPressure');
        }
        this.resetRequested = false;
    }

    checkDivergence(encoder: GPUCommandEncoder) {
        if (!this.enabled) {
            return;
        }
        const result = this.checker?.capture(encoder, { velocity: this.fields.dustVelocity }, this.openBits, this.weather.atmosphereBuffer);
        void result?.then(report => { this.lastDivergence = report; }).catch(console.error);
        return result;
    }

    private bindGroup(name: KernelName) {
        const kernel = this.kernel(name);
        const key = `${this.fields.dustStorageVersion}/${this.fields.dustVelocity.index}/${this.fields.dustDensity.index}/${this.fields.velocity.index}/${this.allocation?.pressure.index ?? 0}`;
        const cached = kernel.groups.get(key);
        if (cached) {
            return cached;
        }
        const resources: Record<number, () => GPUBindingResource> = {
            0: () => ({ buffer: this.parameters }),
            1: () => this.fields.dustVelocity.read,
            3: () => this.openBits.createView(),
            4: () => ({ buffer: this.weather.atmosphereBuffer }),
            5: () => this.fields.dustVelocity.write,
            9: () => this.storage.pressure.read,
            10: () => this.storage.divergenceView,
            11: () => ['divergence', 'buildDustViscosity', 'buildDragTime'].includes(name)
                ? this.storage.divergenceView : this.storage.pressure.write,
            16: () => this.storage.forceView,
            17: () => this.storage.forceView,
            19: () => this.fields.windForcingView,
            20: () => this.fields.dustDensity.read,
            21: () => this.fields.dustDensity.write,
            22: () => this.fields.velocity.read,
            23: () => this.storage.gradientViews[0],
            24: () => this.storage.gradientViews[0],
            25: () => this.storage.gradientViews[1],
            26: () => this.storage.gradientViews[1],
            27: () => this.storage.gradientViews[2],
            28: () => this.storage.gradientViews[2]
        };
        const group = device.createBindGroup({
            label: `dust ${name}`,
            layout: kernel.pipeline.getBindGroupLayout(0),
            entries: bindings[name].map(binding => ({ binding, resource: resources[binding]() }))
        });
        kernel.groups.set(key, group);
        return group;
    }

    private dispatch(encoder: GPUCommandEncoder, name: KernelName) {
        const kernel = this.kernel(name);
        const pass = encoder.beginComputePass({ label: `dust ${name}`, timestampWrites: profiler?.passWrites(`dust ${name}`) });
        pass.setPipeline(kernel.pipeline);
        pass.setBindGroup(0, this.bindGroup(name));
        this.dispatchGrid(pass);
        pass.end();
    }

    private dispatchGrid(pass: GPUComputePassEncoder) {
        pass.dispatchWorkgroups(constants.gridX / 8, constants.gridY / 8, constants.gridZ / 4);
    }

    destroy() {
        this.multigrid?.destroy();
        this.allocation?.pressure.textures.forEach(texture => texture.destroy());
        this.allocation?.textures.forEach(texture => texture.destroy());
        this.readback?.destroy();
    }
}
