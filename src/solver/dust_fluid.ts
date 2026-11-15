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
    clearDustPressure: [11]
} as const;

type KernelName = keyof typeof bindings;
const pipelineCache = new WeakMap<GPUDevice, Record<KernelName, GPUComputePipeline>>();

export class DustFluid {
    enabled = true;
    grainRadiusMicrons = 71;
    grainDensity = 490;
    dynamicViscosity = 1.285;
    loading = 0.02;
    lastDivergence: DivergenceReport | undefined;

    private readonly pressure = new PingPong('dust pressure', 'r32float');
    private readonly divergence = device.createTexture({
        label: 'dust divergence',
        dimension: '3d',
        size: [constants.gridX, constants.gridY, constants.gridZ],
        format: 'r32float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
    });
    private readonly divergenceView = this.divergence.createView();
    private readonly force = device.createTexture({
        label: 'dust force',
        dimension: '3d',
        size: [constants.gridX, constants.gridY, constants.gridZ],
        format: 'rgba16float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
    });
    private readonly forceView = this.force.createView();
    private readonly gradients = ['x', 'y', 'z'].map(axis => device.createTexture({
        label: `dust velocity derivative ${axis}`,
        dimension: '3d',
        size: [constants.gridX, constants.gridY, constants.gridZ],
        format: 'rgba16float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
    }));
    private readonly gradientViews = this.gradients.map(texture => texture.createView());
    private readonly kernels = {} as Record<KernelName, {
        pipeline: GPUComputePipeline;
        groups: Map<string, GPUBindGroup>;
    }>;
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
    ) {
        const names = Object.keys(bindings) as KernelName[];
        let pipelines = pipelineCache.get(device);
        if (!pipelines) {
            const module = device.createShaderModule({
                label: 'two-fluid dust dynamics',
                code: processShaderRaw([commonSource, transportSource, projectionSource, dustSource].join('\n'))
            });
            pipelines = {} as Record<KernelName, GPUComputePipeline>;
            for (const name of names) {
                pipelines[name] = device.createComputePipeline({
                    label: `dust ${name}`,
                    layout: 'auto',
                    compute: { module, entryPoint: name, constants: { dustPhase: 1 } }
                });
            }
            pipelineCache.set(device, pipelines);
        }
        for (const name of names) {
            this.kernels[name] = {
                pipeline: pipelines[name],
                groups: new Map()
            };
        }
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
        if (this.resetRequested) {
            this.reset(encoder);
        }
        // Reuse the divergence scratch for viscosity before the pressure solve.
        this.dispatch(encoder, 'buildDustViscosity');
        this.dispatch(encoder, 'buildDustForce');
        this.dispatch(encoder, 'predictDust');
        this.fields.dustVelocity.swap();
        this.couple(encoder);
    }

    couple(encoder: GPUCommandEncoder) {
        this.dispatch(encoder, 'buildDragTime');
        this.dispatch(encoder, 'coupleDust');
        this.fields.dustVelocity.swap();
    }

    finish(encoder: GPUCommandEncoder, iterations: number, method: PressureMethod = 'multigrid', cycles = 6) {
        if (this.enabled) {
            this.dispatch(encoder, 'divergence');
            if (method === 'multigrid' && iterations > 0) {
                this.multigrid ??= new MultigridPressure('dust multigrid', this.pressure, this.openBits.createView(), this.divergenceView, true);
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
        const kernel = this.kernels.jacobi;
        const pass = encoder.beginComputePass({
            label: 'dust pressure',
            timestampWrites: profiler?.passWrites('dust pressure')
        });
        pass.setPipeline(kernel.pipeline);
        for (let iteration = 0; iteration < iterations; iteration++) {
            pass.setBindGroup(0, this.bindGroup('jacobi'));
            this.dispatchGrid(pass);
            this.pressure.swap();
        }
        pass.end();
    }

    reset(encoder: GPUCommandEncoder) {
        this.dispatch(encoder, 'resetDust');
        this.fields.dustVelocity.swap();
        this.fields.dustDensity.swap();
        this.dispatch(encoder, 'clearDustPressure');
        this.pressure.swap();
        this.dispatch(encoder, 'clearDustPressure');
        this.resetRequested = false;
    }

    checkDivergence(encoder: GPUCommandEncoder) {
        const result = this.checker?.capture(encoder, { velocity: this.fields.dustVelocity }, this.openBits, this.weather.atmosphereBuffer);
        void result?.then(report => { this.lastDivergence = report; }).catch(console.error);
        return result;
    }

    private bindGroup(name: KernelName) {
        const kernel = this.kernels[name];
        const key = `${this.fields.dustVelocity.index}/${this.fields.dustDensity.index}/${this.fields.velocity.index}/${this.pressure.index}`;
        const cached = kernel.groups.get(key);
        if (cached) {
            return cached;
        }
        const resources: Record<number, GPUBindingResource> = {
            0: { buffer: this.parameters },
            1: this.fields.dustVelocity.read,
            3: this.openBits.createView(),
            4: { buffer: this.weather.atmosphereBuffer },
            5: this.fields.dustVelocity.write,
            9: this.pressure.read,
            10: this.divergenceView,
            11: ['divergence', 'buildDustViscosity', 'buildDragTime'].includes(name) ? this.divergenceView : this.pressure.write,
            16: this.forceView,
            17: this.forceView,
            19: this.fields.windForcingView,
            20: this.fields.dustDensity.read,
            21: this.fields.dustDensity.write,
            22: this.fields.velocity.read,
            23: this.gradientViews[0],
            24: this.gradientViews[0],
            25: this.gradientViews[1],
            26: this.gradientViews[1],
            27: this.gradientViews[2],
            28: this.gradientViews[2]
        };
        const group = device.createBindGroup({
            label: `dust ${name}`,
            layout: kernel.pipeline.getBindGroupLayout(0),
            entries: bindings[name].map(binding => ({ binding, resource: resources[binding] }))
        });
        kernel.groups.set(key, group);
        return group;
    }

    private dispatch(encoder: GPUCommandEncoder, name: KernelName) {
        const kernel = this.kernels[name];
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
        this.pressure.textures.forEach(texture => texture.destroy());
        this.divergence.destroy();
        this.force.destroy();
        this.gradients.forEach(texture => texture.destroy());
        this.readback?.destroy();
    }
}
