import { device, profiler } from '../renderer';
import { constants, processShaderRaw } from '../shaders/shaders';
import { PingPong } from '../stage/fields';
import source from './multigrid.wgsl?raw';

export type PressureMethod = 'multigrid' | 'jacobi';

const bindings = {
    buildFineGraph: [0, 7, 8, 13],
    coarsenGraph: [0, 7, 8, 9, 10],
    initializeRhs: [0, 1, 6, 14],
    relaxPressure: [0, 1, 2, 3, 4, 5],
    residual: [0, 1, 2, 3, 4, 6],
    restrictResidual: [0, 1, 5, 6, 9, 11],
    prolongCorrection: [0, 1, 3, 5, 12, 15],
    solveCoarsest: [0, 1, 2, 4, 5]
} as const;

type KernelName = keyof typeof bindings;
const pipelineCache = new WeakMap<GPUDevice, Record<KernelName, GPUComputePipeline>>();

interface Level {
    dims: [number, number, number];
    parameters: GPUBuffer;
    pressure: PingPong;
    textures: GPUTexture[];
    edges: GPUTextureView;
    diagonal: GPUTextureView;
    rhs: GPUTextureView;
    residual: GPUTextureView;
    groups: Map<string, GPUBindGroup>;
}

export class MultigridPressure {
    private readonly levels: Level[] = [];
    private readonly pipelines: Record<KernelName, GPUComputePipeline>;
    private obstacleVersion = -1;

    constructor(
        private readonly label: string,
        pressure: PingPong,
        private readonly openBits: GPUTextureView,
        private readonly divergence: GPUTextureView,
        periodic = false
    ) {
        let pipelines = pipelineCache.get(device);
        if (!pipelines) {
            const module = device.createShaderModule({
                label: 'multigrid pressure',
                code: processShaderRaw(source)
            });
            pipelines = {} as Record<KernelName, GPUComputePipeline>;
            for (const name of Object.keys(bindings) as KernelName[]) {
                pipelines[name] = device.createComputePipeline({
                    label: `multigrid ${name}`,
                    layout: 'auto',
                    compute: { module, entryPoint: name }
                });
            }
            pipelineCache.set(device, pipelines);
        }
        this.pipelines = pipelines;

        let dims: [number, number, number] = [constants.gridX, constants.gridY, constants.gridZ];
        while (true) {
            const fineDims = this.levels.at(-1)?.dims ?? dims;
            const parameters = device.createBuffer({
                label: `${label} multigrid dimensions`,
                size: 32,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            });
            device.queue.writeBuffer(parameters, 0, new Int32Array([
                ...dims, Number(periodic), ...fineDims, 0
            ]));
            const textures = ['rgba32float', 'r32float', 'r32float', 'r32float'].map((format, index) =>
                device.createTexture({
                    label: `${label} multigrid level ${this.levels.length} field ${index}`,
                    dimension: '3d',
                    size: dims,
                    format: format as GPUTextureFormat,
                    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
                })
            );
            const [edges, diagonal, rhs, residual] = textures.map(texture => texture.createView());
            this.levels.push({
                dims, parameters, textures, edges, diagonal, rhs, residual,
                pressure: this.levels.length === 0 ? pressure : new PingPong(`${label} coarse pressure`, 'r32float', dims),
                groups: new Map()
            });
            if (dims[0] * dims[1] * dims[2] <= 256) {
                break;
            }
            dims = dims.map(value => value / 2) as [number, number, number];
        }
    }

    solve(encoder: GPUCommandEncoder, obstacleVersion: number, cycles: number) {
        const pass = encoder.beginComputePass({
            label: this.label,
            timestampWrites: profiler?.passWrites(this.label)
        });
        if (this.obstacleVersion !== obstacleVersion) {
            this.dispatch(pass, 'buildFineGraph', 0);
            for (let level = 1; level < this.levels.length; level++) {
                this.dispatch(pass, 'coarsenGraph', level);
            }
            this.obstacleVersion = obstacleVersion;
        }
        this.dispatch(pass, 'initializeRhs', 0);
        for (let cycle = 0; cycle < cycles; cycle++) {
            this.cycle(pass, 0);
        }
        pass.end();
    }

    private cycle(pass: GPUComputePassEncoder, index: number) {
        const level = this.levels[index];
        if (index === this.levels.length - 1) {
            this.dispatch(pass, 'solveCoarsest', index);
            level.pressure.swap();
            return;
        }
        this.smooth(pass, index);
        this.dispatch(pass, 'residual', index);
        this.dispatch(pass, 'restrictResidual', index + 1);
        this.levels[index + 1].pressure.swap();
        this.cycle(pass, index + 1);
        this.dispatch(pass, 'prolongCorrection', index);
        level.pressure.swap();
        this.smooth(pass, index);
    }

    private smooth(pass: GPUComputePassEncoder, index: number) {
        for (let iteration = 0; iteration < 3; iteration++) {
            this.dispatch(pass, 'relaxPressure', index);
            this.levels[index].pressure.swap();
        }
    }

    private dispatch(pass: GPUComputePassEncoder, name: KernelName, index: number) {
        const level = this.levels[index];
        const fine = this.levels[Math.max(0, index - 1)];
        const coarse = this.levels[Math.min(this.levels.length - 1, index + 1)];
        const key = `${name}/${level.pressure.index}/${coarse.pressure.index}`;
        const pipeline = this.pipelines[name];
        let group = level.groups.get(key);
        if (!group) {
            const resources: Record<number, GPUBindingResource> = {
                0: { buffer: level.parameters },
                1: level.edges,
                2: level.diagonal,
                3: level.pressure.read,
                4: level.rhs,
                5: level.pressure.write,
                6: name === 'residual' ? level.residual : level.rhs,
                7: level.edges,
                8: level.diagonal,
                9: fine.edges,
                10: fine.diagonal,
                11: fine.residual,
                12: coarse.pressure.read,
                13: this.openBits,
                14: this.divergence,
                15: coarse.edges
            };
            group = device.createBindGroup({
                label: `${this.label} ${name} level ${index}`,
                layout: pipeline.getBindGroupLayout(0),
                entries: bindings[name].map(binding => ({ binding, resource: resources[binding] }))
            });
            level.groups.set(key, group);
        }
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, group);
        if (name === 'solveCoarsest') {
            pass.dispatchWorkgroups(1);
        } else {
            pass.dispatchWorkgroups(Math.ceil(level.dims[0] / 8), Math.ceil(level.dims[1] / 8), Math.ceil(level.dims[2] / 4));
        }
    }

    destroy() {
        for (const [index, level] of this.levels.entries()) {
            level.parameters.destroy();
            level.textures.forEach(texture => texture.destroy());
            if (index > 0) {
                level.pressure.textures.forEach(texture => texture.destroy());
            }
        }
    }
}

