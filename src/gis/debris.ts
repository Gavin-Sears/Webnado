import { GUI } from 'lil-gui';

import { canvas, canvasFormat, context, device, profiler } from '../renderer';
import { processShaderRaw } from '../shaders/shaders';
import { Camera } from '../stage/camera';
import { Fields } from '../stage/fields';
import source from './debris.wgsl?raw';

export class Debris {
    readonly count = 4096;
    readonly particles = device.createBuffer({
        label: 'debris particles',
        size: this.count * 32,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });

    enabled = true;
    restitution = 0.15;
    friction = 0.5;
    size = 2;

    private readonly parameters = device.createBuffer({
        label: 'debris parameters',
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    private readonly updatePipeline: GPUComputePipeline;
    private readonly renderPipeline: GPURenderPipeline;
    private updateGroups: GPUBindGroup[];
    private dustStorageVersion: number;
    private renderGroup: GPUBindGroup | undefined;
    private depthTexture: GPUTexture | undefined;
    private spawnRequested = false;
    private clearRequested = false;
    private resetVersion = 0;

    constructor(private readonly fields: Fields, private readonly camera: Camera, private readonly seedDust: () => void) {
        const module = device.createShaderModule({ label: 'debris', code: processShaderRaw(source) });
        this.updatePipeline = device.createComputePipeline({
            label: 'debris update',
            layout: 'auto',
            compute: { module, entryPoint: 'updateDebris' }
        });
        this.renderPipeline = device.createRenderPipeline({
            label: 'debris render',
            layout: 'auto',
            vertex: { module, entryPoint: 'debrisVertex' },
            fragment: { module, entryPoint: 'debrisFragment', targets: [{ format: canvasFormat }] }
        });
        this.updateGroups = this.createUpdateGroups();
        this.dustStorageVersion = fields.dustStorageVersion;
    }

    private createUpdateGroups() {
        return [0, 1].map(index => device.createBindGroup({
            layout: this.updatePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.parameters } },
                { binding: 1, resource: { buffer: this.particles } },
                { binding: 2, resource: this.fields.dustVelocity.view(index) },
                { binding: 3, resource: this.fields.obstacleView }
            ]
        }));
    }

    buildGui(gui: GUI) {
        const folder = gui.addFolder('Dust tracers');
        folder.add(this, 'enabled').name('Enabled');
        folder.add(this, 'restitution', 0, 0.8, 0.05).name('Bounce');
        folder.add(this, 'friction', 0, 1, 0.05).name('Friction');
        folder.add(this, 'size', 1, 6, 0.5).name('Display size (px)');
        folder.add(this, 'seed').name('Seed dust and tracers');
        folder.add(this, 'clear').name('Clear tracers');
    }

    seed() {
        this.seedDust();
        this.spawnRequested = true;
    }

    clear() {
        this.clearRequested = true;
        this.spawnRequested = false;
    }

    step(encoder: GPUCommandEncoder, dt: number, resetVersion: number) {
        if (this.dustStorageVersion !== this.fields.dustStorageVersion) {
            this.updateGroups = this.createUpdateGroups();
            this.dustStorageVersion = this.fields.dustStorageVersion;
        }
        if (resetVersion !== this.resetVersion) {
            this.clearRequested = true;
            this.resetVersion = resetVersion;
        }
        if (this.clearRequested) {
            encoder.clearBuffer(this.particles);
            this.clearRequested = false;
        }
        const data = new ArrayBuffer(32);
        new Float32Array(data).set([
            dt, this.restitution, this.friction, this.size, 0, 0,
            canvas.width, canvas.height
        ]);
        new Uint32Array(data).set([Number(this.spawnRequested), Number(this.enabled)], 4);
        device.queue.writeBuffer(this.parameters, 0, data);
        this.spawnRequested = false;

        const pass = encoder.beginComputePass({ label: 'debris', timestampWrites: profiler?.passWrites('debris') });
        pass.setPipeline(this.updatePipeline);
        pass.setBindGroup(0, this.updateGroups[this.fields.dustVelocity.index]);
        pass.dispatchWorkgroups(this.count / 64);
        pass.end();
    }

    draw(encoder: GPUCommandEncoder, depth: GPUTexture) {
        if (depth !== this.depthTexture) {
            this.renderGroup = device.createBindGroup({
                layout: this.renderPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.parameters } },
                    { binding: 6, resource: { buffer: this.camera.uniformsBuffer } },
                    { binding: 7, resource: depth.createView() },
                    { binding: 9, resource: { buffer: this.particles } }
                ]
            });
            this.depthTexture = depth;
        }
        const pass = encoder.beginRenderPass({
            label: 'debris render',
            colorAttachments: [{ view: context.getCurrentTexture().createView(), loadOp: 'load', storeOp: 'store' }]
        });
        pass.setPipeline(this.renderPipeline);
        pass.setBindGroup(0, this.renderGroup!);
        pass.draw(6, this.count);
        pass.end();
    }

    destroy() {
        this.particles.destroy();
        this.parameters.destroy();
    }
}
