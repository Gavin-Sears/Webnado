import { GUI } from 'lil-gui';

import { canvas, canvasFormat, context, device, profiler } from '../renderer';
import { constants, processShaderRaw } from '../shaders/shaders';
import { Camera } from '../stage/camera';
import { Fields } from '../stage/fields';
import { WeatherData } from '../weather/noaa_loader';
import source from './debris.wgsl?raw';

export class Debris {
    readonly count = 4096;
    readonly particles = device.createBuffer({
        label: 'debris particles',
        size: this.count * 32,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });

    enabled = true;
    response = 0.2;
    restitution = 0.15;
    friction = 0.5;
    mass = 0.02;
    size = 2;

    private readonly parameters = device.createBuffer({
        label: 'debris parameters',
        size: 48,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    private readonly momentum = device.createBuffer({
        label: 'debris momentum transfer',
        size: constants.gridX * constants.gridY * constants.gridZ * 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    private readonly updatePipeline: GPUComputePipeline;
    private readonly feedbackPipeline: GPUComputePipeline;
    private readonly renderPipeline: GPURenderPipeline;
    private readonly updateGroups: GPUBindGroup[];
    private readonly feedbackGroup: GPUBindGroup;
    private renderGroup: GPUBindGroup | undefined;
    private depthTexture: GPUTexture | undefined;
    private spawnRequested = false;
    private clearRequested = false;
    private resetVersion = 0;

    constructor(private readonly fields: Fields, weather: WeatherData, private readonly camera: Camera) {
        const module = device.createShaderModule({ label: 'debris', code: processShaderRaw(source) });
        this.updatePipeline = device.createComputePipeline({
            label: 'debris update',
            layout: 'auto',
            compute: { module, entryPoint: 'updateDebris' }
        });
        this.feedbackPipeline = device.createComputePipeline({
            label: 'debris feedback',
            layout: 'auto',
            compute: { module, entryPoint: 'resolveFeedback' }
        });
        this.renderPipeline = device.createRenderPipeline({
            label: 'debris render',
            layout: 'auto',
            vertex: { module, entryPoint: 'debrisVertex' },
            fragment: { module, entryPoint: 'debrisFragment', targets: [{ format: canvasFormat }] }
        });
        this.updateGroups = [0, 1].map(index => device.createBindGroup({
            layout: this.updatePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.parameters } },
                { binding: 1, resource: { buffer: this.particles } },
                { binding: 2, resource: fields.velocity.view(index) },
                { binding: 3, resource: fields.obstacleView },
                { binding: 4, resource: { buffer: this.momentum } }
            ]
        }));
        this.feedbackGroup = device.createBindGroup({
            layout: this.feedbackPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.parameters } },
                { binding: 4, resource: { buffer: this.momentum } },
                { binding: 5, resource: fields.windForcingView },
                { binding: 8, resource: { buffer: weather.atmosphereBuffer } }
            ]
        });
    }

    buildGui(gui: GUI) {
        const folder = gui.addFolder('Debris');
        folder.add(this, 'enabled').name('Enabled');
        folder.add(this, 'response', 0.02, 2, 0.02).name('Drag response (s)');
        folder.add(this, 'mass', 0.001, 1, 0.001).name('Seed mass (kg)');
        folder.add(this, 'restitution', 0, 0.8, 0.05).name('Bounce');
        folder.add(this, 'friction', 0, 1, 0.05).name('Friction');
        folder.add(this, 'size', 1, 6, 0.5).name('Display size (px)');
        folder.add(this, 'seed').name('Seed debris');
        folder.add(this, 'clear').name('Clear debris');
    }

    seed() {
        this.spawnRequested = true;
    }

    clear() {
        this.clearRequested = true;
        this.spawnRequested = false;
    }

    step(encoder: GPUCommandEncoder, dt: number, resetVersion: number) {
        if (resetVersion !== this.resetVersion) {
            this.clearRequested = true;
            this.resetVersion = resetVersion;
        }
        if (this.clearRequested) {
            encoder.clearBuffer(this.particles);
            this.clearRequested = false;
        }
        const data = new ArrayBuffer(48);
        new Float32Array(data).set([
            dt, this.response, this.restitution, this.friction, this.mass, this.size, 0, 0,
            canvas.width, canvas.height
        ]);
        new Uint32Array(data).set([Number(this.spawnRequested), Number(this.enabled)], 6);
        device.queue.writeBuffer(this.parameters, 0, data);
        this.spawnRequested = false;
        encoder.clearBuffer(this.momentum);

        const pass = encoder.beginComputePass({ label: 'debris', timestampWrites: profiler?.passWrites('debris') });
        pass.setPipeline(this.updatePipeline);
        pass.setBindGroup(0, this.updateGroups[this.fields.velocity.index]);
        pass.dispatchWorkgroups(this.count / 64);
        pass.setPipeline(this.feedbackPipeline);
        pass.setBindGroup(0, this.feedbackGroup);
        pass.dispatchWorkgroups(constants.gridX / 8, constants.gridY / 8, constants.gridZ / 4);
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
        this.momentum.destroy();
    }
}
