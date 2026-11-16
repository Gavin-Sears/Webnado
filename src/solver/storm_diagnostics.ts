import { device } from '../renderer';
import { constants, processShaderRaw } from '../shaders/shaders';
import type { Fields } from '../stage/fields';
import source from './storm_diagnostics.wgsl?raw';

export interface StormReport {
    seconds: number;
    updraft: number;
    speed: number;
    verticalVorticity: number;
    surfaceVorticity: number;
    cloud: number;
    rain: number;
    temperature: number;
    invalid: number;
}

export class StormDiagnostics {
    private readonly groups = Math.ceil(constants.gridX * constants.gridY * constants.gridZ / 256);
    private readonly size = this.groups * 32;
    private readonly output = device.createBuffer({
        size: this.size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    private readonly readback = device.createBuffer({
        size: this.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    private readonly pipeline = device.createComputePipeline({
        label: 'storm diagnostics', layout: 'auto',
        compute: { module: device.createShaderModule({ code: processShaderRaw(source) }), entryPoint: 'main' }
    });

    async measure(fields: Fields, seconds: number): Promise<StormReport> {
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: fields.velocity.read },
                { binding: 1, resource: fields.thermo.read },
                { binding: 2, resource: fields.obstacleView },
                { binding: 3, resource: { buffer: this.output } }
            ]
        }));
        pass.dispatchWorkgroups(this.groups);
        pass.end();
        encoder.copyBufferToBuffer(this.output, 0, this.readback, 0, this.size);
        device.queue.submit([encoder.finish()]);
        await this.readback.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(this.readback.getMappedRange());
        const maxima = new Float32Array(8);
        for (let index = 0; index < data.length; index++) {
            maxima[index % 8] = Math.max(maxima[index % 8], data[index]);
        }
        this.readback.unmap();
        const [updraft, speed, verticalVorticity, surfaceVorticity, cloud, rain, temperature, invalid] = maxima;
        return { seconds, updraft, speed, verticalVorticity, surfaceVorticity, cloud, rain, temperature, invalid };
    }

    destroy() {
        this.output.destroy();
        this.readback.destroy();
    }
}
