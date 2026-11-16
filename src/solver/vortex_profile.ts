import { device } from '../renderer';
import { processShaderRaw } from '../shaders/shaders';
import type { Fields } from '../stage/fields';
import source from './vortex_profile.wgsl?raw';

export interface VortexRing {
    radius: number;
    tangential: number;
    outward: number;
    updraft: number;
}

export class VortexProfile {
    private readonly count = 12;
    private readonly center = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    private readonly output = device.createBuffer({
        size: this.count * 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    private readonly readback = device.createBuffer({
        size: this.count * 16,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    private readonly pipeline = device.createComputePipeline({
        label: 'vortex ring winds',
        layout: 'auto',
        compute: {
            module: device.createShaderModule({ code: processShaderRaw(source) }),
            entryPoint: 'main'
        }
    });

    async measure(fields: Fields, position: number[]): Promise<VortexRing[]> {
        device.queue.writeBuffer(this.center, 0, new Float32Array([...position, 0]));
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: fields.velocity.read },
                { binding: 1, resource: { buffer: this.center } },
                { binding: 2, resource: { buffer: this.output } }
            ]
        }));
        pass.dispatchWorkgroups(this.count);
        pass.end();
        encoder.copyBufferToBuffer(this.output, 0, this.readback, 0, this.count * 16);
        device.queue.submit([encoder.finish()]);
        await this.readback.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(this.readback.getMappedRange());
        const result: VortexRing[] = [];
        for (let index = 0; index < this.count; index++) {
            const [tangential, outward, updraft, radius] = data.subarray(index * 4, index * 4 + 4);
            if (radius > 0) {
                result.push({ radius, tangential, outward, updraft });
            }
        }
        this.readback.unmap();
        return result;
    }

    destroy() {
        this.center.destroy();
        this.output.destroy();
        this.readback.destroy();
    }
}
