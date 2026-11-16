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
    interiorVerticalVorticity: number;
    divergenceRms: number;
    divergenceMax: number;
    surfaceLocation: number[];
    lowestSampleHeight: number;
    updraftLocation: number[];
    rotatingUpdraftScore: number;
    groundVorticity: number;
    groundLocation: number[];
    groundSwirlingStrength: number;
    swirlLocation: number[];
}

export class StormDiagnostics {
    private readonly groups = Math.ceil(constants.gridX * constants.gridY * constants.gridZ / 128);
    private readonly size = this.groups * 112;
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
        pass.dispatchWorkgroups(Math.min(this.groups, 256), Math.ceil(this.groups / 256));
        pass.end();
        encoder.copyBufferToBuffer(this.output, 0, this.readback, 0, this.size);
        device.queue.submit([encoder.finish()]);
        await this.readback.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(this.readback.getMappedRange());
        const maxima = new Float32Array(8);
        let squared = 0;
        let count = 0;
        let divergenceMax = 0;
        let interiorVerticalVorticity = 0;
        let surfacePeak = 0;
        let surfaceLocation = [0, 0, 0];
        let updraftLocation = [0, 0, 0];
        let rotatingUpdraftScore = 0;
        let groundVorticity = 0;
        let groundLocation = [0, 0, 0];
        let groundSwirlingStrength = 0;
        let swirlLocation = [0, 0, 0];
        for (let group = 0; group < this.groups; group++) {
            const offset = group * 28;
            for (let channel = 0; channel < 8; channel++) {
                maxima[channel] = Math.max(maxima[channel], data[offset + channel]);
            }
            squared += data[offset + 8];
            divergenceMax = Math.max(divergenceMax, data[offset + 9]);
            count += data[offset + 10];
            interiorVerticalVorticity = Math.max(interiorVerticalVorticity, data[offset + 11]);
            if (data[offset + 3] > surfacePeak) {
                surfacePeak = data[offset + 3];
                surfaceLocation = Array.from(data.subarray(offset + 12, offset + 15));
            }
            if (data[offset + 19] > rotatingUpdraftScore) {
                rotatingUpdraftScore = data[offset + 19];
                updraftLocation = Array.from(data.subarray(offset + 16, offset + 19));
            }
            if (data[offset + 23] > groundVorticity) {
                groundVorticity = data[offset + 23];
                groundLocation = Array.from(data.subarray(offset + 20, offset + 23));
            }
            if (data[offset + 27] > groundSwirlingStrength) {
                groundSwirlingStrength = data[offset + 27];
                swirlLocation = Array.from(data.subarray(offset + 24, offset + 27));
            }
        }
        this.readback.unmap();
        const [updraft, speed, verticalVorticity, surfaceVorticity, cloud, rain, temperature, invalid] = maxima;
        return {
            seconds, updraft, speed, verticalVorticity, surfaceVorticity, cloud, rain, temperature, invalid,
            interiorVerticalVorticity, divergenceRms: Math.sqrt(squared / Math.max(count, 1)),
            divergenceMax, surfaceLocation, updraftLocation, rotatingUpdraftScore,
            groundVorticity, groundLocation,
            groundSwirlingStrength, swirlLocation,
            lowestSampleHeight: (constants.freeSlipGround ? 0.5 : 1.5) * constants.domainY / constants.gridY
        };
    }

    destroy() {
        this.output.destroy();
        this.readback.destroy();
    }
}
