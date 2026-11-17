import { device, profiler } from '../renderer';
import { constants, processShaderRaw } from '../shaders/shaders';
import { Fields } from '../stage/fields';
import { WeatherData } from '../weather/noaa_loader';
import type { Microphysics } from './fluid_solver';
import commonSource from './common.wgsl?raw';
import microphysicsSource from './microphysics.wgsl?raw';

const names = ['microphysics', 'sedimentRain'] as const;
const pipelineCache = new WeakMap<GPUDevice, GPUComputePipeline[]>();

export class WarmRain implements Microphysics {
    enabled = true;
    rainDrain = false;

    private readonly parameters = device.createBuffer({
        label: 'warm-rain parameters',
        size: 64,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    private readonly pipelines: GPUComputePipeline[];
    private readonly groups = new Map<string, GPUBindGroup>();

    constructor(
        private readonly fields: Fields,
        private readonly weather: WeatherData,
        private readonly openBits: GPUTextureView
    ) {
        let pipelines = pipelineCache.get(device);
        if (!pipelines) {
            const module = device.createShaderModule({
                label: 'warm-rain microphysics',
                code: processShaderRaw(commonSource + '\n' + microphysicsSource)
            });
            pipelines = names.map(name => device.createComputePipeline({
                label: name,
                layout: 'auto',
                compute: { module, entryPoint: name }
            }));
            pipelineCache.set(device, pipelines);
        }
        this.pipelines = pipelines;
    }

    step(encoder: GPUCommandEncoder, dt: number) {
        if (!this.enabled) {
            return;
        }
        device.queue.writeBuffer(this.parameters, 0, new Float32Array([
            dt, 0, 0, 0, 0, 0, 0, Number(this.rainDrain)
        ]));
        for (const [index, pipeline] of this.pipelines.entries()) {
            const key = `${index}/${this.fields.thermo.index}`;
            let group = this.groups.get(key);
            if (!group) {
                group = device.createBindGroup({
                    label: names[index],
                    layout: pipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: this.parameters } },
                        { binding: 2, resource: this.fields.thermo.read },
                        { binding: 3, resource: this.openBits },
                        { binding: 4, resource: { buffer: this.weather.atmosphereBuffer } },
                        { binding: 6, resource: this.fields.thermo.write }
                    ]
                });
                this.groups.set(key, group);
            }
            const pass = encoder.beginComputePass({
                label: names[index],
                timestampWrites: profiler?.passWrites(names[index])
            });
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, group);
            pass.dispatchWorkgroups(constants.gridX / 8, constants.gridY / 8, constants.gridZ / 4);
            pass.end();
            this.fields.thermo.swap();
        }
    }

    destroy() {
        this.parameters.destroy();
    }
}
