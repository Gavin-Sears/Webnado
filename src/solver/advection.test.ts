import { device, initWebGPU } from '../renderer';
import { constants, domainName, resolutionName, processShaderRaw } from '../shaders/shaders';
import { Fields } from '../stage/fields';
import { WeatherData } from '../weather/noaa_loader';
import { ObstacleGrid } from '../gis/obstacle_grid';
import { FluidSolver } from './fluid_solver';
import { halfToFloat } from './divergence_check';

const output = document.getElementById('results')!;
const { gridX, gridY, gridZ } = constants;

function expect(condition: boolean, message: string) {
    if (!condition) {
        throw new Error(message);
    }
    output.textContent += `\nPASS ${message}`;
}

async function cloudStats(fields: Fields) {
    const row = gridX * 8;
    const size = row * gridY * gridZ;
    const buffer = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture: fields.thermo.textures[fields.thermo.index] },
        { buffer, bytesPerRow: row, rowsPerImage: gridY }, [gridX, gridY, gridZ]);
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const data = new DataView(buffer.getMappedRange());
    let mass = 0;
    let moment = 0;
    let peak = 0;
    for (let index = 0; index < gridX * gridY * gridZ; index++) {
        const cloud = halfToFloat(data.getUint16(index * 8 + 4, true));
        if (!Number.isFinite(cloud) || cloud < 0) {
            throw new Error('Transport produced invalid cloud water.');
        }
        const x = ((index % gridX) + 0.5) * constants.domainX / gridX - constants.domainX / 2;
        mass += cloud;
        moment += cloud * x;
        peak = Math.max(peak, cloud);
    }
    buffer.unmap();
    buffer.destroy();
    return { mass, center: moment / mass, peak };
}

async function run() {
    expect(domainName === 'local' && resolutionName === 'low', 'The accuracy fixture uses low/local');
    await initWebGPU();
    device.pushErrorScope('validation');
    const fields = new Fields();
    fields.setObstacles(new ObstacleGrid().materials);
    const weather = new WeatherData({ latitude: 0, longitude: 0 });
    const column = new Float32Array(constants.atmosphereSlices * 8);
    for (let slice = 0; slice < constants.atmosphereSlices; slice++) {
        column.set([300, 10, 20, 0, 100000, 300, 1.16, 0], slice * 8);
    }
    weather.setAtmosphere(column);
    const solver = new FluidSolver(fields, weather);
    solver.dt = 1;
    solver.buoyancy = 0;
    solver.confinement = 0;
    solver.turbulence = false;
    solver.dust.enabled = false;
    solver.warmRain.enabled = false;
    solver.storm.liftSpeed = 0;
    const pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
            entryPoint: 'main',
            module: device.createShaderModule({ code: processShaderRaw(`
                @group(0) @binding(0) var cloudState: texture_storage_3d<rgba16float, write>;
                @compute @workgroup_size(8, 8, 4)
                fn main(@builtin(global_invocation_id) id: vec3u) {
                    let world = domainMin + (vec3f(id) + vec3f(0.5)) * cellSize;
                    let displacement = (world - vec3f(-700, 800, 0)) / 150.0;
                    let cloud = exp(-0.5 * dot(displacement, displacement));
                    textureStore(cloudState, vec3i(id), vec4f(0, 0, cloud, 0));
                }
            `) })
        }
    });

    const transport = async (highOrder: boolean) => {
        solver.reset();
        let encoder = device.createCommandEncoder();
        solver.step(encoder);
        device.queue.submit([encoder.finish()]);
        solver.storm.enabled = true;
        solver.storm.highOrder = highOrder;
        encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: fields.thermo.read }]
        }));
        pass.dispatchWorkgroups(gridX / 8, gridY / 8, gridZ / 4);
        pass.end();
        device.queue.submit([encoder.finish()]);
        const initial = await cloudStats(fields);
        const seedError = await device.popErrorScope();
        if (seedError) {
            throw new Error(seedError.message);
        }
        device.pushErrorScope('validation');
        for (let step = 0; step < 50; step++) {
            encoder = device.createCommandEncoder();
            solver.step(encoder);
            device.queue.submit([encoder.finish()]);
        }
        return { initial, final: await cloudStats(fields) };
    };

    try {
        const first = await transport(false);
        const corrected = await transport(true);
        console.log('Advection comparison ' + JSON.stringify({ first, corrected }));
        expect(corrected.final.peak > first.final.peak * 1.1,
            'Limited MacCormack retains more of a transported smooth cloud');
        expect(Math.abs(corrected.final.center - 300) < 40,
            'The cloud follows the prescribed wind displacement');
        expect(corrected.final.peak <= corrected.initial.peak + 0.002,
            'The limiter prevents a new cloud-water maximum');
        expect(Math.abs(corrected.final.mass / corrected.initial.mass - 1) < 0.05,
            'Cloud mass stays within the accuracy fixture tolerance');
        fields.setObstacles(new Uint8Array(gridX * gridY * gridZ));
        solver.storm.followStorm = true;
        solver.storm.shiftX = 4;
        const encoder = device.createCommandEncoder();
        solver.step(encoder);
        device.queue.submit([encoder.finish()]);
        const shifted = await cloudStats(fields);
        const expected = corrected.final.center - 4 * constants.domainX / gridX + 20;
        expect(Math.abs(shifted.center - expected) < 5,
            'Following the storm translates the cloud by whole cells and preserves its transport');
        expect(Math.abs(shifted.mass / corrected.final.mass - 1) < 0.002,
            'A window translation preserves cloud mass away from open boundaries');
        const error = await device.popErrorScope();
        expect(error === null, error?.message ?? 'No WebGPU validation errors');
        output.textContent += '\nAll checks passed.';
    } finally {
        solver.destroy();
        weather.atmosphereBuffer.destroy();
    }
}

void run().catch(error => {
    output.textContent += `\nFAIL ${error.message}`;
    console.error(error);
});
