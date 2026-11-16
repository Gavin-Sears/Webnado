import { device, initWebGPU } from '../renderer';
import { constants, processShaderRaw } from '../shaders/shaders';
import { Fields } from '../stage/fields';
import { WeatherData } from '../weather/noaa_loader';
import { FluidSolver } from './fluid_solver';

const output = document.getElementById('results')!;

function expect(condition: boolean, message: string) {
    if (!condition) throw new Error(message);
    output.textContent += `\nPASS ${message}`;
}

async function run() {
    await initWebGPU();
    expect(constants.velocityFormat === 'rgba32float', 'The fixture uses precision=32');
    const { gridX, gridY, gridZ } = constants;
    const fields = new Fields(false);
    fields.setObstacles(new Uint8Array(gridX * gridY * gridZ));
    const weather = new WeatherData({ latitude: 0, longitude: 0 });
    const column = new Float32Array(constants.atmosphereSlices * 8);
    for (let slice = 0; slice < constants.atmosphereSlices; slice++) {
        column.set([300, 0, 20, 0, 100000, 300, 1.16, 0], slice * 8);
    }
    weather.setAtmosphere(column);
    const solver = new FluidSolver(fields, weather);
    solver.dt = 1;
    solver.dust.enabled = false;
    solver.turbulence = false;
    solver.confinement = 0;
    solver.warmRain.enabled = false;
    device.pushErrorScope('validation');
    let encoder = device.createCommandEncoder();
    solver.step(encoder);
    device.queue.submit([encoder.finish()]);

    const seed = device.createComputePipeline({ layout: 'auto', compute: {
        module: device.createShaderModule({ code: processShaderRaw(`
            @group(0) @binding(0) var output: texture_storage_3d<rgba32float, write>;
            @compute @workgroup_size(8, 8, 4)
            fn main(@builtin(global_invocation_id) id: vec3u) {
                textureStore(output, vec3i(id), vec4f(20.002, 0, 0, 0));
            }
        `) }), entryPoint: 'main'
    } });
    encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(seed);
    pass.setBindGroup(0, device.createBindGroup({ layout: seed.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: fields.velocity.read }] }));
    pass.dispatchWorkgroups(gridX / 8, gridY / 8, gridZ / 4);
    pass.end();
    device.queue.submit([encoder.finish()]);

    // Isolate transport so pressure and fixed side winds do not remove the perturbation.
    for (let step = 0; step < 40; step++) {
        encoder = device.createCommandEncoder();
        solver['dispatch'](encoder, 'advect');
        fields.velocity.swap();
        fields.thermo.swap();
        device.queue.submit([encoder.finish()]);
    }
    const readback = device.createBuffer({ size: 256,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture: fields.velocity.textures[fields.velocity.index],
        origin: [gridX / 2, gridY / 2, gridZ / 2] },
        { buffer: readback, bytesPerRow: 256 }, [1, 1, 1]);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const wind = new Float32Array(readback.getMappedRange())[0];
    readback.unmap();
    readback.destroy();
    expect(Math.abs(wind - 20.002) < 1e-5,
        'A 0.002 m/s perturbation survives 40 advection passes in a 20 m/s wind');

    encoder = device.createCommandEncoder();
    solver.project(encoder);
    const check = solver.checkDivergence(encoder);
    device.queue.submit([encoder.finish()]);
    expect(Boolean(check), 'The fixture enables checkDivergence');
    const report = await check!;
    expect(report.nonFinite === 0 && report.rms < 5e-4,
        'Projection and CPU readback support the 32-bit velocity layout');
    const error = await device.popErrorScope();
    expect(error === null, error?.message ?? 'No WebGPU validation errors');
    solver.destroy();
    weather.atmosphereBuffer.destroy();
    output.textContent += '\nAll checks passed.';
}

void run().catch(error => {
    output.textContent += `\nFAIL ${error.message}`;
    console.error(error);
});
