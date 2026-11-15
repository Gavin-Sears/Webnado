import { device, initWebGPU } from '../renderer';
import { constants, processShaderRaw } from '../shaders/shaders';
import { Fields } from '../stage/fields';
import { WeatherData } from '../weather/noaa_loader';
import { halfToFloat } from './divergence_check';
import { FluidSolver } from './fluid_solver';

const output = document.getElementById('results')!;
const { gridX, gridY, gridZ } = constants;
const middle = gridX / 2 + gridX * (gridY / 2 + gridY * gridZ / 2);

function expect(condition: boolean, message: string) {
    if (!condition) {
        throw new Error(message);
    }
}

function passed(message: string) {
    output.textContent += `\nPASS ${message}`;
}

async function readTexture(texture: GPUTexture): Promise<Float32Array> {
    const scalar = texture.format === 'r32float';
    const channels = scalar ? 1 : 4;
    const texelBytes = scalar ? 4 : 8;
    const row = Math.ceil(gridX * texelBytes / 256) * 256;
    const buffer = device.createBuffer({ size: row * gridY * gridZ, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow: row, rowsPerImage: gridY }, [gridX, gridY, gridZ]);
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const data = new DataView(buffer.getMappedRange());
    const values = new Float32Array(gridX * gridY * gridZ * channels);
    for (let z = 0; z < gridZ; z++) {
        for (let y = 0; y < gridY; y++) {
            for (let x = 0; x < gridX; x++) {
                const source = row * (y + gridY * z) + x * texelBytes;
                const target = channels * (x + gridX * (y + gridY * z));
                for (let channel = 0; channel < channels; channel++) {
                    values[target + channel] = scalar
                        ? data.getFloat32(source, true)
                        : halfToFloat(data.getUint16(source + channel * 2, true));
                }
            }
        }
    }
    buffer.unmap();
    buffer.destroy();
    expect(values.every(Number.isFinite), 'Nonfinite two-fluid field');
    return values;
}

function seed(fields: Fields, airSpeed: number, dustSpeed: number, concentration: number, mode = 0) {
    const pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
            module: device.createShaderModule({ code: processShaderRaw(`
                override airSpeed: f32;
                override dustSpeed: f32;
                override concentration: f32;
                override mode: u32;
                @group(0) @binding(0) var air: texture_storage_3d<rgba16float, write>;
                @group(0) @binding(1) var dust: texture_storage_3d<rgba16float, write>;
                @group(0) @binding(2) var density: texture_storage_3d<r32float, write>;
                @compute @workgroup_size(8, 8, 4)
                fn main(@builtin(global_invocation_id) id: vec3u) {
                    let cell = vec3i(id);
                    var velocity = vec3f(dustSpeed, 0, 0);
                    var amount = concentration;
                    if (mode == 1u) {
                        velocity = 5.0 * sin(25.1327412 * vec3f(cell) / vec3f(gridDims));
                    }
                    if (mode == 2u) {
                        let center = vec3f(f32(gridDims.x) - 2.0, f32(gridDims.y) / 2.0, f32(gridDims.z) / 2.0);
                        let distance = (vec3f(cell) + vec3f(0.5) - center) / 2.0;
                        amount *= exp(-dot(distance, distance));
                    }
                    if (cell.y == 0) {
                        velocity = vec3f(0);
                        amount = 0.0;
                    }
                    textureStore(air, cell, vec4f(airSpeed, 0, 0, 0));
                    textureStore(dust, cell, vec4f(velocity, 0));
                    textureStore(density, cell, vec4f(amount, 0, 0, 0));
                }
            `) }),
            constants: { airSpeed, dustSpeed, concentration, mode }
        }
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: fields.velocity.read },
            { binding: 1, resource: fields.dustVelocity.read },
            { binding: 2, resource: fields.dustDensity.read }
        ]
    }));
    pass.dispatchWorkgroups(gridX / 8, gridY / 8, gridZ / 4);
    pass.end();
    device.queue.submit([encoder.finish()]);
}

async function run() {
    await initWebGPU();
    output.textContent = '';
    device.pushErrorScope('validation');
    const fields = new Fields();
    const mask = new Uint8Array(gridX * gridY * gridZ);
    for (let z = 0; z < gridZ; z++) {
        mask.fill(1, gridX * gridY * z, gridX * gridY * z + gridX);
    }
    fields.setObstacles(mask);
    const weather = new WeatherData({ latitude: 41.31, longitude: -94.46 });
    const atmosphere = new Float32Array(constants.atmosphereSlices * 8);
    for (let slice = 0; slice < constants.atmosphereSlices; slice++) {
        atmosphere.set([300, 8, 0, 0, 100000, 300, 1.2, 0], slice * 8);
    }
    weather.setAtmosphere(atmosphere);
    const solver = new FluidSolver(fields, weather);
    solver.buoyancy = 0;
    solver.confinement = 0;
    solver.turbulence = false;
    try {
        let encoder = device.createCommandEncoder();
        solver.step(encoder);
        device.queue.submit([encoder.finish()]);
        seed(fields, 2, -4, 0.12);
        encoder = device.createCommandEncoder();
        solver.dust.couple(encoder);
        device.queue.submit([encoder.finish()]);
        let dust = await readTexture(fields.dustVelocity.textures[fields.dustVelocity.index]);
        let reaction = await readTexture(fields.windForcing);
        const initialError = await device.popErrorScope();
        expect(initialError === null, initialError?.message ?? 'Initial two-fluid validation failed');
        device.pushErrorScope('validation');
        const dustSpeed = dust[4 * middle];
        const airSpeed = 2 + solver.dt * reaction[4 * middle];
        const momentumChange = 1.2 * (airSpeed - 2) + 0.12 * (dustSpeed + 4);
        console.log('Drag exchange', { dustSpeed, airSpeed, feedback: reaction[4 * middle], momentumChange });
        expect(Math.abs(momentumChange) < 0.002, 'Drag exchange did not conserve momentum');
        expect(Math.abs(dustSpeed - airSpeed) < 0.005, 'Stiff drag did not approach the common velocity');
        passed('Stiff drag conserves air-plus-dust momentum and remains finite');

        seed(fields, 2, -4, 0);
        encoder = device.createCommandEncoder();
        solver.dust.couple(encoder);
        device.queue.submit([encoder.finish()]);
        reaction = await readTexture(fields.windForcing);
        expect(reaction.every(value => value === 0), 'Empty dust phase pushed on the air');
        passed('Zero dust loading leaves the airflow unchanged');

        solver.dust.grainRadiusMicrons = 500;
        solver.dust.dynamicViscosity = 0.000018;
        solver.dust.writeParameters();
        seed(fields, 2, -4, 0.12);
        encoder = device.createCommandEncoder();
        solver.dust.couple(encoder);
        device.queue.submit([encoder.finish()]);
        dust = await readTexture(fields.dustVelocity.textures[fields.dustVelocity.index]);
        const radius = 0.0005;
        const mass = 4 / 3 * Math.PI * radius ** 3 * solver.dust.grainDensity;
        const response = mass / (3 * Math.PI * radius * solver.dust.dynamicViscosity);
        const relaxation = 1 - Math.exp(-1.1 * solver.dt / response);
        const expectedSpeed = -4 - relaxation * (-6) / 1.1;
        expect(Math.abs(dust[4 * middle] - expectedSpeed) < 0.005, 'Drag missed the analytic relaxation solution');
        passed('Finite drag agrees with an independent analytic solution');

        seed(fields, 0, 0, 0);
        encoder = device.createCommandEncoder();
        solver.dust.predict(encoder);
        device.queue.submit([encoder.finish()]);
        dust = await readTexture(fields.dustVelocity.textures[fields.dustVelocity.index]);
        expect(dust[4 * middle + 1] < -1, 'Gravity did not accelerate the dust phase');
        passed('Dust has its own gravity-driven momentum update');

        seed(fields, 0, 0, 0.02, 1);
        encoder = device.createCommandEncoder();
        solver.dust.finish(encoder, solver.jacobiIterations, solver.pressureMethod, solver.multigridCycles);
        device.queue.submit([encoder.finish()]);
        const measure = async () => {
            const encoder = device.createCommandEncoder();
            const result = solver.dust.checkDivergence(encoder);
            device.queue.submit([encoder.finish()]);
            expect(result !== undefined, 'Run with checkDivergence in the URL');
            return result!;
        };
        await measure();
        const report = await measure();
        console.log('Dust multigrid manufactured flow', report);
        expect(report.rms < 5e-4 && report.maxBlockedFaceSpeed === 0 && report.nonFinite === 0,
            'Independent dust projection failed');
        passed('Dust multigrid enforces periodic sides and solid boundaries');

        encoder = device.createCommandEncoder();
        solver.dust.reset(encoder);
        device.queue.submit([encoder.finish()]);
        seed(fields, 0, 10, 0.02, 2);
        const initial = await readTexture(fields.dustDensity.textures[fields.dustDensity.index]);
        for (let step = 0; step < 60; step++) {
            const encoder = device.createCommandEncoder();
            solver.dust.finish(encoder, 0);
            device.queue.submit([encoder.finish()]);
        }
        const transported = await readTexture(fields.dustDensity.textures[fields.dustDensity.index]);
        const initialMass = initial.reduce((sum, value) => sum + value, 0);
        const finalMass = transported.reduce((sum, value) => sum + value, 0);
        let leftMass = 0;
        for (let index = 0; index < transported.length; index++) {
            if (index % gridX < gridX / 4) {
                leftMass += transported[index];
            }
        }
        expect(transported.every(value => value >= 0), 'Dust concentration became negative');
        expect(Math.abs(finalMass / initialMass - 1) < 0.005, 'Periodic transport lost dust mass in uniform flow');
        expect(leftMass > 0.1 * initialMass, 'Dust did not cross the periodic side');
        passed('Dust concentration crosses periodic sides and preserves mass in uniform flow');

        await device.queue.onSubmittedWorkDone();
        const error = await device.popErrorScope();
        expect(error === null, error?.message ?? 'GPU validation failed');
        passed('No WebGPU validation errors');
        output.textContent += '\n\nAll checks passed.';
    } finally {
        solver.destroy();
        weather.atmosphereBuffer.destroy();
    }
}

void run().catch(error => {
    output.textContent += `\nFAIL ${error.message}`;
    console.error(error);
});
