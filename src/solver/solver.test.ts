import { device, initWebGPU, profiler } from '../renderer';
import { constants, processShaderRaw, resolutionName } from '../shaders/shaders';
import { Fields } from '../stage/fields';
import { WeatherData } from '../weather/noaa_loader';
import { FluidSolver } from './fluid_solver';

const output = document.getElementById('results')!;
const results: { name: string; details?: unknown }[] = [];

function expect(condition: boolean, message: string) {
    if (!condition) {
        throw new Error(message);
    }
}

function passed(name: string, details?: unknown) {
    results.push({ name, details });
    output.textContent = results.map(result => `PASS ${result.name}`).join('\n');
    console.log(name, details ?? 'passed');
}

function atmosphere(windX = 0, windZ = 0): Float32Array {
    const values = new Float32Array(constants.atmosphereSlices * 8);

    for (let level = 0; level < constants.atmosphereSlices; level++) {
        values.set([300, 8, windX, windZ, 100000, 300, 1.16, 0], level * 8);
    }

    return values;
}

function groundMask(): Uint8Array {
    const { gridX, gridY, gridZ } = constants;
    const mask = new Uint8Array(gridX * gridY * gridZ);

    for (let z = 0; z < gridZ; z++) {
        mask.fill(1, gridX * gridY * z, gridX * gridY * z + gridX);
    }

    return mask;
}

const seedSource = `
    override mode: u32;
    @group(0) @binding(0) var velocity: texture_storage_3d<${constants.velocityFormat}, write>;
    @group(0) @binding(1) var thermo: texture_storage_3d<${constants.scalarFormat}, write>;
    @group(0) @binding(2) var energy: texture_storage_3d<${constants.scalarFormat}, write>;

    @compute @workgroup_size(8, 8, 4)
    fn main(@builtin(global_invocation_id) id: vec3u) {
        let cell = vec3i(id);
        let position = vec3f(cell) + vec3f(0.5);
        let world = domainMin + position * cellSize;
        var wind = vec3f(0);
        var state = vec4f(0);

        if (mode == 1u) {
            wind = 5.0 * sin(25.1327412 * vec3f(cell) / vec3f(gridDims));
        }

        if ((mode == 2u || mode == 3u) && cell.y > 0) {
            let distance = (world - vec3f(-1500, 1400, 0)) / vec3f(600);
            state = exp(-dot(distance, distance)) * vec4f(2, 0, 0.2, 0);
        }

        if (mode == 3u) {
            wind.x = 10.0;
        }

        if (mode == 4u) {
            let falloff = exp(-dot(world.xz, world.xz) / 4000000.0);
            wind = vec3f(-world.z, 0, world.x) * 0.01 * falloff;
        }

        if (mode == 5u) { state = vec4f(0, 25, 0, 0); }
        if (mode == 6u) { wind = vec3f(2, 0, 0); }
        textureStore(energy, cell, vec4f(0.2, 0, 0, 0));
        textureStore(velocity, cell, vec4f(wind, 0));
        textureStore(thermo, cell, state);
    }
`;

function seed(fields: Fields, mode: number) {
    const module = device.createShaderModule({
        code: processShaderRaw(seedSource)
    });
    const pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
            module,
            entryPoint: 'main',
            constants: { mode }
        }
    });
    const group = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: fields.velocity.read },
            { binding: 1, resource: fields.thermo.read },
            { binding: 2, resource: fields.turbulence.read }
        ]
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(constants.gridX / 8, constants.gridY / 8, constants.gridZ / 4);
    pass.end();
    device.queue.submit([encoder.finish()]);
    if (mode === 6) {
        const { gridX, gridY, gridZ } = constants;
        const dust = new Uint16Array(gridX * gridY * gridZ * 4);
        for (let i = 0; i < dust.length; i += 4) dust[i] = 0xc400;
        const density = new Float32Array(gridX * gridY * gridZ).fill(.12);
        device.queue.writeTexture({ texture: fields.dustVelocity.textures[fields.dustVelocity.index] },
            dust, { bytesPerRow: gridX * 8, rowsPerImage: gridY }, [gridX, gridY, gridZ]);
        device.queue.writeTexture({ texture: fields.dustDensity.textures[fields.dustDensity.index] },
            density, { bytesPerRow: gridX * 4, rowsPerImage: gridY }, [gridX, gridY, gridZ]);
    }
}


async function readField(texture: GPUTexture): Promise<Float32Array> {
    const { gridX, gridY, gridZ } = constants;
    const texelBytes = texture.format === 'rgba32float' ? 16 : 8;
    const bytesPerRow = Math.ceil(gridX * texelBytes / 256) * 256;
    const buffer = device.createBuffer({
        size: bytesPerRow * gridY * gridZ,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer(
        { texture },
        { buffer, bytesPerRow, rowsPerImage: gridY },
        [gridX, gridY, gridZ]
    );
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);

    const data = new DataView(buffer.getMappedRange());
    const values = new Float32Array(gridX * gridY * gridZ * 4);

    for (let z = 0; z < gridZ; z++) {
        for (let y = 0; y < gridY; y++) {
            for (let x = 0; x < gridX; x++) {
                const target = 4 * (x + gridX * (y + gridY * z));
                const source = bytesPerRow * (y + gridY * z) + x * texelBytes;

                for (let channel = 0; channel < 4; channel++) {
                    values[target + channel] = texelBytes === 16
                        ? data.getFloat32(source + channel * 4, true)
                        : halfToFloat(data.getUint16(source + channel * 2, true));
                }
            }
        }
    }

    buffer.unmap();
    buffer.destroy();
    return values;
}

function tracerStats(values: Float32Array) {
    const { gridX, gridY, gridZ } = constants;
    let total = 0;
    let momentX = 0;

    for (let z = 0; z < gridZ; z++) {
        for (let y = 0; y < gridY; y++) {
            for (let x = 0; x < gridX; x++) {
                const index = 4 * (x + gridX * (y + gridY * z));
                const cloud = values[index + 2];
                expect(cloud >= 0, 'Cloud tracer became negative');
                total += cloud;
                momentX += cloud * ((x + 0.5) / gridX * constants.domainX - constants.domainX / 2);
            }
        }
    }

    expect(values.every(Number.isFinite), 'Nonfinite thermodynamic field');
    return { total, centerX: momentX / Math.max(total, 1e-12) };
}

async function runSteps(solver: FluidSolver, count: number) {
    for (let frame = 0; frame < count; frame++) {
        const time = performance.now();
        profiler?.beginFrame(time);
        const encoder = device.createCommandEncoder();
        solver.step(encoder);
        profiler?.resolve(encoder);
        device.queue.submit([encoder.finish()]);
        profiler?.afterSubmit();
        profiler?.endFrame(time);

        if (frame % 8 === 7) {
            await device.queue.onSubmittedWorkDone();
        }
    }

    await device.queue.onSubmittedWorkDone();
}

async function measure(solver: FluidSolver) {
    const encoder = device.createCommandEncoder();
    const report = solver.checkDivergence(encoder);
    device.queue.submit([encoder.finish()]);
    expect(report !== undefined, 'Add checkDivergence to the test page URL');
    return report!;
}

async function runTests() {
    expect(resolutionName === 'low', 'Use res=low for the numerical fixtures');
    await initWebGPU();
    device.pushErrorScope('validation');
    const fields = new Fields();
    const weather = new WeatherData({ latitude: 0, longitude: 0 });
    const column = atmosphere();
    for (let i = 0; i < constants.atmosphereSlices; i++) column[i * 8 + 6] = 1.2;
    weather.setAtmosphere(column);
    fields.setObstacles(groundMask());
    const solver = new FluidSolver(fields, weather);
    const fresh = async () => {
        solver.storm.enabled = false;
        solver.storm.followStorm = false;
        solver.buoyancy = 0;
        solver.confinement = 0;
        solver.turbulence = false;
        solver.warmRain.enabled = false;
        solver.dust.enabled = false;
        solver.dt = 1;
        solver.reset();
        await runSteps(solver, 1);
        await measure(solver);
    };
    const submit = (encode: (encoder: GPUCommandEncoder) => void) => {
        const encoder = device.createCommandEncoder();
        encode(encoder);
        device.queue.submit([encoder.finish()]);
    };
    const middle = 4 * (constants.gridX / 2 + constants.gridX * (
        constants.gridY / 2 + constants.gridY * constants.gridZ / 2));
    try {
        await fresh();
        await runSteps(solver, 4);
        const rest = await measure(solver);
        expect(rest.rms === 0 && rest.nonFinite === 0, 'Resting atmosphere moved');
        passed('Resting atmosphere remains at rest', rest);

        for (const method of ['jacobi', 'multigrid'] as const) {
            await fresh();
            seed(fields, 1);
            const seedError = await device.popErrorScope();
            expect(seedError === null, seedError?.message ?? 'Fixture failed');
            device.pushErrorScope('validation');
            const before = await measure(solver);
            expect(before.rms > .001, 'Manufactured flow was not divergent');
            solver.pressureMethod = method;
            solver.jacobiIterations = 160;
            solver.multigridCycles = 6;
            submit(encoder => solver.project(encoder));
            const after = await measure(solver);
            expect(after.rms < before.rms * .05, `${method} did not remove enough divergence`);
            expect(after.maxBlockedFaceSpeed === 0 && after.nonFinite === 0, `${method} violated solid boundaries`);
            passed(`${method} removes manufactured divergence and respects obstacles`, { before, after });
        }

        await fresh();
        seed(fields, 2);
        const initial = tracerStats(await readField(fields.thermo.textures[fields.thermo.index]));
        await runSteps(solver, 10);
        const resting = tracerStats(await readField(fields.thermo.textures[fields.thermo.index]));
        expect(Math.abs(resting.total / initial.total - 1) < .002, 'Stationary scalar lost mass');
        passed('Zero-flow transport preserves cloud water');

        solver.storm.enabled = true;
        solver.storm.liftSpeed = solver.storm.liftDuration = 0;
        solver.storm.highOrder = true;
        solver.storm.balanceReference = false;
        solver.turbulence = true;
        solver.buoyancy = 1;
        await runSteps(solver, 20);
        const velocity = await readField(fields.velocity.textures[fields.velocity.index]);
        const energy = await readField(fields.turbulence.textures[fields.turbulence.index]);
        const thermo = await readField(fields.thermo.textures[fields.thermo.index]);
        expect(velocity.every(Number.isFinite) && energy.every(Number.isFinite), 'High-order/TKE integration became nonfinite');
        expect(velocity.some((value, index) => index % 4 === 1 && value > .01), 'Warm air did not rise');
        expect(energy.every((value, index) => index % 4 !== 0 || value >= 0), 'TKE became negative');
        tracerStats(thermo);
        passed('Buoyant transport, high-order correction and TKE work together');

        await fresh();
        seed(fields, 5);
        const beforePhase = await readField(fields.thermo.textures[fields.thermo.index]);
        solver.warmRain.enabled = true;
        submit(encoder => solver.warmRain.step(encoder, 1));
        const afterPhase = await readField(fields.thermo.textures[fields.thermo.index]);
        const water = (a: Float32Array) => a[middle + 1] + a[middle + 2] + a[middle + 3];
        const heat = (a: Float32Array) => 1004 * a[middle] + 2500 * a[middle + 1];
        expect(afterPhase[middle] > 0 && afterPhase[middle + 2] > 0, 'Supersaturation did not condense and warm');
        expect(Math.abs(water(afterPhase) - water(beforePhase)) < .05, 'Phase change lost water');
        expect(Math.abs(heat(afterPhase) - heat(beforePhase)) < 60, 'Latent heat and vapor exchange disagree');
        expect(afterPhase.every(Number.isFinite), 'Microphysics became nonfinite');
        passed('Condensation conserves water and latent energy');

        await fresh();
        solver.dust.enabled = true;
        solver.dust.writeParameters();
        seed(fields, 6);
        submit(encoder => solver.dust.couple(encoder));
        const dust = await readField(fields.dustVelocity.textures[fields.dustVelocity.index]);
        const feedback = await readField(fields.windForcing);
        const airSpeed = 2 + solver.dt * feedback[middle];
        const momentumError = 1.2 * (airSpeed - 2) + .12 * (dust[middle] + 4);
        expect(Math.abs(momentumError) < .005, 'Two-fluid drag lost momentum');
        expect(Math.abs(dust[middle] - airSpeed) < .01, 'Stiff drag did not approach a common velocity');
        passed('Two-fluid drag exchanges momentum without instability', { momentumError });

        await fresh();
        solver.storm.enabled = true;
        solver.storm.followStorm = true;
        solver.storm.elapsed = 1;
        const pending = solver.storm.observe(fields, weather.atmosphereBuffer);
        const encoder = device.createCommandEncoder();
        expect(solver.step(encoder) === false, 'Physics advanced while window tracking was pending');
        device.queue.submit([encoder.finish()]);
        await pending;
        solver.reset();
        expect(solver.storm.elapsed === 0 && solver.storm.windowEast === 0, 'Reset retained the old experiment clock/window');
        passed('Tracking pauses physics until readback completes; reset clears the experiment');

        await device.queue.onSubmittedWorkDone();
        const error = await device.popErrorScope();
        expect(error === null, error?.message ?? 'WebGPU validation failed');
        passed('No WebGPU validation errors');
        output.textContent += '\n\nAll checks passed.';
        return results;
    } finally {
        solver.destroy();
        weather.atmosphereBuffer.destroy();
    }
}

function halfToFloat(bits: number): number {
    const sign = bits & 0x8000 ? -1 : 1;
    const exponent = (bits >>> 10) & 31;
    const mantissa = bits & 1023;

    if (exponent === 31) {
        return mantissa ? NaN : sign * Infinity;
    }

    if (exponent === 0) {
        return sign * 2 ** -14 * mantissa / 1024;
    }

    return sign * 2 ** (exponent - 15) * (1 + mantissa / 1024);
}

const promise = runTests();
Object.assign(window, { solverTests: promise });

void promise.catch(error => {
    output.textContent += `\n\nFAIL ${error.message}`;
    console.error(error);
});
