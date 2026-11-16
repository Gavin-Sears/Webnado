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
    @group(0) @binding(0) var velocity: texture_storage_3d<rgba16float, write>;
    @group(0) @binding(1) var thermo: texture_storage_3d<rgba16float, write>;

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
            { binding: 1, resource: fields.thermo.read }
        ]
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(constants.gridX / 8, constants.gridY / 8, constants.gridZ / 4);
    pass.end();
    device.queue.submit([encoder.finish()]);
}

async function readField(texture: GPUTexture): Promise<Float32Array> {
    const { gridX, gridY, gridZ } = constants;
    const bytesPerRow = Math.ceil(gridX * 8 / 256) * 256;
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
                const source = bytesPerRow * (y + gridY * z) + x * 8;

                for (let channel = 0; channel < 4; channel++) {
                    values[target + channel] = halfToFloat(data.getUint16(source + channel * 2, true));
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
    await initWebGPU();
    device.pushErrorScope('validation');

    const fields = new Fields();
    const weather = new WeatherData({ latitude: 41.31, longitude: -94.46 });
    weather.setAtmosphere(atmosphere());
    fields.setObstacles(groundMask());

    let solver = new FluidSolver(fields, weather);
    const replaceSolver = () => {
        solver.destroy();
        solver = new FluidSolver(fields, weather);
        return solver;
    };

    try {
        if (resolutionName !== 'low') {
            seed(fields, 2);
            await runSteps(solver, 3);
            await measure(solver);
            const report = await measure(solver);
            expect(report.rms < 5e-4 && report.nonFinite === 0, 'Large-grid smoke check failed');
            expect(report.maxBlockedFaceSpeed === 0, 'Large-grid boundary check failed');
            passed(`${resolutionName} grid completes simulation and readback`, report);
            const error = await device.popErrorScope();
            expect(error === null, error?.message ?? 'GPU validation failed');
            output.textContent += '\n\nSmoke check passed.';
            return results;
        }

        seed(fields, 0);
        await runSteps(solver, 4);
        const resting = await measure(solver);
        expect(resting.rms === 0 && resting.nonFinite === 0, 'Resting flow did not remain zero');
        passed('Resting atmosphere stays at rest', resting);

        seed(fields, 1);
        solver.jacobiIterations = 41;
        let encoder = device.createCommandEncoder();
        solver.project(encoder);
        device.queue.submit([encoder.finish()]);
        const firstProjection = await measure(solver);

        seed(fields, 1);
        encoder = device.createCommandEncoder();
        solver.project(encoder);
        device.queue.submit([encoder.finish()]);
        const warmProjection = await measure(solver);
        expect(warmProjection.rms < firstProjection.rms, 'Retained pressure did not improve the repeated solve');
        passed('Pressure stays warm across odd iteration counts', { firstProjection, warmProjection });

        seed(fields, 1);
        solver.jacobiIterations = 1200;
        encoder = device.createCommandEncoder();
        solver.project(encoder);
        device.queue.submit([encoder.finish()]);
        const projected = await measure(solver);
        expect(projected.rms < 5e-4, 'Manufactured gradient projection missed the divergence target');
        expect(projected.maxBlockedFaceSpeed === 0, 'Projection changed blocked face velocities');
        passed('Manufactured gradient flow projects below 5e-4', projected);

        replaceSolver();
        seed(fields, 2);
        solver.buoyancy = 0;
        solver.confinement = 0;
        const before = tracerStats(await readField(fields.thermo.textures[fields.thermo.index]));
        await runSteps(solver, 8);
        const after = tracerStats(await readField(fields.thermo.textures[fields.thermo.index]));
        expect(Math.abs(after.total / before.total - 1) < 0.001, 'Resting tracer changed during advection');
        passed('Zero-flow scalar advection preserves the tracer', after);

        weather.setAtmosphere(atmosphere(10, 0));
        replaceSolver();
        seed(fields, 3);
        solver.buoyancy = 0;
        solver.confinement = 0;
        const start = tracerStats(await readField(fields.thermo.textures[fields.thermo.index]));
        await runSteps(solver, 10);
        const moved = tracerStats(await readField(fields.thermo.textures[fields.thermo.index]));
        const displacement = moved.centerX - start.centerX;
        expect(displacement > 40 && displacement < 60, 'Scalar transport did not follow the 10 m/s wind');
        const windReport = await measure(solver);
        expect(windReport.rms < 5e-4, 'Weather-driven flow missed the divergence target');
        passed('Weather wind transports the tracer in the expected direction', { displacement, windReport });

        const windVelocity = await readField(fields.velocity.textures[fields.velocity.index]);
        const velocityAt = (x: number, y: number, z: number, axis: number) => {
            return windVelocity[axis + 4 * (x + constants.gridX * (y + constants.gridY * z))];
        };
        const middleX = constants.gridX / 2;
        const middleZ = constants.gridZ / 2;
        expect(velocityAt(middleX, 1, middleZ, 0) === 0, 'Ground did not enforce no-slip');
        expect(velocityAt(middleX, constants.gridY - 1, middleZ, 0) === 10,
            'Ceiling did not preserve tangential wind');
        passed('Ground is no-slip and ceiling preserves tangential flow');

        weather.setAtmosphere(atmosphere(-10, 0));
        await runSteps(solver, 1);
        const changedWind = await readField(fields.velocity.textures[fields.velocity.index]);
        const inlet = 4 * (constants.gridX * (constants.gridY / 2 + constants.gridY * middleZ));
        expect(changedWind[inlet] === -10, 'Weather updates did not reach the side boundary');
        passed('Weather updates reach the existing side boundary bindings');

        const profile = atmosphere();

        for (let level = 0; level < constants.atmosphereSlices; level++) {
            profile[level * 8 + 2] = level * 0.5;
        }

        weather.setAtmosphere(profile);
        encoder = device.createCommandEncoder();
        solver.project(encoder);
        device.queue.submit([encoder.finish()]);
        const profileVelocity = await readField(fields.velocity.textures[fields.velocity.index]);
        const slice = (constants.gridY / 2 + 0.5) / constants.gridY * constants.atmosphereSlices - 0.5;
        expect(Math.abs(profileVelocity[inlet] - slice * 0.5) < 0.01,
            'Atmosphere interpolation did not use slice centers');
        passed('Side winds interpolate the atmosphere at slice centers');

        weather.setAtmosphere(atmosphere());
        replaceSolver();
        seed(fields, 2);
        await runSteps(solver, 100);
        await measure(solver);
        const plumeReport = await measure(solver);
        const velocity = await readField(fields.velocity.textures[fields.velocity.index]);
        let maxUpdraft = 0;

        for (let index = 1; index < velocity.length; index += 4) {
            maxUpdraft = Math.max(maxUpdraft, velocity[index]);
        }

        expect(maxUpdraft > 0.05, 'Warm air did not produce an updraft');
        expect(plumeReport.rms < 5e-4 && plumeReport.nonFinite === 0, 'Evolving plume failed the divergence target');
        expect(plumeReport.maxBlockedFaceSpeed === 0, 'Plume crossed a blocked face');
        passed('Buoyancy evolves a finite, projected plume', { maxUpdraft, plumeReport });

        const stressSteps = Number(new URLSearchParams(location.search).get('stress') ?? 0);

        if (stressSteps > 0) {
            await runSteps(solver, stressSteps);
            await measure(solver);
            const stressReport = await measure(solver);
            const thermo = await readField(fields.thermo.textures[fields.thermo.index]);
            expect(stressReport.nonFinite === 0 && thermo.every(Number.isFinite), 'Long run produced nonfinite state');
            expect(stressReport.rms < 5e-4, 'Long run exceeded the divergence target');
            expect(stressReport.maxBlockedFaceSpeed === 0, 'Long run crossed a blocked face');
            passed(`Plume stays stable for ${stressSteps} additional steps`, stressReport);
        }

        replaceSolver();
        seed(fields, 4);
        solver.buoyancy = 0;
        solver.confinement = 0;
        await runSteps(solver, 8);
        const unconstrained = await readField(fields.velocity.textures[fields.velocity.index]);

        replaceSolver();
        seed(fields, 4);
        solver.buoyancy = 0;
        await runSteps(solver, 8);
        const confined = await readField(fields.velocity.textures[fields.velocity.index]);
        let difference = 0;

        for (let index = 0; index < confined.length; index++) {
            difference += Math.abs(confined[index] - unconstrained[index]);
        }

        expect(confined.every(Number.isFinite), 'Vorticity confinement produced nonfinite velocity');
        expect(difference > 1, 'Vorticity confinement did not affect the flow');
        passed('Vorticity confinement feeds back into velocity', { difference });

        replaceSolver();
        seed(fields, 0);
        solver.buoyancy = 0;
        solver.confinement = 0;
        solver.addThermal();
        const initialThermoIndex = fields.thermo.index;
        await runSteps(solver, 1);
        expect(fields.thermo.index === 1 - initialThermoIndex, 'Newest thermo state was not swapped into read');
        const injected = tracerStats(await readField(fields.thermo.textures[fields.thermo.index]));
        await runSteps(solver, 3);
        const retained = tracerStats(await readField(fields.thermo.textures[fields.thermo.index]));
        expect(injected.total > 0, 'Add thermal did not initialize a visible cloud tracer');
        expect(Math.abs(retained.total / injected.total - 1) < 0.001, 'Thermal source was applied more than once');
        passed('Thermal injection runs once and leaves the newest scalar in read');

        if (resolutionName === 'low') {
            const mask = groundMask();
            const { gridX, gridY, gridZ } = constants;

            for (let z = 0; z < gridZ; z++) {
                for (let y = 0; y < gridY; y++) {
                    mask[gridX / 2 + gridX * (y + gridY * z)] = 2;
                }
            }

            fields.setObstacles(mask);
            seed(fields, 1);
            solver.jacobiIterations = 1200;
            encoder = device.createCommandEncoder();
            solver.project(encoder);
            device.queue.submit([encoder.finish()]);
            const obstacleReport = await measure(solver);
            expect(obstacleReport.maxBlockedFaceSpeed === 0, 'Changed obstacles were not applied');
            expect(obstacleReport.rms < 5e-4, 'Projection failed with disconnected fluid regions');
            passed('Obstacle changes rebuild the open bits', obstacleReport);
        }

        fields.setObstacles(new Uint8Array(constants.gridX * constants.gridY * constants.gridZ).fill(1));
        encoder = device.createCommandEncoder();
        solver.project(encoder);
        device.queue.submit([encoder.finish()]);
        const solidReport = await measure(solver);
        expect(solidReport.fluidCells === 0 && solidReport.nonFinite === 0, 'Fully solid domain was not finite');
        expect(solidReport.maxBlockedFaceSpeed === 0, 'Velocity remained inside solids');
        passed('Fully solid domains remain finite', solidReport);

        await device.queue.onSubmittedWorkDone();
        const error = await device.popErrorScope();
        expect(error === null, error?.message ?? 'GPU validation failed');
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
