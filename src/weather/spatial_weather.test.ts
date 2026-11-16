import { device, initWebGPU } from '../renderer';
import { constants, processShaderRaw } from '../shaders/shaders';
import { Fields } from '../stage/fields';
import { FluidSolver } from '../solver/fluid_solver';
import { StormDiagnostics } from '../solver/storm_diagnostics';
import { WeatherData } from './noaa_loader';
import { halfToFloat } from '../solver/divergence_check';
import common from '../solver/common.wgsl?raw';
import { checkSurfaceExchange } from './surface_exchange.test';
import { checkMovingWeather } from './moving_weather.test';
import { checkScalarPrecision } from './scalar_precision.test';

const output = document.getElementById('results')!;

function expect(condition: boolean, message: string) {
    if (!condition) throw new Error(message);
    output.textContent += `\nPASS ${message}`;
}

async function run() {
    expect(!!constants.spatialWeather && constants.velocityFormat === 'rgba32float',
        'The fixture uses spatial weather and 32-bit velocity');
    const url = new URLSearchParams(location.search).get('weather');
    if (!url) throw new Error('Missing weather manifest URL.');
    await initWebGPU();
    device.pushErrorScope('validation');
    const weather = new WeatherData({ latitude: 41.31, longitude: -94.46 });
    await weather.loadSpatial(url);
    const spatial = weather.spatial!;
    expect(spatial.values.slice(20, 20 + spatial.manifest.frames).every(Number.isFinite),
        'Lateral weather flux is projected onto the domain mass constraint');
    const positions = new Float32Array([
        constants.gridX * 0.2, 0.5, constants.gridZ * 0.3, 0,
        constants.gridX * 0.5, constants.gridY * 0.2, constants.gridZ * 0.5, 0,
        constants.gridX * 0.8, constants.gridY * 0.6, constants.gridZ * 0.7, 0,
        0, constants.gridY * 0.1, constants.gridZ * 0.9, 0,
        constants.gridX, constants.gridY * 0.9, 0, 0
    ]);
    const points = device.createBuffer({ size: positions.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(points, 0, positions);
    const resultSize = positions.length / 4 * 80;
    const result = device.createBuffer({ size: resultSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readback = device.createBuffer({ size: resultSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const module = device.createShaderModule({ code: processShaderRaw(common + `
        @group(0) @binding(34) var<storage, read> points: array<vec4f>;
        @group(0) @binding(35) var<storage, read_write> output: array<vec4f>;
        @compute @workgroup_size(1)
        fn sampleWeather(@builtin(global_invocation_id) id: vec3u) {
            let state = spatialState(points[id.x].xyz);
            output[5u * id.x] = state.wind;
            output[5u * id.x + 1u] = state.thermo;
            output[5u * id.x + 2u] = state.air;
            output[5u * id.x + 3u] = vec4f(spatialCoordinate(points[id.x].xyz), state.energy);
            output[5u * id.x + 4u] = spatialSurface(points[id.x].xyz);
        }
    `) });
    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'sampleWeather' } });
    const bindings = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 4, resource: { buffer: weather.atmosphereBuffer } },
        { binding: 34, resource: { buffer: points } },
        { binding: 35, resource: { buffer: result } }
    ] });
    for (const seconds of [0, 1800, 3600, 7200]) {
        weather.writeSpatialTime(seconds);
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindings);
        pass.dispatchWorkgroups(positions.length / 4);
        pass.end();
        encoder.copyBufferToBuffer(result, 0, readback, 0, resultSize);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const gpu = new Float32Array(readback.getMappedRange());
        const time = Math.min(seconds / spatial.manifest.intervalSeconds, spatial.manifest.frames - 1);
        let maxError = 0;
        for (let point = 0; point < positions.length / 4; point++) {
            const x = positions[4 * point] / constants.gridX * constants.domainX - constants.domainX / 2;
            const height = positions[4 * point + 1] / constants.gridY * constants.domainY;
            const z = positions[4 * point + 2] / constants.gridZ * constants.domainZ - constants.domainZ / 2;
            const a = spatial.sample(x, height, z, Math.floor(time));
            const b = spatial.sample(x, height, z, Math.min(Math.floor(time) + 1, spatial.manifest.frames - 1));
            const channels = (sample: typeof a) => [sample.east, sample.vertical, sample.south, sample.terrain,
                sample.theta, sample.vapor, sample.cloud, sample.rain,
                sample.pressure, sample.temperature, sample.density, sample.ice];
            channels(a).forEach((value, channel) => {
                let expected = value + (channels(b)[channel] - value) * (time % 1);
                const lowerFrame = Math.floor(time);
                const upperFrame = Math.min(lowerFrame + 1, spatial.manifest.frames - 1);
                const correction = spatial.values[20 + lowerFrame] * (1 - time % 1)
                    + spatial.values[20 + upperFrame] * (time % 1);
                if (channel === 0 && x === -constants.domainX / 2) expected += correction;
                if (channel === 0 && x === constants.domainX / 2) expected -= correction;
                if (channel === 2 && z === -constants.domainZ / 2) expected += correction;
                if (channel === 2 && z === constants.domainZ / 2) expected -= correction;
                const tolerance = channel === 8 ? 5 : channel === 3 ? 0.1 : 0.015;
                const error = Math.abs(gpu[20 * point + channel] - expected);
                maxError = Math.max(maxError, error / tolerance);
                if (error > tolerance) throw new Error(`At ${seconds}s, point ${point}, channel ${channel}: GPU ${gpu[20 * point + channel]}, CPU ${expected}`);
            });
            const expectedEnergy = a.energy + (b.energy - a.energy) * (time % 1);
            if (Math.abs(gpu[20 * point + 15] - expectedEnergy) > 0.015) {
                throw new Error('The source turbulent energy was not interpolated correctly.');
            }
            spatial.surface(x, z, seconds).forEach((expected, channel) => {
                const tolerance = channel === 1 || channel === 2 ? 1 : 0.01;
                if (Math.abs(gpu[20 * point + 16 + channel] - expected) > tolerance) {
                    throw new Error('The surface flux was not interpolated correctly.');
                }
            });
        }
        readback.unmap();
        expect(maxError <= 1, `GPU spatial interpolation matches CPU sampling at ${seconds}s`);
    }

    const fields = new Fields(false);
    fields.setObstacles(new Uint8Array(constants.gridX * constants.gridY * constants.gridZ));
    const solver = new FluidSolver(fields, weather);
    solver.startSpatialExperiment();
    solver.paused = true;
    const reset = device.createCommandEncoder();
    solver.step(reset);
    device.queue.submit([reset.finish()]);
    expect(solver.storm.liftSpeed === 0 && solver.storm.liftDuration === 0 && !solver.storm.balanceReference,
        'Synthetic lifting and reference-wind forcing are disabled');
    const diagnostics = new StormDiagnostics(weather.atmosphereBuffer);
    const initial = await diagnostics.measure(fields, 0);
    expect(initial.invalid === 0 && initial.speed > 0 && initial.temperature > 0,
        'Reset initializes finite spatial wind and thermodynamic fields');
    const texture = fields.velocity.textures[fields.velocity.index];
    const thermodynamics = fields.thermo.textures[fields.thermo.index];
    const probes = [[1, 1, 1], [constants.gridX / 2, constants.gridY / 3 | 0, constants.gridZ / 2],
        [constants.gridX - 2, constants.gridY - 2, constants.gridZ - 2]];
    const samples = device.createBuffer({ size: probes.length * 512,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const sampleEncoder = device.createCommandEncoder();
    probes.forEach(([x, y, z], index) => {
        sampleEncoder.copyTextureToBuffer({ texture, origin: [x, y, z] },
            { buffer: samples, offset: index * 512, bytesPerRow: 256 }, [1, 1, 1]);
        sampleEncoder.copyTextureToBuffer({ texture: thermodynamics, origin: [x, y, z] },
            { buffer: samples, offset: index * 512 + 256, bytesPerRow: 256 }, [1, 1, 1]);
    });
    device.queue.submit([sampleEncoder.finish()]);
    await samples.mapAsync(GPUMapMode.READ);
    const bytes = samples.getMappedRange();
    const reference = spatial.referenceColumn(constants.atmosphereSlices,
        constants.domainY, constants.domainX, constants.domainZ);
    probes.forEach((cell, index) => {
        const wind = new Float32Array(bytes, index * 512, 4);
        for (let axis = 0; axis < 3; axis++) {
            const position = cell.map((value, component) => value + (axis === component ? 0 : 0.5));
            const sample = spatial.sample(position[0] / constants.gridX * constants.domainX - constants.domainX / 2,
                position[1] / constants.gridY * constants.domainY,
                position[2] / constants.gridZ * constants.domainZ - constants.domainZ / 2);
            const expected = [sample.east, sample.vertical, sample.south][axis];
            if (Math.abs(wind[axis] - expected) > 0.015) throw new Error('Reset misplaced a staggered wind component.');
        }
        const source = spatial.sample((cell[0] + 0.5) / constants.gridX * constants.domainX - constants.domainX / 2,
            (cell[1] + 0.5) / constants.gridY * constants.domainY,
            (cell[2] + 0.5) / constants.gridZ * constants.domainZ - constants.domainZ / 2);
        const slice = Math.max(0, Math.min(constants.atmosphereSlices - 1,
            (cell[1] + 0.5) / constants.gridY * constants.atmosphereSlices - 0.5));
        const lower = Math.floor(slice);
        const upper = Math.min(lower + 1, constants.atmosphereSlices - 1);
        const theta = reference[lower * 8] * (1 - slice % 1) + reference[upper * 8] * (slice % 1);
        const vapor = reference[lower * 8 + 1] * (1 - slice % 1) + reference[upper * 8 + 1] * (slice % 1);
        const state = constants.preciseScalars ? new Float32Array(bytes, index * 512 + 256, 4)
            : new Uint16Array(bytes, index * 512 + 256, 4);
        [source.theta - theta, source.vapor - vapor, source.cloud, source.rain].forEach((expected, channel) => {
            const actual = constants.preciseScalars ? state[channel] : halfToFloat(state[channel]);
            if (Math.abs(actual - expected) > 0.02) {
                throw new Error('Reset did not preserve the source thermodynamic state.');
            }
        });
    });
    samples.unmap();
    expect(true, 'Reset preserves sampled source winds, temperature, moisture, cloud, and rain');
    await checkSurfaceExchange(solver, fields, weather);
    if (spatial.manifest.version >= 2) {
        expect(true, 'Source heat and moisture budgets close, and roughness drag removes momentum');
    }
    await checkMovingWeather(solver, fields, weather);
    if (spatial.manifest.version >= 3) {
        expect(true, 'Moving the window preserves the interior and samples real weather at the new edges');
    }
    solver.paused = false;
    if (spatial.manifest.version >= 3 && weather.spatialFlat) {
        solver.storm.followStorm = true;
        solver.storm.track({ ...initial, seconds: 1, rotatingUpdraftScore: 1,
            updraftLocation: [constants.domainX / 3, 1000, -constants.domainZ / 3] });
        const dt = solver.dt;
        solver.dt = 0.05;
        const tracking = device.createCommandEncoder();
        solver.step(tracking);
        device.queue.submit([tracking.finish()]);
        await device.queue.onSubmittedWorkDone();
        expect(solver.storm.windowEast === spatial.windowEast && solver.storm.windowNorth === spatial.windowNorth
            && solver.storm.windowEast !== 0, 'Tracking updates the simulation and weather coordinates together');
        solver.dt = dt;
        solver.reset();
    }
    for (let step = 0; step < 20; step++) {
        const encoder = device.createCommandEncoder();
        solver.step(encoder);
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
    }
    const evolved = await diagnostics.measure(fields, solver.storm.elapsed);
    expect(evolved.invalid === 0 && evolved.divergenceRms < 1e-5,
        'The weather state evolves with finite values and controlled mass divergence');
    solver.storm.elapsed = (spatial.manifest.frames - 1) * spatial.manifest.intervalSeconds;
    const finished = device.createCommandEncoder();
    solver.step(finished);
    device.queue.submit([finished.finish()]);
    expect(solver.paused, 'The simulation stops at the end of the supplied forecasts');
    await checkScalarPrecision(weather);
    if (constants.preciseScalars) {
        expect(true, 'Precise scalar transport preserves small signals and turbulence follows analytic decay');
    }
    const error = await device.popErrorScope();
    expect(!error, `No WebGPU validation errors${error ? ': ' + error.message : ''}`);
    console.log('Spatial input diagnostics', { initial, evolved });
    output.textContent += '\n\nAll checks passed.';
}

void run().catch(error => {
    output.textContent += `\n\nFAIL ${error.message}`;
    console.error(error);
});
