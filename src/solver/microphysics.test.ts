import { device, initWebGPU } from '../renderer';
import { constants, processShaderRaw } from '../shaders/shaders';
import { Fields } from '../stage/fields';
import { WeatherData } from '../weather/noaa_loader';
import { atmosphereColumn, pointsFromHour, type OpenMeteoForecast } from '../weather/sounding';
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

async function read(texture: GPUTexture) {
    const row = Math.ceil(gridX * 8 / 256) * 256;
    const buffer = device.createBuffer({ size: row * gridY * gridZ, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow: row, rowsPerImage: gridY }, [gridX, gridY, gridZ]);
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const data = new DataView(buffer.getMappedRange());
    const values = new Float32Array(gridX * gridY * gridZ * 4);
    for (let z = 0; z < gridZ; z++) {
        for (let y = 0; y < gridY; y++) {
            for (let x = 0; x < gridX; x++) {
                for (let channel = 0; channel < 4; channel++) {
                    const offset = row * (y + gridY * z) + x * 8 + channel * 2;
                    values[4 * (x + gridX * (y + gridY * z)) + channel] = halfToFloat(data.getUint16(offset, true));
                }
            }
        }
    }
    buffer.unmap();
    buffer.destroy();
    expect(values.every(Number.isFinite), 'Nonfinite thermodynamic state');
    return values;
}

function seed(fields: Fields, vapor: number, cloud: number, rain: number, packet = false) {
    const pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
            module: device.createShaderModule({ code: processShaderRaw(`
                override vapor: f32;
                override cloud: f32;
                override rain: f32;
                override packet: bool;
                @group(0) @binding(0) var state: texture_storage_3d<rgba16float, write>;
                @compute @workgroup_size(8, 8, 4)
                fn main(@builtin(global_invocation_id) id: vec3u) {
                    var value = vec4f(0, vapor - 10.0, cloud, rain);
                    if (packet && id.y != 10u) {
                        value.w = 0.0;
                    }
                    if (id.y == 0u) {
                        value = vec4f(0);
                    }
                    textureStore(state, vec3i(id), value);
                }
            `) }),
            entryPoint: 'main',
            constants: { vapor, cloud, rain, packet: Number(packet) }
        }
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: fields.thermo.read }]
    }));
    pass.dispatchWorkgroups(gridX / 8, gridY / 8, gridZ / 4);
    pass.end();
    device.queue.submit([encoder.finish()]);
}

async function run() {
    await initWebGPU();
    device.pushErrorScope('validation');
    const fields = new Fields();
    const mask = new Uint8Array(gridX * gridY * gridZ);
    for (let z = 0; z < gridZ; z++) {
        mask.fill(1, gridX * gridY * z, gridX * gridY * z + gridX);
    }
    fields.setObstacles(mask);
    const weather = new WeatherData({ latitude: 41.31, longitude: -94.46 });
    const column = new Float32Array(constants.atmosphereSlices * 8);
    for (let slice = 0; slice < constants.atmosphereSlices; slice++) {
        column.set([300, 10, 0, 0, 100000, 300, 1.16, 0], slice * 8);
    }
    weather.setAtmosphere(column);
    const solver = new FluidSolver(fields, weather);
    weather.onApplied = () => solver.reset();
    output.textContent = '';
    const submitStep = () => {
        const encoder = device.createCommandEncoder();
        solver.step(encoder);
        device.queue.submit([encoder.finish()]);
    };
    const phaseChange = async (vapor: number, cloud: number, rain: number, packet = false) => {
        seed(fields, vapor, cloud, rain, packet);
        const before = await read(fields.thermo.textures[fields.thermo.index]);
        const encoder = device.createCommandEncoder();
        solver.warmRain.step(encoder, 1);
        device.queue.submit([encoder.finish()]);
        const after = await read(fields.thermo.textures[fields.thermo.index]);
        for (let i = 0; i < after.length; i += 4) {
            expect(after[i + 1] >= -10 && after[i + 2] >= 0 && after[i + 3] >= 0,
                'Microphysics created negative water');
        }
        if (!packet) {
            const i = 4 * middle;
            const water = (state: Float32Array) => state[i + 1] + state[i + 2] + state[i + 3];
            const enthalpy = (state: Float32Array) => 1004 * state[i] + 2500 * state[i + 1];
            expect(Math.abs(water(after) - water(before)) < 0.02, 'Phase changes lost water');
            expect(Math.abs(enthalpy(after) - enthalpy(before)) < 30, 'Latent heating did not balance vapor exchange');
        }
        return { before, after };
    };

    try {
        submitStep();
        let result = await phaseChange(25, 0, 0);
        expect(result.after[4 * middle] > 0 && result.after[4 * middle + 2] > 0, 'Supersaturated air did not form cloud and warm');
        passed('Condensation warms the air and conserves water and latent energy');

        result = await phaseChange(5, 1, 0);
        expect(result.after[4 * middle] < 0 && result.after[4 * middle + 2] < 1, 'Cloud did not evaporate and cool');
        passed('Cloud evaporation cools the air without creating negative water');

        result = await phaseChange(5, 0, 1);
        expect(result.after[4 * middle] < 0 && result.after[4 * middle + 3] < 1, 'Rain did not evaporate and cool');
        passed('Rain evaporation returns vapor and cools dry air');

        const saturation = 22.79;
        result = await phaseChange(saturation, 3, 1);
        expect(result.after[4 * middle + 3] > 1, 'Cloud conversion and collection did not grow rain');
        passed('Cloud conversion and collection grow rain');

        result = await phaseChange(saturation, 0, 1, true);
        const rainStats = (state: Float32Array) => {
            let mass = 0;
            let moment = 0;
            for (let z = 0; z < gridZ; z++) {
                for (let y = 0; y < gridY; y++) {
                    for (let x = 0; x < gridX; x++) {
                        const amount = state[4 * (x + gridX * (y + gridY * z)) + 3];
                        mass += amount;
                        moment += y * amount;
                    }
                }
            }
            return { mass, height: moment / mass };
        };
        const beforeRain = rainStats(result.before);
        const afterRain = rainStats(result.after);
        expect(afterRain.height < beforeRain.height && Math.abs(afterRain.mass / beforeRain.mass - 1) < 0.005,
            'Rain sedimentation did not move water downward conservatively');
        passed('A rain layer falls downward and preserves water in a uniform-density column');

        for (let slice = 0; slice < constants.atmosphereSlices; slice++) {
            column[slice * 8 + 6] = 1.4 - 0.5 * (slice + 0.5) / constants.atmosphereSlices;
        }
        weather.setAtmosphere(column);
        result = await phaseChange(saturation, 0, 1, true);
        const weightedRain = (state: Float32Array) => {
            let mass = 0;
            for (let index = 0; index < state.length / 4; index++) {
                const y = Math.floor(index / gridX) % gridY;
                const density = 1.4 - 0.5 * (y + 0.5) / gridY;
                mass += density * state[4 * index + 3];
            }
            return mass;
        };
        expect(Math.abs(weightedRain(result.after) / weightedRain(result.before) - 1) < 0.005,
            'Sedimentation changed rain mass across density levels');
        passed('Rain sedimentation conserves density-weighted mass in a stratified column');

        for (let slice = 0; slice < constants.atmosphereSlices; slice++) {
            column[slice * 8 + 6] = 1.16;
        }
        weather.setAtmosphere(column);
        solver.warmRain.rainDrain = true;
        result = await phaseChange(saturation, 0, 1);
        expect(rainStats(result.after).mass < rainStats(result.before).mass * 0.999,
            'Precipitation did not leave through the ground');
        passed('The atmospheric experiment removes rain reaching the ground');
        solver.warmRain.rainDrain = false;

        const fixture: OpenMeteoForecast = {
            elevation: 100,
            hourly: {
                time: ['2024-05-21T21:00'], temperature_2m: [20], relative_humidity_2m: [50],
                surface_pressure: [1000], wind_speed_10m: [20], wind_direction_10m: [270],
                temperature_900hPa: [10], relative_humidity_900hPa: [80], geopotential_height_900hPa: [1100],
                wind_speed_900hPa: [10], wind_direction_900hPa: [180]
            }
        };
        const points = pointsFromHour(fixture, '2024-05-21', 21);
        expect(points.length === 2 && points[1].height === 1000, 'Sounding heights were not relative to the ground');
        expect(Math.abs(points[0].windX - 20) < 1e-6 && Math.abs(points[1].windZ + 10) < 1e-6,
            'Meteorological wind directions were reversed');
        const sounding = atmosphereColumn(points, 8, 1000);
        expect(sounding.every(Number.isFinite) && sounding[4] > 90000, 'Sounding units or interpolation are invalid');
        passed('Sounding conversion preserves SI units, height offsets, and wind directions');

        const version = solver.resetVersion;
        weather.useDryCalm();
        expect(solver.resetVersion === version + 1, 'Applying weather did not request a full solver reset');
        submitStep();
        const cleared = await read(fields.thermo.textures[fields.thermo.index]);
        expect(cleared.every(value => Math.abs(value) < 0.001), 'Weather reset retained old cloud, rain, or temperature');
        const dust = await read(fields.dustVelocity.textures[fields.dustVelocity.index]);
        expect(dust.every(Number.isFinite), 'Weather reset left invalid dust momentum');
        passed('Applying dry weather resets the coupled simulation and clears old cloud');

        const originalFetch = window.fetch;
        try {
            let requestedUrl = '';
            window.fetch = async input => {
                requestedUrl = String(input);
                return new Response(JSON.stringify(fixture));
            };
            await weather.load();
            expect(weather.status.startsWith('2024-05-21 21:00 UTC'), 'Loaded sounding status has the wrong hour');
            const url = new URL(requestedUrl);
            expect(url.searchParams.get('timezone') === 'UTC' && url.searchParams.get('wind_speed_unit') === 'ms',
                'Weather request has incorrect time or wind units');
            passed('Weather loading requests UTC and SI wind units and applies the chosen hour');

            const versionBeforeFailure = solver.resetVersion;
            window.fetch = async () => new Response('Unavailable', { status: 503 });
            await weather.load();
            expect(solver.resetVersion === versionBeforeFailure && weather.status.includes('503'),
                'A failed request replaced the current sounding');
            passed('A failed weather request preserves the active simulation');

            let finishRequest!: (response: Response) => void;
            window.fetch = () => new Promise(resolve => { finishRequest = resolve; });
            const loading = weather.load();
            weather.useWestWind();
            finishRequest(new Response(JSON.stringify(fixture)));
            await loading;
            expect(weather.status.includes('toward the west'), 'An old request overwrote the selected preset');
            passed('Selecting a preset prevents a pending sounding from overwriting it');
        } finally {
            window.fetch = originalFetch;
        }

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
