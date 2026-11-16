import { device } from '../renderer';
import { constants } from '../shaders/shaders';
import type { Fields } from '../stage/fields';
import type { FluidSolver } from '../solver/fluid_solver';
import type { WeatherData } from './noaa_loader';
import { halfToFloat } from '../solver/divergence_check';
import { readTexture } from './surface_exchange.test';

export async function checkMovingWeather(solver: FluidSolver, fields: Fields, weather: WeatherData) {
    if (weather.spatial!.manifest.version < 3) return;
    const { gridX, gridY, gridZ, domainX, domainY, domainZ } = constants;
    const beforeWind = new Float32Array(await readTexture(fields.velocity.textures[fields.velocity.index]));
    const beforeThermo = new Uint16Array(await readTexture(fields.thermo.textures[fields.thermo.index]));
    const dx = 4;
    const dz = -3;
    solver.storm.shiftX = dx;
    solver.storm.shiftZ = dz;
    solver.storm.write(0);
    if (!weather.moveSpatialWindow(dx * domainX / gridX, -dz * domainZ / gridZ)) {
        throw new Error('The test translation does not fit in the source region.');
    }
    const encoder = device.createCommandEncoder();
    solver['dispatch'](encoder, 'shiftStormWindow');
    fields.velocity.swap();
    fields.thermo.swap();
    fields.turbulence.swap();
    device.queue.submit([encoder.finish()]);
    const wind = new Float32Array(await readTexture(fields.velocity.textures[fields.velocity.index]));
    const thermo = new Uint16Array(await readTexture(fields.thermo.textures[fields.thermo.index]));
    for (let z = -dz; z < gridZ; z++) {
        for (let y = 0; y < gridY; y++) {
            for (let x = 0; x < gridX - dx; x++) {
                const target = 4 * (x + gridX * (y + gridY * z));
                const source = 4 * (x + dx + gridX * (y + gridY * (z + dz)));
                for (let channel = 0; channel < 4; channel++) {
                    if (wind[target + channel] !== beforeWind[source + channel]
                        || thermo[target + channel] !== beforeThermo[source + channel]) {
                        throw new Error('The moving window changed a copied interior value.');
                    }
                }
            }
        }
    }
    const cell = [gridX - 2, 2, gridZ / 2];
    const index = 4 * (cell[0] + gridX * (cell[1] + gridY * cell[2]));
    for (let axis = 0; axis < 3; axis++) {
        const position = cell.map((value, component) => value + (component === axis ? 0 : 0.5));
        const sample = weather.spatial!.sample(position[0] / gridX * domainX - domainX / 2,
            position[1] / gridY * domainY, position[2] / gridZ * domainZ - domainZ / 2);
        if (Math.abs(wind[index + axis] - [sample.east, sample.vertical, sample.south][axis]) > 0.015) {
            throw new Error('The newly exposed strip does not match the translated weather location.');
        }
    }
    const sourceState = weather.spatial!.sample((cell[0] + 0.5) / gridX * domainX - domainX / 2,
        (cell[1] + 0.5) / gridY * domainY, (cell[2] + 0.5) / gridZ * domainZ - domainZ / 2);
    if (Math.abs(halfToFloat(thermo[index + 2]) - sourceState.cloud) > 0.02) {
        throw new Error('The newly exposed cloud field does not match the real weather input.');
    }
    if (weather.moveSpatialWindow(1e9, 1e9)) {
        throw new Error('The weather window accepted a location outside the extracted region.');
    }
    solver.reset();
    const reset = device.createCommandEncoder();
    solver.step(reset);
    device.queue.submit([reset.finish()]);
    await device.queue.onSubmittedWorkDone();
    if (weather.spatial!.windowEast !== 0 || weather.spatial!.windowNorth !== 0) {
        throw new Error('Reset did not restore the original weather coordinates.');
    }
}
