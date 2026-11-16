import { device } from '../renderer';
import { constants } from '../shaders/shaders';
import { Fields } from '../stage/fields';
import { FluidSolver } from '../solver/fluid_solver';
import type { WeatherData } from './noaa_loader';
import { readScalarField } from './surface_exchange.test';

export async function checkScalarPrecision(weather: WeatherData) {
    if (!constants.preciseScalars) return;
    const fields = new Fields(false);
    const solver = new FluidSolver(fields, weather);
    solver.startSpatialExperiment();
    solver.paused = true;
    const reset = device.createCommandEncoder();
    solver.step(reset);
    device.queue.submit([reset.finish()]);
    await device.queue.onSubmittedWorkDone();

    const { gridX, gridY, gridZ } = constants;
    const size = [gridX, gridY, gridZ];
    const layout = { bytesPerRow: gridX * 16, rowsPerImage: gridY };
    const uniform = (value: number[]) => {
        const data = new Float32Array(gridX * gridY * gridZ * 4);
        for (let index = 0; index < data.length; index += 4) data.set(value, index);
        return data;
    };
    const zero = uniform([0, 0, 0, 0]);
    device.queue.writeTexture({ texture: fields.velocity.textures[fields.velocity.index] }, zero, layout, size);
    device.queue.writeTexture({ texture: fields.thermo.textures[fields.thermo.index] },
        uniform([4.0002, 1.0001, 0.0001, 0]), layout, size);
    device.queue.writeTexture({ texture: fields.turbulence.textures[fields.turbulence.index] }, zero, layout, size);
    solver.buoyancy = 0;
    solver.mixing = 0;
    solver.turbulence = false;
    solver.transportedVorticity = false;
    solver.microphysics = undefined;
    solver.storm.followStorm = false;
    solver.storm.highOrder = true;
    solver.dt = 0.05;
    solver.paused = false;
    const step = device.createCommandEncoder();
    solver.step(step);
    device.queue.submit([step.finish()]);
    const transported = await readScalarField(fields.thermo.textures[fields.thermo.index]);
    const probe = 4 * (gridX / 2 + gridX * (gridY / 2 + gridY * (gridZ / 2)));
    if (Math.abs(transported[probe] - 4.0002) > 0.00005
        || Math.abs(transported[probe + 1] - 1.0001) > 0.00002) {
        throw new Error('High-order transport rounded precise scalars through a half-precision scratch texture.');
    }

    device.queue.writeTexture({ texture: fields.velocity.textures[fields.velocity.index] }, zero, layout, size);
    device.queue.writeTexture({ texture: fields.turbulence.textures[fields.turbulence.index] },
        uniform([4, 0, 0, 0]), layout, size);
    device.queue.writeBuffer(solver['parameters'], 0, new Float32Array([1, 0, 0, 0, 0, 1, 0, 0]));
    const decay = device.createCommandEncoder();
    solver['dispatch'](decay, 'buildVelocityGradients');
    for (let index = 0; index < 200; index++) {
        solver['dispatch'](decay, 'energy');
        fields.turbulence.swap();
    }
    device.queue.submit([decay.finish()]);
    const energy = await readScalarField(fields.turbulence.textures[fields.turbulence.index]);
    const expected = 4 / (1 + 0.5 * 0.2 * 2 * 200 / (constants.domainX / gridX)) ** 2;
    if (Math.abs(energy[probe] / expected - 1) > 0.01) {
        throw new Error(`Uniform turbulence decay disagrees with its analytic solution: ${energy[probe]} versus ${expected}.`);
    }
    solver.destroy();
}
