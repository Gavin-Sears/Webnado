import { device } from '../renderer';
import { constants } from '../shaders/shaders';
import type { Fields } from '../stage/fields';
import type { FluidSolver } from '../solver/fluid_solver';
import { halfToFloat } from '../solver/divergence_check';
import type { WeatherData } from './noaa_loader';

export async function readTexture(texture: GPUTexture): Promise<ArrayBuffer> {
    const bytesPerTexel = texture.format === 'rgba32float' ? 16 : 8;
    const buffer = device.createBuffer({ size: constants.gridX * constants.gridY * constants.gridZ * bytesPerTexel,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow: constants.gridX * bytesPerTexel,
        rowsPerImage: constants.gridY }, [constants.gridX, constants.gridY, constants.gridZ]);
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const result = buffer.getMappedRange().slice(0);
    buffer.unmap();
    buffer.destroy();
    return result;
}

export async function readBuffer(source: GPUBuffer): Promise<ArrayBuffer> {
    const buffer = device.createBuffer({ size: source.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, buffer, 0, source.size);
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const result = buffer.getMappedRange().slice(0);
    buffer.unmap();
    buffer.destroy();
    return result;
}

export async function checkSurfaceExchange(solver: FluidSolver, fields: Fields, weather: WeatherData) {
    if (weather.spatial!.manifest.version < 2) return;
    const spatial = weather.spatial!;
    const { gridX, gridY, gridZ, domainX, domainY, domainZ, atmosphereSlices } = constants;
    const dt = 1;
    const steps = 120;
    const before = new Uint16Array(await readTexture(fields.thermo.textures[fields.thermo.index]));
    weather.writeSpatialTime(0);
    device.queue.writeBuffer(solver['parameters'], 0, new Float32Array([dt, 0, 0, 0, 0, 0, 0, 0]));
    const encoder = device.createCommandEncoder();
    solver['prepareBoundaries'](encoder);
    for (let step = 0; step < steps; step++) {
        solver['exchangeSurface'](encoder);
    }
    device.queue.submit([encoder.finish()]);
    const after = new Uint16Array(await readTexture(fields.thermo.textures[fields.thermo.index]));
    const pending = new Float32Array(await readBuffer(solver['surfaceResiduals'][solver['surfaceResidualIndex']]));
    const column = spatial.referenceColumn(atmosphereSlices, domainY, domainX, domainZ);
    const slice = 0.5 / gridY * atmosphereSlices - 0.5;
    const lower = Math.floor(slice);
    const upper = lower + 1;
    const ambient = (channel: number) => column[lower * 8 + channel] * (1 - slice % 1)
        + column[upper * 8 + channel] * (slice % 1);
    const exner = Math.pow(ambient(4) / 100000, 287.05 / 1004);
    const area = domainX / gridX * domainZ / gridZ;
    const dy = domainY / gridY;
    let expectedHeat = 0;
    let actualHeat = 0;
    let expectedWater = 0;
    let actualWater = 0;
    for (let z = 0; z < gridZ; z++) {
        for (let x = 0; x < gridX; x++) {
            const index = 4 * (x + gridX * gridY * z);
            const flux = spatial.surface((x + 0.5) / gridX * domainX - domainX / 2,
                (z + 0.5) / gridZ * domainZ - domainZ / 2, 0);
            const dryDensity = ambient(6) / (1 + 0.001 * ambient(1));
            const residualIndex = 4 * (x + gridX * z);
            const capacity = [dryDensity * 1004 * exner * dy, dryDensity * dy * 0.001];
            actualHeat += ((halfToFloat(after[index]) - halfToFloat(before[index])) * capacity[0]
                + pending[residualIndex]) * area;
            actualWater += ((halfToFloat(after[index + 1]) - halfToFloat(before[index + 1])) * capacity[1]
                + pending[residualIndex + 1]) * area;
            expectedHeat += flux[1] * dt * steps * area;
            expectedWater += flux[2] / 2.5e6 * dt * steps * area;
            for (let channel = 0; channel < 2; channel++) {
                const value = halfToFloat(after[index + channel]);
                const quantum = Math.max(2 ** -24, 2 ** (Math.floor(Math.log2(Math.abs(value))) - 10));
                const roundingBound = Math.max(1.01 * quantum, 2 ** -14) + 1e-6;
                if (Math.abs(pending[residualIndex + channel] / capacity[channel]) > roundingBound) {
                    throw new Error(`Pending surface flux at ${x},${z}, channel ${channel}: remainder ${pending[residualIndex + channel]}, value ${value}, quantum ${quantum}`);
                }
            }
        }
    }
    const heatError = Math.abs(actualHeat / expectedHeat - 1);
    const waterError = Math.abs(actualWater / expectedWater - 1);
    if (heatError > 0.03 || waterError > 0.03) {
        throw new Error(`Surface budgets disagree: heat ${heatError}, water ${waterError}`);
    }
    for (let z = 0; z < gridZ; z++) {
        for (let y = 1; y < gridY; y++) {
            for (let x = 0; x < gridX; x++) {
                const index = 4 * (x + gridX * (y + gridY * z));
                if (before[index] !== after[index] || before[index + 1] !== after[index + 1]) {
                    throw new Error('Surface exchange changed a cell above the surface layer.');
                }
            }
        }
    }
    const zeros = new Uint16Array(gridX * gridY * gridZ * 4);
    device.queue.writeTexture({ texture: fields.turbulence.textures[fields.turbulence.index] }, zeros,
        { bytesPerRow: gridX * 8, rowsPerImage: gridY }, [gridX, gridY, gridZ]);
    const forceEncoder = device.createCommandEncoder();
    solver['dispatch'](forceEncoder, 'atmosphericForces');
    device.queue.submit([forceEncoder.finish()]);
    const forces = new Float32Array(await readTexture(solver['atmosphericForce']));
    const velocity = new Float32Array(await readTexture(fields.velocity.textures[fields.velocity.index]));
    const x = gridX / 2;
    const z = gridZ / 2;
    const index = 4 * (x + gridX * gridY * z);
    const wind = [0.5 * (velocity[index] + velocity[index + 4]),
        0.5 * (velocity[index + 2] + velocity[index + 4 * gridX * gridY + 2])];
    const roughness = spatial.surface((x + 0.5) / gridX * domainX - domainX / 2,
        (z + 0.5) / gridZ * domainZ - domainZ / 2, 0)[0];
    const coefficient = Math.pow(0.4 / Math.log((0.5 * dy + roughness) / roughness), 2);
    const rate = coefficient * Math.hypot(...wind) / dy;
    [0, 2].forEach((axis, component) => {
        const expected = -wind[component] * rate / (1 + dt * rate);
        if (Math.abs(forces[index + axis] - expected) > 1e-5) {
            throw new Error('The surface momentum sink does not match the roughness drag law.');
        }
    });
    console.log('Surface budgets and drag', { heatError, waterError, coefficient });
    solver.reset();
    const reset = device.createCommandEncoder();
    solver.step(reset);
    device.queue.submit([reset.finish()]);
    await device.queue.onSubmittedWorkDone();
}
