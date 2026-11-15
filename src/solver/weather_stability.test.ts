import { BoxGeometry, Mesh } from 'three';

import { device, initWebGPU } from '../renderer';
import { constants, domainName } from '../shaders/shaders';
import { ObstacleGrid } from '../gis/obstacle_grid';
import { Fields } from '../stage/fields';
import { WeatherData } from '../weather/noaa_loader';
import { halfToFloat } from './divergence_check';
import { FluidSolver } from './fluid_solver';
import sounding from '../../tests/fixtures/greenfield-local.json';

const output = document.getElementById('results')!;
const { gridX, gridY, gridZ } = constants;

function expect(condition: boolean, message: string) {
    if (!condition) {
        throw new Error(message);
    }
}

async function measure(fields: Fields) {
    const row = Math.ceil(gridX * 8 / 256) * 256;
    const volumeBytes = row * gridY * gridZ;
    const buffer = device.createBuffer({
        size: volumeBytes * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    const encoder = device.createCommandEncoder();
    const textures = [fields.thermo, fields.velocity, fields.vorticity, fields.turbulence];
    for (const [index, field] of textures.entries()) {
        encoder.copyTextureToBuffer(
            { texture: field.textures[field.index] },
            { buffer, offset: index * volumeBytes, bytesPerRow: row, rowsPerImage: gridY },
            [gridX, gridY, gridZ]
        );
    }
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const data = new DataView(buffer.getMappedRange());
    const report = { maxTemperature: 0, maxSpeed: 0, maxVorticity: 0, maxEnergy: 0, nonFinite: 0 };
    const valueAt = (field: number, offset: number, channel = 0) => {
        const value = halfToFloat(data.getUint16(field * volumeBytes + offset + channel * 2, true));
        if (!Number.isFinite(value)) {
            report.nonFinite++;
        }
        return value;
    };
    for (let z = 0; z < gridZ; z++) {
        for (let y = 0; y < gridY; y++) {
            for (let x = 0; x < gridX; x++) {
                const offset = row * (y + gridY * z) + x * 8;
                report.maxTemperature = Math.max(report.maxTemperature, Math.abs(valueAt(0, offset)));
                const u = valueAt(1, offset);
                const v = valueAt(1, offset, 1);
                const w = valueAt(1, offset, 2);
                report.maxSpeed = Math.max(report.maxSpeed, Math.sqrt(u * u + v * v + w * w));
                report.maxVorticity = Math.max(report.maxVorticity, valueAt(2, offset, 3));
                report.maxEnergy = Math.max(report.maxEnergy, valueAt(3, offset));
            }
        }
    }
    buffer.unmap();
    buffer.destroy();
    return report;
}

async function run() {
    expect(domainName === 'local', 'Use domain=local for the captured sounding');
    await initWebGPU();
    device.pushErrorScope('validation');
    const fields = new Fields();
    const grid = new ObstacleGrid();
    for (const [x, z, height] of [[-1200, 0, 800], [1100, -600, 600], [700, 1300, 450]]) {
        const building = new Mesh(new BoxGeometry(112.5, height / 4, 150));
        building.position.set(x / 4, height / 8, z / 4);
        building.updateMatrixWorld(true);
        await grid.addMesh(building, 2);
        building.geometry.dispose();
    }
    fields.setObstacles(grid.materials);
    const weather = new WeatherData({ latitude: sounding.latitude, longitude: sounding.longitude });
    weather.setAtmosphere(new Float32Array(sounding.profile.flat()));
    const solver = new FluidSolver(fields, weather);
    solver.reset();
    solver.addThermal();
    const steps = Number(new URLSearchParams(location.search).get('steps') ?? 800);
    output.textContent = '';

    try {
        for (let step = 1; step <= steps; step++) {
            const encoder = device.createCommandEncoder();
            solver.step(encoder);
            device.queue.submit([encoder.finish()]);
            if (step % 100 === 0 || step === 1) {
                const report = await measure(fields);
                console.log('Loaded weather stability', { step, ...report });
                expect(report.nonFinite === 0, 'Loaded weather produced nonfinite fields');
                expect(report.maxSpeed < 100, 'Air velocity ran away in the captured sounding');
                expect(report.maxVorticity < 5, 'Transported vorticity drifted away from the airflow');
                expect(report.maxTemperature < 10, 'Temperature perturbations ran away');
                expect(report.maxEnergy < 500, 'Turbulent energy ran away');
                output.textContent = `Completed ${step}/${steps} steps.\n${JSON.stringify(report, null, 2)}`;
            } else if (step % 8 === 0) {
                await device.queue.onSubmittedWorkDone();
            }
        }
        const encoder = device.createCommandEncoder();
        const divergence = solver.checkDivergence(encoder);
        device.queue.submit([encoder.finish()]);
        expect(divergence !== undefined, 'Add checkDivergence to the test URL');
        const report = await divergence!;
        expect(report.rms < 5e-4 && report.maxBlockedFaceSpeed === 0, 'Loaded weather lost pressure or solid boundaries');
        const error = await device.popErrorScope();
        expect(error === null, error?.message ?? 'GPU validation failed');
        output.textContent += `\n\nPASS Loaded weather and a warm thermal stay bounded for ${steps} steps.\nAll checks passed.`;
    } finally {
        solver.destroy();
        weather.atmosphereBuffer.destroy();
    }
}

void run().catch(error => {
    output.textContent += `\nFAIL ${error.message}`;
    console.error(error);
});
