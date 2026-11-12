import { BoxGeometry, Group, Mesh, Vector3 } from 'three';

import { device, initWebGPU } from '../renderer';
import { constants } from '../shaders/shaders';
import { Camera } from '../stage/camera';
import { Fields } from '../stage/fields';
import { WeatherData } from '../weather/noaa_loader';
import { ObstacleGrid } from './obstacle_grid';
import { Debris } from './debris';

const output = document.getElementById('results')!;
const h = constants.domainX / constants.gridX;

function expect(condition: boolean, message: string) {
    if (!condition) {
        throw new Error(message);
    }
}

function passed(message: string) {
    output.textContent += `\nPASS ${message}`;
}

async function run() {
    output.textContent = '';
    const grid = new ObstacleGrid();
    const building = new Mesh(new BoxGeometry(2 * h, 4 * h, 2 * h));
    building.position.set(0, 2 * h, 0);
    building.updateMatrixWorld(true);
    await grid.addMesh(building, 2);
    const indexAt = (position: Vector3) => {
        const x = Math.floor((position.x + constants.domainX / 2) / h);
        const y = Math.floor(position.y / h);
        const z = Math.floor((position.z + constants.domainZ / 2) / h);
        return x + constants.gridX * (y + constants.gridY * z);
    };
    expect(grid.materials[indexAt(new Vector3(0, 2 * h, 0))] === 2, 'Building interior stayed empty');
    expect(grid.materials[indexAt(new Vector3(5 * h, 2 * h, 0))] === 0, 'Building filled unrelated space');
    passed('Building surfaces and interiors become solid cells');

    const parent = new Group();
    const translated = new Mesh(new BoxGeometry(h, 3 * h, h));
    translated.position.y = 1.5 * h;
    parent.position.set(8 * h, 0, 5 * h);
    parent.rotation.y = Math.PI / 4;
    parent.add(translated);
    parent.updateMatrixWorld(true);
    await grid.addMesh(translated, 2);
    expect(grid.materials[indexAt(new Vector3(8 * h, 1.5 * h, 5 * h))] === 2,
        'World transforms were lost during collision baking');
    const snapshot = grid.materials.slice();
    parent.remove(translated);
    translated.geometry.dispose();
    expect(grid.materials.every((value, index) => value === snapshot[index]), 'Unloading changed the collision snapshot');
    passed('Collision snapshots retain world transforms and survive unloading');

    await initWebGPU();
    device.pushErrorScope('validation');
    const fields = new Fields();
    fields.setObstacles(grid.materials);
    const weather = new WeatherData({ latitude: 41.31, longitude: -94.46 });
    const debris = new Debris(fields, weather, new Camera());
    const readback = device.createBuffer({
        size: debris.particles.size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    const forcingRow = Math.ceil(constants.gridX * 8 / 256) * 256;
    const forcingReadback = device.createBuffer({
        size: forcingRow * constants.gridY * constants.gridZ,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    const readForcing = async () => {
        const encoder = device.createCommandEncoder();
        encoder.copyTextureToBuffer(
            { texture: fields.windForcing },
            { buffer: forcingReadback, bytesPerRow: forcingRow, rowsPerImage: constants.gridY },
            [constants.gridX, constants.gridY, constants.gridZ]
        );
        device.queue.submit([encoder.finish()]);
        await forcingReadback.mapAsync(GPUMapMode.READ);
        const values = new Uint16Array(forcingReadback.getMappedRange()).slice();
        forcingReadback.unmap();
        return values;
    };

    const inspect = async () => {
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(debris.particles, 0, readback, 0, readback.size);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const values = new Float32Array(readback.getMappedRange()).slice();
        readback.unmap();
        expect(values.every(Number.isFinite), 'Debris produced nonfinite state');
        for (let index = 0; index < values.length; index += 8) {
            if (values[index + 3] !== 0) {
                const material = grid.materials[indexAt(new Vector3(values[index], values[index + 1], values[index + 2]))];
                expect(material === 0, 'Debris ended inside an obstacle');
            }
        }
        return values;
    };

    try {
        debris.seed();
        for (let frame = 0; frame < 60; frame++) {
            const encoder = device.createCommandEncoder();
            debris.step(encoder, 0.5, 0);
            device.queue.submit([encoder.finish()]);
            if (frame % 10 === 9) {
                await device.queue.onSubmittedWorkDone();
            }
        }
        const settled = await inspect();
        expect(settled.some((value, index) => index % 8 === 3 && value === 1), 'No active debris was seeded');
        passed('GPU debris settles without entering terrain or buildings');

        const seed = new Float32Array(debris.count * 8);
        seed.set([-5 * h, 2 * h, 0, 1, 100 * h, 0, 0, 0.02]);
        device.queue.writeBuffer(debris.particles, 0, seed);
        debris.response = 1000;
        const encoder = device.createCommandEncoder();
        debris.step(encoder, 0.5, 0);
        device.queue.submit([encoder.finish()]);
        const swept = await inspect();
        expect(swept[0] < -h, 'Fast debris tunneled through the building');
        passed('Swept collision checks stop debris crossing a building in one step');
        const reaction = await readForcing();
        const originX = constants.gridX / 2 - 5;
        const originZ = constants.gridZ / 2;
        const offset = forcingRow / 2 * (2 + constants.gridY * originZ) + originX * 4;
        expect(reaction[offset] > 0 && (reaction[offset] & 0x8000) === 0,
            'Drag did not return positive x momentum to the air');
        passed('Particle drag transfers opposite momentum back to the airflow');

        debris.clear();
        const clearEncoder = device.createCommandEncoder();
        debris.step(clearEncoder, 0.5, 0);
        device.queue.submit([clearEncoder.finish()]);
        const cleared = await inspect();
        expect(cleared.every(value => value === 0), 'Clear left active debris behind');
        expect((await readForcing()).every(value => value === 0), 'Clear left a momentum source behind');
        passed('Clear removes particles and their momentum source');

        const error = await device.popErrorScope();
        expect(error === null, error?.message ?? 'GPU validation failed');
        passed('No WebGPU validation errors');
        output.textContent += '\n\nAll checks passed.';
    } finally {
        readback.destroy();
        forcingReadback.destroy();
        debris.destroy();
        weather.atmosphereBuffer.destroy();
    }
}

void run().catch(error => {
    output.textContent += `\nFAIL ${error.message}`;
    console.error(error);
});
