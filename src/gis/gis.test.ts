import { BoxGeometry, Group, Mesh, Vector3 } from 'three';

import { device, initWebGPU } from '../renderer';
import { constants, processShaderRaw } from '../shaders/shaders';
import { Camera } from '../stage/camera';
import { Fields } from '../stage/fields';
import { WeatherData } from '../weather/noaa_loader';
import { ObstacleGrid } from './obstacle_grid';
import { Debris } from './debris';
import { FluidSolver } from '../solver/fluid_solver';
import { TileLayer } from './tile_layer';
import type { Tile } from '3d-tiles-renderer/core';

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

    const layer = new TileLayer('', 2, undefined, undefined, () => {}, () => {});
    layer.tiles.group.position.set(-1000, -2000, -3000);
    layer.tiles.group.updateMatrixWorld(true);
    const detached = new Mesh(new BoxGeometry(20, 20, 20));
    detached.position.set(1000, 2010, 3000);
    layer.tiles.dispatchEvent({ type: 'load-model', scene: detached, tile: {} as Tile, url: '' });
    expect(Math.abs(layer.tiles.group.position.y + 2000) < 1e-6,
        'Detached tile content was aligned using Earth-space bounds');
    layer.dispose();
    detached.geometry.dispose();
    passed('Tile placement handles model events before scene attachment');

    await initWebGPU();
    device.pushErrorScope('validation');
    const fields = new Fields();
    fields.setObstacles(grid.materials);
    const weather = new WeatherData({ latitude: 41.31, longitude: -94.46 });
    const debris = new Debris(fields, new Camera(), () => {});
    const readback = device.createBuffer({
        size: debris.particles.size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    const forcingRow = Math.ceil(constants.gridX * 8 / 256) * 256;
    const forcingReadback = device.createBuffer({
        size: forcingRow * constants.gridY * constants.gridZ,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    const readVolume = async (texture: GPUTexture) => {
        const encoder = device.createCommandEncoder();
        encoder.copyTextureToBuffer(
            { texture },
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
        passed('Dust tracers remain outside terrain and buildings');

        const seed = new Float32Array(debris.count * 8);
        seed.set([-5 * h, 2 * h, 0, 1, 100 * h, 0, 0, 0.02]);
        device.queue.writeBuffer(debris.particles, 0, seed);
        const flowPipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: device.createShaderModule({ code: processShaderRaw(`
                @group(0) @binding(0) var output: texture_storage_3d<rgba16float, write>;
                @compute @workgroup_size(8, 8, 4)
                fn main(@builtin(global_invocation_id) id: vec3u) {
                    textureStore(output, vec3i(id), vec4f(${100 * h}, 0, 0, 0));
                }
            `) }) }
        });
        const encoder = device.createCommandEncoder();
        const flowPass = encoder.beginComputePass();
        flowPass.setPipeline(flowPipeline);
        flowPass.setBindGroup(0, device.createBindGroup({
            layout: flowPipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: fields.dustVelocity.read }]
        }));
        flowPass.dispatchWorkgroups(constants.gridX / 8, constants.gridY / 8, constants.gridZ / 4);
        flowPass.end();
        debris.step(encoder, 0.5, 0);
        device.queue.submit([encoder.finish()]);
        const swept = await inspect();
        expect(swept[0] < -h, 'Fast debris tunneled through the building');
        passed('Swept collision checks stop debris crossing a building in one step');
        const reaction = await readVolume(fields.windForcing);
        expect(reaction.every(value => value === 0), 'Visual tracers added a second momentum exchange');
        passed('Visual tracers leave momentum exchange to the two-fluid solver');

        debris.clear();
        const clearEncoder = device.createCommandEncoder();
        debris.step(clearEncoder, 0.5, 0);
        device.queue.submit([clearEncoder.finish()]);
        const cleared = await inspect();
        expect(cleared.every(value => value === 0), 'Clear left active debris behind');
        expect((await readVolume(fields.windForcing)).every(value => value === 0), 'Clear left a momentum source behind');
        passed('Clear removes the visual tracers');

        const solver = new FluidSolver(fields, weather);
        solver.startShearExperiment();
        const settings = new URLSearchParams(location.search);
        solver.confinement = Number(settings.get('confinement') ?? solver.confinement);
        solver.dust.loading = Number(settings.get('loading') ?? solver.dust.loading);
        solver.jacobiIterations = Number(settings.get('pressure') ?? solver.jacobiIterations);
        const coupledSteps = Number(new URLSearchParams(location.search).get('steps') ?? 250);
        expect(Number.isInteger(coupledSteps) && coupledSteps > 0, 'steps must be a positive integer');
        try {
            for (let frame = 0; frame < coupledSteps; frame++) {
                const encoder = device.createCommandEncoder();
                solver.step(encoder);
                if (frame === 0) {
                    debris.seed();
                    solver.dust.seed();
                }
                debris.step(encoder, solver.dt, solver.resetVersion);
                device.queue.submit([encoder.finish()]);
                if (frame % 8 === 7) {
                    await device.queue.onSubmittedWorkDone();
                }
            }
            await device.queue.onSubmittedWorkDone();
            await inspect();
            for (const field of [fields.velocity, fields.thermo, fields.turbulence, fields.vorticity, fields.dustVelocity]) {
                const values = await readVolume(field.textures[field.index]);
                expect(values.every(value => (value & 0x7c00) !== 0x7c00), 'Coupled experiment produced nonfinite fields');
            }
            const measure = async (dust = false) => {
                const encoder = device.createCommandEncoder();
                const report = dust ? solver.dust.checkDivergence(encoder) : solver.checkDivergence(encoder);
                device.queue.submit([encoder.finish()]);
                expect(report !== undefined, 'Run the GIS test with checkDivergence in the URL');
                return report!;
            };
            await measure();
            const report = await measure();
            await measure(true);
            const dustReport = await measure(true);
            console.log('Coupled shear experiment', JSON.stringify({ air: report, dust: dustReport }));
            expect(report.rms < 5e-4 && report.nonFinite === 0,
                `Coupled air RMS ${report.rms}, nonfinite ${report.nonFinite} exceeded the target`);
            expect(dustReport.rms < 5e-4 && dustReport.maxBlockedFaceSpeed === 0,
                'Dust projection missed the target or crossed a solid face');
            passed(`Air and dust stay projected around buildings for ${coupledSteps} coupled steps`);
        } finally {
            solver.destroy();
        }

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
