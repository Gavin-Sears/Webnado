import { device, initWebGPU } from '../renderer';
import { constants, processShaderRaw } from '../shaders/shaders';
import { Fields } from '../stage/fields';
import { WeatherData } from '../weather/noaa_loader';
import { FluidSolver } from './fluid_solver';
import { StormDiagnostics } from './storm_diagnostics';
import { halfToFloat } from './divergence_check';

const output = document.getElementById('results')!;
const { gridX, gridY, gridZ, domainX, domainY, domainZ } = constants;

function expect(condition: boolean, message: string) {
    if (!condition) throw new Error(message);
    output.textContent += `\nPASS ${message}`;
}

async function run() {
    expect(Boolean(constants.anelasticAir), 'The fixture uses case=elreno&projection=anelastic');
    await initWebGPU();
    device.pushErrorScope('validation');
    const fields = new Fields(false);
    fields.setObstacles(new Uint8Array(gridX * gridY * gridZ));
    const weather = new WeatherData({ latitude: 0, longitude: 0 });
    const solver = new FluidSolver(fields, weather);
    solver.dust.enabled = false;
    solver.multigridCycles = 8;
    const diagnostics = new StormDiagnostics(weather.atmosphereBuffer);
    const module = device.createShaderModule({ code: processShaderRaw(`
        override circulation: bool;
        override scaleHeight: f32;
        @group(0) @binding(0) var velocityOut: texture_storage_3d<${constants.velocityFormat}, write>;
        fn potential(x: f32) -> f32 {
            return 100000.0 * cos(3.14159265359 * (x + 0.5) / f32(gridDims.x));
        }
        fn stream(x: f32, y: f32) -> f32 {
            let a = sin(3.14159265359 * x / f32(gridDims.x));
            let b = sin(3.14159265359 * y / f32(gridDims.y));
            return 20000.0 * a * a * b * b;
        }
        @compute @workgroup_size(8, 8, 4)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            let x = f32(id.x);
            let y = f32(id.y);
            var velocity = vec3f((potential(x) - potential(x - 1.0)) / cellSize.x, 0, 0);
            if (id.x == 0u) {
                velocity.x = 0.0;
            }
            if (circulation) {
                let rhoCenter = exp(-(y + 0.5) * cellSize.y / scaleHeight);
                let rhoFace = exp(-y * cellSize.y / scaleHeight);
                velocity.x = (stream(x, y + 1.0) - stream(x, y)) / (cellSize.y * rhoCenter);
                velocity.y = -(stream(x + 1.0, y) - stream(x, y)) / (cellSize.x * rhoFace);
            }
            textureStore(velocityOut, vec3i(id), vec4f(velocity, 0));
        }
    `) });
    const setAtmosphere = (scale: number) => {
        const values = new Float32Array(constants.atmosphereSlices * 8);
        for (let slice = 0; slice < constants.atmosphereSlices; slice++) {
            const height = (slice + 0.5) / constants.atmosphereSlices * domainY;
            const density = Math.exp(-height / scale);
            values.set([300, 0, 0, 0, 100000 * density, 300, density, 0], slice * 8);
        }
        weather.setAtmosphere(values);
    };
    const seed = (circulation: boolean, scale: number) => {
        const pipeline = device.createComputePipeline({ layout: 'auto',
            compute: { module, entryPoint: 'main', constants: { circulation: Number(circulation), scaleHeight: scale } } });
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: fields.velocity.read }] }));
        pass.dispatchWorkgroups(gridX / 8, gridY / 8, gridZ / 4);
        pass.end();
        device.queue.submit([encoder.finish()]);
    };
    const project = () => {
        const encoder = device.createCommandEncoder();
        solver.project(encoder);
        device.queue.submit([encoder.finish()]);
    };
    const read = async () => {
        const precise = constants.velocityFormat === 'rgba32float';
        const texelBytes = precise ? 16 : 8;
        const size = gridX * gridY * gridZ * texelBytes;
        const buffer = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const encoder = device.createCommandEncoder();
        encoder.copyTextureToBuffer({ texture: fields.velocity.textures[fields.velocity.index] },
            { buffer, bytesPerRow: gridX * texelBytes, rowsPerImage: gridY }, [gridX, gridY, gridZ]);
        device.queue.submit([encoder.finish()]);
        await buffer.mapAsync(GPUMapMode.READ);
        const data = buffer.getMappedRange();
        const values = precise ? new Float32Array(data).slice() : Float32Array.from(new Uint16Array(data), halfToFloat);
        buffer.unmap();
        buffer.destroy();
        return values;
    };
    try {
        setAtmosphere(8500);
        seed(false, 8500);
        project();
        const gradient = await diagnostics.measure(fields, 0);
        let error = await device.popErrorScope();
        if (error) throw new Error(error.message);
        device.pushErrorScope('validation');
        expect(gradient.massWeighted && gradient.divergenceRms < 3e-5 && gradient.speed < 0.05,
            'Variable-density multigrid removes a manufactured pressure-gradient flow');

        seed(true, 8500);
        const before = await read();
        project();
        const after = await read();
        const circulation = await diagnostics.measure(fields, 0);
        let difference = 0;
        let energy = 0;
        let volumeSquared = 0;
        let samples = 0;
        const spacing = [domainX / gridX, domainY / gridY, domainZ / gridZ];
        for (let z = 0; z < gridZ - 1; z++) {
            for (let y = 0; y < gridY - 1; y++) {
                for (let x = 0; x < gridX - 1; x++) {
                    const index = 4 * (x + gridX * (y + gridY * z));
                    let divergence = 0;
                    for (let axis = 0; axis < 3; axis++) {
                        const offset = [4, 4 * gridX, 4 * gridX * gridY][axis];
                        difference += (before[index + axis] - after[index + axis]) ** 2;
                        energy += before[index + axis] ** 2;
                        divergence += (after[index + offset + axis] - after[index + axis]) / spacing[axis];
                    }
                    volumeSquared += divergence * divergence;
                    samples++;
                }
            }
        }
        expect(Math.sqrt(difference / energy) < 0.01 && circulation.divergenceRms < 3e-5,
            'Projection preserves a discrete mass-conserving circulation');
        expect(Math.sqrt(volumeSquared / samples) > 1e-4,
            'Mass conservation permits the expected expansion as air rises');

        setAtmosphere(6000);
        seed(false, 6000);
        project();
        const changed = await diagnostics.measure(fields, 0);
        expect(changed.divergenceRms < 3e-5 && changed.speed < 0.05,
            'A changed density profile rebuilds the multigrid coefficients');
        error = await device.popErrorScope();
        expect(error === null, error?.message ?? 'No WebGPU validation errors');
        output.textContent += '\nAll checks passed.';
    } finally {
        diagnostics.destroy();
        solver.destroy();
        weather.atmosphereBuffer.destroy();
    }
}

void run().catch(error => {
    output.textContent += `\nFAIL ${error.message}`;
    console.error(error);
});
