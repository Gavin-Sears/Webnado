import { device, initWebGPU } from '../renderer';
import { constants } from '../shaders/shaders';
import { PingPong } from '../stage/fields';
import { MultigridPressure } from './multigrid_pressure';

const output = document.getElementById('results')!;
const { gridX, gridY, gridZ, domainX } = constants;
const count = gridX * gridY * gridZ;

function expect(condition: boolean, message: string) {
    if (!condition) throw new Error(message);
    output.textContent += `\nPASS ${message}`;
}

async function run() {
    await initWebGPU();
    device.pushErrorScope('validation');
    const pressure = new PingPong('nullspace fixture pressure', 'r32float');
    const create = (format: GPUTextureFormat) => device.createTexture({
        dimension: '3d', format, size: [gridX, gridY, gridZ],
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST
    });
    const mask = create('r32uint');
    const divergence = create('r32float');
    const write = (texture: GPUTexture, values: Uint32Array | Float32Array) => device.queue.writeTexture(
        { texture }, values, { bytesPerRow: gridX * 4, rowsPerImage: gridY }, [gridX, gridY, gridZ]);
    write(mask, new Uint32Array(count).fill(127));
    const atmosphere = device.createBuffer({ size: constants.atmosphereSlices * 32,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const column = new Float32Array(constants.atmosphereSlices * 8);
    for (let slice = 0; slice < constants.atmosphereSlices; slice++) column[slice * 8 + 6] = 1;
    device.queue.writeBuffer(atmosphere, 0, column);
    const solver = new MultigridPressure('nullspace fixture', pressure,
        mask.createView(), divergence.createView(), atmosphere);
    const read = async () => {
        const buffer = device.createBuffer({ size: count * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const encoder = device.createCommandEncoder();
        encoder.copyTextureToBuffer({ texture: pressure.textures[pressure.index] },
            { buffer, bytesPerRow: gridX * 4, rowsPerImage: gridY }, [gridX, gridY, gridZ]);
        device.queue.submit([encoder.finish()]);
        await buffer.mapAsync(GPUMapMode.READ);
        const values = new Float32Array(buffer.getMappedRange()).slice();
        buffer.unmap();
        buffer.destroy();
        return values;
    };
    const solve = async () => {
        for (let iteration = 0; iteration < 12; iteration++) {
            const encoder = device.createCommandEncoder();
            solver.solve(encoder, 0, 6);
            device.queue.submit([encoder.finish()]);
        }
        await device.queue.onSubmittedWorkDone();
        const error = await device.popErrorScope();
        if (error) throw new Error(error.message);
        device.pushErrorScope('validation');
    };
    try {
        write(divergence, new Float32Array(count).fill(1e-4));
        await solve();
        const constant = await read();
        let maximum = 0;
        let gradient = 0;
        for (let index = 0; index < count; index++) {
            maximum = Math.max(maximum, Math.abs(constant[index]));
            if (index % gridX < gridX - 1) {
                gradient = Math.max(gradient, Math.abs(constant[index + 1] - constant[index]) / (domainX / gridX));
            }
        }
        console.log('Nullspace fixture', JSON.stringify({ maximum, gradient }));
        expect(gradient < 1e-6, 'A constant incompatible source does not create airflow');
        expect(maximum < 0.01, 'The pressure gauge stays bounded during repeated solves');

        const amplitude = 500;
        const eigenvalue = 4 * Math.sin(Math.PI / (2 * gridX)) ** 2 / (domainX / gridX) ** 2;
        const source = new Float32Array(count);
        for (let index = 0; index < count; index++) {
            const expected = amplitude * Math.cos(Math.PI * ((index % gridX) + 0.5) / gridX);
            source[index] = -eigenvalue * expected + 1e-4;
        }
        write(divergence, source);
        await solve();
        const result = await read();
        let squared = 0;
        for (let index = 0; index < count; index++) {
            const expected = amplitude * Math.cos(Math.PI * ((index % gridX) + 0.5) / gridX);
            squared += (result[index] - expected) ** 2;
        }
        console.log('Nullspace manufactured', JSON.stringify({ rms: Math.sqrt(squared / count),
            samples: [result[0], result[gridX / 2], result[gridX - 1]] }));
        expect(Math.sqrt(squared / count) < 0.01 * amplitude,
            'A biased source retains the independently known Neumann pressure solution');
        const error = await device.popErrorScope();
        expect(error === null, error?.message ?? 'No WebGPU validation errors');
        output.textContent += '\nAll checks passed.';
    } finally {
        solver.destroy();
        pressure.textures.forEach(texture => texture.destroy());
        mask.destroy();
        divergence.destroy();
        atmosphere.destroy();
    }
}

void run().catch(error => {
    output.textContent += `\nFAIL ${error.message}`;
    console.error(error);
});
