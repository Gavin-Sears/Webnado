import { device, initWebGPU } from '../renderer';
import { constants, processShaderRaw } from '../shaders/shaders';
import { Fields } from '../stage/fields';
import { StormDiagnostics } from './storm_diagnostics';
import { WeatherData } from '../weather/noaa_loader';

const output = document.getElementById('results')!;

function expect(condition: boolean, message: string) {
    if (!condition) throw new Error(message);
    output.textContent += `\nPASS ${message}`;
}

async function run() {
    await initWebGPU();
    device.pushErrorScope('validation');
    const fields = new Fields(false);
    const weather = new WeatherData({ latitude: 0, longitude: 0 });
    const diagnostics = new StormDiagnostics(weather.atmosphereBuffer);
    const module = device.createShaderModule({ code: processShaderRaw(`
        override rotation: bool;
        @group(0) @binding(0) var velocityOut: texture_storage_3d<rgba16float, write>;
        @compute @workgroup_size(8, 8, 4)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            let world = domainMin + (vec3f(id) + vec3f(0.5)) * cellSize;
            var velocity = vec3f(0.04 * world.z, 0, 0);
            if (rotation) {
                velocity = vec3f(10.0 + 0.02 * world.z, 4, -5.0 - 0.02 * world.x);
            }
            textureStore(velocityOut, vec3i(id), vec4f(velocity, 0));
        }
    `) });
    const measure = async (rotation: boolean) => {
        const pipeline = device.createComputePipeline({ layout: 'auto',
            compute: { module, entryPoint: 'main', constants: { rotation: Number(rotation) } } });
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: fields.velocity.read }] }));
        pass.dispatchWorkgroups(constants.gridX / 8, constants.gridY / 8, constants.gridZ / 4);
        pass.end();
        device.queue.submit([encoder.finish()]);
        const result = await diagnostics.measure(fields, 0);
        const error = await device.popErrorScope();
        if (error) throw new Error(error.message);
        device.pushErrorScope('validation');
        return result;
    };
    try {
        const shear = await measure(false);
        expect(shear.groundVorticity > 0.03 && shear.groundSwirlingStrength < 1e-6,
            'A strong shear layer has curl but no swirling strength');
        const rotating = await measure(true);
        expect(Math.abs(rotating.groundSwirlingStrength - 0.02) < 0.002,
            'Translated solid-body rotation retains its known swirling strength');
        expect(rotating.cloudOverheadSwirl === 0,
            'A dry vortex is not reported as rotation beneath rising cloud');
        const cloud = new Uint16Array(constants.gridX * constants.gridY * constants.gridZ * 4);
        for (let index = 2; index < cloud.length; index += 4) cloud[index] = 0x3c00;
        device.queue.writeTexture({ texture: fields.thermo.textures[fields.thermo.index] }, cloud,
            { bytesPerRow: constants.gridX * 8, rowsPerImage: constants.gridY },
            [constants.gridX, constants.gridY, constants.gridZ]);
        const cloudy = await measure(true);
        expect(Math.abs(cloudy.cloudOverheadSwirl - 0.02) < 0.002,
            'The under-cloud diagnostic retains the known swirl beneath a rising cloud');
        const error = await device.popErrorScope();
        expect(error === null, error?.message ?? 'No WebGPU validation errors');
        output.textContent += '\nAll checks passed.';
    } finally {
        diagnostics.destroy();
        weather.atmosphereBuffer.destroy();
    }
}

void run().catch(error => {
    output.textContent += `\nFAIL ${error.message}`;
    console.error(error);
});
