import * as renderer from "../renderer";
import * as shaders from "../shaders/shaders";
import type { Stage } from "../stage/stage";

const TEST_VOLUMES = { ellipsoid: 0, fbm: 1 };
type TestVolume = keyof typeof TEST_VOLUMES | "none";
const requested = new URLSearchParams(location.search).get("test") ?? "";
export const TEST_VOLUME: TestVolume = Object.keys(TEST_VOLUMES).includes(requested) ? requested as TestVolume : "none";

export function fillTestVolume(stage: Stage) {
    if (TEST_VOLUME === "none") {
        return;
    }

    const { fields, solver, weather } = stage;
    solver.paused = true;
    weather.onApplied = () => {};

    const device = renderer.device;
    const { gridX, gridY, gridZ } = shaders.constants;
    const pipeline = device.createComputePipeline({
        label: "test volume",
        layout: "auto",
        compute: {
            module: device.createShaderModule({ label: "test volume shader", code: shaders.testVolumeComputeSrc }),
            constants: { testVolume: TEST_VOLUMES[TEST_VOLUME] }
        }
    });

    const encoder = device.createCommandEncoder({ label: "test volume" });
    const pass = encoder.beginComputePass({ label: "test volume pass" });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: fields.thermo.write }]
    }));
    pass.dispatchWorkgroups(gridX / 4, gridY / 4, gridZ / 4);
    pass.end();
    device.queue.submit([encoder.finish()]);
    fields.thermo.swap();
}
