import GUI from 'lil-gui';

// import { initWebGPU, showFatalError } from './renderer';
// replace below with above after adding new logic
import { context, device, initWebGPU, Renderer, showFatalError } from './renderer';
// import { VolumetricRenderer } from './renderers/volumetric';
import { resolutionName, resolutions } from './shaders/shaders';

import { Stage } from './stage/stage';

async function start() {
    await initWebGPU();

    const stage = await Stage.create();
    // const renderer = new VolumetricRenderer(stage);
    new ClearRenderer(stage);

    const gui = new GUI();

    // When we change sim resolution, just reload page.
    // Deleting and re-creating all the buffers is something we can add later.
    gui.add({ resolution: resolutionName }, 'resolution', Object.keys(resolutions)).onChange((resolution: string) => {
        const url = new URL(window.location.href);
        url.searchParams.set('res', resolution);
        window.location.href = url.href;
    });

    // stage.weather.buildGui(gui);
    // stage.gis.buildGui(gui);
    // stage.solver.buildGui(gui);
    // renderer.buildGui(gui);
}

// Temporary clear screen renderer
class ClearRenderer extends Renderer {
    protected override draw() {
        const encoder = device.createCommandEncoder();
        encoder.beginRenderPass({
            colorAttachments: [{
                view: context.getCurrentTexture().createView(),
                clearValue: [0.32, 0.42, 0.58, 1],
                loadOp: "clear",
                storeOp: "store"
            }]
        }).end();
        device.queue.submit([encoder.finish()]);
    }
}

start().catch((error) => {
    console.error(error);
    showFatalError(`Webnado couldn't start: ${error}`);
});
