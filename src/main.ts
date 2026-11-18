import { initWebGPU, showFatalError } from './renderer';
import { VolumetricRenderer } from './renderers/volumetric';
import { Stage } from './stage/stage';
import { buildControls } from './ui/controls';

async function start() {
    await initWebGPU();
    const stage = await Stage.create();
    const renderer = new VolumetricRenderer(stage);
    buildControls(stage, renderer);
}

start().catch((error) => {
    console.error(error);
    showFatalError(`Webnado couldn't start: ${error}`);
});
