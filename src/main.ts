import GUI from 'lil-gui';

import { initWebGPU, showFatalError } from './renderer';
import { VolumetricRenderer } from './renderers/volumetric';
import { constants, domainName, domains, resolutionName, resolutions } from './shaders/shaders';

import { Stage } from './stage/stage';

async function start() {
    await initWebGPU();

    const stage = await Stage.create();
    const renderer = new VolumetricRenderer(stage);

    const gui = new GUI();

    // When we change sim resolution, just reload page.
    // Deleting and re-creating all the buffers is something we can add later.
    gui.add({ resolution: resolutionName }, 'resolution', Object.keys(resolutions)).onChange((resolution: string) => {
        const url = new URL(window.location.href);
        url.searchParams.set('res', resolution);
        window.location.href = url.href;
    });
    const precision = { velocity: constants.velocityFormat === 'rgba32float' ? '32' : '16' };
    gui.add(precision, 'velocity', { '16 bit': '16', '32 bit': '32' }).name('Velocity precision').onChange((value: string) => {
        const url = new URL(window.location.href);
        url.searchParams.set('precision', value);
        window.location.href = url.href;
    });
    gui.add({ domain: domainName }, 'domain', Object.keys(domains)).onChange((domain: string) => {
        const url = new URL(window.location.href);
        url.searchParams.set('domain', domain);
        url.searchParams.delete('case');
        window.location.href = url.href;
    });

    stage.weather.buildGui(gui);
    stage.gis.buildGui(gui);
    stage.solver.buildGui(gui);
    stage.debris.buildGui(gui, stage.solver.storm.enabled
        ? () => stage.solver.storm.cloudRotationLocation ?? stage.solver.storm.rotationLocation : undefined);
    renderer.buildGui(gui);
}

start().catch((error) => {
    console.error(error);
    showFatalError(`Webnado couldn't start: ${error}`);
});
