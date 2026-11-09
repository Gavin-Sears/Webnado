import { GUI } from 'lil-gui';

import * as renderer from '../renderer';
import { Stage } from '../stage/stage';

// TODO:
// On each frame: gis.render(camera), solver.step(encoder), then raymarch.
// Output: the canvas (renderer.canvasFormat). The raymarch fragment shader should also convert
// the final color from linear to sRGB.
//
// Raymarcher inputs, in bind group 0:
//   0  camera.uniformsBuffer    CameraUniforms (common.wgsl). Meters, +X east, +Y up, +Z south.
//   1  fields.thermo.read       texture_3d<f32>, rgba16float, gridX x gridY x gridZ, cell centered
//                               r theta' (K), g qv' (g/kg), b qc cloud water (g/kg), a qr rain (g/kg)
//                               Sample at (world - domainMin) / domainSize. Check thermo.index at draw time, since it swaps.
//   2  linear sampler           blends between cells when sampling thermo
// The rasterized scene from Three.js:
//   3  gis.sceneColorTexture    texture_2d<f32>, rgba16float, canvas size, linear color
//   4  gis.sceneDepthTexture    texture_depth_2d, depth32float, canvas size, 0 near to 1 far, 1 where empty
export class VolumetricRenderer extends renderer.Renderer {
    constructor(stage: Stage) {
        super(stage);
    }

    buildGui(_gui: GUI) {
    }

    override draw() {
    }
}
