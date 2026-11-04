// CHECKITOUT: prepended to all shaders

// World: +X east, +Y up, +Z south, in meters
const gridDims = vec3i(${gridX}, ${gridY}, ${gridZ});
const domainSize = vec3f(${domainX}, ${domainY}, ${domainZ});
const domainMin = vec3f(-0.5 * domainSize.x, 0.0, -0.5 * domainSize.z);
const cellSize = domainSize / vec3f(gridDims);

// Shared fields (see src/stage/fields.ts):
//   velocity: u, v, w (m/s), free channel. Staggered: texel (i, j, k) holds u at x = i, v at y = j, w at z = k.
//   thermo:   theta' (K), qv', qc, qr (g/kg), cell centered. theta' and qv' are relative to the atmosphere, so 0 is ordinary air.
//   obstacle: 0 air, 1 terrain, 2 building

// size: 208
struct CameraUniforms {
    viewProjMat: mat4x4f,
    invViewProjMat: mat4x4f,
    prevViewProjMat: mat4x4f,   // last frame's, for reprojection
    cameraPos: vec3f,
    frameIndex: u32
}

// One height slice of background air. Slices are evenly spaced from the ground to the top of the box.
// size: 32
struct AtmosphereSample {
    state: vec4f,   // theta (K), qv (g/kg), wind x (m/s), wind z (m/s)
    air: vec4f      // pressure (Pa), temperature (K), density (kg/m^3, unused)
}
