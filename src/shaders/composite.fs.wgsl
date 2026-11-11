// Upsamples the raymarch target, puts it over the scene, then converts to sRGB.

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var sceneColor: texture_2d<f32>;
@group(0) @binding(2) var sceneDepth: texture_depth_2d;
@group(0) @binding(3) var volume: texture_2d<f32>;     // rgb light, a transmittance (not opacity)

// The difference in depth between two texels we average must be less than this value.
const depthTolerance = 0.1;

// Canvas pixel under a target texel's center. Must match fullResPixel in raymarch.fs.wgsl.
fn fullResPixel(pixel: vec2i, fullSize: vec2i, lowSize: vec2i) -> vec2i {
    let scale = vec2f(fullSize) / vec2f(lowSize);
    return min(vec2i((vec2f(pixel) + 0.5) * scale), fullSize - 1);
}

// Meters from the camera to the surface a canvas pixel shows (about 100 km at the far plane)
fn viewDistance(pixel: vec2i, fullSize: vec2i) -> f32 {
    let uv = (vec2f(pixel) + 0.5) / vec2f(fullSize);
    let clip = vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, textureLoad(sceneDepth, pixel, 0), 1.0);
    let world = camera.invViewProjMat * clip;
    return length(world.xyz / world.w - camera.cameraPos);
}

// Bilinear blend of the 4 nearest texels, skipping ones on a different surface.
// Falls back to the closest in depth (GPU Gems 3 ch. 23).
fn upsample(pixel: vec2i, fullSize: vec2i, lowSize: vec2i) -> vec4f {
    // In target texels, shifted so texel centers are whole numbers
    let lowPos = (vec2f(pixel) + 0.5) * vec2f(lowSize) / vec2f(fullSize) - 0.5;
    // Top left of the 4 neighbors. Can be -1 at the edges, so neighbors are clamped below.
    let base = vec2i(floor(lowPos));
    let f = lowPos - floor(lowPos);
    let center = viewDistance(pixel, fullSize);

    var sum = vec4f(0.0);
    var weightSum = 0.0;
    var nearest = vec4f(0.0, 0.0, 0.0, 1.0);
    var nearestDiff = 1e30;
    // Avoid blurring edges by averaging texels on same surface
    for (var n = 0; n < 4; n++) {
        let offset = vec2i(n & 1, n >> 1);
        let low = clamp(base + offset, vec2i(0), lowSize - 1);
        let value = textureLoad(volume, low, 0);
        let diff = abs(viewDistance(fullResPixel(low, fullSize, lowSize), fullSize) - center) / center;
        let weight = select(1.0 - f.x, f.x, offset.x == 1) * select(1.0 - f.y, f.y, offset.y == 1);
        if (diff < depthTolerance) {
            sum += weight * value;
            weightSum += weight;
        }
        if (diff < nearestDiff) {
            nearestDiff = diff;
            nearest = value;
        }
    }
    // Renormalize, since skipped neighbors leave the weights short of 1
    if (weightSum > 1e-4) {
        return sum / weightSum;
    }
    return nearest;
}

// Filmic curve (Narkowicz's ACES fit): rolls off bright values instead of clipping them
fn ACESToneMap(c: vec3f) -> vec3f {
    return clamp((c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}

fn linearToSrgb(c: vec3f) -> vec3f {
    return select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, 12.92 * c, c <= vec3f(0.0031308));
}

@fragment
fn main(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
    let pixel = vec2i(fragCoord.xy);
    let fullSize = vec2i(textureDimensions(sceneDepth));
    let lowSize = vec2i(textureDimensions(volume));

    var cloud: vec4f;
    if (all(lowSize == fullSize)) {
        cloud = textureLoad(volume, pixel, 0);
    } else {
        cloud = upsample(pixel, fullSize, lowSize);
    }

    let scene = textureLoad(sceneColor, pixel, 0).rgb;
    return vec4f(linearToSrgb(ACESToneMap(scene * cloud.a + cloud.rgb)), 1.0);
}
