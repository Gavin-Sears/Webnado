// Blends this frame's raymarch with last frame's result;
// backtracks to find where same cloud was on screen last frame.

// Output: Cloud RGBA (see raymarch.fs.wgsl)

// must match writeTemporalParams() in volumetric.ts
struct TemporalParams {
    blend: f32,
    historyValid: u32,
    clampHistory: u32,
    padding: u32
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> temporal: TemporalParams;
@group(0) @binding(2) var current: texture_2d<f32>;             // current raymarch result
@group(0) @binding(3) var cloudDepth: texture_2d<f32>;          // meters along each ray (average depth, check out raymarch shader)
@group(0) @binding(4) var history: texture_2d<f32>;
@group(0) @binding(5) var linearSampler: sampler;

fn previousUv(fragCoord: vec2f, size: vec2f, depth: f32) -> vec2f {
    let uv = fragCoord / size;
    let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let farPoint = camera.invViewProjMat * vec4f(ndc, 1.0, 1.0);
    let dir = normalize(farPoint.xyz / farPoint.w - camera.cameraPos);
    let world = camera.cameraPos + dir * depth;

    let previousClip = camera.prevViewProjMat * vec4f(world, 1.0);
    if (previousClip.w <= 0.0) {
        return vec2f(-1.0);
    }
    let previousNdc = previousClip.xy / previousClip.w;
    return vec2f(previousNdc.x * 0.5 + 0.5, 0.5 - previousNdc.y * 0.5);
}

@fragment
fn main(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
    let size = vec2i(textureDimensions(current));
    let pixel = vec2i(fragCoord.xy);
    let now = textureLoad(current, pixel, 0);

    let prevUv = previousUv(fragCoord.xy, vec2f(size), textureLoad(cloudDepth, pixel, 0).r);
    let onScreen = all(prevUv >= vec2f(0.0)) && all(prevUv <= vec2f(1.0));
    if (temporal.historyValid == 0u || !onScreen) {
        return now;
    }

    // note that we cannot use textureSample here because we have an early return (non uniform control flow)
    var past = textureSampleLevel(history, linearSampler, prevUv, 0.0);

    // clamp past values so that they don't fall too far out of acceptable range
    if (temporal.clampHistory != 0u) {
        var low = now;
        var high = now;
        for (var y = -1; y <= 1; y++) {
            for (var x = -1; x <= 1; x++) {
                let neighbor = textureLoad(current, clamp(pixel + vec2i(x, y), vec2i(0), size - 1), 0);
                low = min(low, neighbor);
                high = max(high, neighbor);
            }
        }
        past = clamp(past, low, high);
    }

    return mix(past, now, temporal.blend);
}