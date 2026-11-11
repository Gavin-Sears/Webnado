// Single scattering raymarch into a reduced resolution target (composite.fs.wgsl upsamples)
// Output: RGBA, with rgb light toward the camera, and a as transmittance.

// must match writeParams() in volumetric.ts
struct RaymarchParams {
    sunDir: vec3f,          // toward the sun
    phaseG: f32,            // Henyey-Greenstein asymmetry
    sunColor: vec3f,
    extinction: f32,        // 1/m per g/kg of qc + qr
    ambientColor: vec3f,
    stepCells: f32,         // distance between samples, in cells
    targetSize: vec2u,      // size of this pass's target
    maxSteps: u32,
    shadowSteps: u32,
    shadowDistance: f32,    // meters
    jitterFrames: u32,      // 1 holds the noise still, more cycles it each frame
    brickSkipping: u32,
    debugView: u32,
    debugScale: f32,
    debugBoost: u32         // density view: 1 makes faint cloud visible
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> params: RaymarchParams;
@group(0) @binding(2) var thermo: texture_3d<f32>;
@group(0) @binding(3) var linearSampler: sampler;
@group(0) @binding(4) var sceneDepth: texture_depth_2d;
@group(0) @binding(5) var brickMax: texture_3d<f32>;   // largest qc + qr near each brick (bricks.cs.wgsl)
@group(0) @binding(6) var velocity: texture_3d<f32>;   // debug views only
@group(0) @binding(7) var obstacle: texture_3d<u32>;   // debug views only

const pi = 3.14159265;
const earlyExitTransmittance = 0.01;
const emptyBrick = 1e-4;                // g/kg. Bricks under this value are empty
const boostDepth = 500.0;               // boosted density view
const brickDims = gridDims / ${brickSize};
const brickWorldSize = cellSize * f32(${brickSize});

// MATERIAL
// ========

// Fraction of light blocked per meter at a point (1/m)
fn extinctionAt(world: vec3f) -> f32 {
    let q = textureSampleLevel(thermo, linearSampler, (world - domainMin) / domainSize, 0.0);
    return (q.z + q.w) * params.extinction;
}

// How much scattered light goes cosTheta away from its original direction. Integrates to 1 over the sphere.
fn henyeyGreenstein(cosTheta: f32, g: f32) -> f32 {
    let denom = 1.0 + g * g - 2.0 * g * cosTheta;
    return (1.0 - g * g) / (4.0 * pi * denom * sqrt(denom));
}

// LIGHT
// =====

// Fraction of sunlight that reaches a point. offset (0 to 1) shifts the samples within their steps.
// Texture sampling with light volumes might replace this soon.
fn sunTransmittance(world: vec3f, offset: f32) -> f32 {
    let distance = min(intersectDomain(world, params.sunDir).y, params.shadowDistance);
    let stepSize = distance / f32(params.shadowSteps);
    var opticalDepth = 0.0;
    for (var i = 0u; i < params.shadowSteps; i++) {
        opticalDepth += extinctionAt(world + params.sunDir * ((f32(i) + offset) * stepSize));
    }
    return exp(-opticalDepth * stepSize);
}

// Sun lighting and shadows
fn inScattered(world: vec3f, sunPhase: f32, shadowOffset: f32) -> vec3f {
    return params.sunColor * sunPhase * sunTransmittance(world, shadowOffset) + params.ambientColor;
}

// RAYS
// ====

fn safeInverse(dir: vec3f) -> vec3f {
    return 1.0 / select(dir, vec3f(1e-9), abs(dir) < vec3f(1e-9));
}

// vec2f with t at entry into sim grid and exit respectively. No hit if entry >= exit. 
fn intersectDomain(origin: vec3f, dir: vec3f) -> vec2f {
    let invDir = safeInverse(dir);
    let t0 = (domainMin - origin) * invDir;
    let t1 = (domainMin + domainSize - origin) * invDir;
    let tMin = min(t0, t1);
    let tMax = max(t0, t1);
    return vec2f(max(max(tMin.x, tMin.y), max(tMin.z, 0.0)), min(tMax.x, min(tMax.y, tMax.z)));
}

fn brickOf(world: vec3f) -> vec3i {
    return clamp(vec3i(floor((world - domainMin) / brickWorldSize)), vec3i(0), brickDims - 1);
}

// Ray steps needed to get from current position past the far side of the current brick.
// Used for skipping empty bricks
fn stepsToLeave(brick: vec3i, origin: vec3f, invDir: vec3f, t: f32, stepSize: f32) -> f32 {
    let brickMin = domainMin + vec3f(brick) * brickWorldSize;
    let tFar = max((brickMin - origin) * invDir, (brickMin + brickWorldSize - origin) * invDir);
    let tExit = min(tFar.x, min(tFar.y, tFar.z));
    return max(ceil((tExit - t) / stepSize), 1.0);
}

// Interleaved gradient noise (Jimenez 2014), 0 to 1 per pixel.
fn jitterAt(pixel: vec2f) -> f32 {
    let p = pixel + 5.588238 * f32(camera.frameIndex % params.jitterFrames);
    return fract(52.9829189 * fract(dot(p, vec2f(0.06711056, 0.00583715))));
}

fn march(origin: vec3f, dir: vec3f, tStart: f32, tEnd: f32, jitter: f32) -> vec4f {
    // StepSize becomes larger if we can't reach the end
    let stepSize = max(params.stepCells * cellSize.x, (tEnd - tStart) / f32(params.maxSteps));
    let invDir = safeInverse(dir);
    let sunPhase = 4.0 * pi * henyeyGreenstein(dot(dir, params.sunDir), params.phaseG);

    var t = tStart + jitter * stepSize;
    var light = vec3f(0.0);
    var transmittance = 1.0;
    for (var i = 0u; i < params.maxSteps && t < tEnd; i++) {
        let world = origin + dir * t;

        if (params.brickSkipping != 0u) {
            let brick = brickOf(world);
            if (textureLoad(brickMax, brick, 0).r < emptyBrick) {
                t += stepsToLeave(brick, origin, invDir, t, stepSize) * stepSize;
                continue;
            }
        }

        let extinction = extinctionAt(world);
        if (extinction > 0.0) {
            // Fraction of light that makes it through this step unblocked (Beer-Lambert)
            // Notice that the scene (tEnd) can stop the ray as well
            let stepTransmittance = exp(-extinction * min(stepSize, tEnd - t));
            let stepNumber = floor((t - tStart) / stepSize);
            // Different shadow offset each step to prevent tiling
            let shadowOffset = fract(jitter + stepNumber * 0.618034);
            // Clouds barely absorb light, so the light a step blocks is the light it scatters
            // That fraction is at most 1, so long steps can't over-brighten (Hillaire 2015).
            // Dust will eventually require albedo
            light += transmittance * inScattered(world, sunPhase, shadowOffset) * (1.0 - stepTransmittance);
            transmittance *= stepTransmittance;
            if (transmittance < earlyExitTransmittance) {
                break;
            }
        }
        t += stepSize;
    }
    return vec4f(light, transmittance);
}

// DEBUG VIEWS
// ===========

// black, red, yellow, white
fn heat(x: f32) -> vec3f {
    return clamp(vec3f(3.0 * x, 3.0 * x - 1.0, 3.0 * x - 2.0), vec3f(0.0), vec3f(1.0));
}

// Velocity (m/s) at a point
fn velocityAt(uvw: vec3f) -> vec3f {
    let half = 0.5 / vec3f(gridDims);
    return vec3f(
        textureSampleLevel(velocity, linearSampler, uvw + vec3f(half.x, 0.0, 0.0), 0.0).x,
        textureSampleLevel(velocity, linearSampler, uvw + vec3f(0.0, half.y, 0.0), 0.0).y,
        textureSampleLevel(velocity, linearSampler, uvw + vec3f(0.0, 0.0, half.z), 0.0).z
    );
}

// Curl of velocity (1/s), by central differences one cell apart
fn vorticityAt(uvw: vec3f) -> vec3f {
    let cell = 1.0 / vec3f(gridDims);
    let dx = (velocityAt(uvw + vec3f(cell.x, 0.0, 0.0)) - velocityAt(uvw - vec3f(cell.x, 0.0, 0.0))) / (2.0 * cellSize.x);
    let dy = (velocityAt(uvw + vec3f(0.0, cell.y, 0.0)) - velocityAt(uvw - vec3f(0.0, cell.y, 0.0))) / (2.0 * cellSize.y);
    let dz = (velocityAt(uvw + vec3f(0.0, 0.0, cell.z)) - velocityAt(uvw - vec3f(0.0, 0.0, cell.z))) / (2.0 * cellSize.z);
    return vec3f(dy.z - dz.y, dz.x - dx.z, dx.y - dy.x);
}

// Unlit, half a cell per step, unaffected by quality settings
fn debugMarch(origin: vec3f, dir: vec3f, tStart: f32, tEnd: f32) -> vec4f {
    let stepSize = 0.5 * cellSize.x;
    var color = vec3f(0.0);
    var transmittance = 1.0;
    var t = tStart;
    for (var i = 0u; i < 2048u && t < tEnd && transmittance > earlyExitTransmittance; i++) {
        let world = origin + dir * t;
        let uvw = (world - domainMin) / domainSize;
        var sampleColor = vec3f(0.0);
        var opacity = 0.0;

        switch params.debugView {
            // qc + qr, white at debugScale. Opacity matches the real render, unless boosted: then it follows
            // debugScale instead, so faint clouds show up.
            case 1u: {
                let q = textureSampleLevel(thermo, linearSampler, uvw, 0.0);
                let amount = (q.z + q.w) / params.debugScale;
                sampleColor = heat(amount);
                let extinction = select((q.z + q.w) * params.extinction, amount / boostDepth, params.debugBoost != 0u);
                opacity = 1.0 - exp(-extinction * stepSize);
            }
            // theta': red warmer than the background, blue colder, fully opaque at debugScale K
            case 2u: {
                let theta = textureSampleLevel(thermo, linearSampler, uvw, 0.0).x;
                sampleColor = select(vec3f(0.2, 0.4, 1.0), vec3f(1.0, 0.3, 0.15), theta > 0.0);
                opacity = 0.05 * clamp(abs(theta) / params.debugScale, 0.0, 1.0);
            }
            // velocity (m/s). Color shows direction, and opacity grows with speed up to debugScale
            case 3u: {
                let v = velocityAt(uvw);
                let speed = length(v);
                sampleColor = 0.5 + 0.5 * v / max(speed, 1e-6);
                opacity = 0.05 * clamp(speed / params.debugScale, 0.0, 1.0);
            }
            // spin rate (1/s), colored like density and opaque at debugScale
            case 4u: {
                let spin = length(vorticityAt(uvw)) / params.debugScale;
                sampleColor = heat(spin);
                opacity = 0.05 * clamp(spin, 0.0, 1.0);
            }
            // brown terrain, gray buildings
            case 5u: {
                let cell = clamp(vec3i(floor(uvw * vec3f(gridDims))), vec3i(0), gridDims - 1);
                let material = textureLoad(obstacle, cell, 0).r;
                sampleColor = select(vec3f(0.75), vec3f(0.45, 0.32, 0.18), material == 1u);
                opacity = select(0.0, 1.0, material != 0u);
            }
            // bricks the raymarcher won't skip, in checkerboard pattern
            default: {
                let brick = brickOf(world);
                sampleColor = select(vec3f(0.2, 0.6, 1.0), vec3f(1.0, 0.6, 0.2), ((brick.x + brick.y + brick.z) & 1) == 1);
                opacity = select(0.0, 0.03, textureLoad(brickMax, brick, 0).r >= emptyBrick);
            }
        }

        color += transmittance * opacity * sampleColor;
        transmittance *= 1.0 - opacity;
        t += stepSize;
    }
    return vec4f(color, transmittance);
}

// MAIN
// ====

// converts low res to full res. Same mapping in composite.fs.wgsl
fn fullResPixel(pixel: vec2i, fullSize: vec2i) -> vec2i {
    let scale = vec2f(fullSize) / vec2f(params.targetSize);
    return min(vec2i((vec2f(pixel) + 0.5) * scale), fullSize - 1);
}

@fragment
fn main(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
    let fullSize = vec2i(textureDimensions(sceneDepth));
    let pixel = fullResPixel(vec2i(fragCoord.xy), fullSize);
    let uv = (vec2f(pixel) + 0.5) / vec2f(fullSize);
    let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

    let farPoint = camera.invViewProjMat * vec4f(ndc, 1.0, 1.0);
    let origin = camera.cameraPos;
    let dir = normalize(farPoint.xyz / farPoint.w - origin);

    var hit = intersectDomain(origin, dir);

    // stop at opaque scene geometry
    let depth = textureLoad(sceneDepth, pixel, 0);
    if (depth < 1.0) {
        let scenePoint = camera.invViewProjMat * vec4f(ndc, depth, 1.0);
        hit.y = min(hit.y, length(scenePoint.xyz / scenePoint.w - origin));
    }

    if (hit.x >= hit.y) {
        return vec4f(0.0, 0.0, 0.0, 1.0);
    }
    if (params.debugView != 0u) {
        return debugMarch(origin, dir, hit.x, hit.y);
    }
    return march(origin, dir, hit.x, hit.y, jitterAt(fragCoord.xy));
}
