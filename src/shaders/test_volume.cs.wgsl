// Creates test volumes for raymarcher

@group(0) @binding(0) var thermoOut: texture_storage_3d<rgba16float, write>;

override testVolume: u32 = 0;

// Each returns thermo at a point: theta', qv', qc, qr
// theta' = potential temperature as difference from background
// qv' = vapour mixing ratio (grams vapour per kg dry air) as difference from background
// qc = cloud water mixing ratio
// qr = rain water mixing ratio

// You'll notice that these are just clouds for now.

fn ellipsoid(world: vec3f) -> vec4f {
    let offset = (world - vec3f(0.0, 2500.0, 0.0)) / vec3f(2500.0, 1200.0, 2500.0);
    let cloud = 3.0 * (1.0 - smoothstep(0.6, 1.0, length(offset)));
    return vec4f(0.0, 0.0, cloud, 0.0);
}

fn hash(cell: vec3i) -> f32 {
    var h = bitcast<vec3u>(cell) * vec3u(73856093u, 19349663u, 83492791u);
    var x = h.x ^ h.y ^ h.z;
    x = (x ^ (x >> 16u)) * 0x7feb352du;
    x = (x ^ (x >> 15u)) * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return f32(x) / 4294967295.0;
}

fn valueNoise(p: vec3f) -> f32 {
    let i = vec3i(floor(p));
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let x00 = mix(hash(i), hash(i + vec3i(1, 0, 0)), u.x);
    let x10 = mix(hash(i + vec3i(0, 1, 0)), hash(i + vec3i(1, 1, 0)), u.x);
    let x01 = mix(hash(i + vec3i(0, 0, 1)), hash(i + vec3i(1, 0, 1)), u.x);
    let x11 = mix(hash(i + vec3i(0, 1, 1)), hash(i + vec3i(1, 1, 1)), u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}

// 5 octaves of noise, lacunarity of 2.0, gain of 0.5
fn fbm(world: vec3f) -> vec4f {
    var p = world / 2000.0;     // meters per noise cell of the largest octave
    var amplitude = 0.5;
    var noise = 0.0;
    for (var octave = 0; octave < 5; octave++) {
        noise += amplitude * valueNoise(p);
        p *= 2.0;
        amplitude *= 0.5;
    }

    let layer = smoothstep(800.0, 1600.0, world.y) * (1.0 - smoothstep(3000.0, 4500.0, world.y));
    let toWall = max(abs(world.x) / (0.5 * domainSize.x), abs(world.z) / (0.5 * domainSize.z));
    let cloud = 3.0 * clamp((noise * layer * (1.0 - smoothstep(0.85, 1.0, toWall)) - 0.45) / 0.2, 0.0, 1.0);
    return vec4f(0.0, 0.0, cloud, 0.0);
}

@compute
@workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let world = domainMin + (vec3f(id) + 0.5) * cellSize;
    var q: vec4f;
    switch testVolume {
        case 1u: { q = fbm(world); }
        default: { q = ellipsoid(world); }
    }
    textureStore(thermoOut, vec3i(id), q);
}
