@group(0) @binding(0) var velocity: texture_3d<f32>;
@group(0) @binding(1) var<uniform> center: vec4f;
@group(0) @binding(2) var<storage, read_write> rings: array<vec4f>;

var<workgroup> samples: array<vec4f, 128>;

fn componentAt(world: vec3f, axis: i32) -> f32 {
    var offset = vec3f(0.5);
    offset[axis] = 0.0;
    let coordinate = clamp((world - domainMin) / cellSize - offset,
        vec3f(0), vec3f(gridDims - 1));
    let base = vec3i(floor(coordinate));
    let fraction = fract(coordinate);
    var value = 0.0;
    for (var z = 0; z < 2; z++) {
        for (var y = 0; y < 2; y++) {
            for (var x = 0; x < 2; x++) {
                let corner = vec3i(x, y, z);
                let weight = mix(vec3f(1) - fraction, fraction, vec3f(corner));
                value += weight.x * weight.y * weight.z
                    * textureLoad(velocity, min(base + corner, gridDims - 1), 0)[axis];
            }
        }
    }
    return value;
}

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) local: u32, @builtin(workgroup_id) group: vec3u) {
    let radius = f32(group.x + 1u) * max(cellSize.x, cellSize.z);
    let edge = min(min(center.x - domainMin.x, -domainMin.x - center.x),
        min(center.z - domainMin.z, -domainMin.z - center.z));
    let valid = radius <= 1500.0 && radius + max(cellSize.x, cellSize.z) < edge;
    var sample = vec4f(0);
    if (valid) {
        let angle = 6.28318530718 * f32(local) / 128.0;
        let tangent = vec2f(-sin(angle), -cos(angle));
        let outward = vec2f(cos(angle), -sin(angle));
        let position = center.xyz + vec3f(radius * outward.x, 0, radius * outward.y);
        let wind = vec2f(componentAt(position, 0), componentAt(position, 2));
        sample = vec4f(dot(wind, tangent), dot(wind, outward), componentAt(position, 1), radius);
    }
    samples[local] = sample;
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride /= 2u) {
        if (local < stride) {
            samples[local] += samples[local + stride];
        }
        workgroupBarrier();
    }
    if (local == 0u) {
        rings[group.x] = samples[0] / 128.0;
    }
}
