@group(0) @binding(0) var velocity: texture_3d<f32>;
@group(0) @binding(1) var thermo: texture_3d<f32>;
@group(0) @binding(2) var materials: texture_3d<u32>;
@group(0) @binding(3) var<storage, read_write> results: array<vec4f>;

var<workgroup> first: array<vec4f, 256>;
var<workgroup> second: array<vec4f, 256>;

fn center(cell: vec3i) -> vec3f {
    let here = clamp(cell, vec3i(0), gridDims - 1);
    let base = textureLoad(velocity, here, 0).xyz;
    return 0.5 * (base + vec3f(
        textureLoad(velocity, min(here + vec3i(1, 0, 0), gridDims - 1), 0).x,
        textureLoad(velocity, min(here + vec3i(0, 1, 0), gridDims - 1), 0).y,
        textureLoad(velocity, min(here + vec3i(0, 0, 1), gridDims - 1), 0).z
    ));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u,
    @builtin(local_invocation_index) local: u32,
    @builtin(workgroup_id) group: vec3u) {
    let index = i32(id.x);
    let cell = vec3i(index % gridDims.x, (index / gridDims.x) % gridDims.y,
        index / (gridDims.x * gridDims.y));
    var a = vec4f(0);
    var b = vec4f(0);
    if (cell.z < gridDims.z && textureLoad(materials, cell, 0).x == 0u) {
        let state = textureLoad(thermo, cell, 0);
        let wind = center(cell);
        let dx = (center(cell + vec3i(1, 0, 0)) - center(cell - vec3i(1, 0, 0))) / (2.0 * cellSize.x);
        let dz = (center(cell + vec3i(0, 0, 1)) - center(cell - vec3i(0, 0, 1))) / (2.0 * cellSize.z);
        let verticalCurl = abs(dz.x - dx.z);
        let height = (f32(cell.y) + 0.5) * cellSize.y;
        a = vec4f(max(wind.y, 0), length(wind), verticalCurl,
            select(0.0, verticalCurl, height < 1000.0 && cell.y > 1));
        b = vec4f(state.z, state.w, abs(state.x), 0);
        if (any(abs(state) > vec4f(60000)) || any(abs(wind) > vec3f(60000))
            || any(state != state) || any(wind != wind)) {
            b.w = 1;
        }
    }
    first[local] = a;
    second[local] = b;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (local < stride) {
            first[local] = max(first[local], first[local + stride]);
            second[local] = max(second[local], second[local + stride]);
        }
        workgroupBarrier();
    }
    if (local == 0u) {
        results[2u * group.x] = first[0];
        results[2u * group.x + 1u] = second[0];
    }
}
