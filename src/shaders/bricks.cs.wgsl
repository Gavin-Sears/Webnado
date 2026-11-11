// populates brickMaxOut, which stores largest qc + qr per brick, so that raymarcher can skip bricks with low value.

@group(0) @binding(0) var thermo: texture_3d<f32>;
@group(0) @binding(1) var brickMaxOut: texture_storage_3d<r32float, write>;

// One cell of border on each side, because linear sampling near a brick's edge reads its neighbors' cells
const span = ${brickSize} + 2;
const spanCells = u32(span * span * span);

// Power of two (for reduction), at most 256 (default limit)
const threads = 64u;

var<workgroup> partial: array<f32, threads>;

@compute
@workgroup_size(threads)
fn main(@builtin(workgroup_id) brick: vec3u, @builtin(local_invocation_index) local: u32) {
    let low = vec3i(brick) * ${brickSize} - 1;
    var largest = 0.0;
    for (var n = local; n < spanCells; n += threads) {
        let offset = vec3i(i32(n) % span, (i32(n) / span) % span, i32(n) / (span * span));
        let q = textureLoad(thermo, clamp(low + offset, vec3i(0), gridDims - 1), 0);
        largest = max(largest, q.z + q.w);
    }

    partial[local] = largest;
    workgroupBarrier();
    for (var stride = threads / 2u; stride > 0u; stride >>= 1u) {
        if (local < stride) {
            partial[local] = max(partial[local], partial[local + stride]);
        }
        workgroupBarrier();
    }

    if (local == 0u) {
        textureStore(brickMaxOut, vec3i(brick), vec4f(partial[0], 0.0, 0.0, 0.0));
    }
}
