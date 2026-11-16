@group(0) @binding(0) var velocity: texture_3d<f32>;
@group(0) @binding(1) var thermo: texture_3d<f32>;
@group(0) @binding(2) var materials: texture_3d<u32>;
@group(0) @binding(3) var<storage, read_write> results: array<vec4f>;
@group(0) @binding(4) var<storage, read> atmosphere: array<AtmosphereSample>;

fn densityAt(height: f32) -> f32 {
    if (${anelasticAir} == 0) {
        return 1.0;
    }
    let slice = clamp(height / f32(gridDims.y) * f32(${atmosphereSlices}) - 0.5,
        0.0, f32(${atmosphereSlices} - 1));
    let lower = u32(floor(slice));
    let upper = min(lower + 1u, ${atmosphereSlices}u - 1u);
    return mix(atmosphere[lower].air.z, atmosphere[upper].air.z, fract(slice));
}

var<workgroup> first: array<vec4f, 128>;
var<workgroup> second: array<vec4f, 128>;
var<workgroup> third: array<vec4f, 128>;
var<workgroup> locations: array<vec4f, 128>;
var<workgroup> updraftLocations: array<vec4f, 128>;
var<workgroup> groundLocations: array<vec4f, 128>;
var<workgroup> swirlLocations: array<vec4f, 128>;

fn center(cell: vec3i) -> vec3f {
    let here = clamp(cell, vec3i(0), gridDims - 1);
    let base = textureLoad(velocity, here, 0).xyz;
    return 0.5 * (base + vec3f(
        textureLoad(velocity, min(here + vec3i(1, 0, 0), gridDims - 1), 0).x,
        textureLoad(velocity, min(here + vec3i(0, 1, 0), gridDims - 1), 0).y,
        textureLoad(velocity, min(here + vec3i(0, 0, 1), gridDims - 1), 0).z
    ));
}

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) local: u32,
    @builtin(workgroup_id) group: vec3u) {
    let groupIndex = group.x + 256u * group.y;
    let index = i32(128u * groupIndex + local);
    let cell = vec3i(index % gridDims.x, (index / gridDims.x) % gridDims.y,
        index / (gridDims.x * gridDims.y));
    var a = vec4f(0);
    var b = vec4f(0);
    var c = vec4f(0);
    var location = vec4f(0);
    var updraftLocation = vec4f(0);
    var groundLocation = vec4f(0);
    var swirlLocation = vec4f(0);
    if (cell.z < gridDims.z && textureLoad(materials, cell, 0).x == 0u) {
        let state = textureLoad(thermo, cell, 0);
        let wind = center(cell);
        let dx = (center(cell + vec3i(1, 0, 0)) - center(cell - vec3i(1, 0, 0))) / (2.0 * cellSize.x);
        let dz = (center(cell + vec3i(0, 0, 1)) - center(cell - vec3i(0, 0, 1))) / (2.0 * cellSize.z);
        let signedCurl = dz.x - dx.z;
        let verticalCurl = abs(signedCurl);
        let height = (f32(cell.y) + 0.5) * cellSize.y;
        let world = domainMin + (vec3f(cell) + vec3f(0.5)) * cellSize;
        let sideDistance = min(min(world.x - domainMin.x, -domainMin.x - world.x),
            min(world.z - domainMin.z, -domainMin.z - world.z));
        let interior = sideDistance > max(4.0 * cellSize.x, 0.05 * domainSize.x);
        a = vec4f(max(wind.y, 0), length(wind), verticalCurl,
            select(0.0, verticalCurl, interior && height < 1000.0));
        b = vec4f(state.z, state.w, abs(state.x), 0);
        location = vec4f(world, a.w);
        if (interior && height < 200.0) {
            groundLocation = vec4f(world, verticalCurl);
            let trace = dx.x + dz.z;
            let determinant = dx.x * dz.z - dz.x * dx.z;
            let strength = 0.5 * sqrt(max(0.0, 4.0 * determinant - trace * trace));
            swirlLocation = vec4f(world, strength);
        }
        if (interior && height > 500.0 && height < 6000.0 && wind.y > 5.0) {
            updraftLocation = vec4f(world, wind.y * max(signedCurl, 0.0001));
        }
        c.w = select(0.0, verticalCurl, interior);
        if (all(cell < gridDims - 1)) {
            let base = textureLoad(velocity, cell, 0).xyz;
            let upper = vec3f(
                textureLoad(velocity, cell + vec3i(1, 0, 0), 0).x,
                textureLoad(velocity, cell + vec3i(0, 1, 0), 0).y,
                textureLoad(velocity, cell + vec3i(0, 0, 1), 0).z);
            let density = densityAt(f32(cell.y) + 0.5);
            let lowerDensity = vec3f(density, densityAt(f32(cell.y)), density);
            let upperDensity = vec3f(density, densityAt(f32(cell.y) + 1.0), density);
            let divergence = dot((upper * upperDensity - base * lowerDensity) / cellSize, vec3f(1)) / density;
            c.x = divergence * divergence;
            c.y = abs(divergence);
            c.z = 1;
        }
        if (any(abs(state) > vec4f(60000)) || any(abs(wind) > vec3f(60000))
            || any(state != state) || any(wind != wind)) {
            b.w = 1;
        }
    }
    first[local] = a;
    second[local] = b;
    third[local] = c;
    locations[local] = location;
    updraftLocations[local] = updraftLocation;
    groundLocations[local] = groundLocation;
    swirlLocations[local] = swirlLocation;
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride >>= 1u) {
        if (local < stride) {
            if (swirlLocations[local + stride].w > swirlLocations[local].w) {
                swirlLocations[local] = swirlLocations[local + stride];
            }
            if (groundLocations[local + stride].w > groundLocations[local].w) {
                groundLocations[local] = groundLocations[local + stride];
            }
            if (updraftLocations[local + stride].w > updraftLocations[local].w) {
                updraftLocations[local] = updraftLocations[local + stride];
            }
            if (first[local + stride].w > first[local].w) {
                locations[local] = locations[local + stride];
            }
            first[local] = max(first[local], first[local + stride]);
            second[local] = max(second[local], second[local + stride]);
            third[local] = vec4f(third[local].x + third[local + stride].x,
                max(third[local].y, third[local + stride].y),
                third[local].z + third[local + stride].z,
                max(third[local].w, third[local + stride].w));
        }
        workgroupBarrier();
    }
    if (local == 0u) {
        results[7u * groupIndex] = first[0];
        results[7u * groupIndex + 1u] = second[0];
        results[7u * groupIndex + 2u] = third[0];
        results[7u * groupIndex + 3u] = locations[0];
        results[7u * groupIndex + 4u] = updraftLocations[0];
        results[7u * groupIndex + 5u] = groundLocations[0];
        results[7u * groupIndex + 6u] = swirlLocations[0];
    }
}
