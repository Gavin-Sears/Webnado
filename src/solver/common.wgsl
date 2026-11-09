struct SolverParameters {
    dt: f32,
    buoyancy: f32,
    confinement: f32,
    addThermal: f32,
}

@group(0) @binding(0) var<uniform> parameters: SolverParameters;
@group(0) @binding(1) var velocityIn: texture_3d<f32>;
@group(0) @binding(2) var thermoIn: texture_3d<f32>;
@group(0) @binding(3) var openBits: texture_3d<u32>;
@group(0) @binding(4) var<storage, read> atmosphere: array<AtmosphereSample>;
@group(0) @binding(5) var velocityOut: texture_storage_3d<rgba16float, write>;
@group(0) @binding(6) var thermoOut: texture_storage_3d<rgba16float, write>;
@group(0) @binding(7) var vorticityIn: texture_3d<f32>;
@group(0) @binding(8) var vorticityOut: texture_storage_3d<rgba16float, write>;
@group(0) @binding(9) var pressureIn: texture_3d<f32>;
@group(0) @binding(10) var divergenceIn: texture_3d<f32>;
@group(0) @binding(11) var scalarOut: texture_storage_3d<r32float, write>;
@group(0) @binding(12) var obstacles: texture_3d<u32>;
@group(0) @binding(13) var openBitsOut: texture_storage_3d<r32uint, write>;

const fluidBit = 64u;

fn inside(cell: vec3i) -> bool {
    return all(cell >= vec3i(0)) && all(cell < gridDims);
}

fn axisOffset(axis: i32) -> vec3i {
    var offset = vec3i(0);
    offset[axis] = 1;
    return offset;
}

fn fluid(cell: vec3i) -> bool {
    if (!inside(cell)) {
        return false;
    }

    return (textureLoad(openBits, cell, 0).x & fluidBit) != 0u;
}

fn connection(cell: vec3i, axis: i32, positive: bool) -> bool {
    let bit = u32(2 * axis) + select(0u, 1u, positive);
    return (textureLoad(openBits, cell, 0).x & (1u << bit)) != 0u;
}

fn faceOpen(cell: vec3i, axis: i32) -> bool {
    if (inside(cell)) {
        return connection(cell, axis, false);
    }

    let lower = cell - axisOffset(axis);
    if (cell[axis] == gridDims[axis] && inside(lower)) {
        return connection(lower, axis, true);
    }

    return false;
}

fn background(height: f32) -> AtmosphereSample {
    // WeatherData places its samples at slice centers, rather than domain endpoints.
    let slice = clamp(height / f32(gridDims.y) * f32(${atmosphereSlices}) - 0.5,
        0.0, f32(${atmosphereSlices} - 1));
    let lower = u32(floor(slice));
    let upper = min(lower + 1u, ${atmosphereSlices}u - 1u);

    return AtmosphereSample(
        mix(atmosphere[lower].state, atmosphere[upper].state, fract(slice)),
        mix(atmosphere[lower].air, atmosphere[upper].air, fract(slice))
    );
}

fn faceVelocity(cell: vec3i, axis: i32) -> f32 {
    if (!faceOpen(cell, axis)) {
        return 0.0;
    }

    if (axis != 1 && (cell[axis] == 0 || cell[axis] == gridDims[axis])) {
        let wind = background(f32(cell.y) + 0.5).state.zw;
        return select(wind.x, wind.y, axis == 2);
    }

    return textureLoad(velocityIn, cell, 0)[axis];
}

fn centerVelocity(cell: vec3i) -> vec3f {
    let nearest = clamp(cell, vec3i(0), gridDims - 1);
    var velocity = vec3f(0);

    for (var axis = 0; axis < 3; axis++) {
        velocity[axis] = 0.5 * (faceVelocity(nearest, axis)
            + faceVelocity(nearest + axisOffset(axis), axis));
    }

    return velocity;
}
