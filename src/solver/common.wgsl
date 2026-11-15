struct SolverParameters {
    dt: f32,
    buoyancy: f32,
    confinement: f32,
    addThermal: f32,
    mixing: f32,
    turbulence: f32,
    transportedVorticity: f32,
    vorticityPadding: f32,
    dustEnabled: f32,
    grainRadius: f32,
    grainDensity: f32,
    dustViscosity: f32,
    seedDust: f32,
    dustLoading: f32,
    dustPadding: vec2f,
}

override dustPhase: bool = false;

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
@group(0) @binding(14) var turbulenceIn: texture_3d<f32>;
@group(0) @binding(15) var turbulenceOut: texture_storage_3d<rgba16float, write>;
@group(0) @binding(16) var forceIn: texture_3d<f32>;
@group(0) @binding(17) var forceOut: texture_storage_3d<rgba16float, write>;
@group(0) @binding(18) var windForcing: texture_3d<f32>;
@group(0) @binding(19) var windForcingOut: texture_storage_3d<rgba16float, write>;
@group(0) @binding(20) var dustDensityIn: texture_3d<f32>;
@group(0) @binding(21) var dustDensityOut: texture_storage_3d<r32float, write>;
@group(0) @binding(22) var airVelocityIn: texture_3d<f32>;
@group(0) @binding(23) var dustGradientX: texture_3d<f32>;
@group(0) @binding(24) var dustGradientXOut: texture_storage_3d<rgba16float, write>;
@group(0) @binding(25) var dustGradientY: texture_3d<f32>;
@group(0) @binding(26) var dustGradientYOut: texture_storage_3d<rgba16float, write>;
@group(0) @binding(27) var dustGradientZ: texture_3d<f32>;
@group(0) @binding(28) var dustGradientZOut: texture_storage_3d<rgba16float, write>;

const fluidBit = 64u;

fn phaseCell(cell: vec3i) -> vec3i {
    if (dustPhase) {
        return vec3i((cell.x % gridDims.x + gridDims.x) % gridDims.x,
            cell.y, (cell.z % gridDims.z + gridDims.z) % gridDims.z);
    }
    return cell;
}

fn phasePosition(position: vec3f) -> vec3f {
    if (dustPhase) {
        return vec3f(position.x - floor(position.x / f32(gridDims.x)) * f32(gridDims.x),
            position.y, position.z - floor(position.z / f32(gridDims.z)) * f32(gridDims.z));
    }
    return position;
}

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
    let open = (textureLoad(openBits, cell, 0).x & (1u << bit)) != 0u;
    let neighbor = cell + select(-1, 1, positive) * axisOffset(axis);
    if (dustPhase && axis != 1 && !inside(neighbor)) {
        let opposite = textureLoad(openBits, phaseCell(neighbor), 0).x;
        let oppositeBit = u32(2 * axis) + select(1u, 0u, positive);
        return open && (opposite & fluidBit) != 0u && (opposite & (1u << oppositeBit)) != 0u;
    }
    return open;
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

    if (!dustPhase && axis != 1 && (cell[axis] == 0 || cell[axis] == gridDims[axis])) {
        let wind = background(f32(cell.y) + 0.5).state.zw;
        return select(wind.x, wind.y, axis == 2);
    }

    return textureLoad(velocityIn, phaseCell(cell), 0)[axis];
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
