const freeSlipGround = ${freeSlipGround} != 0;
const anelasticAir = ${anelasticAir} != 0;
const spatialWeather = ${spatialWeather} != 0;

struct SolverParameters {
    dt: f32,
    buoyancy: f32,
    confinement: f32,
    addThermal: f32,
    mixing: f32,
    turbulence: f32,
    transportedVorticity: f32,
    rainDrain: f32,
    dustEnabled: f32,
    grainRadius: f32,
    grainDensity: f32,
    dustViscosity: f32,
    seedDust: f32, // 0 off, 1 broad source, 2 local source
    dustLoading: f32,
    dustSeedCenter: vec2f,
}

override dustPhase: bool = false;

@group(0) @binding(0) var<uniform> parameters: SolverParameters;
@group(0) @binding(1) var velocityIn: texture_3d<f32>;
@group(0) @binding(2) var thermoIn: texture_3d<f32>;
@group(0) @binding(3) var openBits: texture_3d<u32>;
@group(0) @binding(4) var<storage, read> atmosphere: array<AtmosphereSample>;
@group(0) @binding(5) var velocityOut: texture_storage_3d<${velocityFormat}, write>;
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
@group(0) @binding(17) var forceOut: texture_storage_3d<${velocityFormat}, write>;
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

struct StormParameters {
    timing: vec4f,  // enabled, elapsed, lifting duration, target updraft
    region: vec4f,  // horizontal radius, vertical radius, unused
    damping: vec4f, // damping height, inverse timescale, unused, reference stress balance
}
@group(0) @binding(29) var<uniform> storm: StormParameters;
@group(0) @binding(30) var airGradientZOut: texture_storage_3d<rgba16float, write>;
@group(0) @binding(31) var airGradientX: texture_3d<f32>;
@group(0) @binding(32) var airGradientY: texture_3d<f32>;
@group(0) @binding(33) var airGradientZ: texture_3d<f32>;

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

fn airFaceDensity(cell: vec3i, axis: i32) -> f32 {
    if (!anelasticAir || dustPhase) {
        return 1.0;
    }
    let height = f32(cell.y) + select(0.5, 0.0, axis == 1);
    return background(height).air.z;
}

struct SpatialState {
    wind: vec4f, // east, up, south, terrain elevation
    thermo: vec4f,
    air: vec4f,
    energy: f32,
}

fn spatialCoordinate(position: vec3f) -> vec3f {
    let map = atmosphere[${atmosphereSlices}u + 1u];
    let grid = atmosphere[${atmosphereSlices}u];
    var world = domainMin + position * cellSize;
    if (atmosphere[${atmosphereSlices}u + 2u].state.z >= 3.0) {
        let offset = atmosphere[${atmosphereSlices}u + 3u].state.xy;
        world += vec3f(offset.x, 0, -offset.y);
    }
    let latitude = map.air.x;
    let longitude = map.air.y;
    let radial = map.state.w * cos(latitude) + world.z * sin(latitude);
    let x = radial * cos(longitude) - world.x * sin(longitude);
    let y = radial * sin(longitude) + world.x * cos(longitude);
    let z = map.state.w * sin(latitude) - world.z * cos(latitude);
    let lat = atan2(z, length(vec2f(x, y)));
    let lon = atan2(y, x);
    let rho = map.state.w * map.state.y / pow(tan(0.7853981634 + 0.5 * lat), map.state.x);
    let angle = map.state.x * (lon - map.air.w);
    let projected = vec2f(rho * sin(angle), map.state.z - rho * cos(angle));
    let horizontal = (projected - grid.air.xy) / grid.air.zw;
    return vec3f(horizontal.x, world.y + map.air.z, horizontal.y);
}

fn spatialDataStart() -> u32 {
    let version = atmosphere[${atmosphereSlices}u + 2u].state.z;
    return ${atmosphereSlices}u + select(3u, 4u, version >= 3.0);
}

fn spatialIndex(cell: vec3u, frame: u32) -> u32 {
    let dims = vec3u(atmosphere[${atmosphereSlices}u].state.xyz);
    return spatialDataStart() + 2u * (((frame * dims.z + cell.y) * dims.y + cell.z) * dims.x + cell.x);
}

fn spatialColumn(cell: vec2u, height: f32, frame: u32) -> SpatialState {
    let levels = u32(atmosphere[${atmosphereSlices}u].state.z);
    var lower = 0u;
    var upper = levels - 1u;
    for (var iteration = 0; iteration < 8 && upper - lower > 1u; iteration++) {
        let middle = (lower + upper) / 2u;
        if (atmosphere[spatialIndex(vec3u(cell.x, middle, cell.y), frame)].state.x < height) {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    let a = spatialIndex(vec3u(cell.x, lower, cell.y), frame);
    let b = spatialIndex(vec3u(cell.x, upper, cell.y), frame);
    let fraction = clamp((height - atmosphere[a].state.x)
        / (atmosphere[b].state.x - atmosphere[a].state.x), 0.0, 1.0);
    let first = mix(atmosphere[a].state, atmosphere[b].state, fraction);
    let second = mix(atmosphere[a].air, atmosphere[b].air, fraction);
    return SpatialState(vec4f(first.w, second.y, second.x, atmosphere[a + 1u].air.x),
        vec4f(first.yz, second.zw), mix(atmosphere[a + 1u].state, atmosphere[b + 1u].state, fraction),
        mix(atmosphere[a + 1u].air.y, atmosphere[b + 1u].air.y, fraction));
}

fn spatialFrame(position: vec3f, frame: u32) -> SpatialState {
    let coordinate = spatialCoordinate(position);
    let dims = atmosphere[${atmosphereSlices}u].state.xy;
    let horizontal = clamp(coordinate.xz, vec2f(0), dims - vec2f(1));
    let base = min(vec2u(floor(horizontal)), vec2u(dims) - vec2u(2));
    let fraction = horizontal - vec2f(base);
    var result = SpatialState(vec4f(0), vec4f(0), vec4f(0), 0);
    for (var z = 0u; z < 2u; z++) {
        for (var x = 0u; x < 2u; x++) {
            let amount = mix(vec2f(1) - fraction, fraction, vec2f(f32(x), f32(z)));
            let weight = amount.x * amount.y;
            let point = spatialColumn(base + vec2u(x, z), coordinate.y, frame);
            result.wind += weight * point.wind;
            result.thermo += weight * point.thermo;
            result.air += weight * point.air;
            result.energy += weight * point.energy;
        }
    }
    return result;
}

fn spatialState(position: vec3f) -> SpatialState {
    let count = atmosphere[${atmosphereSlices}u].state.w;
    let timing = atmosphere[${atmosphereSlices}u + 2u].state;
    let time = clamp(timing.x / timing.y, 0.0, count - 1.0);
    let lower = u32(floor(time));
    let a = spatialFrame(position, lower);
    let b = spatialFrame(position, min(lower + 1u, u32(count) - 1u));
    var wind = mix(a.wind, b.wind, fract(time));
    // Project the imported lateral flux onto the closed-top domain's mass constraint.
    let corrections = atmosphere[${atmosphereSlices}u + 2u].air;
    let correction = mix(corrections[lower], corrections[min(lower + 1u, u32(count) - 1u)], fract(time));
    if (position.x <= 0.0) { wind.x += correction; }
    if (position.x >= f32(gridDims.x)) { wind.x -= correction; }
    if (position.z <= 0.0) { wind.z += correction; }
    if (position.z >= f32(gridDims.z)) { wind.z -= correction; }
    return SpatialState(wind,
        mix(a.thermo, b.thermo, fract(time)), mix(a.air, b.air, fract(time)), mix(a.energy, b.energy, fract(time)));
}

fn spatialSurface(position: vec3f) -> vec4f {
    let timing = atmosphere[${atmosphereSlices}u + 2u].state;
    if (timing.z < 2.0) {
        return vec4f(0);
    }
    let dims = vec4u(atmosphere[${atmosphereSlices}u].state);
    let horizontal = clamp(spatialCoordinate(position).xz, vec2f(0), vec2f(dims.xy) - vec2f(1));
    let base = min(vec2u(floor(horizontal)), dims.xy - vec2u(2));
    let fraction = horizontal - vec2f(base);
    let time = clamp(timing.x / timing.y, 0.0, f32(dims.w - 1u));
    let start = spatialDataStart() + 2u * dims.x * dims.y * dims.z * dims.w;
    var result = vec4f(0);
    for (var t = 0u; t < 2u; t++) {
        let frame = min(u32(floor(time)) + t, dims.w - 1u);
        let timeWeight = select(1.0 - fract(time), fract(time), t == 1u);
        for (var z = 0u; z < 2u; z++) {
            for (var x = 0u; x < 2u; x++) {
                let amount = mix(vec2f(1) - fraction, fraction, vec2f(f32(x), f32(z)));
                let index = start + ((frame * dims.y + base.y + z) * dims.x + base.x + x);
                result += timeWeight * amount.x * amount.y * atmosphere[index].state;
            }
        }
    }
    return result;
}

fn fieldFaceVelocity(field: texture_3d<f32>, cell: vec3i, axis: i32) -> f32 {
    if (!faceOpen(cell, axis)) {
        return 0.0;
    }

    if (!dustPhase && axis != 1 && (cell[axis] == 0 || cell[axis] == gridDims[axis])) {
        if (spatialWeather) {
            var offset = vec3f(0.5);
            offset[axis] = 0.0;
            return spatialState(vec3f(cell) + offset).wind[axis];
        }
        let wind = quantizeToF16(background(f32(cell.y) + 0.5).state.zw);
        return select(wind.x, wind.y, axis == 2);
    }

    return textureLoad(field, phaseCell(cell), 0)[axis];
}

fn faceVelocity(cell: vec3i, axis: i32) -> f32 {
    return fieldFaceVelocity(velocityIn, cell, axis);
}

fn centerVelocity(cell: vec3i) -> vec3f {
    var nearest = clamp(cell, vec3i(0), gridDims - 1);
    let reflected = freeSlipGround && cell.y < 0;
    if (reflected) {
        nearest.y = 0;
    }
    var velocity = vec3f(0);

    for (var axis = 0; axis < 3; axis++) {
        velocity[axis] = 0.5 * (faceVelocity(nearest, axis)
            + faceVelocity(nearest + axisOffset(axis), axis));
    }

    if (reflected) {
        velocity.y = -velocity.y;
    }
    return velocity;
}
