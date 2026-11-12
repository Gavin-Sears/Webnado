fn sampleDynamic(field: texture_3d<f32>, position: vec3f) -> vec4f {
    let coordinate = clamp(position - vec3f(0.5), vec3f(0), vec3f(gridDims - 1));
    let base = vec3i(floor(coordinate));
    let fraction = fract(coordinate);
    var result = vec4f(0);
    var total = 0.0;

    for (var z = 0; z < 2; z++) {
        for (var y = 0; y < 2; y++) {
            for (var x = 0; x < 2; x++) {
                let corner = vec3i(x, y, z);
                let cell = min(base + corner, gridDims - 1);
                if (fluid(cell)) {
                    let weight = mix(vec3f(1) - fraction, fraction, vec3f(corner));
                    let amount = weight.x * weight.y * weight.z;
                    result += amount * textureLoad(field, cell, 0);
                    total += amount;
                }
            }
        }
    }

    return result / max(total, 1e-6);
}

fn scalarNeighbor(field: texture_3d<f32>, cell: vec3i, neighbor: vec3i) -> vec4f {
    if (!fluid(neighbor)) {
        return textureLoad(field, cell, 0);
    }
    return textureLoad(field, neighbor, 0);
}

fn eddyViscosity(k: f32) -> f32 {
    let inverseSize = 1.0 / (cellSize * cellSize);
    // Explicit diffusion stays within its stability bound, including the symmetric stress term.
    let limit = 0.2 / (parameters.dt * (inverseSize.x + inverseSize.y + inverseSize.z)
        * max(1.0, 3.0 * parameters.mixing));
    return min(0.2 * cellSize.x * sqrt(max(k, 1e-4)), limit) * parameters.turbulence;
}

fn buoyancyAt(cell: vec3i) -> f32 {
    let state = textureLoad(thermoIn, cell, 0);
    let air = background(f32(cell.y) + 0.5);
    return parameters.buoyancy * 9.81 * (state.x / air.state.x
        + 0.00061 * state.y - 0.001 * (state.z + state.w));
}

fn virtualTemperature(cell: vec3i) -> f32 {
    let nearest = clamp(cell, vec3i(0), gridDims - 1);
    let state = textureLoad(thermoIn, nearest, 0);
    let air = background(f32(nearest.y) + 0.5);
    return (air.state.x + state.x)
        * (1.0 + 0.00061 * (air.state.y + state.y) - 0.001 * (state.z + state.w));
}

@compute @workgroup_size(8, 8, 4)
fn energy(@builtin(global_invocation_id) id: vec3u) {
    let cell = vec3i(id);
    if (!inside(cell)) {
        return;
    }

    var result = vec4f(0);
    if (fluid(cell) && parameters.turbulence > 0.0) {
        let position = vec3f(cell) + vec3f(0.5);
        let previous = sampleDynamic(turbulenceIn, traceBack(position)).x;
        let viscosity = eddyViscosity(previous);
        let mixing = 3.0 * parameters.mixing * viscosity;
        let dx = velocityDerivative(cell, 0);
        let dy = velocityDerivative(cell, 1);
        let dz = velocityDerivative(cell, 2);
        let strain = 2.0 * (dx.x * dx.x + dy.y * dy.y + dz.z * dz.z)
            + pow(dx.y + dy.x, 2.0) + pow(dx.z + dz.x, 2.0) + pow(dy.z + dz.y, 2.0);
        let theta = background(position.y).state.x;
        let stratification = parameters.buoyancy * 9.81 / theta
            * (virtualTemperature(cell + vec3i(0, 1, 0)) - virtualTemperature(cell - vec3i(0, 1, 0)))
            / (2.0 * cellSize.y);
        var diffusion = 0.0;
        let center = textureLoad(turbulenceIn, cell, 0).x;

        for (var axis = 0; axis < 3; axis++) {
            let offset = axisOffset(axis);
            let lower = scalarNeighbor(turbulenceIn, cell, cell - offset).x;
            let upper = scalarNeighbor(turbulenceIn, cell, cell + offset).x;
            diffusion += viscosity * (upper - 2.0 * center + lower) / pow(cellSize[axis], 2.0);
        }

        // Warm-water, dimensionally normalized one-equation closure for Eqs. 5–6.
        let source = viscosity * strain - mixing * stratification + diffusion;
        let energyValue = max(0.0, previous + parameters.dt * source)
            / (1.0 + parameters.dt * 0.2 * sqrt(max(previous, 0.0)) / cellSize.x);
        let updatedViscosity = eddyViscosity(energyValue);
        result = vec4f(energyValue, updatedViscosity,
            3.0 * parameters.mixing * updatedViscosity, 0);
    }

    textureStore(turbulenceOut, cell, result);
}

fn stress(cell: vec3i, axis: i32) -> vec3f {
    let neighbor = cell + axisOffset(axis);
    if (!fluid(cell) || !fluid(neighbor)) {
        return vec3f(0);
    }

    let viscosity = 0.5 * (textureLoad(turbulenceIn, cell, 0).y
        + textureLoad(turbulenceIn, neighbor, 0).y);
    let derivative = (centerVelocity(neighbor) - centerVelocity(cell)) / cellSize[axis];
    var transpose = vec3f(0);
    for (var component = 0; component < 3; component++) {
        transpose[component] = 0.5 * (velocityDerivative(cell, component)[axis]
            + velocityDerivative(neighbor, component)[axis]);
    }
    return viscosity * (derivative + transpose);
}

@compute @workgroup_size(8, 8, 4)
fn atmosphericForces(@builtin(global_invocation_id) id: vec3u) {
    let cell = vec3i(id);
    if (!inside(cell)) {
        return;
    }

    var force = vec3f(0);
    if (fluid(cell)) {
        force.y = buoyancyAt(cell);
        for (var axis = 0; axis < 3; axis++) {
            let offset = axisOffset(axis);
            force += (stress(cell, axis) - stress(cell - offset, axis)) / cellSize[axis];
            let lower = scalarNeighbor(turbulenceIn, cell, cell - offset).x;
            let upper = scalarNeighbor(turbulenceIn, cell, cell + offset).x;
            force[axis] -= (upper - lower) / (3.0 * cellSize[axis]);
        }
    }
    textureStore(forceOut, cell, vec4f(force, 0));
}

fn measuredCurl(cell: vec3i) -> vec3f {
    let dx = velocityDerivative(cell, 0);
    let dy = velocityDerivative(cell, 1);
    let dz = velocityDerivative(cell, 2);
    return vec3f(dy.z - dz.y, dz.x - dx.z, dx.y - dy.x);
}

@compute @workgroup_size(8, 8, 4)
fn evolveVorticity(@builtin(global_invocation_id) id: vec3u) {
    let cell = vec3i(id);
    if (!inside(cell)) {
        return;
    }

    var rotation = vec3f(0);
    if (fluid(cell)) {
        let position = vec3f(cell) + vec3f(0.5);
        rotation = sampleDynamic(vorticityIn, traceBack(position)).xyz;
        if (parameters.initializeVorticity > 0.0) {
            rotation = measuredCurl(cell);
        }

        let stretching = rotation.x * velocityDerivative(cell, 0)
            + rotation.y * velocityDerivative(cell, 1)
            + rotation.z * velocityDerivative(cell, 2);
        let fx = (scalarNeighbor(forceIn, cell, cell + vec3i(1, 0, 0)).xyz
            - scalarNeighbor(forceIn, cell, cell - vec3i(1, 0, 0)).xyz) / (2.0 * cellSize.x);
        let fy = (scalarNeighbor(forceIn, cell, cell + vec3i(0, 1, 0)).xyz
            - scalarNeighbor(forceIn, cell, cell - vec3i(0, 1, 0)).xyz) / (2.0 * cellSize.y);
        let fz = (scalarNeighbor(forceIn, cell, cell + vec3i(0, 0, 1)).xyz
            - scalarNeighbor(forceIn, cell, cell - vec3i(0, 0, 1)).xyz) / (2.0 * cellSize.z);
        rotation += parameters.dt * (stretching + vec3f(fy.z - fz.y, fz.x - fx.z, fx.y - fy.x));
    }

    textureStore(vorticityOut, cell, vec4f(rotation, length(rotation)));
}

@compute @workgroup_size(8, 8, 4)
fn mixThermo(@builtin(global_invocation_id) id: vec3u) {
    let cell = vec3i(id);
    if (!inside(cell)) {
        return;
    }

    var state = vec4f(0);
    if (fluid(cell)) {
        state = textureLoad(thermoIn, cell, 0);
        var flux = vec4f(0);
        let mixing = textureLoad(turbulenceIn, cell, 0).z;
        for (var axis = 0; axis < 3; axis++) {
            let offset = axisOffset(axis);
            for (var sign = -1; sign <= 1; sign += 2) {
                let neighbor = cell + sign * offset;
                if (fluid(neighbor)) {
                    let coefficient = 0.5 * (mixing + textureLoad(turbulenceIn, neighbor, 0).z);
                    flux += coefficient * (textureLoad(thermoIn, neighbor, 0) - state) / pow(cellSize[axis], 2.0);
                }
            }
        }
        state += parameters.dt * flux;
    }
    textureStore(thermoOut, cell, state);
}

@compute @workgroup_size(8, 8, 4)
fn resetFields(@builtin(global_invocation_id) id: vec3u) {
    let cell = vec3i(id);
    if (!inside(cell)) {
        return;
    }
    let wind = background(f32(cell.y) + 0.5).state.zw;
    textureStore(velocityOut, cell, vec4f(wind.x, 0, wind.y, 0));
    textureStore(thermoOut, cell, vec4f(0));
    textureStore(vorticityOut, cell, vec4f(0));
    textureStore(turbulenceOut, cell, vec4f(0));
}

@compute @workgroup_size(8, 8, 4)
fn clearPressure(@builtin(global_invocation_id) id: vec3u) {
    if (inside(vec3i(id))) {
        textureStore(scalarOut, vec3i(id), vec4f(0));
    }
}
