@compute @workgroup_size(8, 8, 4)
fn shiftStormWindow(@builtin(global_invocation_id) id: vec3u) {
    let cell = vec3i(id);
    if (!inside(cell)) {
        return;
    }
    let source = cell + vec3i(i32(storm.region.z), 0, i32(storm.region.w));
    let wind = quantizeToF16(background(f32(cell.y) + 0.5).state.zw);
    var velocity = vec4f(wind.x, 0, wind.y, 0);
    var thermo = vec4f(0);
    var energyValue = vec4f(0);
    if (inside(source)) {
        velocity = textureLoad(velocityIn, source, 0);
        thermo = textureLoad(thermoIn, source, 0);
        energyValue = textureLoad(turbulenceIn, source, 0);
    }
    textureStore(velocityOut, cell, velocity);
    textureStore(thermoOut, cell, thermo);
    textureStore(turbulenceOut, cell, energyValue);
}

@compute @workgroup_size(8, 8, 4)
fn buildVelocityGradients(@builtin(global_invocation_id) id: vec3u) {
    let cell = vec3i(id);
    if (!inside(cell)) {
        return;
    }
    textureStore(velocityOut, cell, vec4f(velocityDerivative(cell, 0), 0));
    textureStore(vorticityOut, cell, vec4f(velocityDerivative(cell, 1), 0));
    textureStore(airGradientZOut, cell, vec4f(velocityDerivative(cell, 2), 0));
}

fn cachedDerivative(cell: vec3i, axis: i32) -> vec3f {
    if (axis == 0) {
        return textureLoad(airGradientX, cell, 0).xyz;
    }
    if (axis == 1) {
        return textureLoad(airGradientY, cell, 0).xyz;
    }
    return textureLoad(airGradientZ, cell, 0).xyz;
}

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
    // A small energy floor lets shear start producing turbulence from a resting state.
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
        let dx = cachedDerivative(cell, 0);
        let dy = cachedDerivative(cell, 1);
        let dz = cachedDerivative(cell, 2);
        let shear = vec3f(dx.y + dy.x, dx.z + dz.x, dy.z + dz.y);
        let strain = 2.0 * (dx.x * dx.x + dy.y * dy.y + dz.z * dz.z) + dot(shear, shear);
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

        // Warm-water, dimensionally normalized one-equation closure for Eqs. 5 and 6.
        let source = viscosity * strain - mixing * stratification + diffusion;
        let energyValue = max(0.0, previous + parameters.dt * source)
            / (1.0 + parameters.dt * 0.2 * sqrt(max(previous, 0.0)) / cellSize.x);
        let updatedViscosity = eddyViscosity(energyValue);
        result = vec4f(energyValue, updatedViscosity,
            3.0 * parameters.mixing * updatedViscosity, 0);
    }

    textureStore(turbulenceOut, cell, result);
}

fn referenceVelocity(cell: vec3i) -> vec3f {
    let height = f32(clamp(cell.y, 0, gridDims.y - 1)) + 0.5;
    let air = background(height);
    return quantizeToF16(vec3f(air.state.z, 0, air.state.w));
}

fn referenceDerivative(cell: vec3i, axis: i32) -> vec3f {
    if (axis != 1) {
        return vec3f(0);
    }
    let offset = axisOffset(axis);
    return quantizeToF16((referenceVelocity(cell + offset) - referenceVelocity(cell - offset))
        / (2.0 * cellSize[axis]));
}

fn stress(rawCell: vec3i, axis: i32) -> vec3f {
    var cell = rawCell;
    var neighbor = cell + axisOffset(axis);
    if (storm.timing.x > 0.0 && axis != 1) {
        // Continue the shear stress through open sides instead of creating an artificial edge force.
        cell = clamp(cell, vec3i(0), gridDims - 1);
        neighbor = clamp(neighbor, vec3i(0), gridDims - 1);
    }
    if (!fluid(cell) || !fluid(neighbor)) {
        return vec3f(0);
    }

    let viscosity = 0.5 * (textureLoad(turbulenceIn, cell, 0).y
        + textureLoad(turbulenceIn, neighbor, 0).y);
    var derivative = (centerVelocity(neighbor) - centerVelocity(cell)) / cellSize[axis];
    if (storm.timing.x > 0.0 && storm.damping.w > 0.0) {
        // The experiment maintains its reference column against background turbulent stress.
        derivative -= (referenceVelocity(neighbor) - referenceVelocity(cell)) / cellSize[axis];
    }
    var transpose = vec3f(0);
    for (var component = 0; component < 3; component++) {
        transpose[component] = 0.5 * (cachedDerivative(cell, component)[axis]
            + cachedDerivative(neighbor, component)[axis]);
        if (storm.timing.x > 0.0 && storm.damping.w > 0.0) {
            transpose[component] -= 0.5 * (referenceDerivative(cell, component)[axis]
                + referenceDerivative(neighbor, component)[axis]);
        }
    }
    return viscosity * (derivative + transpose);
}

@compute @workgroup_size(4, 4, 4)
fn atmosphericForces(@builtin(global_invocation_id) id: vec3u) {
    let cell = vec3i(id);
    if (!inside(cell)) {
        return;
    }

    var force = vec3f(0);
    if (fluid(cell)) {
        force = vec3f(0, buoyancyAt(cell), 0);
        for (var axis = 0; axis < 3; axis++) {
            let offset = axisOffset(axis);
            force += (stress(cell, axis) - stress(cell - offset, axis)) / cellSize[axis];
            // The balanced experiment absorbs the isotropic 2k/3 stress into pressure.
            if (storm.timing.x == 0.0 || storm.damping.w == 0.0) {
                let lower = scalarNeighbor(turbulenceIn, cell, cell - offset).x;
                let upper = scalarNeighbor(turbulenceIn, cell, cell + offset).x;
                force[axis] -= (upper - lower) / (3.0 * cellSize[axis]);
            }
        }
        if (storm.timing.x > 0.0) {
            let world = domainMin + (vec3f(cell) + vec3f(0.5)) * cellSize;
            let velocity = centerVelocity(cell);
            let air = background(f32(cell.y) + 0.5);
            let damping = smoothstep(storm.damping.x, domainSize.y, world.y) * storm.damping.y;
            let ambient = quantizeToF16(vec3f(air.state.z, 0, air.state.w));
            force -= damping * (velocity - ambient);

            // Temporary lifting initiates convection without prescribing circulation.
            let radius = vec3f(storm.region.x, storm.region.y, storm.region.x);
            let distance = length((world - vec3f(0, storm.region.y, 0)) / radius);
            let shape = pow(max(0.0, cos(min(distance, 1.0) * 1.5707963)), 2.0);
            let fade = 1.0 - clamp((storm.timing.y - 0.75 * storm.timing.z) / (0.25 * storm.timing.z), 0.0, 1.0);
            let rate = (1.0 - exp(-0.5 * fade * parameters.dt)) / parameters.dt;
            force.y += rate * max(storm.timing.w * shape - velocity.y, 0.0);
        }
    }
    textureStore(forceOut, cell, vec4f(force, 0));
}

fn totalForceAt(cell: vec3i) -> vec3f {
    let nearest = clamp(cell, vec3i(0), gridDims - 1);
    var feedback = vec3f(0);
    for (var axis = 0; axis < 3; axis++) {
        let upper = min(nearest + axisOffset(axis), gridDims - 1);
        feedback[axis] = 0.5 * (textureLoad(windForcing, nearest, 0)[axis]
            + textureLoad(windForcing, upper, 0)[axis]);
    }
    return textureLoad(forceIn, nearest, 0).xyz + feedback;
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

        let dx = velocityDerivative(cell, 0);
        let dy = velocityDerivative(cell, 1);
        let dz = velocityDerivative(cell, 2);
        var stretching = rotation.x * dx + rotation.y * dy + rotation.z * dz;
        if (anelasticAir) {
            stretching -= rotation * (dx.x + dy.y + dz.z);
        }
        let fx = (totalForceAt(cell + vec3i(1, 0, 0)) - totalForceAt(cell - vec3i(1, 0, 0))) / (2.0 * cellSize.x);
        let fy = (totalForceAt(cell + vec3i(0, 1, 0)) - totalForceAt(cell - vec3i(0, 1, 0))) / (2.0 * cellSize.y);
        let fz = (totalForceAt(cell + vec3i(0, 0, 1)) - totalForceAt(cell - vec3i(0, 0, 1))) / (2.0 * cellSize.z);
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
    let wind = quantizeToF16(background(f32(cell.y) + 0.5).state.zw);
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

@compute @workgroup_size(8, 8, 4)
fn clearForcing(@builtin(global_invocation_id) id: vec3u) {
    if (inside(vec3i(id))) {
        textureStore(windForcingOut, vec3i(id), vec4f(0));
    }
}
