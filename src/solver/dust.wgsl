fn dustDerivative(cell: vec3i, axis: i32) -> vec3f {
    let offset = axisOffset(axis);
    return (centerVelocity(phaseCell(cell + offset)) - centerVelocity(phaseCell(cell - offset)))
        / (2.0 * cellSize[axis]);
}

fn dustViscosity(cell: vec3i) -> f32 {
    let height = (f32(cell.y) + 0.5) * cellSize.y;
    let mixingLength = 0.4 * height;
    let shear = length(dustDerivative(cell, 1));
    let coefficient = parameters.dustViscosity / parameters.grainDensity
        + mixingLength * mixingLength * shear;
    let inverseSize = 1.0 / (cellSize * cellSize);
    let limit = 0.2 / (parameters.dt * (inverseSize.x + inverseSize.y + inverseSize.z));
    return min(coefficient, limit);
}

@compute @workgroup_size(8, 8, 4)
fn buildDustViscosity(@builtin(global_invocation_id) id: vec3u) {
    let cell = vec3i(id);
    if (inside(cell)) {
        var viscosity = 0.0;
        if (fluid(cell) && parameters.dustEnabled > 0.0) {
            viscosity = dustViscosity(cell);
        }
        textureStore(scalarOut, cell, vec4f(viscosity, 0, 0, 0));
    }
}

fn dustStress(cell: vec3i, axis: i32) -> vec3f {
    let neighbor = phaseCell(cell + axisOffset(axis));
    if (!fluid(cell) || !fluid(neighbor) || !connection(cell, axis, true)) {
        return vec3f(0);
    }
    let viscosity = 0.5 * (textureLoad(divergenceIn, cell, 0).x + textureLoad(divergenceIn, neighbor, 0).x);
    let derivative = (centerVelocity(neighbor) - centerVelocity(cell)) / cellSize[axis];
    var transpose = vec3f(0);
    for (var component = 0; component < 3; component++) {
        transpose[component] = 0.5 * (dustDerivative(cell, component)[axis]
            + dustDerivative(neighbor, component)[axis]);
    }
    return viscosity * (derivative + transpose);
}

fn dustForce(cell: vec3i) -> vec3f {
    if (!fluid(cell)) {
        return vec3f(0);
    }
    var force = vec3f(0, -9.81, 0);
    for (var axis = 0; axis < 3; axis++) {
        let lower = phaseCell(cell - axisOffset(axis));
        force += (dustStress(cell, axis) - dustStress(lower, axis)) / cellSize[axis];
    }
    return force;
}

@compute @workgroup_size(8, 8, 4)
fn buildDustForce(@builtin(global_invocation_id) id: vec3u) {
    let cell = vec3i(id);
    if (inside(cell)) {
        textureStore(forceOut, cell, vec4f(dustForce(cell), 0));
    }
}

@compute @workgroup_size(8, 8, 4)
fn predictDust(@builtin(global_invocation_id) id: vec3u) {
    let cell = vec3i(id);
    if (!inside(cell)) {
        return;
    }
    var velocity = vec3f(0);
    if (parameters.dustEnabled > 0.0) {
        let force = textureLoad(forceIn, cell, 0).xyz;
        for (var axis = 0; axis < 3; axis++) {
            if (faceOpen(cell, axis)) {
                let position = vec3f(cell) + faceOffset(axis);
                let lower = phaseCell(cell - axisOffset(axis));
                velocity[axis] = sampleComponent(traceBack(position), axis)
                    + parameters.dt * 0.5 * (force[axis] + textureLoad(forceIn, lower, 0)[axis]);
            }
        }
    }
    textureStore(velocityOut, cell, vec4f(velocity, 0));
}

fn airFace(cell: vec3i, axis: i32) -> f32 {
    if (axis == 1 && cell.y == gridDims.y) {
        return 0.0;
    }
    if (axis != 1 && (cell[axis] == 0 || cell[axis] == gridDims[axis])) {
        let wind = background(f32(cell.y) + 0.5).state.zw;
        return select(wind.x, wind.y, axis == 2);
    }
    return textureLoad(airVelocityIn, clamp(cell, vec3i(0), gridDims - 1), 0)[axis];
}

fn airAt(cell: vec3i) -> vec3f {
    let nearest = clamp(cell, vec3i(0), gridDims - 1);
    var result = vec3f(0);
    for (var axis = 0; axis < 3; axis++) {
        result[axis] = 0.5 * (airFace(nearest, axis) + airFace(nearest + axisOffset(axis), axis));
    }
    return result;
}

fn dragTime(cell: vec3i) -> f32 {
    var laplacian = vec3f(0);
    let velocity = airAt(cell);
    for (var axis = 0; axis < 3; axis++) {
        let offset = axisOffset(axis);
        laplacian += (airAt(cell + offset) - 2.0 * velocity + airAt(cell - offset))
            / (cellSize[axis] * cellSize[axis]);
    }
    let radius = parameters.grainRadius;
    let viscosity = parameters.dustViscosity;
    let grainMass = 4.0 / 3.0 * 3.14159265 * radius * radius * radius * parameters.grainDensity;
    let reynolds = parameters.grainDensity * radius * length(laplacian) / viscosity;
    // Cyclogenesis Eq. 14. The stiff drag exchange is integrated analytically below.
    return grainMass / (3.0 * 3.14159265 * radius * viscosity)
        * (1.0 + reynolds / 60.0 + 0.25 * reynolds / (1.0 + sqrt(reynolds)));
}

@compute @workgroup_size(8, 8, 4)
fn buildDragTime(@builtin(global_invocation_id) id: vec3u) {
    let cell = vec3i(id);
    if (inside(cell)) {
        textureStore(scalarOut, cell, vec4f(dragTime(cell), 0, 0, 0));
    }
}

@compute @workgroup_size(8, 8, 4)
fn coupleDust(@builtin(global_invocation_id) id: vec3u) {
    let cell = vec3i(id);
    if (!inside(cell)) {
        return;
    }
    var velocity = vec3f(0);
    var feedback = vec3f(0);
    if (parameters.dustEnabled > 0.0 && fluid(cell)) {
        let airDensity = background(f32(cell.y) + 0.5).air.z;
        let concentration = textureLoad(dustDensityIn, cell, 0).x;
        for (var axis = 0; axis < 3; axis++) {
            if (faceOpen(cell, axis)) {
                let neighbor = phaseCell(cell - axisOffset(axis));
                let faceDensity = 0.5 * (concentration + textureLoad(dustDensityIn, neighbor, 0).x);
                let ratio = faceDensity / airDensity;
                let response = 0.5 * (textureLoad(divergenceIn, cell, 0).x + textureLoad(divergenceIn, neighbor, 0).x);
                let relaxation = 1.0 - exp(-(1.0 + ratio) * parameters.dt / max(response, 1e-9));
                let slip = faceVelocity(cell, axis) - airFace(cell, axis);
                let change = relaxation * slip / (1.0 + ratio);
                velocity[axis] = faceVelocity(cell, axis) - change;
                feedback[axis] = ratio * change / parameters.dt;
            }
        }
    }
    textureStore(velocityOut, cell, vec4f(velocity, 0));
    textureStore(windForcingOut, cell, vec4f(feedback, 0));
}

fn sampleDustDensity(position: vec3f) -> f32 {
    let location = phasePosition(position);
    if (location.y < 0.0 || location.y >= f32(gridDims.y)) {
        return 0.0;
    }
    let coordinate = location - vec3f(0.5);
    let base = vec3i(floor(coordinate));
    let fraction = fract(coordinate);
    var value = 0.0;
    var total = 0.0;
    for (var z = 0; z < 2; z++) {
        for (var y = 0; y < 2; y++) {
            for (var x = 0; x < 2; x++) {
                let corner = vec3i(x, y, z);
                let neighbor = phaseCell(base + corner);
                if (fluid(neighbor)) {
                    let weight = mix(vec3f(1) - fraction, fraction, vec3f(corner));
                    let amount = weight.x * weight.y * weight.z;
                    value += amount * textureLoad(dustDensityIn, neighbor, 0).x;
                    total += amount;
                }
            }
        }
    }
    return value / max(total, 1e-6);
}

@compute @workgroup_size(8, 8, 4)
fn transportDustDensity(@builtin(global_invocation_id) id: vec3u) {
    let cell = vec3i(id);
    if (!inside(cell)) {
        return;
    }
    var concentration = 0.0;
    if (fluid(cell) && parameters.dustEnabled > 0.0) {
        let position = vec3f(cell) + vec3f(0.5);
        concentration = sampleDustDensity(traceBack(position));
        let world = domainMin + position * cellSize;
        let nearGround = !fluid(cell - vec3i(0, 1, 0)) || !fluid(cell - vec3i(0, 2, 0));
        if (parameters.seedDust > 0.0 && nearGround
            && abs(world.x) < 0.325 * domainSize.x && abs(world.z) < 0.325 * domainSize.z) {
            concentration += parameters.dustLoading;
        }
    }
    textureStore(dustDensityOut, cell, vec4f(concentration, 0, 0, 0));
}

@compute @workgroup_size(8, 8, 4)
fn resetDust(@builtin(global_invocation_id) id: vec3u) {
    if (inside(vec3i(id))) {
        textureStore(velocityOut, vec3i(id), vec4f(0));
        textureStore(dustDensityOut, vec3i(id), vec4f(0));
    }
}

@compute @workgroup_size(8, 8, 4)
fn clearDustPressure(@builtin(global_invocation_id) id: vec3u) {
    if (inside(vec3i(id))) {
        textureStore(scalarOut, vec3i(id), vec4f(0));
    }
}
