fn velocityDerivative(cell: vec3i, axis: i32) -> vec3f {
    let offset = axisOffset(axis);
    return (centerVelocity(cell + offset) - centerVelocity(cell - offset)) / (2.0 * cellSize[axis]);
}

@compute @workgroup_size(8, 8, 4)
fn curl(@builtin(global_invocation_id) id: vec3u) {
    let cell = vec3i(id);
    if (!inside(cell)) {
        return;
    }

    var rotation = vec3f(0);

    if (fluid(cell)) {
        let dx = velocityDerivative(cell, 0);
        let dy = velocityDerivative(cell, 1);
        let dz = velocityDerivative(cell, 2);
        rotation = vec3f(dy.z - dz.y, dz.x - dx.z, dx.y - dy.x);
    }

    textureStore(vorticityOut, cell, vec4f(rotation, length(rotation)));
}

fn rotationMagnitude(cell: vec3i) -> f32 {
    if (!fluid(cell)) {
        return 0.0;
    }

    return textureLoad(vorticityIn, cell, 0).w;
}

fn centerForce(cell: vec3i) -> vec3f {
    if (!fluid(cell)) {
        return vec3f(0);
    }

    let thermo = textureLoad(thermoIn, cell, 0);
    let air = background(f32(cell.y) + 0.5);
    let buoyancy = 9.81 * (thermo.x / air.state.x
        + 0.61 * thermo.y * 0.001 - (thermo.z + thermo.w) * 0.001);
    var gradient = vec3f(0);

    for (var axis = 0; axis < 3; axis++) {
        let offset = axisOffset(axis);
        gradient[axis] = (rotationMagnitude(cell + offset) - rotationMagnitude(cell - offset))
            / (2.0 * cellSize[axis]);
    }

    let normal = gradient / max(length(gradient), 1e-6);
    let rotation = textureLoad(vorticityIn, cell, 0).xyz;
    let confinement = parameters.confinement * cellSize.x * cross(normal, rotation);

    return vec3f(0, parameters.buoyancy * buoyancy, 0) + confinement;
}

@compute @workgroup_size(8, 8, 4)
fn forces(@builtin(global_invocation_id) id: vec3u) {
    let cell = vec3i(id);
    if (!inside(cell)) {
        return;
    }

    var velocity = vec3f(0);
    let force = centerForce(cell);

    for (var axis = 0; axis < 3; axis++) {
        if (!faceOpen(cell, axis)) {
            continue;
        }

        velocity[axis] = faceVelocity(cell, axis);
        let neighbor = cell - axisOffset(axis);

        if (inside(neighbor)) {
            let faceForce = 0.5 * (force + centerForce(neighbor));
            velocity[axis] += parameters.dt * faceForce[axis];
        }
    }

    textureStore(velocityOut, cell, vec4f(velocity, 0));
}
