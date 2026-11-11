@compute @workgroup_size(8, 8, 4)
fn divergence(@builtin(global_invocation_id) id: vec3u) {
    let cell = vec3i(id);
    if (!inside(cell)) {
        return;
    }

    var value = 0.0;

    if (fluid(cell)) {
        for (var axis = 0; axis < 3; axis++) {
            let upper = faceVelocity(cell + axisOffset(axis), axis);
            let lower = faceVelocity(cell, axis);
            value += (upper - lower) / cellSize[axis];
        }
    }

    textureStore(scalarOut, cell, vec4f(value, 0, 0, 0));
}

// Pressure is stored as an impulse in m²/s, not atmospheric pressure in Pa.
@compute @workgroup_size(8, 8, 4)
fn jacobi(@builtin(global_invocation_id) id: vec3u) {
    let cell = vec3i(id);
    if (!inside(cell)) {
        return;
    }

    var pressure = 0.0;

    if (fluid(cell)) {
        var sum = 0.0;
        var diagonal = 0.0;

        for (var direction = 0; direction < 6; direction++) {
            let axis = direction / 2;
            let positive = direction % 2 == 1;
            let sign = select(-1, 1, positive);
            let neighbor = cell + sign * axisOffset(axis);

            if (inside(neighbor) && connection(cell, axis, positive)) {
                let weight = 1.0 / (cellSize[axis] * cellSize[axis]);
                sum += weight * textureLoad(pressureIn, neighbor, 0).x;
                diagonal += weight;
            }
        }

        pressure = textureLoad(pressureIn, cell, 0).x;

        if (diagonal > 0.0) {
            let candidate = (sum - textureLoad(divergenceIn, cell, 0).x) / diagonal;
            pressure = mix(pressure, candidate, 2.0 / 3.0);
        }
    }

    textureStore(scalarOut, cell, vec4f(pressure, 0, 0, 0));
}

@compute @workgroup_size(8, 8, 4)
fn project(@builtin(global_invocation_id) id: vec3u) {
    let cell = vec3i(id);
    if (!inside(cell)) {
        return;
    }

    var velocity = vec3f(0);

    for (var axis = 0; axis < 3; axis++) {
        velocity[axis] = faceVelocity(cell, axis);
        let neighbor = cell - axisOffset(axis);

        if (faceOpen(cell, axis) && inside(neighbor)) {
            let upper = textureLoad(pressureIn, cell, 0).x;
            let lower = textureLoad(pressureIn, neighbor, 0).x;
            velocity[axis] -= (upper - lower) / cellSize[axis];
        }
    }

    textureStore(velocityOut, cell, vec4f(velocity, 0));
}
