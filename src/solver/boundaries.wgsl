fn obstacleFluid(cell: vec3i) -> bool {
    if (!inside(cell)) {
        return false;
    }

    return textureLoad(obstacles, cell, 0).x == 0u;
}

fn touchesGround(cell: vec3i) -> bool {
    return !obstacleFluid(cell - vec3i(0, 1, 0));
}

@compute @workgroup_size(8, 8, 4)
fn buildOpenBits(@builtin(global_invocation_id) id: vec3u) {
    let cell = vec3i(id);
    if (!inside(cell)) {
        return;
    }

    var bits = 0u;

    if (obstacleFluid(cell)) {
        bits = fluidBit;

        for (var direction = 0; direction < 6; direction++) {
            let axis = direction / 2;
            let sign = select(-1, 1, direction % 2 == 1);
            let neighbor = cell + sign * axisOffset(axis);
            var open = obstacleFluid(neighbor);

            if (!inside(neighbor) && axis != 1) {
                open = true;
            }

            // Close ground-adjacent tangential faces so projection preserves no-slip.
            if (axis != 1 && (touchesGround(cell)
                || (inside(neighbor) && touchesGround(neighbor)))) {
                open = false;
            }

            if (open) {
                bits |= 1u << u32(direction);
            }
        }
    }

    textureStore(openBitsOut, cell, vec4u(bits, 0, 0, 0));
}

@compute @workgroup_size(8, 8, 4)
fn enforceBoundaries(@builtin(global_invocation_id) id: vec3u) {
    let cell = vec3i(id);
    if (!inside(cell)) {
        return;
    }

    var velocity = vec3f(0);

    for (var axis = 0; axis < 3; axis++) {
        velocity[axis] = faceVelocity(cell, axis);
    }

    textureStore(velocityOut, cell, vec4f(velocity, 0));
}
