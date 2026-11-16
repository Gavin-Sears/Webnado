fn insidePosition(position: vec3f) -> bool {
    return all(position >= vec3f(0)) && all(position < vec3f(gridDims));
}

fn absoluteThermo(field: texture_3d<f32>, position: vec3f) -> vec4f {
    return sampleThermoField(field, position) + vec4f(background(position.y).state.xy, 0, 0);
}

@compute @workgroup_size(8, 8, 4)
fn correctAdvection(@builtin(global_invocation_id) id: vec3u) {
    let cell = vec3i(id);
    if (!inside(cell)) {
        return;
    }
    var velocity = vec3f(0);
    for (var axis = 0; axis < 3; axis++) {
        if (!faceOpen(cell, axis)) {
            continue;
        }
        let position = vec3f(cell) + faceOffset(axis);
        let departure = traceField(forceIn, position, parameters.dt);
        let returning = traceField(forceIn, position, -parameters.dt);
        let forward = faceVelocity(cell, axis);
        velocity[axis] = forward;
        if (!insidePosition(departure) || !insidePosition(returning)) {
            continue;
        }
        var corrected = forward + 0.5 * (fieldFaceVelocity(forceIn, cell, axis)
            - sampleComponent(returning, axis));
        let base = vec3i(floor(departure - faceOffset(axis)));
        var lower = 1e20;
        var upper = -1e20;
        for (var z = 0; z < 2; z++) {
            for (var y = 0; y < 2; y++) {
                for (var x = 0; x < 2; x++) {
                    let value = fieldFaceVelocity(forceIn, base + vec3i(x, y, z), axis);
                    lower = min(lower, value);
                    upper = max(upper, value);
                }
            }
        }
        corrected = clamp(corrected, lower, upper);
        velocity[axis] = corrected;
    }
    textureStore(velocityOut, cell, vec4f(velocity, 0));

    var state = vec4f(0);
    if (fluid(cell)) {
        let position = vec3f(cell) + vec3f(0.5);
        let departure = traceField(forceIn, position, parameters.dt);
        let returning = traceField(forceIn, position, -parameters.dt);
        let ambient = vec4f(background(position.y).state.xy, 0, 0);
        state = textureLoad(thermoIn, cell, 0);
        if (insidePosition(departure) && insidePosition(returning)) {
            let original = textureLoad(vorticityIn, cell, 0) + ambient;
            let backward = absoluteThermo(thermoIn, returning);
            let corrected = state + ambient + 0.5 * (original - backward);
            let base = vec3i(floor(departure - vec3f(0.5)));
            var lower = vec4f(1e20);
            var upper = vec4f(-1e20);
            var count = 0u;
            for (var z = 0; z < 2; z++) {
                for (var y = 0; y < 2; y++) {
                    for (var x = 0; x < 2; x++) {
                        let neighbor = base + vec3i(x, y, z);
                        if (fluid(neighbor)) {
                            let value = textureLoad(vorticityIn, neighbor, 0)
                                + vec4f(background(f32(neighbor.y) + 0.5).state.xy, 0, 0);
                            lower = min(lower, value);
                            upper = max(upper, value);
                            count++;
                        }
                    }
                }
            }
            if (count > 0u) {
                state = clamp(corrected, lower, upper) - ambient;
            }
        }
        state.y = max(state.y, -ambient.y);
        state = vec4f(state.xy, max(state.zw, vec2f(0)));
    }
    textureStore(thermoOut, cell, state);
}
