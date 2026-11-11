fn faceOffset(axis: i32) -> vec3f {
    var offset = vec3f(0.5);
    offset[axis] = 0.0;
    return offset;
}

fn sampleComponent(position: vec3f, axis: i32) -> f32 {
    if (position.x < 0.0 || position.z < 0.0
        || position.x > f32(gridDims.x) || position.z > f32(gridDims.z)) {
        if (axis == 1) {
            return 0.0;
        }

        let wind = background(position.y).state.zw;
        return select(wind.x, wind.y, axis == 2);
    }

    var upper = vec3f(gridDims - 1);
    upper[axis] = f32(gridDims[axis]);

    let coordinate = clamp(position - faceOffset(axis), vec3f(0), upper);
    let base = vec3i(floor(coordinate));
    let fraction = fract(coordinate);
    var result = 0.0;

    for (var z = 0; z < 2; z++) {
        for (var y = 0; y < 2; y++) {
            for (var x = 0; x < 2; x++) {
                let corner = vec3i(x, y, z);
                let weight = mix(vec3f(1) - fraction, fraction, vec3f(corner));
                result += faceVelocity(base + corner, axis) * weight.x * weight.y * weight.z;
            }
        }
    }

    return result;
}

fn sampleVelocity(position: vec3f) -> vec3f {
    return vec3f(
        sampleComponent(position, 0),
        sampleComponent(position, 1),
        sampleComponent(position, 2)
    );
}

fn sampleThermo(position: vec3f) -> vec4f {
    if (any(position < vec3f(0)) || any(position >= vec3f(gridDims))) {
        return vec4f(0);
    }

    let coordinate = clamp(position - vec3f(0.5), vec3f(0), vec3f(gridDims - 1));
    let base = vec3i(floor(coordinate));
    let fraction = fract(coordinate);
    var result = vec4f(0);
    var totalWeight = 0.0;

    for (var z = 0; z < 2; z++) {
        for (var y = 0; y < 2; y++) {
            for (var x = 0; x < 2; x++) {
                let corner = vec3i(x, y, z);
                let cell = min(base + corner, gridDims - 1);

                if (fluid(cell)) {
                    let weight = mix(vec3f(1) - fraction, fraction, vec3f(corner));
                    let contribution = weight.x * weight.y * weight.z;
                    result += textureLoad(thermoIn, cell, 0) * contribution;
                    totalWeight += contribution;
                }
            }
        }
    }

    return result / max(totalWeight, 1e-6);
}

fn traceBack(position: vec3f) -> vec3f {
    let midpoint = position - 0.5 * parameters.dt * sampleVelocity(position) / cellSize;
    let departure = position - parameters.dt * sampleVelocity(midpoint) / cellSize;
    let segments = max(1, i32(ceil(length(departure - position) * 2.0)));
    var previous = position;

    for (var segment = 1; segment <= segments; segment++) {
        let point = mix(position, departure, f32(segment) / f32(segments));
        let cell = vec3i(floor(point));

        if (!inside(cell)) {
            return departure;
        }

        if (!fluid(cell)) {
            return previous;
        }

        previous = point;
    }

    return departure;
}

@compute @workgroup_size(8, 8, 4)
fn advect(@builtin(global_invocation_id) id: vec3u) {
    let cell = vec3i(id);
    if (!inside(cell)) {
        return;
    }

    var velocity = vec3f(0);

    for (var axis = 0; axis < 3; axis++) {
        if (!faceOpen(cell, axis)) {
            continue;
        }

        if (axis != 1 && cell[axis] == 0) {
            velocity[axis] = faceVelocity(cell, axis);
        } else {
            let position = vec3f(cell) + faceOffset(axis);
            velocity[axis] = sampleComponent(traceBack(position), axis);
        }
    }

    textureStore(velocityOut, cell, vec4f(velocity, 0));

    var thermo = vec4f(0);

    if (fluid(cell)) {
        let position = vec3f(cell) + vec3f(0.5);
        let departure = traceBack(position);
        thermo = sampleThermo(departure);
        thermo += vec4f(background(departure.y).state.xy - background(position.y).state.xy, 0.0, 0.0);

        if (parameters.addThermal > 0.0) {
            let world = domainMin + position * cellSize;
            let radius = max(700.0, cellSize.x);
            let height = max(600.0, 2.0 * cellSize.y);
            let distance = vec3f(world.x / radius, (world.y - height) / height, world.z / radius);
            let weight = exp(-dot(distance, distance));
            thermo += weight * vec4f(2.0, 0, 0.2, 0);
        }

        thermo.y = max(thermo.y, -background(position.y).state.y);
        thermo = vec4f(thermo.xy, max(thermo.zw, vec2f(0)));
    }

    textureStore(thermoOut, cell, thermo);
}
