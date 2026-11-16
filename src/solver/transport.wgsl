fn faceOffset(axis: i32) -> vec3f {
    var offset = vec3f(0.5);
    offset[axis] = 0.0;
    return offset;
}

fn sampleFieldComponent(field: texture_3d<f32>, rawPosition: vec3f, axis: i32) -> f32 {
    let position = phasePosition(rawPosition);
    if (!dustPhase && (position.x < 0.0 || position.z < 0.0
        || position.x > f32(gridDims.x) || position.z > f32(gridDims.z))) {
        if (axis == 1) {
            return 0.0;
        }

        let wind = quantizeToF16(background(position.y).state.zw);
        return select(wind.x, wind.y, axis == 2);
    }

    var upper = vec3f(gridDims - 1);
    upper[axis] = f32(gridDims[axis]);

    var coordinate = clamp(position - faceOffset(axis), vec3f(0), upper);
    if (dustPhase) {
        coordinate.x = position.x - faceOffset(axis).x;
        coordinate.z = position.z - faceOffset(axis).z;
    }
    let base = vec3i(floor(coordinate));
    let fraction = fract(coordinate);
    let reference = fieldFaceVelocity(field, phaseCell(base), axis);
    var correction = 0.0;

    for (var z = 0; z < 2; z++) {
        for (var y = 0; y < 2; y++) {
            for (var x = 0; x < 2; x++) {
                let corner = vec3i(x, y, z);
                let weight = mix(vec3f(1) - fraction, fraction, vec3f(corner));
                correction += (fieldFaceVelocity(field, phaseCell(base + corner), axis) - reference)
                    * weight.x * weight.y * weight.z;
            }
        }
    }

    return reference + correction;
}

fn sampleComponent(position: vec3f, axis: i32) -> f32 {
    return sampleFieldComponent(velocityIn, position, axis);
}

fn sampleFieldVelocity(field: texture_3d<f32>, position: vec3f) -> vec3f {
    return vec3f(
        sampleFieldComponent(field, position, 0),
        sampleFieldComponent(field, position, 1),
        sampleFieldComponent(field, position, 2)
    );
}

fn sampleVelocity(position: vec3f) -> vec3f {
    return sampleFieldVelocity(velocityIn, position);
}

fn sampleThermoField(field: texture_3d<f32>, position: vec3f) -> vec4f {
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
                    result += textureLoad(field, cell, 0) * contribution;
                    totalWeight += contribution;
                }
            }
        }
    }

    return result / max(totalWeight, 1e-6);
}

fn sampleThermo(position: vec3f) -> vec4f {
    return sampleThermoField(thermoIn, position);
}

fn traceField(field: texture_3d<f32>, position: vec3f, dt: f32) -> vec3f {
    let midpoint = position - 0.5 * dt * sampleFieldVelocity(field, position) / cellSize;
    let departure = position - dt * sampleFieldVelocity(field, midpoint) / cellSize;
    let segments = max(1, i32(ceil(length(departure - position) * 2.0)));
    var previous = position;

    for (var segment = 1; segment <= segments; segment++) {
        let point = mix(position, departure, f32(segment) / f32(segments));
        let cell = vec3i(floor(phasePosition(point)));

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

fn traceBack(position: vec3f) -> vec3f {
    return traceField(velocityIn, position, parameters.dt);
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

        if (storm.timing.x > 0.0 && (departure.x < 0.5 || departure.z < 0.5
            || departure.x > f32(gridDims.x) - 0.5 || departure.z > f32(gridDims.z) - 0.5)) {
            thermo = vec4f(0);
        }

        if (parameters.addThermal > 0.0) {
            let world = domainMin + position * cellSize;
            let radius = max(min(700.0, 0.08 * domainSize.x), cellSize.x);
            let height = max(min(600.0, 0.1 * domainSize.y), 2.0 * cellSize.y);
            let distance = vec3f(world.x / radius, (world.y - height) / height, world.z / radius);
            let weight = exp(-dot(distance, distance));
            thermo += weight * vec4f(2.0, 0, 0.2, 0);
        }

        thermo.y = max(thermo.y, -background(position.y).state.y);
        thermo = vec4f(thermo.xy, max(thermo.zw, vec2f(0)));
    }

    textureStore(thermoOut, cell, thermo);
}
