struct DebrisParameters {
    dt: f32,
    restitution: f32,
    friction: f32,
    size: f32,
    spawn: u32,
    enabled: u32,
    viewport: vec2f,
    seedRegion: vec4f,
}

struct Particle {
    position: vec4f,
    velocity: vec4f,
}

@group(0) @binding(0) var<uniform> params: DebrisParameters;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(2) var dustVelocity: texture_3d<f32>;
@group(0) @binding(3) var obstacle: texture_3d<u32>;
@group(0) @binding(6) var<uniform> camera: CameraUniforms;
@group(0) @binding(7) var sceneDepth: texture_depth_2d;
@group(0) @binding(9) var<storage, read> visibleParticles: array<Particle>;

const particleCount = 4096u;

fn random(seed: u32) -> f32 {
    var value = seed * 747796405u + 2891336453u;
    value = ((value >> ((value >> 28u) + 4u)) ^ value) * 277803737u;
    return f32((value >> 22u) ^ value) / 4294967295.0;
}

fn worldCell(position: vec3f) -> vec3i {
    return vec3i(floor((position - domainMin) / cellSize));
}

fn inDomain(cell: vec3i) -> bool {
    return all(cell >= vec3i(0)) && all(cell < gridDims);
}

fn wrapPosition(position: vec3f) -> vec3f {
    let offset = position - domainMin;
    return vec3f(domainMin.x + offset.x - floor(offset.x / domainSize.x) * domainSize.x,
        position.y, domainMin.z + offset.z - floor(offset.z / domainSize.z) * domainSize.z);
}

fn solid(position: vec3f) -> bool {
    let cell = worldCell(position);
    return inDomain(cell) && textureLoad(obstacle, cell, 0).x != 0u;
}

fn dustVelocityAt(position: vec3f) -> vec3f {
    let grid = (position - domainMin) / cellSize;
    var velocity = vec3f(0);
    for (var axis = 0; axis < 3; axis++) {
        var offset = vec3f(0.5);
        offset[axis] = 0.0;
        var coordinate = grid - offset;
        coordinate.y = clamp(coordinate.y, 0.0, f32(gridDims.y - 1));
        let base = vec3i(floor(coordinate));
        let fraction = fract(coordinate);
        for (var z = 0; z < 2; z++) {
            for (var y = 0; y < 2; y++) {
                for (var x = 0; x < 2; x++) {
                    let corner = vec3i(x, y, z);
                    let weight = mix(vec3f(1) - fraction, fraction, vec3f(corner));
                    let sample = base + corner;
                    let cell = vec3i((sample.x % gridDims.x + gridDims.x) % gridDims.x,
                        min(sample.y, gridDims.y - 1), (sample.z % gridDims.z + gridDims.z) % gridDims.z);
                    let value = textureLoad(dustVelocity, cell, 0)[axis];
                    velocity[axis] += value * weight.x * weight.y * weight.z;
                }
            }
        }
    }
    return velocity;
}

fn surfaceHeight(x: f32, z: f32) -> f32 {
    let column = clamp(worldCell(vec3f(x, 0, z)), vec3i(0), gridDims - 1);
    var height = select(cellSize.y, 0.0, ${freeSlipGround} != 0);
    for (var y = 0; y < gridDims.y; y++) {
        if (textureLoad(obstacle, vec3i(column.x, y, column.z), 0).x != 0u) {
            height = f32(y + 1) * cellSize.y;
        }
    }
    return height;
}

@compute @workgroup_size(64)
fn updateDebris(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= particleCount) {
        return;
    }
    var particle = particles[id.x];
    if (params.spawn != 0u) {
        var x = (random(id.x * 3u) - 0.5) * domainSize.x * 0.65;
        var z = (random(id.x * 3u + 1u) - 0.5) * domainSize.z * 0.65;
        if (params.seedRegion.w > 0.0) {
            let radius = params.seedRegion.z * sqrt(random(id.x * 3u));
            let angle = 6.2831853 * random(id.x * 3u + 1u);
            x = params.seedRegion.x + radius * cos(angle);
            z = params.seedRegion.y + radius * sin(angle);
        }
        let y = surfaceHeight(x, z) + cellSize.y * (0.1 + random(id.x * 3u + 2u));
        particle.position = vec4f(x, y, z, 1);
        particle.velocity = vec4f(0);
    }
    if (params.enabled == 0u || particle.position.w == 0.0) {
        particles[id.x] = particle;
        return;
    }

    var position = particle.position.xyz;
    if (solid(position)) {
        position.y = surfaceHeight(position.x, position.z) + 0.1 * cellSize.y;
    }
    var velocity = dustVelocityAt(position);
    let displacement = velocity * params.dt;
    let segments = max(1, i32(ceil(max(max(abs(displacement.x), abs(displacement.y)), abs(displacement.z))
        / (0.4 * cellSize.x))));

    for (var segment = 0; segment < segments; segment++) {
        for (var axis = 0; axis < 3; axis++) {
            var next = position;
            next[axis] += velocity[axis] * params.dt / f32(segments);
            next = wrapPosition(next);
            if (solid(next)) {
                velocity[axis] *= -params.restitution;
                for (var tangent = 0; tangent < 3; tangent++) {
                    if (tangent != axis) {
                        velocity[tangent] *= 1.0 - params.friction;
                    }
                }
            } else {
                position = next;
            }
        }
    }

    particle.position = vec4f(position, select(0.0, 1.0, inDomain(worldCell(position))));
    particle.velocity = vec4f(velocity, 0);
    particles[id.x] = particle;
}

struct DebrisVertex {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn debrisVertex(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> DebrisVertex {
    let corners = array<vec2f, 6>(vec2f(-1, -1), vec2f(1, -1), vec2f(-1, 1),
        vec2f(-1, 1), vec2f(1, -1), vec2f(1, 1));
    let particle = visibleParticles[instance];
    var clip = camera.viewProjMat * vec4f(particle.position.xyz, 1);
    clip = vec4f(clip.xy + corners[vertex] * params.size * 2.0 / params.viewport * clip.w, clip.zw);
    if (particle.position.w == 0.0 || params.enabled == 0u) {
        clip = vec4f(2, 2, 2, 1);
    }
    return DebrisVertex(clip, corners[vertex]);
}

@fragment
fn debrisFragment(input: DebrisVertex) -> @location(0) vec4f {
    let depth = textureLoad(sceneDepth, vec2i(input.position.xy), 0);
    if (dot(input.uv, input.uv) > 1.0 || input.position.z > depth + 1e-5) {
        discard;
    }
    return vec4f(0.65, 0.42, 0.16, 1);
}
