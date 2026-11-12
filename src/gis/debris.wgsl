struct DebrisParameters {
    dt: f32,
    response: f32,
    restitution: f32,
    friction: f32,
    mass: f32,
    size: f32,
    spawn: u32,
    enabled: u32,
    viewport: vec2f,
    unused: vec2f,
}

struct Particle {
    position: vec4f,
    velocity: vec4f,
}

struct Momentum {
    x: atomic<i32>,
    y: atomic<i32>,
    z: atomic<i32>,
    unused: atomic<i32>,
}

@group(0) @binding(0) var<uniform> params: DebrisParameters;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(2) var wind: texture_3d<f32>;
@group(0) @binding(3) var obstacle: texture_3d<u32>;
@group(0) @binding(4) var<storage, read_write> momentum: array<Momentum>;
@group(0) @binding(5) var feedback: texture_storage_3d<rgba16float, write>;
@group(0) @binding(6) var<uniform> camera: CameraUniforms;
@group(0) @binding(7) var sceneDepth: texture_depth_2d;
@group(0) @binding(8) var<storage, read> atmosphere: array<AtmosphereSample>;
@group(0) @binding(9) var<storage, read> visibleParticles: array<Particle>;

const particleCount = 4096u;
const momentumScale = 256.0;

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

fn solid(position: vec3f) -> bool {
    let cell = worldCell(position);
    return inDomain(cell) && textureLoad(obstacle, cell, 0).x != 0u;
}

fn windAt(position: vec3f) -> vec3f {
    let grid = (position - domainMin) / cellSize;
    var velocity = vec3f(0);
    for (var axis = 0; axis < 3; axis++) {
        var offset = vec3f(0.5);
        offset[axis] = 0.0;
        let coordinate = clamp(grid - offset, vec3f(0), vec3f(gridDims - 1));
        let base = vec3i(floor(coordinate));
        let fraction = fract(coordinate);
        for (var z = 0; z < 2; z++) {
            for (var y = 0; y < 2; y++) {
                for (var x = 0; x < 2; x++) {
                    let corner = vec3i(x, y, z);
                    let weight = mix(vec3f(1) - fraction, fraction, vec3f(corner));
                    let value = textureLoad(wind, min(base + corner, gridDims - 1), 0)[axis];
                    velocity[axis] += value * weight.x * weight.y * weight.z;
                }
            }
        }
    }
    return velocity;
}

fn surfaceHeight(x: f32, z: f32) -> f32 {
    let column = clamp(worldCell(vec3f(x, 0, z)), vec3i(0), gridDims - 1);
    var height = cellSize.y;
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
        let x = (random(id.x * 3u) - 0.5) * domainSize.x * 0.65;
        let z = (random(id.x * 3u + 1u) - 0.5) * domainSize.z * 0.65;
        let y = surfaceHeight(x, z) + cellSize.y * (0.1 + random(id.x * 3u + 2u));
        particle.position = vec4f(x, y, z, 1);
        particle.velocity = vec4f(0, 0, 0, params.mass);
    }
    if (params.enabled == 0u || particle.position.w == 0.0) {
        particles[id.x] = particle;
        return;
    }

    var position = particle.position.xyz;
    if (solid(position)) {
        position.y = surfaceHeight(position.x, position.z) + 0.1 * cellSize.y;
    }
    let air = windAt(position);
    let previousVelocity = particle.velocity.xyz;
    let gravity = vec3f(0, -9.81, 0);
    let decay = exp(-params.dt / params.response);
    var velocity = air + (previousVelocity - air) * decay
        + gravity * params.response * (1.0 - decay);
    let impulse = particle.velocity.w * (velocity - previousVelocity - gravity * params.dt);
    let origin = worldCell(position);
    let displacement = velocity * params.dt;
    let segments = max(1, i32(ceil(max(max(abs(displacement.x), abs(displacement.y)), abs(displacement.z))
        / (0.4 * cellSize.x))));

    for (var segment = 0; segment < segments; segment++) {
        for (var axis = 0; axis < 3; axis++) {
            var next = position;
            next[axis] += velocity[axis] * params.dt / f32(segments);
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

    if (inDomain(origin)) {
        let index = u32(origin.x + gridDims.x * (origin.y + gridDims.y * origin.z));
        // This bound keeps the fixed-point sum within i32 even if every particle shares a cell.
        let bound = 2147483000.0 / f32(particleCount);
        let reaction = vec3i(clamp(-impulse * momentumScale, vec3f(-bound), vec3f(bound)));
        atomicAdd(&momentum[index].x, reaction.x);
        atomicAdd(&momentum[index].y, reaction.y);
        atomicAdd(&momentum[index].z, reaction.z);
    }
    particle.position = vec4f(position, select(0.0, 1.0, inDomain(worldCell(position))));
    particle.velocity = vec4f(velocity, particle.velocity.w);
    particles[id.x] = particle;
}

@compute @workgroup_size(8, 8, 4)
fn resolveFeedback(@builtin(global_invocation_id) id: vec3u) {
    let cell = vec3i(id);
    if (!inDomain(cell)) {
        return;
    }
    let index = u32(cell.x + gridDims.x * (cell.y + gridDims.y * cell.z));
    let slice = min(u32((f32(cell.y) + 0.5) / f32(gridDims.y) * f32(${atmosphereSlices})), ${atmosphereSlices}u - 1u);
    let mass = atmosphere[slice].air.z * cellSize.x * cellSize.y * cellSize.z;
    let impulse = vec3f(f32(atomicLoad(&momentum[index].x)),
        f32(atomicLoad(&momentum[index].y)), f32(atomicLoad(&momentum[index].z))) / momentumScale;
    let acceleration = impulse / max(mass * params.dt, 1e-6);
    textureStore(feedback, cell, vec4f(acceleration, 0));
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
