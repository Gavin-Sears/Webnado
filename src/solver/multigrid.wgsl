struct GridParameters {
    dims: vec3i,
    periodic: i32,
    fineDims: vec3i,
    unused: i32,
}

@group(0) @binding(0) var<uniform> grid: GridParameters;
@group(0) @binding(1) var edges: texture_3d<f32>;
@group(0) @binding(2) var diagonal: texture_3d<f32>;
@group(0) @binding(3) var pressure: texture_3d<f32>;
@group(0) @binding(4) var rhs: texture_3d<f32>;
@group(0) @binding(5) var pressureOut: texture_storage_3d<r32float, write>;
@group(0) @binding(6) var scalarOut: texture_storage_3d<r32float, write>;
@group(0) @binding(7) var edgesOut: texture_storage_3d<rgba32float, write>;
@group(0) @binding(8) var diagonalOut: texture_storage_3d<r32float, write>;
@group(0) @binding(9) var fineEdges: texture_3d<f32>;
@group(0) @binding(10) var fineDiagonal: texture_3d<f32>;
@group(0) @binding(11) var fineResidual: texture_3d<f32>;
@group(0) @binding(12) var coarsePressure: texture_3d<f32>;
@group(0) @binding(13) var openBits: texture_3d<u32>;
@group(0) @binding(14) var divergence: texture_3d<f32>;
@group(0) @binding(15) var coarseEdges: texture_3d<f32>;
@group(0) @binding(16) var<storage, read_write> pressureMoments: array<vec4f>;
@group(0) @binding(17) var<storage, read_write> pressureMeans: array<vec2f>;
@group(0) @binding(18) var<storage, read> atmosphere: array<AtmosphereSample>;

fn faceDensity(cell: vec3i, axis: i32, positive: bool) -> f32 {
    if (${anelasticAir} == 0 || grid.periodic != 0) {
        return 1.0;
    }
    let height = f32(cell.y) + select(0.5, select(0.0, 1.0, positive), axis == 1);
    let slice = clamp(height / f32(grid.dims.y) * f32(${atmosphereSlices}) - 0.5,
        0.0, f32(${atmosphereSlices} - 1));
    let lower = u32(floor(slice));
    let upper = min(lower + 1u, ${atmosphereSlices}u - 1u);
    return mix(atmosphere[lower].air.z, atmosphere[upper].air.z, fract(slice));
}

var<workgroup> momentSums: array<vec4f, 256>;

fn reduceMoments(local: u32) {
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (local < stride) {
            momentSums[local] += momentSums[local + stride];
        }
        workgroupBarrier();
    }
}

@compute @workgroup_size(256)
fn reduceFineState(@builtin(local_invocation_index) local: u32,
    @builtin(workgroup_id) group: vec3u) {
    let groupIndex = group.x + 256u * group.y;
    let index = i32(256u * groupIndex + local);
    let cell = vec3i(index % grid.dims.x, (index / grid.dims.x) % grid.dims.y,
        index / (grid.dims.x * grid.dims.y));
    var value = vec4f(0);
    if (insideGrid(cell, grid.dims) && (textureLoad(openBits, cell, 0).x & 64u) != 0u) {
        value = vec4f(textureLoad(divergence, cell, 0).x, textureLoad(pressure, cell, 0).x, 1, 0);
    }
    momentSums[local] = value;
    reduceMoments(local);
    if (local == 0u) {
        pressureMoments[groupIndex] = momentSums[0];
    }
}

@compute @workgroup_size(256)
fn finishReduction(@builtin(local_invocation_index) local: u32) {
    let groups = u32((grid.dims.x * grid.dims.y * grid.dims.z + 255) / 256);
    var value = vec4f(0);
    for (var index = local; index < groups; index += 256u) {
        value += pressureMoments[index];
    }
    momentSums[local] = value;
    reduceMoments(local);
    if (local == 0u) {
        pressureMeans[0] = momentSums[0].xy / max(momentSums[0].z, 1.0);
    }
}

fn insideGrid(cell: vec3i, dims: vec3i) -> bool {
    return all(cell >= vec3i(0)) && all(cell < dims);
}

fn offsetFor(axis: i32) -> vec3i {
    var offset = vec3i(0);
    offset[axis] = 1;
    return offset;
}

fn neighborOf(cell: vec3i, axis: i32, positive: bool, dims: vec3i) -> vec3i {
    var neighbor = cell + select(-1, 1, positive) * offsetFor(axis);
    if (grid.periodic != 0 && axis != 1) {
        neighbor[axis] = (neighbor[axis] + dims[axis]) % dims[axis];
    }
    return neighbor;
}

fn baseWeight(cell: vec3i, axis: i32, positive: bool) -> f32 {
    let bits = textureLoad(openBits, cell, 0).x;
    let bit = u32(axis * 2) + select(0u, 1u, positive);
    let neighbor = neighborOf(cell, axis, positive, grid.dims);
    if ((bits & 64u) == 0u || (bits & (1u << bit)) == 0u || !insideGrid(neighbor, grid.dims)) {
        return 0.0;
    }
    let peer = textureLoad(openBits, neighbor, 0).x;
    let opposite = u32(axis * 2) + select(1u, 0u, positive);
    if ((peer & 64u) == 0u || (peer & (1u << opposite)) == 0u) {
        return 0.0;
    }
    return faceDensity(cell, axis, positive) / (cellSize[axis] * cellSize[axis]);
}

@compute @workgroup_size(8, 8, 4)
fn buildFineGraph(@builtin(global_invocation_id) id: vec3u) {
    let cell = vec3i(id);
    if (!insideGrid(cell, grid.dims)) {
        return;
    }
    var weights = vec3f(0);
    var sum = 0.0;
    for (var axis = 0; axis < 3; axis++) {
        weights[axis] = baseWeight(cell, axis, true);
        sum += weights[axis] + baseWeight(cell, axis, false);
    }
    let volume = select(0.0, 1.0, (textureLoad(openBits, cell, 0).x & 64u) != 0u);
    textureStore(edgesOut, cell, vec4f(weights, volume));
    textureStore(diagonalOut, cell, vec4f(sum, 0, 0, 0));
}

fn graphWeight(field: texture_3d<f32>, cell: vec3i, axis: i32, positive: bool, dims: vec3i) -> f32 {
    let neighbor = neighborOf(cell, axis, positive, dims);
    if (!insideGrid(neighbor, dims)) {
        return 0.0;
    }
    let current = textureLoad(field, cell, 0);
    let peer = textureLoad(field, neighbor, 0);
    if (current.w == 0.0 || peer.w == 0.0) {
        return 0.0;
    }
    if (positive) {
        return current[axis];
    }
    return peer[axis] * peer.w / current.w;
}

fn childOffset(index: u32) -> vec3i {
    return vec3i(i32(index & 1u), i32((index >> 1u) & 1u), i32(index >> 2u));
}

@compute @workgroup_size(8, 8, 4)
fn coarsenGraph(@builtin(global_invocation_id) id: vec3u) {
    let cell = vec3i(id);
    if (!insideGrid(cell, grid.dims)) {
        return;
    }
    let first = 2 * cell;
    var fluidChildren = 0u;
    var volume = 0.0;
    for (var index = 0u; index < 8u; index++) {
        let amount = textureLoad(fineEdges, first + childOffset(index), 0).w;
        if (amount > 0.0) {
            fluidChildren |= 1u << index;
            volume += amount;
        }
    }

    // An aggregate must contain one connected fluid region, so thin walls survive coarsening.
    var visited = 0u;
    if (fluidChildren != 0u) {
        visited = 1u << firstTrailingBit(fluidChildren);
    }
    for (var iteration = 0; iteration < 8; iteration++) {
        for (var index = 0u; index < 8u; index++) {
            if ((visited & (1u << index)) != 0u) {
                let child = first + childOffset(index);
                for (var axis = 0; axis < 3; axis++) {
                    let neighbor = index ^ (1u << u32(axis));
                    let positive = (index & (1u << u32(axis))) == 0u;
                    if ((fluidChildren & (1u << neighbor)) != 0u
                        && graphWeight(fineEdges, child, axis, positive, grid.fineDims) > 0.0) {
                        visited |= 1u << neighbor;
                    }
                }
            }
        }
    }

    var weights = vec3f(0);
    var sum = 0.0;
    if (fluidChildren != 0u && visited == fluidChildren) {
        for (var index = 0u; index < 8u; index++) {
            let offset = childOffset(index);
            let child = first + offset;
            let amount = textureLoad(fineEdges, child, 0).w;
            sum += amount * textureLoad(fineDiagonal, child, 0).x;
            for (var axis = 0; axis < 3; axis++) {
                let internalPositive = offset[axis] == 0;
                let internal = graphWeight(fineEdges, child, axis, internalPositive, grid.fineDims);
                sum -= amount * internal;
                if (offset[axis] == 1) {
                    weights[axis] += amount * graphWeight(fineEdges, child, axis, true, grid.fineDims);
                }
            }
        }
        // Average fine face conductance and account for twice the distance between coarse centers.
        weights /= 2.0 * volume;
        sum = max(0.0, sum / (2.0 * volume));
    } else {
        volume = 0.0;
    }
    textureStore(edgesOut, cell, vec4f(weights, volume));
    textureStore(diagonalOut, cell, vec4f(sum, 0, 0, 0));
}

@compute @workgroup_size(8, 8, 4)
fn initializeRhs(@builtin(global_invocation_id) id: vec3u) {
    let cell = vec3i(id);
    if (insideGrid(cell, grid.dims)) {
        let fluidCell = textureLoad(edges, cell, 0).w > 0.0;
        let value = select(0.0, pressureMeans[0].x - textureLoad(divergence, cell, 0).x, fluidCell);
        textureStore(scalarOut, cell, vec4f(value, 0, 0, 0));
        let centered = select(0.0, textureLoad(pressure, cell, 0).x - pressureMeans[0].y, fluidCell);
        textureStore(pressureOut, cell, vec4f(centered, 0, 0, 0));
    }
}

fn neighborSum(cell: vec3i) -> f32 {
    var sum = 0.0;
    for (var direction = 0; direction < 6; direction++) {
        let axis = direction / 2;
        let positive = direction % 2 == 1;
        let weight = graphWeight(edges, cell, axis, positive, grid.dims);
        if (weight > 0.0) {
            sum += weight * textureLoad(pressure, neighborOf(cell, axis, positive, grid.dims), 0).x;
        }
    }
    return sum;
}

@compute @workgroup_size(8, 8, 4)
fn relaxPressure(@builtin(global_invocation_id) id: vec3u) {
    let cell = vec3i(id);
    if (insideGrid(cell, grid.dims)) {
        let d = textureLoad(diagonal, cell, 0).x;
        var value = 0.0;
        if (d > 0.0) {
            let candidate = (neighborSum(cell) + textureLoad(rhs, cell, 0).x) / d;
            value = mix(textureLoad(pressure, cell, 0).x, candidate, 2.0 / 3.0);
        }
        textureStore(pressureOut, cell, vec4f(value, 0, 0, 0));
    }
}

@compute @workgroup_size(8, 8, 4)
fn residual(@builtin(global_invocation_id) id: vec3u) {
    let cell = vec3i(id);
    if (insideGrid(cell, grid.dims)) {
        let applied = textureLoad(diagonal, cell, 0).x * textureLoad(pressure, cell, 0).x - neighborSum(cell);
        let value = textureLoad(rhs, cell, 0).x - applied;
        textureStore(scalarOut, cell, vec4f(value, 0, 0, 0));
    }
}

@compute @workgroup_size(8, 8, 4)
fn restrictResidual(@builtin(global_invocation_id) id: vec3u) {
    let cell = vec3i(id);
    if (!insideGrid(cell, grid.dims)) {
        return;
    }
    let volume = textureLoad(edges, cell, 0).w;
    var sum = 0.0;
    if (volume > 0.0) {
        for (var index = 0u; index < 8u; index++) {
            let child = 2 * cell + childOffset(index);
            sum += textureLoad(fineEdges, child, 0).w * textureLoad(fineResidual, child, 0).x;
        }
        sum /= volume;
    }
    textureStore(scalarOut, cell, vec4f(sum, 0, 0, 0));
    textureStore(pressureOut, cell, vec4f(0));
}

@compute @workgroup_size(8, 8, 4)
fn prolongCorrection(@builtin(global_invocation_id) id: vec3u) {
    let cell = vec3i(id);
    if (!insideGrid(cell, grid.dims)) {
        return;
    }
    var value = 0.0;
    if (textureLoad(edges, cell, 0).w > 0.0) {
        value = textureLoad(pressure, cell, 0).x;
        let parent = cell / 2;
        if (textureLoad(coarseEdges, parent, 0).w > 0.0) {
            let dims = vec3i(textureDimensions(coarseEdges));
            var correction = 0.0;
            // Cell-centered interpolation follows open faces, including periodic seams.
            for (var corner = 0u; corner < 8u; corner++) {
                var sampleCell = parent;
                var weight = 1.0;
                for (var axis = 0; axis < 3; axis++) {
                    if ((corner & (1u << u32(axis))) == 0u) {
                        weight *= 0.75;
                    } else {
                        weight *= 0.25;
                        let positive = cell[axis] % 2 == 1;
                        if (graphWeight(coarseEdges, sampleCell, axis, positive, dims) > 0.0) {
                            sampleCell = neighborOf(sampleCell, axis, positive, dims);
                        }
                    }
                }
                correction += weight * textureLoad(coarsePressure, sampleCell, 0).x;
            }
            value += correction;
        }
    }
    textureStore(pressureOut, cell, vec4f(value, 0, 0, 0));
}

var<workgroup> coarseA: array<f32, 256>;
var<workgroup> coarseB: array<f32, 256>;

@compute @workgroup_size(256)
fn solveCoarsest(@builtin(local_invocation_index) index: u32) {
    let cell = vec3i(i32(index) % grid.dims.x, (i32(index) / grid.dims.x) % grid.dims.y,
        i32(index) / (grid.dims.x * grid.dims.y));
    let valid = insideGrid(cell, grid.dims);
    var d = 0.0;
    var b = 0.0;
    var weights: array<f32, 6>;
    var indices: array<u32, 6>;
    if (valid) {
        d = textureLoad(diagonal, cell, 0).x;
        b = textureLoad(rhs, cell, 0).x;
        for (var direction = 0; direction < 6; direction++) {
            let axis = direction / 2;
            let positive = direction % 2 == 1;
            weights[direction] = graphWeight(edges, cell, axis, positive, grid.dims);
            if (weights[direction] > 0.0) {
                let neighbor = neighborOf(cell, axis, positive, grid.dims);
                indices[direction] = u32(neighbor.x + grid.dims.x * (neighbor.y + grid.dims.y * neighbor.z));
            }
        }
    }
    coarseA[index] = 0.0;
    coarseB[index] = 0.0;
    workgroupBarrier();
    for (var iteration = 0; iteration < 64; iteration++) {
        var sum = b;
        for (var direction = 0; direction < 6; direction++) {
            let neighbor = indices[direction];
            if (iteration % 2 == 0) {
                sum += weights[direction] * coarseA[neighbor];
            } else {
                sum += weights[direction] * coarseB[neighbor];
            }
        }
        var value = 0.0;
        if (d > 0.0) {
            var previous = coarseB[index];
            if (iteration % 2 == 0) {
                previous = coarseA[index];
            }
            value = mix(previous, sum / d, 2.0 / 3.0);
        }
        if (iteration % 2 == 0) {
            coarseB[index] = value;
        } else {
            coarseA[index] = value;
        }
        workgroupBarrier();
    }
    if (valid) {
        textureStore(pressureOut, cell, vec4f(coarseA[index], 0, 0, 0));
    }
}

