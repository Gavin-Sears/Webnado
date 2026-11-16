export interface SpatialWeatherManifest {
    version: number;
    state: string;
    bytes: number;
    sha256: string;
    center: { latitude: number; longitude: number };
    groundMsl: number;
    shape: [number, number, number];
    frames: number;
    intervalSeconds: number;
    forecastHours: number[];
    sources: { source: string; initializationUtc: string }[];
    limitations: string[];
}

export interface SpatialSample {
    theta: number;
    vapor: number;
    east: number;
    south: number;
    vertical: number;
    cloud: number;
    rain: number;
    pressure: number;
    temperature: number;
    density: number;
    ice: number;
    terrain: number;
    energy: number;
}

export class SpatialWeather {
    readonly width: number;
    readonly depth: number;
    readonly levels: number;

    constructor(readonly manifest: SpatialWeatherManifest, readonly values: Float32Array) {
        this.width = values[0];
        this.depth = values[1];
        this.levels = values[2];
        const surfaceValues = manifest.version === 2 ? this.width * this.depth * manifest.frames * 8 : 0;
        const expected = 24 + this.width * this.depth * this.levels * manifest.frames * 16 + surfaceValues;
        if (![1, 2].includes(manifest.version) || values.length !== expected
            || manifest.bytes !== values.byteLength || values[3] !== manifest.frames
            || manifest.frames < 2 || manifest.frames > 4
            || manifest.shape.join() !== [this.width, this.levels, this.depth].join()) {
            throw new Error('Spatial weather dimensions do not match the manifest.');
        }
    }

    static async load(manifestUrl: string): Promise<SpatialWeather> {
        const url = new URL(manifestUrl, location.href);
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Spatial weather manifest returned HTTP ${response.status}.`);
        }
        const manifest = await response.json() as SpatialWeatherManifest;
        const stateResponse = await fetch(new URL(manifest.state, url));
        if (!stateResponse.ok) {
            throw new Error(`Spatial weather state returned HTTP ${stateResponse.status}.`);
        }
        const buffer = await stateResponse.arrayBuffer();
        const digest = await crypto.subtle.digest('SHA-256', buffer);
        const checksum = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
        if (checksum !== manifest.sha256) {
            throw new Error('Spatial weather checksum does not match its source manifest.');
        }
        return new SpatialWeather(manifest, new Float32Array(buffer));
    }

    gridCoordinate(east: number, south: number): [number, number] {
        const a = this.values;
        const latitude = a[12];
        const longitude = a[13];
        const radius = a[11];
        const north = -south;
        const radial = radius * Math.cos(latitude) - north * Math.sin(latitude);
        const x = radial * Math.cos(longitude) - east * Math.sin(longitude);
        const y = radial * Math.sin(longitude) + east * Math.cos(longitude);
        const z = radius * Math.sin(latitude) + north * Math.cos(latitude);
        const lat = Math.atan2(z, Math.hypot(x, y));
        const lon = Math.atan2(y, x);
        const rho = radius * a[9] / Math.pow(Math.tan(Math.PI / 4 + lat / 2), a[8]);
        const angle = a[8] * (lon - a[15]);
        return [(rho * Math.sin(angle) - a[4]) / a[6],
                (a[10] - rho * Math.cos(angle) - a[5]) / a[7]];
    }

    sample(east: number, height: number, south: number, frame = 0): SpatialSample {
        const [x, z] = this.gridCoordinate(east, south);
        if (x < 0 || z < 0 || x > this.width - 1 || z > this.depth - 1) {
            throw new Error('The simulation extends outside the extracted weather region.');
        }
        const baseX = Math.min(Math.floor(x), this.width - 2);
        const baseZ = Math.min(Math.floor(z), this.depth - 2);
        const result = new Float64Array(16);
        for (let dz = 0; dz <= 1; dz++) {
            for (let dx = 0; dx <= 1; dx++) {
                const weight = (dx ? x - baseX : 1 - x + baseX)
                    * (dz ? z - baseZ : 1 - z + baseZ);
                const column = this.column(baseX + dx, baseZ + dz, height + this.values[14], frame);
                for (let channel = 0; channel < 16; channel++) {
                    result[channel] += weight * column[channel];
                }
            }
        }
        return {
            theta: result[1], vapor: result[2], east: result[3], south: result[4],
            vertical: result[5], cloud: result[6], rain: result[7], pressure: result[8],
            temperature: result[9], density: result[10], ice: result[11], terrain: result[12], energy: result[13]
        };
    }

    surface(east: number, south: number, seconds: number): number[] {
        if (this.manifest.version !== 2) return [0, 0, 0, 0];
        const [x, z] = this.gridCoordinate(east, south);
        const baseX = Math.min(Math.floor(x), this.width - 2);
        const baseZ = Math.min(Math.floor(z), this.depth - 2);
        const time = Math.min(seconds / this.manifest.intervalSeconds, this.manifest.frames - 1);
        const start = 24 + this.width * this.depth * this.levels * this.manifest.frames * 16;
        const result = [0, 0, 0, 0];
        for (let t = 0; t <= 1; t++) {
            const frame = Math.min(Math.floor(time) + t, this.manifest.frames - 1);
            const timeWeight = t ? time % 1 : 1 - time % 1;
            for (let dz = 0; dz <= 1; dz++) {
                for (let dx = 0; dx <= 1; dx++) {
                    const weight = timeWeight * (dx ? x - baseX : 1 - x + baseX)
                        * (dz ? z - baseZ : 1 - z + baseZ);
                    const index = start + 8 * ((frame * this.depth + baseZ + dz) * this.width + baseX + dx);
                    for (let channel = 0; channel < 4; channel++) {
                        result[channel] += weight * this.values[index + channel];
                    }
                }
            }
        }
        return result;
    }

    referenceColumn(slices: number, height: number, width: number, depth: number): Float32Array {
        const result = new Float32Array(slices * 8);
        for (let slice = 0; slice < slices; slice++) {
            for (let z = 0; z < 5; z++) {
                for (let x = 0; x < 5; x++) {
                    const sample = this.sample((x / 4 - 0.5) * width,
                        (slice + 0.5) / slices * height, (z / 4 - 0.5) * depth);
                    const values = [sample.theta, sample.vapor, sample.east, sample.south,
                        sample.pressure, sample.temperature, sample.density, 0];
                    values.forEach((value, channel) => result[slice * 8 + channel] += value / 25);
                }
            }
        }
        return result;
    }

    balanceBoundaryFlux(column: Float32Array, grid: [number, number, number],
        size: [number, number, number], stratified: boolean): number[] {
        const [nx, ny, nz] = grid;
        const [width, height, depth] = size;
        const slices = column.length / 8;
        const faces: { east: number; south: number; height: number; normal: number; axis: 'east' | 'south'; weight: number }[] = [];
        for (let y = 0; y < ny; y++) {
            const level = (y + 0.5) / ny * height;
            const slice = Math.max(0, Math.min(slices - 1, (y + 0.5) / ny * slices - 0.5));
            const lower = Math.floor(slice);
            const upper = Math.min(lower + 1, slices - 1);
            const density = stratified ? column[lower * 8 + 6] * (1 - slice % 1)
                + column[upper * 8 + 6] * (slice % 1) : 1;
            for (const normal of [-1, 1]) {
                for (let z = 0; z < nz; z++) {
                    const east = normal * width / 2;
                    const south = (z + 0.5) / nz * depth - depth / 2;
                    const terrain = this.sample(east - normal * width / nx / 2, level, south).terrain;
                    if (level + this.manifest.groundMsl >= terrain) {
                        faces.push({ east, south, height: level, normal, axis: 'east',
                            weight: density * depth / nz * height / ny });
                    }
                }
                for (let x = 0; x < nx; x++) {
                    const east = (x + 0.5) / nx * width - width / 2;
                    const south = normal * depth / 2;
                    const terrain = this.sample(east, level, south - normal * depth / nz / 2).terrain;
                    if (level + this.manifest.groundMsl >= terrain) {
                        faces.push({ east, south, height: level, normal, axis: 'south',
                            weight: density * width / nx * height / ny });
                    }
                }
            }
        }
        const area = faces.reduce((total, face) => total + face.weight, 0);
        const corrections: number[] = [];
        for (let frame = 0; frame < this.manifest.frames; frame++) {
            const flux = faces.reduce((total, face) => total + face.normal * face.weight
                * this.sample(face.east, face.height, face.south, frame)[face.axis], 0);
            const correction = flux / area;
            corrections.push(correction);
            this.values[20 + frame] = correction;
        }
        return corrections;
    }

    private column(x: number, z: number, height: number, frame: number): Float64Array {
        const index = (level: number) => 24 + 16 * (
            ((frame * this.levels + level) * this.depth + z) * this.width + x);
        let lower = 0;
        let upper = this.levels - 1;
        while (upper - lower > 1) {
            const middle = Math.floor((lower + upper) / 2);
            if (this.values[index(middle)] < height) {
                lower = middle;
            } else {
                upper = middle;
            }
        }
        const a = index(lower);
        const b = index(upper);
        const fraction = Math.max(0, Math.min(1, (height - this.values[a]) / (this.values[b] - this.values[a])));
        return Float64Array.from({ length: 16 }, (_, channel) =>
            this.values[a + channel] * (1 - fraction) + this.values[b + channel] * fraction);
    }
}
