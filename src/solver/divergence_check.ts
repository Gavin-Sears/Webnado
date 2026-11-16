import { constants } from '../shaders/shaders';
import { Fields } from '../stage/fields';

export interface DivergenceReport {
    rms: number;
    maxAbsolute: number;
    maxBlockedFaceSpeed: number;
    nonFinite: number;
    fluidCells: number;
    massWeighted: boolean;
}

export class DivergenceCheck {
    static get bufferSize(): number {
        const texelBytes = constants.velocityFormat === 'rgba32float' ? 16 : 8;
        const velocityRow = Math.ceil(constants.gridX * texelBytes / 256) * 256;
        const maskRow = Math.ceil(constants.gridX * 4 / 256) * 256;
        const layers = constants.gridY * constants.gridZ;

        return (velocityRow + maskRow) * layers + constants.atmosphereSlices * 32;
    }

    private get velocityStride() {
        return !this.periodic && constants.velocityFormat === 'rgba32float' ? 16 : 8;
    }
    private get velocityRow() {
        return Math.ceil(constants.gridX * this.velocityStride / 256) * 256;
    }
    private readonly maskRow = Math.ceil(constants.gridX * 4 / 256) * 256;
    private get velocityBytes() {
        return this.velocityRow * constants.gridY * constants.gridZ;
    }
    private readonly maskBytes = this.maskRow * constants.gridY * constants.gridZ;
    private get atmosphereOffset() {
        return this.velocityBytes + this.maskBytes;
    }

    private pending: Promise<DivergenceReport> | undefined;

    constructor(private readonly readback: GPUBuffer, private readonly periodic = false) {}

    capture(
        encoder: GPUCommandEncoder,
        fields: Pick<Fields, 'velocity'>,
        openBits: GPUTexture,
        atmosphere: GPUBuffer
    ): Promise<DivergenceReport> {
        if (this.pending) {
            return this.pending;
        }

        const size = [constants.gridX, constants.gridY, constants.gridZ];

        encoder.copyTextureToBuffer(
            { texture: fields.velocity.textures[fields.velocity.index] },
            {
                buffer: this.readback,
                bytesPerRow: this.velocityRow,
                rowsPerImage: constants.gridY
            },
            size
        );

        encoder.copyTextureToBuffer(
            { texture: openBits },
            {
                buffer: this.readback,
                offset: this.velocityBytes,
                bytesPerRow: this.maskRow,
                rowsPerImage: constants.gridY
            },
            size
        );

        encoder.copyBufferToBuffer(
            atmosphere,
            0,
            this.readback,
            this.atmosphereOffset,
            constants.atmosphereSlices * 32
        );

        this.pending = new Promise<DivergenceReport>((resolve, reject) => {
            // The renderer submits the encoder before this microtask runs.
            queueMicrotask(() => {
                this.read().then(resolve, reject);
            });
        }).finally(() => {
            this.pending = undefined;
        });

        return this.pending;
    }

    private async read(): Promise<DivergenceReport> {
        await this.readback.mapAsync(GPUMapMode.READ);

        try {
            const data = new DataView(this.readback.getMappedRange());
            return this.calculate(data);
        } finally {
            this.readback.unmap();
        }
    }

    private calculate(data: DataView): DivergenceReport {
        const { gridX, gridY, gridZ, domainX, domainY, domainZ } = constants;
        const dimensions = [gridX, gridY, gridZ];
        const cellSizes = [domainX / gridX, domainY / gridY, domainZ / gridZ];
        const readVelocity = (offset: number) => this.velocityStride === 16
            ? data.getFloat32(offset, true) : halfToFloat(data.getUint16(offset, true));

        const bitsAt = (x: number, y: number, z: number) => {
            const offset = this.velocityBytes + this.maskRow * (y + gridY * z) + x * 4;
            return data.getUint32(offset, true);
        };

        const sampleAt = (height: number, channel: number) => {
            const slice = Math.max(0, Math.min(constants.atmosphereSlices - 1,
                height / gridY * constants.atmosphereSlices - 0.5));
            const lower = Math.floor(slice);
            const upper = Math.min(lower + 1, constants.atmosphereSlices - 1);
            const a = data.getFloat32(this.atmosphereOffset + lower * 32 + channel * 4, true);
            const b = data.getFloat32(this.atmosphereOffset + upper * 32 + channel * 4, true);

            return a + (b - a) * (slice - lower);
        };
        const windAt = (y: number, axis: number) => sampleAt(y + 0.5, axis === 0 ? 2 : 3);
        const massWeighted = Boolean(constants.anelasticAir) && !this.periodic;
        const densityAt = (y: number, axis: number) => massWeighted
            ? sampleAt(y + (axis === 1 ? 0 : 0.5), 6) : 1;

        const faceOpen = (position: number[], axis: number, positive: boolean) => {
            const bit = 1 << (axis * 2 + Number(positive));
            const open = (bitsAt(position[0], position[1], position[2]) & bit) !== 0;
            if (this.periodic && axis !== 1
                && position[axis] === (positive ? dimensions[axis] - 1 : 0)) {
                const opposite = [...position];
                opposite[axis] = positive ? 0 : dimensions[axis] - 1;
                const bits = bitsAt(opposite[0], opposite[1], opposite[2]);
                return open && (bits & 64) !== 0 && (bits & (1 << (axis * 2 + Number(!positive)))) !== 0;
            }
            return open;
        };

        const faceAt = (x: number, y: number, z: number, axis: number) => {
            const position = [x, y, z];

            if (position[axis] === dimensions[axis]) {
                if (axis === 1) {
                    return 0;
                }

                position[axis]--;
                const open = faceOpen(position, axis, true);
                if (this.periodic) {
                    position[axis] = 0;
                    const offset = this.velocityRow * (position[1] + gridY * position[2])
                        + position[0] * this.velocityStride + axis * (this.velocityStride / 4);
                    return open ? readVelocity(offset) : 0;
                }
                return open ? windAt(y, axis) : 0;
            }

            const offset = this.velocityRow * (y + gridY * z)
                + x * this.velocityStride + axis * (this.velocityStride / 4);
            return readVelocity(offset);
        };

        let sumSquares = 0;
        let maxAbsolute = 0;
        let maxBlockedFaceSpeed = 0;
        let nonFinite = 0;
        let fluidCells = 0;

        for (let z = 0; z < gridZ; z++) {
            for (let y = 0; y < gridY; y++) {
                for (let x = 0; x < gridX; x++) {
                    const bits = bitsAt(x, y, z);
                    let divergence = 0;

                    for (let axis = 0; axis < 3; axis++) {
                        const lower = faceAt(x, y, z, axis);
                        const upperPosition = [x, y, z];
                        upperPosition[axis]++;
                        const upper = faceAt(upperPosition[0], upperPosition[1], upperPosition[2], axis);

                        if (!Number.isFinite(lower)) {
                            nonFinite++;
                        }

                        if (!faceOpen([x, y, z], axis, false)) {
                            maxBlockedFaceSpeed = Math.max(maxBlockedFaceSpeed, Math.abs(lower));
                        }

                        const upperDensity = densityAt(y + Number(axis === 1), axis);
                        divergence += (upper * upperDensity - lower * densityAt(y, axis)) / cellSizes[axis];
                    }

                    if ((bits & 64) !== 0) {
                        if (massWeighted) divergence /= sampleAt(y + 0.5, 6);
                        sumSquares += divergence * divergence;
                        maxAbsolute = Math.max(maxAbsolute, Math.abs(divergence));
                        fluidCells++;
                    }
                }
            }
        }

        return {
            rms: Math.sqrt(sumSquares / Math.max(fluidCells, 1)),
            maxAbsolute,
            maxBlockedFaceSpeed,
            nonFinite,
            fluidCells,
            massWeighted
        };
    }

}

export function halfToFloat(bits: number): number {
    const sign = bits & 0x8000 ? -1 : 1;
    const exponent = (bits >>> 10) & 31;
    const mantissa = bits & 1023;

    if (exponent === 31) {
        return mantissa ? NaN : sign * Infinity;
    }

    if (exponent === 0) {
        return sign * 2 ** -14 * mantissa / 1024;
    }

    return sign * 2 ** (exponent - 15) * (1 + mantissa / 1024);
}
