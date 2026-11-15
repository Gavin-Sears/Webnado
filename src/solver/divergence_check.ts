import { constants } from '../shaders/shaders';
import { Fields } from '../stage/fields';

export interface DivergenceReport {
    rms: number;
    maxAbsolute: number;
    maxBlockedFaceSpeed: number;
    nonFinite: number;
    fluidCells: number;
}

export class DivergenceCheck {
    static get bufferSize(): number {
        const velocityRow = Math.ceil(constants.gridX * 8 / 256) * 256;
        const maskRow = Math.ceil(constants.gridX * 4 / 256) * 256;
        const layers = constants.gridY * constants.gridZ;

        return (velocityRow + maskRow) * layers + constants.atmosphereSlices * 32;
    }

    private readonly velocityRow = Math.ceil(constants.gridX * 8 / 256) * 256;
    private readonly maskRow = Math.ceil(constants.gridX * 4 / 256) * 256;
    private readonly velocityBytes = this.velocityRow * constants.gridY * constants.gridZ;
    private readonly maskBytes = this.maskRow * constants.gridY * constants.gridZ;
    private readonly atmosphereOffset = this.velocityBytes + this.maskBytes;

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

        const bitsAt = (x: number, y: number, z: number) => {
            const offset = this.velocityBytes + this.maskRow * (y + gridY * z) + x * 4;
            return data.getUint32(offset, true);
        };

        const windAt = (y: number, axis: number) => {
            const slice = Math.max(0, Math.min(constants.atmosphereSlices - 1,
                (y + 0.5) / gridY * constants.atmosphereSlices - 0.5));
            const lower = Math.floor(slice);
            const upper = Math.min(lower + 1, constants.atmosphereSlices - 1);
            const channel = axis === 0 ? 2 : 3;
            const a = data.getFloat32(this.atmosphereOffset + lower * 32 + channel * 4, true);
            const b = data.getFloat32(this.atmosphereOffset + upper * 32 + channel * 4, true);

            return a + (b - a) * (slice - lower);
        };

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
                    const offset = this.velocityRow * (position[1] + gridY * position[2]) + position[0] * 8 + axis * 2;
                    return open ? halfToFloat(data.getUint16(offset, true)) : 0;
                }
                return open ? windAt(y, axis) : 0;
            }

            const offset = this.velocityRow * (y + gridY * z) + x * 8 + axis * 2;
            return halfToFloat(data.getUint16(offset, true));
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

                        divergence += (upper - lower) / cellSizes[axis];
                    }

                    if ((bits & 64) !== 0) {
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
            fluidCells
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
