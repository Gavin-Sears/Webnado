import { device } from "../renderer";
import { constants } from "../shaders/shaders";

const fieldUsage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST;

function createField(
    label: string,
    format: GPUTextureFormat,
    size: GPUExtent3D = [constants.gridX, constants.gridY, constants.gridZ]
): GPUTexture {
    return device.createTexture({
        label,
        dimension: "3d",
        size,
        format,
        usage: fieldUsage
    });
}

// Two copies of a field. Passes read `read`, write `write`, then swap(). When a class finishes its frame, `read` must be
// the newest. Cache one bind group per `index`.
export class PingPong {
    index = 0;
    readonly textures: GPUTexture[];    // for debug readback and copies
    private views: GPUTextureView[];

    constructor(
        label: string,
        format: GPUTextureFormat,
        size: GPUExtent3D = [constants.gridX, constants.gridY, constants.gridZ]
    ) {
        this.textures = [
            createField(`${label} 0`, format, size),
            createField(`${label} 1`, format, size)
        ];
        this.views = this.textures.map(texture => texture.createView());
    }

    view(index: number): GPUTextureView {
        return this.views[index];
    }

    get read(): GPUTextureView {
        return this.views[this.index];
    }

    get write(): GPUTextureView {
        return this.views[1 - this.index];
    }

    swap() {
        this.index = 1 - this.index;
    }
}

// CHECKITOUT: the fields shared between teammates, with layouts in common.wgsl.
// These formats work as storage textures on every device, unlike r8uint or rg16float.
export class Fields {
    velocity = new PingPong("velocity", "rgba16float");
    dustVelocity: PingPong;
    dustDensity: PingPong;
    dustStorageVersion = 0;
    private dustAllocated: boolean;
    thermo = new PingPong("thermo", "rgba16float");
    // Turbulence channels are k (m²/s²), viscosity and mixing (m²/s), unused.
    turbulence = new PingPong("turbulence", "rgba16float");
    vorticity = new PingPong("transported vorticity", "rgba16float");
    readonly windForcing = createField("dust feedback acceleration", "rgba16float");
    readonly windForcingView = this.windForcing.createView();

    private obstacle = createField("obstacle", "r32uint");
    obstacleView = this.obstacle.createView();
    obstacleVersion = 0;
    hasObstacles = false;

    constructor(withDust = true) {
        this.dustAllocated = withDust;
        const size = withDust ? undefined : [1, 1, 1];
        this.dustVelocity = new PingPong('dust velocity', 'rgba16float', size);
        this.dustDensity = new PingPong('dust concentration (kg/m3)', 'r32float', size);
    }

    enableDustFields() {
        if (this.dustAllocated) {
            return;
        }
        this.dustVelocity.textures.forEach(texture => texture.destroy());
        this.dustDensity.textures.forEach(texture => texture.destroy());
        this.dustVelocity = new PingPong('dust velocity', 'rgba16float');
        this.dustDensity = new PingPong('dust concentration (kg/m3)', 'r32float');
        this.dustAllocated = true;
        this.dustStorageVersion++;
    }

    // The idea behind the obstacles is that we will have an entry for
    // each spatial grid cell. In each cell, we will denote what material
    // is present with the Uint8.
    // We convert to array index with: i + gridX * (j + gridY * k)
    // We convert back using: 
    // i = index % gridX;
    // j = Math.floor(index / gridX) % gridY;
    // k = Math.floor(index / (gridX * gridY));
    // Loading the obstacle array will be the task of the GIS person.
    setObstacles(materials: Uint8Array) {
        this.hasObstacles = materials.some(material => material !== 0);
        const { gridX, gridY, gridZ } = constants;
        device.queue.writeTexture(
            { texture: this.obstacle },
            Uint32Array.from(materials),
            { bytesPerRow: gridX * 4, rowsPerImage: gridY },
            { width: gridX, height: gridY, depthOrArrayLayers: gridZ }
        );
        this.obstacleVersion++;
    }
}
