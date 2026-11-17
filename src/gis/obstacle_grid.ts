import { Box3, InstancedMesh, Matrix4, Mesh, Triangle, Vector3 } from 'three';

import { constants } from '../shaders/shaders';

export class ObstacleGrid {
    readonly materials = new Uint8Array(constants.gridX * constants.gridY * constants.gridZ);
    private readonly cellSize = new Vector3(
        constants.domainX / constants.gridX,
        constants.domainY / constants.gridY,
        constants.domainZ / constants.gridZ
    );
    private readonly origin = new Vector3(-constants.domainX / 2, 0, -constants.domainZ / 2);

    constructor() {
        if (constants.freeSlipGround) {
            return;
        }
        const { gridX, gridY, gridZ } = constants;
        for (let z = 0; z < gridZ; z++) {
            this.materials.fill(1, gridX * gridY * z, gridX * gridY * z + gridX);
        }
    }

    async addMesh(mesh: Mesh, material: number) {
        const { gridX, gridY, gridZ } = constants;
        const geometry = mesh.geometry;
        const positions = geometry.getAttribute('position');
        if (!positions) {
            return;
        }
        const indices = geometry.getIndex();
        const triangle = new Triangle();
        const bounds = new Box3();
        const cellBounds = new Box3();
        const low = new Vector3();
        const high = new Vector3();
        const minimum = new Vector3();
        const maximum = new Vector3(gridX - 1, gridY - 1, gridZ - 1);
        const top = new Int32Array(gridX * gridZ).fill(-1);
        const matrix = mesh.matrixWorld.clone();
        const instanceMatrix = new Matrix4();
        const count = mesh instanceof InstancedMesh ? mesh.count : 1;

        for (let instance = 0; instance < count; instance++) {
            const transform = matrix.clone();
            if (mesh instanceof InstancedMesh) {
                mesh.getMatrixAt(instance, instanceMatrix);
                transform.multiply(instanceMatrix);
            }
            const vertices = indices?.count ?? positions.count;
            for (let start = 0; start + 2 < vertices; start += 3) {
                if (start % 12288 === 0) {
                    await new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
                }
                triangle.a.fromBufferAttribute(positions, indices ? indices.getX(start) : start).applyMatrix4(transform);
                triangle.b.fromBufferAttribute(positions, indices ? indices.getX(start + 1) : start + 1).applyMatrix4(transform);
                triangle.c.fromBufferAttribute(positions, indices ? indices.getX(start + 2) : start + 2).applyMatrix4(transform);
                bounds.setFromPoints([triangle.a, triangle.b, triangle.c]);
                low.copy(bounds.min).sub(this.origin).divide(this.cellSize).floor();
                high.copy(bounds.max).sub(this.origin).divide(this.cellSize).floor();
                if (high.x < 0 || high.y < 0 || high.z < 0
                    || low.x >= gridX || low.y >= gridY || low.z >= gridZ) {
                    continue;
                }
                low.max(minimum);
                high.min(maximum);
                for (let z = low.z; z <= high.z; z++) {
                    for (let y = low.y; y <= high.y; y++) {
                        for (let x = low.x; x <= high.x; x++) {
                            cellBounds.min.set(x, y, z).multiply(this.cellSize).add(this.origin);
                            cellBounds.max.copy(cellBounds.min).add(this.cellSize);
                            if (cellBounds.intersectsTriangle(triangle)) {
                                top[x + gridX * z] = Math.max(top[x + gridX * z], y);
                            }
                        }
                    }
                }
            }
        }

        // Solid columns are conservative proxies at the simulation's voxel resolution.
        for (let z = 0; z < gridZ; z++) {
            for (let x = 0; x < gridX; x++) {
                for (let y = 0; y <= top[x + gridX * z]; y++) {
                    const index = x + gridX * (y + gridY * z);
                    this.materials[index] = Math.max(this.materials[index], material);
                }
            }
        }
    }
}
