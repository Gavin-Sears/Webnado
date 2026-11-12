import { Box3, Matrix4, Mesh, Triangle, Vector3 } from 'three';

import { constants } from '../shaders/shaders';

export class ObstacleGrid {
    readonly materials = new Uint8Array(constants.gridX * constants.gridY * constants.gridZ);
    private readonly cellSize = constants.domainX / constants.gridX;
    private readonly origin = new Vector3(-constants.domainX / 2, 0, -constants.domainZ / 2);

    constructor() {
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
        const top = new Int32Array(gridX * gridZ).fill(-1);
        const matrix = mesh.matrixWorld.clone();
        const instanceMatrix = new Matrix4();
        const count = 'isInstancedMesh' in mesh ? (mesh as import('three').InstancedMesh).count : 1;

        for (let instance = 0; instance < count; instance++) {
            const transform = matrix.clone();
            if ('isInstancedMesh' in mesh) {
                (mesh as import('three').InstancedMesh).getMatrixAt(instance, instanceMatrix);
                transform.multiply(instanceMatrix);
            }
            const vertices = indices?.count ?? positions.count;
            for (let start = 0; start + 2 < vertices; start += 3) {
                triangle.a.fromBufferAttribute(positions, indices ? indices.getX(start) : start).applyMatrix4(transform);
                triangle.b.fromBufferAttribute(positions, indices ? indices.getX(start + 1) : start + 1).applyMatrix4(transform);
                triangle.c.fromBufferAttribute(positions, indices ? indices.getX(start + 2) : start + 2).applyMatrix4(transform);
                bounds.setFromPoints([triangle.a, triangle.b, triangle.c]);
                const low = bounds.min.clone().sub(this.origin).divideScalar(this.cellSize).floor();
                const high = bounds.max.clone().sub(this.origin).divideScalar(this.cellSize).floor();
                if (high.x < 0 || high.y < 0 || high.z < 0
                    || low.x >= gridX || low.y >= gridY || low.z >= gridZ) {
                    continue;
                }
                low.max(new Vector3(0, 0, 0));
                high.min(new Vector3(gridX - 1, gridY - 1, gridZ - 1));
                for (let z = low.z; z <= high.z; z++) {
                    for (let y = low.y; y <= high.y; y++) {
                        for (let x = low.x; x <= high.x; x++) {
                            cellBounds.min.set(x, y, z).multiplyScalar(this.cellSize).add(this.origin);
                            cellBounds.max.copy(cellBounds.min).addScalar(this.cellSize);
                            if (cellBounds.intersectsTriangle(triangle)) {
                                top[x + gridX * z] = Math.max(top[x + gridX * z], y);
                            }
                        }
                    }
                }
                if (start % 12288 === 0) {
                    await new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
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
