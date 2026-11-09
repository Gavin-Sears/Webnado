import { GUI } from "lil-gui";
import * as THREE from "three/webgpu";

import { canvas, device, profiler } from "../renderer";
import { constants } from "../shaders/shaders";
import { Camera } from "../stage/camera";
import type { Site } from "../stage/stage";

// Draws terrain and buildings with Three.js into its own render target
// Input: a Site and the camera
// Output: sceneColorTexture and sceneDepthTexture for the raymarch renderer, 
//         and obstacleMaterials() for the solver
export class GisTerrain {
    scene = new THREE.Scene();

    private target: THREE.RenderTarget;
    private sceneChanged = true;
    private readingTimestamp = false;

    private constructor(private threeRenderer: THREE.WebGPURenderer, readonly site: Site) {
        // GPU errors are already reported, so this will stop Three from logging the errors twice.
        threeRenderer.onError = () => {};

        // An offscreen texture for Three to draw into. HalfFloatType gives rgba16float color, and FloatType gives depth32float.
        // These textures will then get fed into the raymarcher shader to create the final image.
        this.target = new THREE.RenderTarget(canvas.width, canvas.height, {
            type: THREE.HalfFloatType,
            samples: 0,
            depthTexture: new THREE.DepthTexture(canvas.width, canvas.height, THREE.FloatType)
        });

        // TODO: replace this flat ground with 3D tiles for this.site, using 3d-tiles-renderer@0.5.3.
        //   Test with the public sample (no key): https://raw.githubusercontent.com/CesiumGS/3d-tiles-samples/main/1.0/TilesetWithDiscreteLOD/tileset.json
        //   You'll need to use Cesium ion for the real data. Ask me (Gavin) for the token, put it in .env 
        //   as VITE_CESIUM_ION_TOKEN, and read it with import.meta.env.VITE_CESIUM_ION_TOKEN.
        //   (asset ID for buildings is 96188, and terrain is 1)
        //   DO NOT push the token. Let me know if this happens, and I can make a new one. 
        //   Add tiles.group to this.scene here, and call tiles.update() at the top of render().
        this.scene.background = new THREE.Color(0.32, 0.42, 0.58);
        this.scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 3));
        const ground = new THREE.PlaneGeometry(60000, 60000).rotateX(-Math.PI / 2);
        this.scene.add(new THREE.Mesh(ground, new THREE.MeshStandardMaterial({ color: 0x4a5a3c, roughness: 1 })));
    }

    static async create(site: Site): Promise<GisTerrain> {
        const threeRenderer = new THREE.WebGPURenderer({ canvas: document.createElement("canvas"), device, trackTimestamp: profiler !== undefined });
        await threeRenderer.init();

        // Three silently falls back to WebGL2, which can't share the device
        if (!(threeRenderer.backend as any).isWebGPUBackend) {
            throw new Error("Three.js did not start its WebGPU backend");
        }
        return new GisTerrain(threeRenderer, site);
    }

    // One material per cell (0 air, 1 terrain, 2 building), indexed i + gridX * (j + gridY * k)
    // TODO: build this from the real ground. This example has the bottom layer as ground.
    obstacleMaterials(): Uint8Array {
        const { gridX, gridY, gridZ } = constants;
        const materials = new Uint8Array(gridX * gridY * gridZ);
        for (let k = 0; k < gridZ; ++k) {
            materials.fill(1, gridX * gridY * k, gridX * gridY * k + gridX);
        }
        return materials;
    }

    buildGui(_gui: GUI) {
    }

    // Call after anything that changes how the scene looks.
    // Camera and window changes are already handled.
    invalidate() {
        this.sceneChanged = true;
    }

    // When nothing changed since our last frame, we can skip rendering and save resources.
    render(camera: Camera) {
        if (this.target.width !== canvas.width || this.target.height !== canvas.height) {
            this.target.setSize(canvas.width, canvas.height); // Three destroys the old textures
            this.sceneChanged = true;
        }
        if (!this.sceneChanged && !camera.moved) {
            return;
        }
        this.sceneChanged = false;

        this.threeRenderer.setRenderTarget(this.target);
        this.threeRenderer.render(this.scene, camera.threeCamera);
        this.recordGpuTime();
    }

    // Only with ?profile. Three creates its passes internally, so it times them itself.
    private async recordGpuTime() {
        if (!profiler || this.readingTimestamp) {
            return;
        }

        this.readingTimestamp = true;
        try {
            const ms = await this.threeRenderer.resolveTimestampsAsync(THREE.TimestampQuery.RENDER);
            if (ms) {
                profiler.recordPass("three", ms);
            }
        } finally {
            this.readingTimestamp = false;
        }
    }

    // Only valid after render(). These textures change on resize, so compare them 
    // before reusing bind groups.
    get sceneColorTexture(): GPUTexture {
        return (this.threeRenderer.backend as any).get(this.target.texture).texture;
    }

    get sceneDepthTexture(): GPUTexture {
        return (this.threeRenderer.backend as any).get(this.target.depthTexture).texture;
    }
}
