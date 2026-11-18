import { groundElevation } from './elevation';
import { GUI } from "lil-gui";
import * as THREE from "three/webgpu";

import { canvas, device, profiler } from "../renderer";
import { constants } from "../shaders/shaders";
import { Camera } from "../stage/camera";
import type { Site } from "../stage/stage";
import { Fields } from "../stage/fields";
import { ObstacleGrid } from "./obstacle_grid";
import { TileLayer } from "./tile_layer";
import type { SpatialWeather } from '../weather/spatial_weather';

function initialSource() {
    const query = new URLSearchParams(location.search);
    const scene = query.get('scene') ?? '';
    if (['flat', 'demo'].includes(scene)) {
        return scene;
    }
    return ['elreno', 'hrrr'].includes(query.get('case') ?? '') ? 'flat' : 'demo';
}

// Draws terrain and buildings with Three.js into its own render target
// Input: a Site and the camera
// Output: sceneColorTexture and sceneDepthTexture for the raymarch renderer, 
//         and obstacleMaterials() for the solver
export class GisTerrain {
    scene = new THREE.Scene();

    private target: THREE.RenderTarget;
    private sceneChanged = true;
    private readingTimestamp = false;
    private usedElevationModel = false;
    private readonly layers: TileLayer[] = [];
    private readonly demo = new THREE.Group();
    private fields: Fields | undefined;
    private baking = false;
    private materials = new ObstacleGrid().materials;
    private generation = 0;
    private rebakeRequested = false;
    private readonly attribution = document.createElement('div');

    source = initialSource();
    tilesetUrl = 'https://raw.githubusercontent.com/CesiumGS/3d-tiles-samples/main/1.0/TilesetWithDiscreteLOD/tileset.json';
    latitude: number;
    longitude: number;
    elevation: number;
    automaticElevation = true;
    status = 'Local building scene';

    private constructor(private threeRenderer: THREE.WebGPURenderer, readonly site: Site) {
        this.latitude = site.latitude;
        this.longitude = site.longitude;
        this.elevation = site.elevation ?? 0;
        // GPU errors are already reported, so this will stop Three from logging the errors twice.
        threeRenderer.onError = () => {};

        // An offscreen texture for Three to draw into. HalfFloatType gives rgba16float color, and FloatType gives depth32float.
        // These textures will then get fed into the raymarcher shader to create the final image.
        this.target = new THREE.RenderTarget(canvas.width, canvas.height, {
            type: THREE.HalfFloatType,
            samples: 0,
            depthTexture: new THREE.DepthTexture(canvas.width, canvas.height, THREE.FloatType)
        });

        this.scene.background = new THREE.Color(0.32, 0.42, 0.58);
        this.scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 3));
        const ground = new THREE.PlaneGeometry(60000, 60000).rotateX(-Math.PI / 2);
        this.scene.add(new THREE.Mesh(ground, new THREE.MeshStandardMaterial({ color: 0x4a5a3c, roughness: 1 })));
        this.scene.add(this.demo);
        this.demo.visible = this.source === 'demo';
        const scale = constants.domainX / 12800;
        for (const [x, z, height] of [[-1200, 0, 800], [1100, -600, 600], [700, 1300, 450]]) {
            const building = new THREE.Mesh(
                new THREE.BoxGeometry(450 * scale, height * scale, 600 * scale),
                new THREE.MeshStandardMaterial({ color: 0x99958a, roughness: 0.9 })
            );
            building.position.set(x * scale, height * scale / 2, z * scale);
            this.demo.add(building);
        }
        this.attribution.style.cssText = 'position:fixed;bottom:4px;right:8px;color:white;font:11px sans-serif;max-width:70%;text-align:right';
        document.body.append(this.attribution);
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
        return this.materials;
    }

    async connect(fields: Fields) {
        this.fields = fields;
        fields.setObstacles(this.materials);
        await this.bakeCollisions();
    }

    useSpatialTerrain(weather: SpatialWeather) {
        const { gridX, gridY, gridZ, domainX, domainY, domainZ } = constants;
        const ground = weather.manifest.groundMsl;
        const materials = new Uint8Array(gridX * gridY * gridZ);
        for (let z = 0; z < gridZ; z++) {
            for (let x = 0; x < gridX; x++) {
                const sample = weather.sample((x + 0.5) / gridX * domainX - domainX / 2,
                    0, (z + 0.5) / gridZ * domainZ - domainZ / 2);
                for (let y = 0; y < gridY; y++) {
                    if ((y + 0.5) / gridY * domainY + ground < sample.terrain) {
                        materials[x + gridX * (y + gridY * z)] = 1;
                    }
                }
            }
        }
        const geometry = new THREE.PlaneGeometry(domainX, domainZ, gridX, gridZ).rotateX(-Math.PI / 2);
        const positions = geometry.getAttribute('position');
        for (let index = 0; index < positions.count; index++) {
            positions.setY(index, weather.sample(positions.getX(index), 0, positions.getZ(index)).terrain - ground);
        }
        geometry.computeVertexNormals();
        this.scene.add(new THREE.Mesh(geometry,
            new THREE.MeshStandardMaterial({ color: 0x4a5a3c, roughness: 1 })));
        this.materials = materials;
        this.fields?.setObstacles(materials);
        this.status = 'HRRR terrain Ã‚Â· elevations from the weather grid';
        this.invalidate();
    }

    buildGui(gui: GUI) {
        const folder = gui.addFolder('GIS and collisions');
        if (constants.spatialWeather) {
            folder.add(this, 'status').name('Source').disable();
            return;
        }
        folder.add(this, 'source', ['flat', 'demo', 'sample', 'cesium', 'url']).name('Source');
        folder.add(this, 'tilesetUrl').name('Tileset URL');
        folder.add(this, 'latitude', -85, 85, 0.0001).name('Latitude').listen();
        folder.add(this, 'longitude', -180, 180, 0.0001).name('Longitude').listen();
        folder.add(this, 'automaticElevation').name('Automatic ground height');
        folder.add(this, 'elevation', -500, 5000, 1).name('Ground elevation (m)').listen();
        folder.add(this, 'load').name('Load scene');
        folder.add(this, 'bakeCollisions').name('Bake visible collisions');
        folder.add(this, 'status').name('Status').listen().disable();
    }

    async load() {
        const request = ++this.generation;
        if (this.source === 'cesium' && this.automaticElevation) {
            this.status = 'Looking up ground elevation';
            try {
                const height = await groundElevation(this.latitude, this.longitude);
                if (request !== this.generation) return;
                this.elevation = height;
            } catch (error) {
                if (request === this.generation) this.status = `${error}; set ground height manually and retry`;
                return;
            }
        }
        if (this.source === 'cesium' && !import.meta.env.VITE_CESIUM_ION_TOKEN) {
            this.status = 'Set VITE_CESIUM_ION_TOKEN in .env to load Cesium';
            return;
        }
        if (request !== this.generation) return;
        for (const layer of this.layers) {
            this.scene.remove(layer.tiles.group);
            layer.dispose();
        }
        this.layers.length = 0;
        this.usedElevationModel = false;
        this.demo.visible = this.source === 'demo';
        this.materials = new ObstacleGrid().materials;
        this.fields?.setObstacles(this.materials);
        this.attribution.textContent = '';
        this.site.latitude = this.latitude;
        this.site.longitude = this.longitude;
        this.site.elevation = this.elevation;

        if (this.source === 'demo' || this.source === 'flat') {
            void this.bakeCollisions();
        } else {
            const generation = this.generation;
            const specs = this.source === 'cesium'
                ? [{ material: 1, assetId: '1' }, { material: 2, assetId: '96188' }]
                : [{ material: 2, assetId: undefined }];
            for (const spec of specs) {
                const layer = new TileLayer(
                    this.tilesetUrl,
                    spec.material,
                    this.source === 'cesium' ? this.site : undefined,
                    spec.assetId,
                    () => this.invalidate(),
                    message => {
                        if (generation === this.generation) {
                            this.status = `Loading failed: ${message}`;
                        }
                    },
                    this.automaticElevation,
                    (latitude, longitude, elevation) => {
                        if (generation !== this.generation) return;
                        this.latitude = this.site.latitude = latitude;
                        this.longitude = this.site.longitude = longitude;
                        this.elevation = this.site.elevation = elevation;
                        this.usedElevationModel = this.automaticElevation;
                        this.status = this.automaticElevation
                            ? 'Ground from Open-Meteo / Copernicus DEM; bake collisions after tiles load'
                            : 'Manual ground height; bake collisions after tiles load';
                    },
                    this.elevation
                );
                this.layers.push(layer);
                this.scene.add(layer.tiles.group);
            }
            this.status = 'Loading tiles; bake collisions when the scene is ready';
        }
        this.invalidate();
    }

    async bakeCollisions() {
        if (this.baking) {
            this.rebakeRequested = true;
            return;
        }
        this.baking = true;
        const generation = this.generation;
        this.status = 'Building collision snapshot';
        try {
            this.scene.updateMatrixWorld(true);
            const grid = new ObstacleGrid();
            const objects = this.demo.visible
                ? [{ scene: this.demo, material: 2 }]
                : this.layers.flatMap(layer => layer.loadedScenes().map(scene => ({ scene, material: layer.material })));
            if (objects.length === 0 && this.source !== 'flat') {
                this.status = 'No visible tiles to bake; wait for loading or move toward the scene';
                return;
            }
            for (const object of objects) {
                const meshes: THREE.Mesh[] = [];
                object.scene.traverse(child => {
                    if (child instanceof THREE.Mesh) {
                        meshes.push(child);
                    }
                });
                for (const mesh of meshes) {
                    if (generation !== this.generation) {
                        return;
                    }
                    await grid.addMesh(mesh, object.material);
                }
            }
            if (generation === this.generation) {
                this.materials = grid.materials;
                this.fields?.setObstacles(this.materials);
                this.status = `Collision snapshot ready (${objects.length} loaded models)`;
            }
        } catch (error) {
            if (generation === this.generation) {
                this.status = `Collision bake failed: ${String(error)}`;
            }
        } finally {
            this.baking = false;
            if (this.rebakeRequested) {
                this.rebakeRequested = false;
                void this.bakeCollisions();
            }
        }
    }

    // Call after anything that changes how the scene looks.
    // Camera and window changes are already handled.
    invalidate() {
        this.sceneChanged = true;
    }

    // When nothing changed since our last frame, we can skip rendering and save resources.
    render(camera: Camera) {
        for (const layer of this.layers) {
            layer.tiles.setCamera(camera.threeCamera);
            layer.tiles.setResolution(camera.threeCamera, canvas.width, canvas.height);
            layer.tiles.update();
        }
        if (this.layers.length > 0) {
            const credits = this.layers.flatMap(layer => layer.tiles.getAttributions());
            this.attribution.textContent = credits.map(credit => String(credit.value).replace(/<[^>]*>/g, '')).join(' | ') + (this.usedElevationModel ? ' | Ground: Open-Meteo / Copernicus DEM' : '');
        }
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
