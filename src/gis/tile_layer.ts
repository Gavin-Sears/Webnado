import { groundElevation } from './elevation';
import { Box3, Matrix4, MathUtils, Object3D, Sphere } from 'three';
import { TilesRenderer } from '3d-tiles-renderer';
import { CesiumIonAuthPlugin, GLTFExtensionsPlugin, QuantizedMeshPlugin } from '3d-tiles-renderer/plugins';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';

import type { Site } from '../stage/stage';

export class TileLayer {
    readonly tiles: TilesRenderer;
    private readonly draco = new DRACOLoader();

    constructor(
        url: string,
        readonly material: number,
        site: Site | undefined,
        assetId: string | undefined,
        onChange: () => void,
        onError: (message: string) => void,
        automaticElevation = true,
        onAnchor?: (latitude: number, longitude: number, elevation: number) => void,
        manualElevation?: number
    ) {
        this.tiles = new TilesRenderer(url);
        this.tiles.errorTarget = 8;
        this.draco.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.5.7/');
        this.tiles.registerPlugin(new GLTFExtensionsPlugin({ dracoLoader: this.draco }));
        if (assetId) {
            this.tiles.registerPlugin(new CesiumIonAuthPlugin({
                apiToken: import.meta.env.VITE_CESIUM_ION_TOKEN,
                assetId,
                autoRefreshToken: true,
                assetTypeHandler(type, tiles) {
                    if (type === 'TERRAIN') {
                        tiles.registerPlugin(new QuantizedMeshPlugin({}));
                    } else {
                        throw new Error(`Unsupported Cesium asset type: ${type}`);
                    }
                }
            }));
        }
        let aligned = site !== undefined;
        this.tiles.addEventListener('load-root-tileset', async () => {
            let latitude = site?.latitude ?? 0;
            let longitude = site?.longitude ?? 0;
            let height = site?.elevation ?? 0;
            if (!site) {
                const sphere = new Sphere();
                this.tiles.getBoundingSphere(sphere);
                if (sphere.center.length() < this.tiles.ellipsoid.radius.x * 0.5) {
                    const local = new Matrix4().makeRotationX(-Math.PI / 2);
                    const center = sphere.center.clone().applyMatrix4(local);
                    local.setPosition(center.negate());
                    local.decompose(this.tiles.group.position, this.tiles.group.quaternion, this.tiles.group.scale);
                    this.tiles.group.updateMatrixWorld(true);
                    onChange();
                    return;
                }
                const coordinates = this.tiles.ellipsoid.getPositionToCartographic(sphere.center, {});
                latitude = MathUtils.radToDeg(coordinates.lat);
                longitude = MathUtils.radToDeg(coordinates.lon);
                height = coordinates.height;
            }
            aligned = true;
            if (!site && automaticElevation) {
                this.tiles.group.visible = false;
                try {
                    height = await groundElevation(latitude, longitude);
                } catch (error) {
                    onError(String(error));
                    return;
                }
            }
            if (!site && !automaticElevation && manualElevation !== undefined) height = manualElevation;
            onAnchor?.(latitude, longitude, height);
            this.tiles.group.visible = true;
            const frame = this.tiles.ellipsoid.getEastNorthUpFrame(
                MathUtils.degToRad(latitude), MathUtils.degToRad(longitude), height, new Matrix4()
            );
            const local = new Matrix4().makeRotationX(-Math.PI / 2).multiply(frame.invert());
            local.decompose(this.tiles.group.position, this.tiles.group.quaternion, this.tiles.group.scale);
            this.tiles.group.updateMatrixWorld(true);
            onChange();
        });
        this.tiles.addEventListener('load-model', event => {
            if (!aligned) {
                this.tiles.group.updateMatrixWorld(true);
                event.scene.updateWorldMatrix(true, true);
                const bounds = new Box3().setFromObject(event.scene);
                if (event.scene.parent !== this.tiles.group) {
                    bounds.applyMatrix4(this.tiles.group.matrixWorld);
                }
                if (!bounds.isEmpty()) {
                    this.tiles.group.position.y -= bounds.min.y;
                    this.tiles.group.updateMatrixWorld(true);
                    aligned = true;
                }
            }
            onChange();
        });
        this.tiles.addEventListener('tile-visibility-change', onChange);
        this.tiles.addEventListener('load-error', event => onError(event.error.message));
    }

    loadedScenes(): Object3D[] {
        const scenes: Object3D[] = [];
        this.tiles.forEachLoadedModel((scene, tile) => {
            if (!this.tiles.visibleTiles.has(tile)) {
                return;
            }
            const bounds = new Box3().setFromObject(scene);
            if (!bounds.isEmpty()) {
                scenes.push(scene);
            }
        });
        return scenes;
    }

    dispose() {
        this.tiles.dispose();
        this.draco.dispose();
    }
}
