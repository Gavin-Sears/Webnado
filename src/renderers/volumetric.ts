import { GUI } from 'lil-gui';

import * as renderer from '../renderer';
import * as shaders from '../shaders/shaders';
import { Stage } from '../stage/stage';
import { fillTestVolume } from './test_volume';
import { help } from '../ui/help';
import blueNoiseUrl from '../assets/blue_noise_64.png';

// Draws the clouds in fields.thermo over the Three.js scene. Each frame, draw() runs gis.render(camera),
// solver.step(encoder), then three passes of its own:
//   1. bricks      bricks.cs.wgsl      finds largest density per 8^3 brick, so rays can skip empty space
//   2. raymarch    raymarch.fs.wgsl    single scattering into a target smaller than the canvas: rgb light, a transmittance
//   3. temporal    temporal.fs.wgsl    blends with last frame's result, reprojected
//   3. composite   composite.fs.wgsl   depth-aware upsample, over the scene, tone map, then linear to sRGB for the canvas
// A light volume pass would go between 1 and 2 (see sunTransmittance in raymarch.fs.wgsl).

export interface Quality {
    resolutionScale: number;
    stepCells: number;
    maxSteps: number;
    shadowSteps: number;
}

// Pick one with ?quality=low, medium or high. Otherwise platformQuality() picks.
export const qualityPresets: Record<string, Quality> = {
    low: { resolutionScale: 0.35, stepCells: 2, maxSteps: 96, shadowSteps: 4 },
    medium: { resolutionScale: 0.5, stepCells: 1, maxSteps: 192, shadowSteps: 6 },
    high: { resolutionScale: 0.75, stepCells: 0.75, maxSteps: 384, shadowSteps: 8 }
};

// Quality presets for each GPU vendor
function platformQuality(): string {
    if (/Android|iPhone|iPad/i.test(navigator.userAgent)) {
        return 'low';
    }
    switch (renderer.device.adapterInfo.vendor) {
        case 'nvidia':
        case 'amd':
            return 'high';
        case 'apple':
            return 'medium';
        default:
            return 'low';   // intel, mobile GPUs, and anything unknown
    }
}

// The numbers match the cases in debugMarch() (raymarch.fs.wgsl)
const DEBUG_VIEWS: Record<string, number> = {
    off: 0, density: 1, temperature: 2, velocity: 3, vorticity: 4, obstacles: 5, bricks: 6,
    'dust density': 7, 'dust velocity': 8, 'vertical rotation': 9, 'swirling strength': 10
};

// Starting scale and max value for debug views
const DEBUG_SCALES: Record<string, { value: number, max: number, unit: string }> = {
    density: { value: 3, max: 10, unit: 'g/kg' },
    densityBoosted: { value: 0.1, max: 2, unit: 'g/kg' },
    temperature: { value: 1, max: 10, unit: 'K' },          // theta'. The thermal starts at +2 K.
    velocity: { value: 5, max: 50, unit: 'm/s' },
    vorticity: { value: 0.1, max: 1, unit: '1/s' },
    'dust density': { value: 0.02, max: 0.2, unit: 'kg/m3' },
    'dust velocity': { value: 5, max: 50, unit: 'm/s' },
    'vertical rotation': { value: 0.05, max: 0.5, unit: '1/s' },
    'swirling strength': { value: 0.03, max: 0.2, unit: '1/s' }
};

const VOLUME_FORMAT: GPUTextureFormat = "rgba16float";
const PARAMS_SIZE = 112; // RaymarchParams in raymarch.fs.wgsl
const CLOUD_DEPTH_FORMAT: GPUTextureFormat = "r32float";
const NO_HIT_DEPTH = 100000; // meters, must match noHitDepth in raymarch.fs.wgsl

const { gridX, gridY, gridZ, brickSize } = shaders.constants;
const BRICK_DIMS: [number, number, number] = [gridX / brickSize, gridY / brickSize, gridZ / brickSize];

export class VolumetricRenderer extends renderer.Renderer {
    // Settings. The GUI changes these, and draw() sends them to the GPU every frame.
    qualityName: string;
    quality: Quality;
    lighting = {
        sunAzimuth: 31,         // degrees from +X (east) toward +Z (south)
        sunElevation: 54,       // degrees above the horizon. With the azimuth, this is normalize(0.5, 0.8, 0.3).
        sunColor: [1.0, 0.96, 0.88],
        sunIntensity: 1.5,
        ambientColor: [0.5, 0.6, 0.78],
        ambientIntensity: 0.5,
        phaseG: 0.3,            // Henyey-Greenstein: 0 scatters evenly, near 1 mostly forward
        extinction: 0.0025      // 1/m per g/kg of qc + qr
    };
    shadowDistance = 6000;      // meters
    brickSkipping = true;
    animateJitter = true;
    blueNoiseToggle = true;
    multipleScattering = {
        octaves: 3,             // degree of detail we use to approximate scattering. 1 is for single scattering
        attenuation: 0.5,       // shadow optical depth multiplier per octave (should be <= contribution)
        contribution: 0.5,      // light multiplier per octave
        eccentricity: 0.5,      // phase g multiplier per octave
    }
    temporal = {
        enabled: true,
        blend: 0.1,
        clampHistory: true
    };
    debugView = 'off';
    debugScale = 1;
    debugBoost = false;         // density view: makes faint cloud visible

    private raymarchPipeline: GPURenderPipeline;
    private compositePipeline: GPURenderPipeline;
    private bricksPipeline: GPUComputePipeline;
    private linearSampler: GPUSampler;

    private paramsBuffer: GPUBuffer;
    private paramsData = new ArrayBuffer(PARAMS_SIZE);
    private paramsFloats = new Float32Array(this.paramsData);
    private paramsUints = new Uint32Array(this.paramsData);

    // 1/8 of the grid on each side, r32float
    private brickTexture: GPUTexture;
    private brickView: GPUTextureView;
    private bricksBindGroups: GPUBindGroup[] = [];

    // The raymarch target, recreated when the canvas or resolution scale changes
    private volumeTexture: GPUTexture | undefined;
    private volumeView: GPUTextureView | undefined;

    // Rebuilt when a resize changes the textures they hold
    private boundColor: GPUTexture | undefined;
    private boundDepth: GPUTexture | undefined;
    private boundBlueNoise: GPUTexture | undefined;
    private boundVolume: GPUTexture | undefined;
    private boundDustVersion = -1;
    private raymarchBindGroups: GPUBindGroup[] = [];    // by 2 * thermo.index + velocity.index
    private compositeBindGroups: GPUBindGroup[] = [];

    private blueNoiseTexture: GPUTexture;
    private blueNoiseView: GPUTextureView;
    private blueNoiseLoaded = false;

    private temporalPipeline: GPURenderPipeline;
    private temporalParamsBuffer: GPUBuffer;
    private temporalParamsData = new ArrayBuffer(16);
    private cloudDepthTexture: GPUTexture | undefined;
    private cloudDepthView: GPUTextureView | undefined;
    private historyTextures: GPUTexture[] = [];
    private historyViews: GPUTextureView[] = [];
    private historyIndex = 0;
    private historyValid = false;
    private historyResetVersion = -1;
    private historyWindowEast = 0;
    private historyWindowNorth = 0;
    private temporalBindGroups: GPUBindGroup[] = [];

    constructor(stage: Stage) {
        super(stage);
        const device = renderer.device;

        const requested = new URLSearchParams(window.location.search).get('quality') ?? '';
        this.qualityName = Object.hasOwn(qualityPresets, requested) ? requested : platformQuality();
        this.quality = { ...qualityPresets[this.qualityName] };

        const vertex = { module: device.createShaderModule({ label: "fullscreen vert shader", code: shaders.fullscreenVertSrc }) };
        const fragment = (label: string, code: string, ...formats: GPUTextureFormat[]): GPUFragmentState => ({
            module: device.createShaderModule({ label: `${label} frag shader`, code }),
            targets: formats.map(format => ({ format }))
        });
        this.raymarchPipeline = device.createRenderPipeline({
            label: "raymarch",
            layout: "auto",
            vertex,
            fragment: fragment("raymarch", shaders.raymarchFragSrc, VOLUME_FORMAT, CLOUD_DEPTH_FORMAT)
        });
        this.temporalPipeline = device.createRenderPipeline({
            label: "temporal",
            layout: "auto",
            vertex,
            fragment: fragment("temporal", shaders.temporalFragSrc, VOLUME_FORMAT)
        });
        this.compositePipeline = device.createRenderPipeline({
            label: "composite",
            layout: "auto",
            vertex,
            fragment: fragment("composite", shaders.compositeFragSrc, renderer.canvasFormat)
        });
        this.bricksPipeline = device.createComputePipeline({
            label: "bricks",
            layout: "auto",
            compute: { module: device.createShaderModule({ label: "bricks shader", code: shaders.bricksComputeSrc }) }
        });

        this.linearSampler = device.createSampler({ label: "volume linear sampler", magFilter: "linear", minFilter: "linear" });
        this.paramsBuffer = device.createBuffer({
            label: "raymarch params",
            size: PARAMS_SIZE,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.brickTexture = device.createTexture({
            label: "brick max",
            dimension: "3d",
            size: { width: BRICK_DIMS[0], height: BRICK_DIMS[1], depthOrArrayLayers: BRICK_DIMS[2] },
            format: "r32float",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
        });
        this.brickView = this.brickTexture.createView();

        this.temporalParamsBuffer = device.createBuffer({
            label: "temporal params",
            size: this.temporalParamsData.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.blueNoiseTexture = this.createBlueNoiseTexture(1, 1);
        this.blueNoiseView = this.blueNoiseTexture.createView();
        this.loadBlueNoise();

        fillTestVolume(stage);
    }

    private createBlueNoiseTexture(width: number, height: number): GPUTexture {
        return renderer.device.createTexture({
            label: "blue noise",
            size: { width, height },
            format: "rgba8unorm",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
        });
    }

    private async loadBlueNoise() {
        try {
            const response = await fetch(blueNoiseUrl);
            const bitmap = await createImageBitmap(
                await response.blob(), 
                { 
                    colorSpaceConversion: 'none',
                    premultiplyAlpha: 'none'
                }
            );
            const texture = this.createBlueNoiseTexture(bitmap.width, bitmap.height);
            renderer.device.queue.copyExternalImageToTexture({ source: bitmap }, { texture }, [bitmap.width, bitmap.height]);
            bitmap.close();

            this.blueNoiseTexture.destroy();
            this.blueNoiseTexture = texture;
            this.blueNoiseView = texture.createView();

            this.blueNoiseLoaded = true;
        } catch (error) {
            console.warn('Blue noise failed to load, using interleaved gradient noise', error);
        }
    }

    buildGui(gui: GUI, inspection?: GUI) {
        const folder = gui.addFolder('Raymarcher');

        const quality = folder.addFolder('Quality');
        quality.add(this, 'qualityName', Object.keys(qualityPresets)).name('preset').onChange((name: string) => {
            Object.assign(this.quality, qualityPresets[name]);
            quality.controllers.forEach(controller => controller.updateDisplay());
        });
        quality.add(this.quality, 'resolutionScale', 0.25, 1, 0.05).name('resolution scale');
        quality.add(this.quality, 'stepCells', 0.25, 4, 0.05).name('step (cells)');
        quality.add(this.quality, 'maxSteps', 16, 1024, 1).name('max steps');
        quality.add(this.quality, 'shadowSteps', 1, 32, 1).name('shadow steps');
        quality.add(this, 'shadowDistance', 500, 12800, 100).name('shadow distance (m)');
        quality.add(this, 'brickSkipping').name('brick skipping');
        quality.add(this, 'animateJitter').name('animate jitter');
        quality.add(this, 'blueNoiseToggle').name('blue noise jitter');

        const temporal = folder.addFolder('Temporal');
        temporal.add(this.temporal, 'enabled');
        temporal.add(this.temporal, 'blend', 0.02, 1, 0.01).name('new frame weight');
        temporal.add(this.temporal, 'clampHistory').name('clamp history');

        const lighting = folder.addFolder('Lighting');
        lighting.add(this.lighting, 'sunAzimuth', -180, 180, 1).name('sun azimuth');
        lighting.add(this.lighting, 'sunElevation', 0, 90, 1).name('sun elevation');
        lighting.addColor(this.lighting, 'sunColor').name('sun color');
        lighting.add(this.lighting, 'sunIntensity', 0, 5, 0.05).name('sun intensity');
        lighting.addColor(this.lighting, 'ambientColor').name('ambient color');
        lighting.add(this.lighting, 'ambientIntensity', 0, 2, 0.05).name('ambient intensity');
        lighting.add(this.lighting, 'phaseG', -0.9, 0.9, 0.01).name('phase g');
        lighting.add(this.lighting, 'extinction', 0.0002, 0.01, 0.0001);

        const scattering = folder.addFolder('Multiple scattering');
        scattering.add(this.multipleScattering, 'octaves', 1, 8, 1);
        scattering.add(this.multipleScattering, 'attenuation', 0, 1, 0.01);
        scattering.add(this.multipleScattering, 'contribution', 0, 1, 0.01);
        scattering.add(this.multipleScattering, 'eccentricity', 0, 1, 0.01);

        const debug = (inspection ?? folder).addFolder('Diagnostic view');
        debug.add({ focus: () => this.focusRotation() }, 'focus').name('Look at near-ground rotation');
        debug.add({ focus: () => this.focusRotation(true) }, 'focus').name('Look at rotation under cloud');
        const view = debug.add(this, 'debugView', {
            'Cloud rendering': 'off',
            'Cloud water': 'density',
            'Temperature anomaly': 'temperature',
            'Air velocity': 'velocity',
            'Vorticity magnitude': 'vorticity',
            'Collision obstacles': 'obstacles',
            'Occupied bricks': 'bricks',
            'Dust density': 'dust density',
            'Dust velocity': 'dust velocity',
            'Vertical rotation': 'vertical rotation',
            'Swirling strength': 'swirling strength'
        }).name('Show');
        const explanation = help(debug, '');
        const boost = debug.add(this, 'debugBoost').name('boost faint cloud');
        const scale = debug.add(this, 'debugScale', 0, 1);
        const explanations: Record<string, string> = {
            off: 'Cloud water rendered with lighting. Clear air can contain wind and rotation without visible cloud.',
            density: 'Cloud liquid water in g/kg. Boost reveals faint cloud; it does not create cloud.',
            temperature: 'Potential-temperature anomaly relative to the surrounding reference atmosphere: orange is warmer, blue is colder.',
            velocity: 'Air velocity components encoded as color. A uniform color can represent a steady background wind.',
            vorticity: 'Magnitude of local rotation, including horizontal rolls and shear. Bright regions are not necessarily tornadoes.',
            obstacles: 'Voxelized solids used for collisions. The ground and buildings should match the scene.',
            bricks: 'Occupied regions used to skip empty space during cloud rendering.',
            'dust density': 'Grid-based dust mass per volume. The separate dots are visualization tracers.',
            'dust velocity': 'Velocity of the dust fluid, including drag from air. It can differ from air velocity.',
            'vertical rotation': 'Signed rotation about the vertical axis. This isolates vertical spin from horizontal rolls.',
            'swirling strength': 'Local swirling strength filters out pure shear. Look for a compact, persistent column near the floor.'
        };
        help(debug, 'Focus buttons work once storm tracking detects rotation. Drag the scene to look around; WASD moves, Q/E moves vertically, Shift moves faster.');
        const showScale = () => {
            explanation.textContent = explanations[this.debugView];
            const name = this.debugView === 'density' && this.debugBoost ? 'densityBoosted' : this.debugView;
            const preset = DEBUG_SCALES[name];
            boost.show(this.debugView === 'density');
            scale.show(preset !== undefined);
            if (preset) {
                this.debugScale = preset.value;
                scale.min(preset.max / 1000).max(preset.max).step(preset.max / 1000).name(`scale (${preset.unit})`);
                scale.updateDisplay();
            }
        };
        view.onChange(showScale);
        boost.onChange(showScale);
        showScale();
    }

    private focusRotation(underCloud = false) {
        const storm = this.stage.solver.storm;
        const target = underCloud ? storm.cloudRotationLocation : storm.rotationLocation;
        if (!target) return;
        const camera = this.stage.camera;
        camera.threeCamera.position.set(target[0] + 2500, 400, target[2] + 2500);
        camera.yaw = -135;
        camera.pitch = Math.atan2(200, Math.hypot(2500, 2500)) * 180 / Math.PI;
        camera.moved = true;
    }

    override stop() {
        super.stop();
        this.volumeTexture?.destroy();
        this.volumeTexture = undefined;
        this.brickTexture.destroy();
        this.paramsBuffer.destroy();
        this.blueNoiseTexture.destroy();
        this.cloudDepthTexture?.destroy();
        this.historyTextures.forEach(texture => texture.destroy());
        this.temporalParamsBuffer.destroy();
    }

    private updateVolumeTarget() {
        const { canvas } = renderer;
        const scale = this.quality.resolutionScale;
        const width = Math.max(1, Math.round(canvas.width * scale));
        const height = Math.max(1, Math.round(canvas.height * scale));
        if (this.volumeTexture?.width === width && this.volumeTexture?.height === height) {
            return;
        }

        this.volumeTexture?.destroy();
        this.volumeTexture = renderer.device.createTexture({
            label: "raymarch target",
            size: { width, height },
            format: VOLUME_FORMAT,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        });
        this.volumeView = this.volumeTexture.createView();

        const target = (label: string, format: GPUTextureFormat) => renderer.device.createTexture({
            label,
            size: { width, height },
            format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        });
        this.cloudDepthTexture?.destroy();
        this.cloudDepthTexture = target("cloud depth", CLOUD_DEPTH_FORMAT);
        this.cloudDepthView = this.cloudDepthTexture.createView();
        this.historyTextures.forEach(texture => texture.destroy());
        this.historyTextures = [target("history 0", VOLUME_FORMAT), target("history 1", VOLUME_FORMAT)];
        this.historyViews = this.historyTextures.map(texture => texture.createView());
        this.historyValid = false;
    }

    private updateBindGroups() {
        const { gis, camera, fields } = this.stage;
        const device = renderer.device;
        const color = gis.sceneColorTexture;
        const depth = gis.sceneDepthTexture;

        if (depth !== this.boundDepth || this.blueNoiseTexture !== this.boundBlueNoise
            || this.boundDustVersion !== fields.dustStorageVersion) {
            this.bricksBindGroups = Array.from({ length: 2 }, (_, i) => device.createBindGroup({
                label: `bricks ${i}`,
                layout: this.bricksPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: fields.thermo.view(i) },
                    { binding: 1, resource: this.brickView }
                ]
            }));
            const depthView = depth.createView();
            this.raymarchBindGroups = Array.from({ length: 16 }, (_, i) => device.createBindGroup({
                label: `raymarch ${i}`,
                layout: this.raymarchPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: camera.uniformsBuffer } },
                    { binding: 1, resource: { buffer: this.paramsBuffer } },
                    { binding: 2, resource: fields.thermo.view((i >> 1) & 1) },
                    { binding: 3, resource: this.linearSampler },
                    { binding: 4, resource: depthView },
                    { binding: 5, resource: this.brickView },
                    { binding: 6, resource: fields.velocity.view(i & 1) },
                    { binding: 7, resource: fields.obstacleView },
                    { binding: 8, resource: fields.dustVelocity.view((i >> 2) & 1) },
                    { binding: 9, resource: fields.dustDensity.view((i >> 3) & 1) },
                    { binding: 10, resource: this.blueNoiseView }
                ]
            }));
        }

        if (color !== this.boundColor || depth !== this.boundDepth || this.volumeTexture !== this.boundVolume) {
            const colorView = color.createView();
            const depthView = depth.createView();
            this.compositeBindGroups = [...this.historyViews, this.volumeView!].map((view, i) => 
                device.createBindGroup({
                    label: `composite ${i}`,
                    layout: this.compositePipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: camera.uniformsBuffer } },
                        { binding: 1, resource: colorView },
                        { binding: 2, resource: depthView },
                        { binding: 3, resource: view }
                    ]
                }));
        }

        if (this.volumeTexture !== this.boundVolume) {
            this.temporalBindGroups = [0, 1].map(i => device.createBindGroup({
                label: `temporal ${i}`,
                layout: this.temporalPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: camera.uniformsBuffer } },
                    { binding: 1, resource: { buffer: this.temporalParamsBuffer } },
                    { binding: 2, resource: this.volumeView! },
                    { binding: 3, resource: this.cloudDepthView! },
                    { binding: 4, resource: this.historyViews[i] },
                    { binding: 5, resource: this.linearSampler }
                ]
            }));
        }

        this.boundColor = color;
        this.boundDepth = depth;
        this.boundBlueNoise = this.blueNoiseTexture;
        this.boundVolume = this.volumeTexture;
        this.boundDustVersion = fields.dustStorageVersion;
    }

    // In the layout of RaymarchParams
    private writeParams() {
        const f = this.paramsFloats;
        const u = this.paramsUints;
        const l = this.lighting;
        const azimuth = l.sunAzimuth * Math.PI / 180;
        const elevation = l.sunElevation * Math.PI / 180;

        f.set([Math.cos(elevation) * Math.cos(azimuth), Math.sin(elevation), Math.cos(elevation) * Math.sin(azimuth)], 0);
        f[3] = l.phaseG;
        f.set(l.sunColor.map(c => c * l.sunIntensity), 4);
        f[7] = l.extinction;
        f.set(l.ambientColor.map(c => c * l.ambientIntensity), 8);
        f[11] = this.quality.stepCells;
        u[12] = this.volumeTexture!.width;
        u[13] = this.volumeTexture!.height;
        u[14] = Math.max(1, Math.round(this.quality.maxSteps));
        u[15] = Math.max(1, Math.round(this.quality.shadowSteps));
        f[16] = this.shadowDistance;
        u[17] = this.animateJitter ? 64 : 1;
        u[18] = this.brickSkipping ? 1 : 0;
        u[19] = DEBUG_VIEWS[this.debugView];
        f[20] = this.debugScale;
        u[21] = this.debugBoost ? 1 : 0;
        const ms = this.multipleScattering;
        u[22] = Math.max(1, Math.round(ms.octaves));
        f[23] = ms.attenuation;
        f[24] = ms.contribution;
        f[25] = ms.eccentricity;
        u[26] = this.blueNoiseToggle && this.blueNoiseLoaded ? 1 : 0;
        renderer.device.queue.writeBuffer(this.paramsBuffer, 0, this.paramsData);
    }

    private writeTemporalParams() {
        const f = new Float32Array(this.temporalParamsData);
        const u = new Uint32Array(this.temporalParamsData);
        f[0] = this.temporal.blend;
        u[1] = this.historyValid ? 1 : 0;
        u[2] = this.temporal.clampHistory ? 1 : 0;

        renderer.device.queue.writeBuffer(this.temporalParamsBuffer, 0, this.temporalParamsData);
    }

    override draw() {
        const { gis, camera, solver, fields, debris } = this.stage;
        const device = renderer.device;
        const profiler = renderer.profiler;

        this.updateVolumeTarget();
        gis.render(camera); // Three submits this itself
        this.writeParams();

        const encoder = device.createCommandEncoder({ label: "frame" });

        const advanced = solver.step(encoder);
        const useTemporal = this.temporal.enabled && this.debugView === 'off';
        // A reset or moving window breaks the previous frame's world-space correspondence.
        if (!useTemporal || solver.resetVersion !== this.historyResetVersion
            || solver.storm.windowEast !== this.historyWindowEast
            || solver.storm.windowNorth !== this.historyWindowNorth) {
            this.historyValid = false;
        }
        this.historyResetVersion = solver.resetVersion;
        this.historyWindowEast = solver.storm.windowEast;
        this.historyWindowNorth = solver.storm.windowNorth;
        this.writeTemporalParams();
        this.updateBindGroups();
        // Tracers follow the projected dust field; the solver owns momentum exchange.
        debris.step(encoder, advanced ? solver.dt : 0, solver.resetVersion);

        // After the solver, so the bricks match the thermo being drawn
        if (this.brickSkipping || this.debugView === 'bricks') {
            const bricksPass = encoder.beginComputePass({ label: "bricks pass", timestampWrites: profiler?.passWrites("bricks") });
            bricksPass.setPipeline(this.bricksPipeline);
            bricksPass.setBindGroup(0, this.bricksBindGroups[fields.thermo.index]);
            bricksPass.dispatchWorkgroups(...BRICK_DIMS);
            bricksPass.end();
        }

        const raymarchPass = encoder.beginRenderPass({
            label: "raymarch pass",
            timestampWrites: profiler?.passWrites("raymarch"),
            colorAttachments: [{
                view: this.volumeView!,
                clearValue: [0, 0, 0, 1],
                loadOp: "clear",
                storeOp: "store"
            }, {
                view: this.cloudDepthView!,
                clearValue: [NO_HIT_DEPTH, 0, 0, 0],
                loadOp: "clear",
                storeOp: "store"
            }]
        });
        raymarchPass.setPipeline(this.raymarchPipeline);
        const fieldState = fields.velocity.index + 2 * fields.thermo.index
            + 4 * fields.dustVelocity.index + 8 * fields.dustDensity.index;
        raymarchPass.setBindGroup(0, this.raymarchBindGroups[fieldState]);
        raymarchPass.draw(3);
        raymarchPass.end();

        let shown = 2;
        if (useTemporal) {
            const read = this.historyIndex;
            const write = 1 - read;
            const temporalPass = encoder.beginRenderPass({
                label: "temporal pass",
                timestampWrites: profiler?.passWrites("temporal"),
                colorAttachments: [{
                    view: this.historyViews[write],
                    loadOp: "clear",
                    storeOp: "store"
                }]
            });

            temporalPass.setPipeline(this.temporalPipeline);
            temporalPass.setBindGroup(0, this.temporalBindGroups[read]);
            temporalPass.draw(3);
            temporalPass.end();

            shown = write;
            this.historyIndex = write;
            this.historyValid = true;
        }

        const compositePass = encoder.beginRenderPass({
            label: "composite pass",
            timestampWrites: profiler?.passWrites("composite"),
            colorAttachments: [{
                view: renderer.context.getCurrentTexture().createView(),
                loadOp: "clear",
                storeOp: "store"
            }]
        });
        compositePass.setPipeline(this.compositePipeline);
        compositePass.setBindGroup(0, this.compositeBindGroups[shown]);
        compositePass.draw(3);
        compositePass.end();
        debris.draw(encoder, gis.sceneDepthTexture);

        profiler?.resolve(encoder);
        device.queue.submit([encoder.finish()]);
        solver.storm.observe(fields, this.stage.weather.atmosphereBuffer);
        profiler?.afterSubmit();
    }
}
