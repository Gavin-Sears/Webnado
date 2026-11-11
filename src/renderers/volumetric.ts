import { GUI } from 'lil-gui';

import * as renderer from '../renderer';
import * as shaders from '../shaders/shaders';
import { Stage } from '../stage/stage';
import { fillTestVolume } from './test_volume';

// Draws the clouds in fields.thermo over the Three.js scene. Each frame, draw() runs gis.render(camera),
// solver.step(encoder), then three passes of its own:
//   1. bricks      bricks.cs.wgsl      finds largest density per 8^3 brick, so rays can skip empty space
//   2. raymarch    raymarch.fs.wgsl    single scattering into a target smaller than the canvas: rgb light, a transmittance
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
    off: 0, density: 1, temperature: 2, velocity: 3, vorticity: 4, obstacles: 5, bricks: 6
};

// Starting scale and max value for debug views
const DEBUG_SCALES: Record<string, { value: number, max: number, unit: string }> = {
    density: { value: 3, max: 10, unit: 'g/kg' },
    densityBoosted: { value: 0.1, max: 2, unit: 'g/kg' },
    temperature: { value: 1, max: 10, unit: 'K' },          // theta'. The thermal starts at +2 K.
    velocity: { value: 5, max: 50, unit: 'm/s' },
    vorticity: { value: 0.1, max: 1, unit: '1/s' }
};

const VOLUME_FORMAT: GPUTextureFormat = "rgba16float";
const PARAMS_SIZE = 96; // RaymarchParams in raymarch.fs.wgsl

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
    animateJitter = false;      // M3: turn on once temporal reprojection averages the noise over frames
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
    private bricksBindGroups: GPUBindGroup[];   // by thermo.index

    // The raymarch target, recreated when the canvas or resolution scale changes
    private volumeTexture: GPUTexture | undefined;
    private volumeView: GPUTextureView | undefined;

    // Rebuilt when a resize changes the textures they hold
    private boundColor: GPUTexture | undefined;
    private boundDepth: GPUTexture | undefined;
    private boundVolume: GPUTexture | undefined;
    private raymarchBindGroups: GPUBindGroup[] = [];    // by 2 * thermo.index + velocity.index
    private compositeBindGroup: GPUBindGroup | undefined;

    constructor(stage: Stage) {
        super(stage);
        const device = renderer.device;

        const requested = new URLSearchParams(window.location.search).get('quality') ?? '';
        this.qualityName = Object.hasOwn(qualityPresets, requested) ? requested : platformQuality();
        this.quality = { ...qualityPresets[this.qualityName] };

        const vertex = { module: device.createShaderModule({ label: "fullscreen vert shader", code: shaders.fullscreenVertSrc }) };
        const fragment = (label: string, code: string, format: GPUTextureFormat): GPUFragmentState => ({
            module: device.createShaderModule({ label: `${label} frag shader`, code }),
            targets: [{ format }]
        });
        this.raymarchPipeline = device.createRenderPipeline({
            label: "raymarch",
            layout: "auto",
            vertex,
            fragment: fragment("raymarch", shaders.raymarchFragSrc, VOLUME_FORMAT)
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
        this.bricksBindGroups = [0, 1].map(i => device.createBindGroup({
            label: `bricks ${i}`,
            layout: this.bricksPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: stage.fields.thermo.view(i) },
                { binding: 1, resource: this.brickView }
            ]
        }));

        fillTestVolume(stage.fields);
    }

    buildGui(gui: GUI) {
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

        const lighting = folder.addFolder('Lighting');
        lighting.add(this.lighting, 'sunAzimuth', -180, 180, 1).name('sun azimuth');
        lighting.add(this.lighting, 'sunElevation', 0, 90, 1).name('sun elevation');
        lighting.addColor(this.lighting, 'sunColor').name('sun color');
        lighting.add(this.lighting, 'sunIntensity', 0, 5, 0.05).name('sun intensity');
        lighting.addColor(this.lighting, 'ambientColor').name('ambient color');
        lighting.add(this.lighting, 'ambientIntensity', 0, 2, 0.05).name('ambient intensity');
        lighting.add(this.lighting, 'phaseG', -0.9, 0.9, 0.01).name('phase g');
        lighting.add(this.lighting, 'extinction', 0.0002, 0.01, 0.0001);

        const debug = folder.addFolder('Debug');
        const view = debug.add(this, 'debugView', Object.keys(DEBUG_VIEWS)).name('view');
        const boost = debug.add(this, 'debugBoost').name('boost faint cloud');
        const scale = debug.add(this, 'debugScale', 0, 1);
        const showScale = () => {
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

    override stop() {
        super.stop();
        this.volumeTexture?.destroy();
        this.volumeTexture = undefined;
        this.brickTexture.destroy();
        this.paramsBuffer.destroy();
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
    }

    private updateBindGroups() {
        const { gis, camera, fields } = this.stage;
        const device = renderer.device;
        const color = gis.sceneColorTexture;
        const depth = gis.sceneDepthTexture;

        if (depth !== this.boundDepth) {
            const depthView = depth.createView();
            this.raymarchBindGroups = [0, 1, 2, 3].map(i => device.createBindGroup({
                label: `raymarch ${i}`,
                layout: this.raymarchPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: camera.uniformsBuffer } },
                    { binding: 1, resource: { buffer: this.paramsBuffer } },
                    { binding: 2, resource: fields.thermo.view(i >> 1) },
                    { binding: 3, resource: this.linearSampler },
                    { binding: 4, resource: depthView },
                    { binding: 5, resource: this.brickView },
                    { binding: 6, resource: fields.velocity.view(i & 1) },
                    { binding: 7, resource: fields.obstacleView }
                ]
            }));
        }

        if (color !== this.boundColor || depth !== this.boundDepth || this.volumeTexture !== this.boundVolume) {
            this.compositeBindGroup = device.createBindGroup({
                label: "composite",
                layout: this.compositePipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: camera.uniformsBuffer } },
                    { binding: 1, resource: color.createView() },
                    { binding: 2, resource: depth.createView() },
                    { binding: 3, resource: this.volumeView! }
                ]
            });
        }

        this.boundColor = color;
        this.boundDepth = depth;
        this.boundVolume = this.volumeTexture;
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
        renderer.device.queue.writeBuffer(this.paramsBuffer, 0, this.paramsData);
    }

    override draw() {
        const { gis, camera, solver, fields } = this.stage;
        const device = renderer.device;
        const profiler = renderer.profiler;

        this.updateVolumeTarget();
        gis.render(camera); // Three submits this itself
        this.updateBindGroups();
        this.writeParams();

        const encoder = device.createCommandEncoder({ label: "frame" });

        solver.step(encoder);

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
            }]
        });
        raymarchPass.setPipeline(this.raymarchPipeline);
        raymarchPass.setBindGroup(0, this.raymarchBindGroups[2 * fields.thermo.index + fields.velocity.index]);
        raymarchPass.draw(3);
        raymarchPass.end();

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
        compositePass.setBindGroup(0, this.compositeBindGroup!);
        compositePass.draw(3);
        compositePass.end();

        profiler?.resolve(encoder);
        device.queue.submit([encoder.finish()]);
        profiler?.afterSubmit();
    }
}
