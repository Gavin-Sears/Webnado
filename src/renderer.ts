import { Profiler } from './profiler';
import { Stage } from './stage/stage';

export var canvas: HTMLCanvasElement;
export var canvasFormat: GPUTextureFormat;
export var context: GPUCanvasContext;
export var device: GPUDevice;

export var aspectRatio: number;

export var profiler: Profiler | undefined;

const MAX_FRAMES_IN_FLIGHT = 2;

let stopped = false;

// Only the first failure is shown
export function showFatalError(message: string) {
    if (stopped) {
        return;
    }
    stopped = true;

    let errorMessageElement = document.createElement("h1");
    errorMessageElement.textContent = message;
    errorMessageElement.style.paddingLeft = '0.4em';
    errorMessageElement.style.color = '#fff';
    document.body.innerHTML = '';
    document.body.appendChild(errorMessageElement);
}

const MAX_GPU_ERROR_MESSAGES = 20;
const gpuErrorMessages = new Set<string>();
let gpuErrorCount = 0;
let gpuErrorBanner: HTMLElement | undefined;

// Logs each message once and keeps a count on screen
function reportGpuError(event: GPUUncapturedErrorEvent) {
    const message = event.error.message;
    gpuErrorCount++;

    if (!gpuErrorMessages.has(message) && gpuErrorMessages.size < MAX_GPU_ERROR_MESSAGES) {
        gpuErrorMessages.add(message);
        console.error(`WebGPU error: ${message}`);
    }

    if (!gpuErrorBanner) {
        gpuErrorBanner = document.createElement("pre");
        gpuErrorBanner.style.cssText = "position:fixed;bottom:0;left:0;right:0;margin:0;padding:6px;font:12px monospace;color:#fff;background:#a00c;white-space:pre-wrap;pointer-events:none";
        document.body.appendChild(gpuErrorBanner);
    }
    gpuErrorBanner.textContent = `WebGPU errors: ${gpuErrorCount}. Latest:\n${message}`;
}

function resizeCanvas(entries?: ResizeObserverEntry[]) {
    const exactSize = entries?.[entries.length - 1].devicePixelContentBoxSize?.[0];
    const width = exactSize?.inlineSize ?? Math.floor(canvas.clientWidth * window.devicePixelRatio);
    const height = exactSize?.blockSize ?? Math.floor(canvas.clientHeight * window.devicePixelRatio);

    const maxDim = device.limits.maxTextureDimension2D;
    canvas.width = Math.max(1, Math.min(width, maxDim));
    canvas.height = Math.max(1, Math.min(height, maxDim));

    aspectRatio = canvas.width / canvas.height;
}

// CHECKITOUT: initializes WebGPU and creates anything shared by all renderers
export async function initWebGPU() {
    canvas = document.getElementById("mainCanvas") as HTMLCanvasElement;

    if (!navigator.gpu)
    {
        showFatalError("This browser doesn't support WebGPU! Try using Google Chrome.");
        throw new Error("WebGPU not supported on this browser");
    }

    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter)
    {
        showFatalError("No appropriate GPU adapter found.");
        throw new Error("no appropriate GPUAdapter found");
    }

    const urlParams = new URLSearchParams(window.location.search);
    const profiling = urlParams.has("profile");

    // Three.js shares this device and can't add features later, so tile texture formats are requested here
    const wantedFeatures: GPUFeatureName[] = ["texture-compression-bc", "texture-compression-etc2", "texture-compression-astc"];
    if (profiling) {
        wantedFeatures.push("timestamp-query");
    }
    device = await adapter.requestDevice({
        requiredFeatures: wantedFeatures.filter(feature => adapter.features.has(feature))
    });

    device.addEventListener("uncapturederror", reportGpuError);

    device.lost.then((info) => {
        console.error(`WebGPU device lost (${info.reason}): ${info.message}`);
        showFatalError("The connection to the GPU was lost. Reload the page to restart.");
    });

    context = canvas.getContext("webgpu")!;
    canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device: device,
        format: canvasFormat,
    });

    resizeCanvas();
    const resizeObserver = new ResizeObserver(resizeCanvas);
    try {
        // this box also reports display scale changes
        resizeObserver.observe(canvas, { box: "device-pixel-content-box" });
    } catch {
        resizeObserver.observe(canvas); // browsers without that box
    }

    if (profiling) {
        profiler = new Profiler(device, adapter.info, canvas, urlParams);
    }

    console.log("WebGPU init successful");
}

export abstract class Renderer {
    protected stage: Stage;

    private prevTime = 0;
    private framesInFlight = 0;
    private frameRequestId: number;

    constructor(stage: Stage) {
        this.stage = stage;

        this.frameRequestId = requestAnimationFrame(this.onFrame);
    }

    // Subclasses that own GPU resources must destroy them here and call super.stop().
    // Simulation resources belong in Stage.
    stop(): void {
        cancelAnimationFrame(this.frameRequestId);
    }

    protected abstract draw(): void;

    private onFrameFinished = () => {
        this.framesInFlight--;
    };

    // CHECKITOUT: the main rendering loop
    // note that we use an arrow function so that scheduling a frame does not create a new closure
    private onFrame = (time: number) => {
        if (stopped) {
            return;
        }

        // This check stops us from queuing more frames when we have hit our max in flight frames
        if (this.framesInFlight >= MAX_FRAMES_IN_FLIGHT) {
            this.frameRequestId = requestAnimationFrame(this.onFrame);
            return;
        }
        this.framesInFlight++;

        // Use a try catch so that the screen doesn't just freeze when rendering fails
        try {
            profiler?.beginFrame(time);

            this.stage.camera.onFrame(this.prevTime === 0 ? 0 : time - this.prevTime);
            this.prevTime = time;

            this.draw();

            profiler?.endFrame(time);
        } catch (error) {
            console.error(error);
            showFatalError(`Rendering stopped: ${error}`);
            return;
        }

        device.queue.onSubmittedWorkDone().then(this.onFrameFinished);

        this.frameRequestId = requestAnimationFrame(this.onFrame);
    };
}
