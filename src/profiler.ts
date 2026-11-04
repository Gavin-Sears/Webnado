// CHECKITOUT: the benchmarking tool.
//
//   ?profile                    live readout and a capture that skips 120 frames and records 600
//   ?profile&warmup=60&frames=2000
//
// Captures are logged as JSON and stored in window.profileResult. For accurate numbers, launch Chrome with
//   --disable-gpu-vsync --disable-frame-rate-limit --enable-webgpu-developer-features
// The last flag turns off GPU time rounding in browsers that round to 100 us. The capture warns when times look rounded.
//
// Usage for each timed pass:
//   encoder.beginComputePass({ timestampWrites: renderer.profiler?.passWrites("advection") });
//   ...
//   renderer.profiler?.resolve(encoder);             // after the last pass, before encoder.finish()
//   renderer.device.queue.submit([encoder.finish()]);
//   renderer.profiler?.afterSubmit();

export interface Stats {
    mean: number;   // mean frame time
    median: number; // median frame time
    p95: number;    // 95% of frames are faster than this
    count: number;  // number of frames/samples
}

const SAMPLE_CAPACITY = 4096;
const READOUT_INTERVAL_MS = 1000;
const TIMESTAMP_QUANTUM_MS = 0.1;

class SampleBuffer {
    private values = new Float64Array(SAMPLE_CAPACITY);
    private next = 0;
    private count = 0;

    add(value: number) {
        this.values[this.next] = value;
        this.next = (this.next + 1) % SAMPLE_CAPACITY;
        this.count = Math.min(this.count + 1, SAMPLE_CAPACITY);
    }

    reset() {
        this.next = 0;
        this.count = 0;
    }

    samples(): Float64Array {
        return this.values.subarray(0, this.count);
    }

    stats(): Stats {
        if (this.count === 0) {
            return { mean: NaN, median: NaN, p95: NaN, count: 0 };
        }
        const sorted = this.samples().slice().sort();
        const at = (q: number) => sorted[Math.min(this.count - 1, Math.floor(q * this.count))];
        return {
            mean: sorted.reduce((a, b) => a + b, 0) / this.count,
            median: at(0.5),
            p95: at(0.95),
            count: this.count
        };
    }
}

export class Profiler {
    // Timed passes per frame, which sizes the query set and its buffers. passWrites() skips any extras.
    private static readonly maxPasses = 16;
    // Limit on staging buffers. resolve() drops a readback when all of them are in flight.
    private static readonly maxReadBuffers = 16;

    private device: GPUDevice;
    private adapterInfo: GPUAdapterInfo;
    private canvas: HTMLCanvasElement;

    // Begin and end timestamps for each pass, with slots assigned by passWrites(). Undefined without timestamp-query.
    private querySet: GPUQuerySet | undefined;
    // Receives querySet's values in resolve(), then copies them to a staging buffer
    private resolveBuffer: GPUBuffer | undefined;
    // Staging buffers ready to reuse. resolve() takes one and afterSubmit() returns it once read.
    private freeReadBuffers: GPUBuffer[] = [];
    // Staging buffers created so far, checked against maxReadBuffers in resolve()
    private liveReadBuffers = 0;

    // Pass names from passWrites() this frame, in slot order. resolve() takes and clears them.
    private frameNames: string[] = [];
    // This frame's staging buffer and pass names. Set by resolve() and mapped by afterSubmit().
    private pendingCopy: { buffer: GPUBuffer, names: string[] } | undefined;

    // Time between beginFrame() calls. Reported as frameIntervalMs and shown in the readout.
    private frameInterval = new SampleBuffer();
    // CPU time from beginFrame() to endFrame(). Reported as cpuEncodeMs and shown in the readout.
    private cpuEncode = new SampleBuffer();
    // GPU time per pass name, added by recordPass(). Reported as gpuPassMs and shown in the readout.
    private passes = new Map<string, SampleBuffer>();
    // Readbacks skipped by resolve(). Reported in the capture.
    private droppedReadbacks = 0;
    // Pass names passWrites() couldn't time. Reported in the capture and shown in the readout.
    private skippedPasses = new Set<string>();

    // Previous frame's time from beginFrame(), used to measure frameInterval
    private prevTime = 0;
    // performance.now() from beginFrame(), used to measure cpuEncode
    private cpuStart = 0;

    // From url params. We discard this number of frames before collecting data
    private warmupFrames: number;
    // From url params. We capture this number of frames
    private captureFrames: number;
    // Frames completed, counted in endFrame() to find the capture's start and end
    private frameIndex = 0;

    // On-screen stats element, created in the constructor and filled by updateReadout()
    private readout: HTMLElement;
    // Time of the last readout update, used so that updating the readout only happens once per READOUT_INTERVAL_MS
    private lastReadout = 0;

    // Reads the capture settings, then creates the GPU timing resources and the readout
    constructor(device: GPUDevice, adapterInfo: GPUAdapterInfo, canvas: HTMLCanvasElement, params: URLSearchParams) {
        this.device = device;
        this.adapterInfo = adapterInfo;
        this.canvas = canvas;

        // capped to the sample buffer size
        this.warmupFrames = Number(params.get("warmup") ?? 120);
        this.captureFrames = Math.min(Number(params.get("frames") ?? 600), SAMPLE_CAPACITY);

        if (device.features.has("timestamp-query")) {
            this.querySet = device.createQuerySet({
                label: "profiler queries",
                type: "timestamp",
                count: Profiler.maxPasses * 2
            });
            this.resolveBuffer = device.createBuffer({
                label: "profiler resolve",
                size: Profiler.maxPasses * 2 * 8,
                usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
            });
        }

        // Creating our stats UI, then appending it to the doc.
        // This also gives us access to the text for updating in real time.
        this.readout = document.createElement("pre");
        this.readout.style.cssText = "position:fixed;top:0;left:0;margin:0;padding:6px;font:12px monospace;color:#0f0;background:#000a;pointer-events:none";
        document.body.appendChild(this.readout);
    }

    // This executes in the renderer at the beginning of a frame
    beginFrame(time: number) {
        if (this.prevTime !== 0) {
            this.frameInterval.add(time - this.prevTime);
        }
        this.prevTime = time;
        this.cpuStart = performance.now();
    }

    // This executes in the renderer after drawing the frame
    endFrame(time: number) {
        this.cpuEncode.add(performance.now() - this.cpuStart);

        this.frameIndex++;
        if (this.frameIndex === this.warmupFrames) {
            this.resetSamples();
        } else if (this.frameIndex === this.warmupFrames + this.captureFrames) {
            this.reportCapture();
            this.resetSamples();
        }

        if (time - this.lastReadout >= READOUT_INTERVAL_MS) {
            this.lastReadout = time;
            this.updateReadout();
        }
    }

    // For GPU time measured elsewhere, like Three.js's passes
    recordPass(name: string, ms: number) {
        let samples = this.passes.get(name);
        if (!samples) {
            samples = new SampleBuffer();
            this.passes.set(name, samples);
        }
        samples.add(ms);
    }

    // Use the result as `timestampWrites` in a pass descriptor
    passWrites(name: string): GPURenderPassTimestampWrites & GPUComputePassTimestampWrites | undefined {
        const passIdx = this.frameNames.length;
        if (!this.querySet) {
            return undefined;
        }
        if (passIdx >= Profiler.maxPasses) {
            this.skippedPasses.add(name);
            return undefined;
        }
        this.frameNames.push(name);
        return {
            querySet: this.querySet,
            beginningOfPassWriteIndex: passIdx * 2,
            endOfPassWriteIndex: passIdx * 2 + 1
        };
    }

    // Once per frame, after the last timed pass and before encoder.finish()
    resolve(encoder: GPUCommandEncoder) {
        const names = this.frameNames;
        this.frameNames = [];
        if (names.length === 0 || !this.querySet || !this.resolveBuffer) {
            return;
        }

        // When every staging buffer is in flight, skip and count this readback instead of creating more
        const buffer = this.freeReadBuffers.pop() ?? (this.liveReadBuffers < Profiler.maxReadBuffers ? this.createReadBuffer() : undefined);
        if (!buffer) {
            this.droppedReadbacks++;
            return;
        }

        encoder.resolveQuerySet(this.querySet, 0, names.length * 2, this.resolveBuffer, 0);
        encoder.copyBufferToBuffer(this.resolveBuffer, 0, buffer, 0, names.length * 2 * 8);
        this.pendingCopy = { buffer, names };
    }

    // Once per frame, after queue.submit()
    afterSubmit() {
        const pending = this.pendingCopy;
        this.pendingCopy = undefined;
        if (!pending) {
            return;
        }

        pending.buffer.mapAsync(GPUMapMode.READ).then(() => {
            const times = new BigUint64Array(pending.buffer.getMappedRange());
            pending.names.forEach((name, i) => {
                const begin = times[i * 2];
                const end = times[i * 2 + 1];
                if (end >= begin) {
                    this.recordPass(name, Number(end - begin) / 1e6);
                }
            });
            pending.buffer.unmap();
            this.freeReadBuffers.push(pending.buffer);
        }).catch(() => {
            // device lost, so drop the sample and buffer
            pending.buffer.destroy();
            this.liveReadBuffers--;
        });
    }

    // Creates a staging buffer the CPU can read. Called by resolve() when none are free.
    private createReadBuffer(): GPUBuffer {
        this.liveReadBuffers++;
        return this.device.createBuffer({
            label: "profiler readback",
            size: Profiler.maxPasses * 2 * 8,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });
    }

    // Clears all samples when the capture starts and after it reports
    private resetSamples() {
        this.frameInterval.reset();
        this.cpuEncode.reset();
        this.passes.forEach(samples => samples.reset());
        this.droppedReadbacks = 0;
        this.skippedPasses.clear();
    }

    // return object with mean, median, p95, and count for each gpu pass
    // Builds the capture's gpuPassMs, with stats for each pass name
    private passStats(): Record<string, Stats> {
        const result: Record<string, Stats> = {};
        this.passes.forEach((samples, name) => result[name] = samples.stats());
        return result;
    }

    // Collect warnings to return for JSON output
    private warnings(frame: Stats): string[] {
        const warnings: string[] = [];
        if (!this.querySet) {
            warnings.push("timestamp-query unavailable: no GPU pass times");
        }
        if (frame.median > 16 && frame.median < 17.4) {
            warnings.push("frame interval looks limited by vsync: launch Chrome with --disable-gpu-vsync --disable-frame-rate-limit");
        }
        // Zeros are skipped because they look the same with or without rounding
        const passSamples = [...this.passes.values()].flatMap(samples => [...samples.samples()]).filter(ms => ms !== 0);
        if (passSamples.length > 0 && passSamples.every(ms => Math.abs(ms / TIMESTAMP_QUANTUM_MS - Math.round(ms / TIMESTAMP_QUANTUM_MS)) < 1e-6)) {
            warnings.push("GPU times look rounded to 100 us: launch Chrome with --enable-webgpu-developer-features");
        }
        if (this.skippedPasses.size > 0) {
            warnings.push(`over the ${Profiler.maxPasses} pass limit, not timed: ${[...this.skippedPasses].join(", ")}`);
        }
        if (this.droppedReadbacks > 0) {
            warnings.push(`${this.droppedReadbacks} frames have no GPU times: all ${Profiler.maxReadBuffers} staging buffers were busy`);
        }
        return warnings;
    }

    // Updates the JSON variable you can access
    // to collect your performance stats, and logs
    // the object in the console as well.
    // Only gets used in endframe()
    private reportCapture() {
        const frame = this.frameInterval.stats();
        const result = {
            url: window.location.search,
            canvas: [this.canvas.width, this.canvas.height],
            adapter: { vendor: this.adapterInfo.vendor, architecture: this.adapterInfo.architecture, description: this.adapterInfo.description },
            frameIntervalMs: frame,
            cpuEncodeMs: this.cpuEncode.stats(),
            gpuPassMs: this.passStats(),
            droppedReadbacks: this.droppedReadbacks,
            warnings: this.warnings(frame)
        };
        (window as any).profileResult = result;
        console.log(JSON.stringify(result));
    }

    // Used in endframe to update UI reading text
    private updateReadout() {
        const frame = this.frameInterval.stats();
        const lines = [
            `frame ${frame.median.toFixed(2)} ms (${(1000 / frame.median).toFixed(0)} fps)`,
            `cpu   ${this.cpuEncode.stats().median.toFixed(3)} ms`
        ];
        this.passes.forEach((samples, name) => lines.push(`${name} ${samples.stats().median.toFixed(3)} ms`));
        if (!this.querySet) {
            lines.push("GPU timing unavailable");
        }
        if (this.skippedPasses.size > 0) {
            lines.push(`over pass limit: ${[...this.skippedPasses].join(", ")}`);
        }
        if (this.droppedReadbacks > 0) {
            lines.push(`dropped readbacks: ${this.droppedReadbacks}`);
        }
        this.readout.textContent = lines.join("\n");
    }
}
