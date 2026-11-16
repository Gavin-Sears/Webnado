import { GUI } from 'lil-gui';

import { device } from '../renderer';
import { constants } from '../shaders/shaders';
import type { Fields } from '../stage/fields';
import { StormDiagnostics, type StormReport } from './storm_diagnostics';

export class StormExperiment {
    enabled = false;
    elapsed = 0;
    liftDuration = 1200;
    liftSpeed = 10;
    radius = 10000;
    liftHeight = 1500;
    highOrder = true;
    balanceReference = new URLSearchParams(location.search).get('balance') !== 'off';
    followStorm = new URLSearchParams(location.search).get('follow') === '1';
    windowEast = 0;
    windowNorth = 0;
    shiftX = 0;
    shiftZ = 0;
    phase = 'Off';
    updraft = 0;
    surfaceVorticity = 0;
    groundSwirl = 0;
    rotationLocation: number[] | undefined;
    cloudOverheadSwirl = 0;
    cloudRotationLocation: number[] | undefined;
    divergence = 0;

    private diagnostics: StormDiagnostics | undefined;
    private pending = false;
    private sampledAt = -Infinity;

    readonly buffer = device.createBuffer({
        label: 'atmospheric experiment',
        size: 48,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    buildGui(gui: GUI, start: (domain: 'supercell' | 'storm-focus') => void) {
        const folder = gui.addFolder('Atmospheric storm');
        folder.add({ start: () => start('supercell') }, 'start').name('El Reno: full storm domain');
        folder.add({ start: () => start('storm-focus') }, 'start').name('El Reno: cropped storm window');
        if (this.enabled) {
            const choices = { 'Constant volume': 'volume', 'Stratified mass': 'anelastic' };
            const settings = { constraint: constants.anelasticAir ? 'anelastic' : 'volume' };
            folder.add(settings, 'constraint', choices).name('Air conservation').onChange((constraint: string) => {
                const url = new URL(location.href);
                url.searchParams.set('projection', constraint);
                location.href = url.href;
            });
        }
        folder.add(this, 'elapsed').name('Simulated seconds').listen().disable();
        folder.add(this, 'phase').name('Phase').listen().disable();
        folder.add(this, 'updraft').name('Peak updraft (m/s)').listen().disable();
        folder.add(this, 'surfaceVorticity').name('Core rotation below 1 km (1/s)').listen().disable();
        folder.add(this, 'groundSwirl').name('Ground swirling strength (1/s)').listen().disable();
        folder.add(this, 'cloudOverheadSwirl').name('Ground swirl under rising cloud (1/s)').listen().disable();
        folder.add(this, 'divergence').name('Constraint error RMS (1/s)').listen().disable();
        folder.add(this, 'liftDuration', 60, 1200, 30).name('Initial lifting duration (s)');
        folder.add(this, 'liftSpeed', 1, 20, 1).name('Initial updraft (m/s)');
        folder.add(this, 'highOrder').name('Higher-order advection');
        folder.add(this, 'followStorm').name('Move window (experimental)');
        folder.add(this, 'windowEast').name('Window east offset (m)').listen().disable();
        folder.add(this, 'windowNorth').name('Window north offset (m)').listen().disable();
    }

    write(dt: number) {
        device.queue.writeBuffer(this.buffer, 0, new Float32Array([
            Number(this.enabled), this.elapsed, this.liftDuration, this.liftSpeed,
            this.radius, this.liftHeight, this.shiftX, this.shiftZ,
            Math.min(14000, 0.8 * constants.domainY), 1 / 300, 0, Number(this.balanceReference)
        ]));
        if (this.enabled) {
            this.elapsed += dt;
        }
        this.phase = !this.enabled ? 'Off' : this.elapsed < this.liftDuration ? 'Initial lifting' : 'Free evolution';
    }

    observe(fields: Fields, atmosphere: GPUBuffer) {
        if (!this.enabled || this.pending || this.elapsed - this.sampledAt < 60) {
            return;
        }
        this.diagnostics ??= new StormDiagnostics(atmosphere);
        this.pending = true;
        this.sampledAt = this.elapsed;
        void this.diagnostics.measure(fields, this.elapsed).then(report => {
            this.updraft = report.updraft;
            this.surfaceVorticity = report.surfaceVorticity;
            this.groundSwirl = report.groundSwirlingStrength;
            this.rotationLocation = report.groundSwirlingStrength > 0.005 ? report.swirlLocation : undefined;
            this.cloudOverheadSwirl = report.cloudOverheadSwirl;
            this.cloudRotationLocation = report.cloudOverheadSwirl > 0.005 ? report.cloudOverheadLocation : undefined;
            this.divergence = report.divergenceRms;
            this.track(report);
        }).catch(error => console.error('Storm diagnostics failed', error)).finally(() => {
            this.pending = false;
        });
    }

    track(report: StormReport) {
        if (!this.followStorm || report.seconds < this.liftDuration || report.rotatingUpdraftScore <= 0) {
            return;
        }
        const [x, , z] = report.updraftLocation;
        this.shiftX = Math.abs(x) > 0.2 * constants.domainX
            ? Math.round(x * constants.gridX / constants.domainX) : 0;
        this.shiftZ = Math.abs(z) > 0.2 * constants.domainZ
            ? Math.round(z * constants.gridZ / constants.domainZ) : 0;
    }

    resetTracking() {
        this.groundSwirl = 0;
        this.rotationLocation = undefined;
        this.cloudOverheadSwirl = 0;
        this.cloudRotationLocation = undefined;
        this.shiftX = 0;
        this.shiftZ = 0;
        this.windowEast = 0;
        this.windowNorth = 0;
        this.sampledAt = -Infinity;
    }

    destroy() {
        this.buffer.destroy();
        this.diagnostics?.destroy();
    }
}
