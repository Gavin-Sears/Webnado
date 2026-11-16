import { GUI } from 'lil-gui';

import { device } from '../renderer';

export class StormExperiment {
    enabled = false;
    elapsed = 0;
    liftDuration = 600;
    liftSpeed = 10;
    radius = 5000;
    liftHeight = 2000;

    readonly buffer = device.createBuffer({
        label: 'atmospheric experiment',
        size: 48,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    buildGui(gui: GUI, start: () => void) {
        const folder = gui.addFolder('Atmospheric storm');
        folder.add({ start }, 'start').name('Start El Reno experiment');
        folder.add(this, 'elapsed').name('Simulated seconds').listen().disable();
        folder.add(this, 'liftDuration', 60, 1200, 30).name('Initial lifting duration (s)');
        folder.add(this, 'liftSpeed', 1, 20, 1).name('Initial updraft (m/s)');
    }

    write(dt: number) {
        device.queue.writeBuffer(this.buffer, 0, new Float32Array([
            Number(this.enabled), this.elapsed, this.liftDuration, this.liftSpeed,
            this.radius, this.liftHeight, 0, 0,
            14000, 1 / 300, 0, 0
        ]));
        if (this.enabled) {
            this.elapsed += dt;
        }
    }
}
