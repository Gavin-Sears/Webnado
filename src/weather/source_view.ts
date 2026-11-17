import { GUI } from 'lil-gui';
import { constants } from '../shaders/shaders';
import type { WeatherData } from './noaa_loader';

export function buildSourceView(gui: GUI, weather: WeatherData, seconds: () => number) {
    if (!weather.spatial) return;
    const panel = document.createElement('div');
    panel.style.cssText = 'position:fixed;bottom:24px;left:12px;background:#14202bee;color:white;padding:8px;font:12px sans-serif;z-index:2';
    panel.hidden = true;
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 64;
    canvas.style.cssText = 'width:256px;height:256px;display:block;image-rendering:pixelated';
    const caption = document.createElement('div');
    panel.append(caption, canvas);
    document.body.append(panel);
    const state = { visible: false, field: 'cloud', height: 1000, refresh };
    const folder = gui.addFolder('Source weather comparison');
    folder.add(state, 'visible').name('Show source slice').onChange(refresh);
    folder.add(state, 'field', ['cloud', 'vertical wind', 'temperature']).onChange(refresh);
    folder.add(state, 'height', 0, constants.domainY, 100).name('Height above floor (m)').onFinishChange(refresh);
    folder.add(state, 'refresh').name('Snapshot at current time');
    function refresh() {
        panel.hidden = !state.visible;
        if (!state.visible || !weather.spatial) return;
        const source = weather.spatial;
        const time = Math.max(0, Math.min(seconds(), (source.manifest.frames - 1) * source.manifest.intervalSeconds));
        const frame = Math.min(Math.floor(time / source.manifest.intervalSeconds), source.manifest.frames - 2);
        const blend = time / source.manifest.intervalSeconds - frame;
        const values: number[] = [];
        for (let z = 0; z < 64; z++) {
            for (let x = 0; x < 64; x++) {
                const east = ((x + .5) / 64 - .5) * constants.domainX;
                const south = ((z + .5) / 64 - .5) * constants.domainZ;
                const a = source.sample(east, state.height, south, frame);
                const b = source.sample(east, state.height, south, frame + 1);
                const key = state.field === 'cloud' ? 'cloud' : state.field === 'temperature' ? 'temperature' : 'vertical';
                values.push(a[key] * (1 - blend) + b[key] * blend);
            }
        }
        const low = Math.min(...values), high = Math.max(...values);
        const context = canvas.getContext('2d')!;
        const pixels = context.createImageData(64, 64);
        values.forEach((value, index) => {
            const t = high > low ? (value - low) / (high - low) : .5;
            pixels.data.set([255 * t, 100, 255 * (1 - t), 255], index * 4);
        });
        context.putImageData(pixels, 0, 0);
        const unit = state.field === 'cloud' ? 'g/kg' : state.field === 'temperature' ? 'K' : 'm/s';
        caption.textContent = `HRRR model source | north up | ${time.toFixed(0)} s\n${state.field}: blue ${low.toFixed(2)} -> red ${high.toFixed(2)} ${unit}`;
        caption.style.whiteSpace = 'pre-line';
    }
}
