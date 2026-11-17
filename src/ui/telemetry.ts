import type GUI from 'lil-gui';
import type { Stage } from '../stage/stage';
import { constants } from '../shaders/shaders';

export function buildTelemetry(gui: GUI, stage: Stage) {
    const header = document.createElement('section');
    header.className = 'mission-overview';
    header.setAttribute('aria-label', 'Simulation telemetry');
    const topline = document.createElement('div');
    topline.className = 'mission-topline';
    const clock = document.createElement('span');
    clock.className = 'mission-clock';
    const status = document.createElement('span');
    status.className = 'mission-status';
    topline.append(clock, status);

    const context = document.createElement('div');
    context.className = 'mission-context';
    const grid = document.createElement('div');
    grid.className = 'mission-metrics';
    function metric(label: string, unit: string) {
        const cell = document.createElement('div');
        const name = document.createElement('span');
        name.className = 'metric-label';
        name.textContent = label;
        const value = document.createElement('span');
        value.className = 'metric-value';
        const suffix = document.createElement('span');
        suffix.className = 'metric-unit';
        suffix.textContent = unit;
        cell.append(name, value, suffix);
        grid.append(cell);
        return value;
    }
    const dt = metric('TIME STEP', 's');
    const updraft = metric('UPDRAFT', 'm/s');
    const wind = metric('RING WIND', 'm/s');
    grid.title = 'Storm measurements are the latest diagnostic snapshots. A dash means no active storm experiment.';
    header.append(topline, context, grid);
    gui.domElement.insertBefore(header, gui.$children);

    const footer = document.createElement('div');
    footer.className = 'mission-footer';
    const system = document.createElement('span');
    system.textContent = `WEBGPU / ${constants.gridX} × ${constants.gridY} × ${constants.gridZ}`;
    const guidance = document.createElement('button');
    guidance.textContent = 'GUIDANCE OFF';
    guidance.setAttribute('aria-label', 'Show control explanations');
    guidance.setAttribute('aria-pressed', 'false');
    guidance.addEventListener('click', () => {
        const visible = gui.domElement.classList.toggle('show-guidance');
        guidance.textContent = visible ? 'GUIDANCE ON' : 'GUIDANCE OFF';
        guidance.setAttribute('aria-pressed', String(visible));
    });
    footer.append(system, guidance);
    gui.domElement.append(footer);

    function write(element: HTMLElement, value: string) {
        if (element.textContent !== value) element.textContent = value;
    }
    function update() {
        if (!gui.domElement.isConnected) return;
        const { solver, weather } = stage;
        const storm = solver.storm;
        const paused = solver.paused;
        const syncing = !paused && storm.waitingForTracking;
        header.dataset.state = paused ? 'paused' : syncing ? 'sync' : 'running';
        write(status, paused ? 'PAUSED' : syncing ? 'SYNCING' : 'RUNNING');
        const seconds = Math.floor(storm.elapsed);
        const hours = Math.floor(seconds / 3600).toString().padStart(2, '0');
        const minutes = Math.floor(seconds / 60 % 60).toString().padStart(2, '0');
        const remainder = (seconds % 60).toString().padStart(2, '0');
        write(clock, storm.enabled ? `T+ ${hours}:${minutes}:${remainder}` : 'LOCAL / SOUNDING');
        const mode = weather.spatial ? 'HRRR 3D' : weather.scenario === 'elreno' ? 'EL RENO / TRIGGERED' : 'VERTICAL SOUNDING';
        const domain = [constants.domainX, constants.domainY, constants.domainZ].map(size => size / 1000).join(' × ');
        write(context, `${mode}  /  ${domain} KM`);
        write(dt, solver.dt.toFixed(2));
        write(updraft, storm.enabled ? storm.updraft.toFixed(1) : '—');
        write(wind, storm.enabled ? storm.rotatingWind.toFixed(1) : '—');
        window.setTimeout(update, 250);
    }
    update();
}
