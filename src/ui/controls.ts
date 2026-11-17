import GUI from 'lil-gui';
import type { Stage } from '../stage/stage';
import type { VolumetricRenderer } from '../renderers/volumetric';
import { constants, domainName, domains, resolutionName, resolutions, tkeModelName } from '../shaders/shaders';
import { buildWeatherSelection } from '../weather/weather_selection';
import { buildSourceView } from '../weather/source_view';
import { help } from './help';
import './controls.css';

export function buildControls(stage: Stage, renderer: VolumetricRenderer) {
    const gui = new GUI({ title: 'Webnado', width: 370 });
    gui.domElement.classList.add('webnado-controls');
    const tabs = document.createElement('nav');
    tabs.className = 'control-tabs';
    tabs.setAttribute('aria-label', 'Control workspace');
    gui.domElement.insertBefore(tabs, gui.$children);

    const run = gui.addFolder('Run');
    const inspect = gui.addFolder('Inspect');
    const advanced = gui.addFolder('Advanced');
    const workspaces = { Run: run, Inspect: inspect, Advanced: advanced };
    for (const [name, folder] of Object.entries(workspaces)) {
        folder.$title.hidden = true;
        const button = document.createElement('button');
        button.textContent = name;
        button.addEventListener('click', () => {
            for (const [label, workspace] of Object.entries(workspaces)) {
                workspace.show(label === name);
            }
            for (const tab of tabs.querySelectorAll('button')) {
                tab.setAttribute('aria-pressed', String(tab === button));
            }
        });
        tabs.append(button);
    }
    (tabs.firstElementChild as HTMLButtonElement).click();

    help(run, `Current mode: ${stage.weather.spatial ? '3D HRRR weather' : stage.weather.scenario === 'elreno' ? 'El Reno sounding experiment' : 'local sounding / sandbox'}.`);
    help(run, 'Choose a weather case, then let the storm evolve. Use Inspect to see airflow and rotation; Advanced contains the full controls.');
    void buildWeatherSelection(run, stage.weather);
    const playback = run.addFolder('Playback');
    playback.add(stage.solver, 'paused').name('Pause simulation').listen();
    playback.add(stage.solver, 'reset').name('Restart current simulation');
    if (stage.solver.storm.enabled) {
        playback.add(stage.solver.storm, 'elapsed').name('Simulation time (s)').listen().disable();
        help(playback, 'Weather cases start from the beginning, not a saved vortex. The measured Iowa and Kansas runs evolved for about 4,300–4,800 simulation seconds.');
    } else {
        playback.add(stage.solver, 'addThermal').name('Add warm thermal');
        playback.add(stage.solver, 'startShearExperiment').name('Start shear experiment');
        help(playback, 'A warm thermal adds a small warm, moist bubble. A sounding alone may not produce visible cloud.');
    }

    help(advanced, 'These controls change the experiment. Loading a weather case restores its baseline settings.');
    help(inspect, 'Colors show a diagnostic field, not a photograph of a tornado. Change the scale if a view becomes saturated.');
    renderer.buildGui(advanced, inspect);
    buildSourceView(inspect, stage.weather, () => stage.solver.storm.elapsed);
    if (stage.solver.storm.enabled) {
        const measurements = inspect.addFolder('Storm measurements');
        measurements.add(stage.solver.storm, 'rotatingWind').name('Ring wind (m/s)').listen().disable();
        measurements.add(stage.solver.storm, 'rotatingWindHeight').name('Ring height (m)').listen().disable();
        measurements.add(stage.solver.storm, 'groundSwirl').name('Swirling strength (1/s)').listen().disable();
        help(measurements, 'These measurements do not establish ground contact or an EF rating.');
        measurements.close();
    }
    const profile = { enabled: new URLSearchParams(location.search).has('profile') };
    inspect.add(profile, 'enabled').name('GPU timing overlay (reload)').onChange((enabled: boolean) => {
        const url = new URL(location.href);
        if (enabled) url.searchParams.set('profile', '');
        else url.searchParams.delete('profile');
        location.assign(url.href);
    });

    const grid = advanced.addFolder('Grid and model (reload)');
    const settings = {
        res: resolutionName, domain: domainName, tke: tkeModelName,
        precision: constants.velocityFormat === 'rgba32float' ? '32' : '16',
        state: constants.preciseScalars ? '32' : '16'
    };
    const reload = (key: string, value: string) => {
        const url = new URL(location.href);
        url.searchParams.set(key, value);
        location.assign(url.href);
    };
    grid.add(settings, 'res', Object.keys(resolutions)).name('Grid resolution').onChange((value: string) => reload('res', value));
    grid.add(settings, 'domain', Object.keys(domains)).name('Domain size').onChange((value: string) => reload('domain', value));
    help(grid, 'Local: 3.2 × 1.6 × 3.2 km. Storm: 12.8 × 6.4 × 12.8 km. Full storm: 64 × 18 × 64 km. Storm focus: 32 × 18 × 32 km.');
    grid.add(settings, 'precision', { '16 bit': '16', '32 bit': '32' }).name('Velocity precision').onChange((value: string) => reload('precision', value));
    grid.add(settings, 'state', { '16 bit': '16', '32 bit': '32' }).name('Scalar precision').onChange((value: string) => reload('state', value));
    grid.add(settings, 'tke', { Legacy: 'legacy', Deardorff: 'deardorff' }).name('TKE coefficients').onChange((value: string) => reload('tke', value));
    stage.weather.buildGui(advanced);
    stage.gis.buildGui(advanced);
    stage.solver.buildGui(advanced);
    stage.debris.buildGui(advanced, stage.solver.storm.enabled
        ? () => stage.solver.storm.cloudRotationLocation ?? stage.solver.storm.rotationLocation : undefined);
    for (const folder of advanced.folders) folder.close();
}
