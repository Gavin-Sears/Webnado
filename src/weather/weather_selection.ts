import type GUI from 'lil-gui';
import type { WeatherData } from './noaa_loader';
import { help } from '../ui/help';

export function experimentUrl(mode: 'hrrr' | 'elreno' | 'sounding', resolution: string, manifest = '') {
    // Start clean so settings from a different experiment cannot leak into a case.
    const url = new URL(location.pathname, location.origin);
    if (new URLSearchParams(location.search).has('profile')) url.searchParams.set('profile', '');
    const settings: Record<string, string> = mode === 'sounding'
        ? { res: resolution, domain: 'local' }
        : { case: mode, res: resolution, domain: 'storm-focus', precision: '32', state: '32', projection: 'anelastic', tke: 'deardorff' };
    if (mode === 'hrrr') Object.assign(settings, { weather: manifest, terrain: 'flat', follow: '1' });
    if (mode === 'elreno') {
        Object.assign(settings, { frame: 'bunkers', lift: '10', duration: '1200', state: '16', tke: 'legacy' });
    }
    for (const [key, value] of Object.entries(settings)) url.searchParams.set(key, value);
    return url;
}

export async function buildWeatherSelection(gui: GUI, weather: WeatherData) {
    const folder = gui.addFolder('Choose an experiment');
    const state = { case: '', resolution: 'medium', status: 'Loading prepared weather…' };
    const description = help(folder, 'Prepared 3D weather contains a modeled storm. A sounding is one vertical profile, shared across the domain.');
    const choices: Record<string, string> = {};
    const descriptions = new Map<string, string>();
    const catalogUrl = new URL(`${import.meta.env.BASE_URL}weather/catalog.json`, location.origin);
    try {
        const response = await fetch(catalogUrl);
        if (!response.ok) throw new Error('Catalog unavailable');
        const cases = await response.json() as { name: string; manifest: string }[];
        for (const item of cases) {
            const manifest = new URL(item.manifest, catalogUrl).href;
            choices[item.name] = manifest;
            descriptions.set(manifest, `${item.name}. NOAA HRRR 3D model weather; no added thermal or prescribed updraft. Rotation may develop, but a ground-reaching tornado is not guaranteed.`);
        }
        state.status = cases.length ? 'Prepared weather ready' : 'No prepared weather on this deployment';
    } catch {
        state.status = 'Prepared weather unavailable; sounding experiments still work';
    }
    choices['El Reno · May 24, 2011 (triggered)'] = 'elreno';
    choices['Custom sounding / local sandbox'] = 'sounding';
    descriptions.set('elreno', 'Idealized El Reno sounding experiment. A nonrotating updraft starts the storm for 20 simulated minutes, then fades. This is not weather-only replay.');
    descriptions.set('sounding', 'Explore any location with a vertical sounding from Open-Meteo, or try the local thermal and shear experiments. This does not import an existing 3D storm.');
    const current = new URLSearchParams(location.search);
    state.case = current.get('case') === 'hrrr' ? current.get('weather') ?? ''
        : current.get('case') === 'elreno' ? 'elreno' : 'sounding';
    if (state.case && !Object.values(choices).includes(state.case)) {
        choices['Current custom 3D weather'] = state.case;
    }
    state.resolution = current.has('case') && current.has('res') ? current.get('res')! : 'medium';
    if (!['medium', 'high'].includes(state.resolution)) state.resolution = 'medium';
    const updateDescription = () => {
        description.textContent = descriptions.get(state.case) ?? 'Custom 3D HRRR weather. The outcome depends on the supplied weather and grid.';
    };
    folder.add(state, 'case', choices).name('Experiment').onChange(updateDescription);
    folder.add(state, 'resolution', { 'Preview · medium grid': 'medium', 'Research · high grid': 'high' }).name('Simulation quality');
    help(folder, 'Research uses the grid from our Iowa and Kansas checks and needs more GPU memory. Preview is cheaper; it may evolve differently. Loading restarts the page.');
    folder.add({ load: () => {
        const mode = state.case === 'elreno' || state.case === 'sounding' ? state.case : 'hrrr';
        location.assign(experimentUrl(mode, state.resolution, state.case).href);
    } }, 'load').name('Load selected experiment');
    const status = help(folder, state.status);
    status.setAttribute('role', 'status');
    updateDescription();

    const custom = gui.addFolder('Another location');
    help(custom, 'Soundings work for other locations today. A full 3D HRRR storm needs prepared data or a connected weather service; choosing a tornado location alone does not recreate its tornado.');
    if (weather.spatial || weather.scenario === 'elreno') {
        custom.add({ open: () => location.assign(experimentUrl('sounding', 'medium').href) }, 'open').name('Open custom sounding mode');
    } else {
        custom.add(weather.site, 'latitude', -90, 90, 0.001).name('Latitude');
        custom.add(weather.site, 'longitude', -180, 180, 0.001).name('Longitude');
        custom.add(weather, 'date').name('Date (YYYY-MM-DD)');
        custom.add(weather, 'hour', 0, 23, 1).name('Hour (UTC)');
        custom.add(weather, 'load').name('Load sounding');
        custom.add(weather, 'status').name('Weather status').listen().disable();
    }
    custom.close();

    const hrrr = custom.addFolder('Full 3D weather');
    const remote = { manifest: '' };
    const weatherStatus = help(hrrr, 'Load a hosted HRRR manifest, or use the weather service below if connected.');
    weatherStatus.setAttribute('role', 'status');
    hrrr.add(remote, 'manifest').name('Manifest URL');
    hrrr.add({ load: () => {
        try {
            const url = new URL(remote.manifest);
            if (!['http:', 'https:'].includes(url.protocol)) throw new Error('Use an HTTP or HTTPS manifest URL');
            location.assign(experimentUrl('hrrr', state.resolution, url.href).href);
        } catch {
            weatherStatus.textContent = 'Enter a full HTTP or HTTPS weather manifest URL.';
        }
    } }, 'load').name('Load 3D weather URL');
    const endpoint = import.meta.env.VITE_WEATHER_API_URL as string | undefined;
    if (!endpoint) {
        help(hrrr, 'This static website has no HRRR preparation service connected. Use a prepared case above or a hosted manifest URL. Python preparation runs outside the browser.');
        return;
    }
    help(hrrr, 'Prepare a CONUS crop from NOAA HRRR. Dates from December 2, 2020 onward; availability varies. This downloads and prepares data on the weather server.');
    const request = { latitude: weather.site.latitude, longitude: weather.site.longitude,
        date: weather.date, hour: 18, forecast: 1 };
    hrrr.add(request, 'latitude', 25, 49, 0.001).name('Center latitude');
    hrrr.add(request, 'longitude', -124, -67, 0.001).name('Center longitude');
    hrrr.add(request, 'date').name('Cycle date (UTC)');
    hrrr.add(request, 'hour', 0, 23, 1).name('Cycle hour (UTC)');
    hrrr.add(request, 'forecast', 1, 16, 1).name('First forecast hour');
    const api = new URL(endpoint.endsWith('/') ? endpoint : endpoint + '/', location.href);
    let busy = false;
    const prepare = hrrr.add({ prepare: async () => {
        if (busy) return;
        busy = true;
        prepare.disable();
        try {
            weatherStatus.textContent = 'Requesting 3D weather…';
            const response = await fetch(new URL('jobs', api), { method: 'POST',
                headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(request) });
            let job = await response.json();
            if (!response.ok) throw new Error(job.error ?? 'Weather request failed');
            while (job.status !== 'ready') {
                if (job.status === 'failed') throw new Error(job.message);
                weatherStatus.textContent = job.message ?? 'Preparing weather';
                await new Promise(resolve => setTimeout(resolve, 2000));
                const poll = await fetch(new URL(`jobs/${job.id}`, api));
                job = await poll.json();
                if (!poll.ok) throw new Error(job.error ?? 'Weather worker unavailable');
            }
            location.assign(experimentUrl('hrrr', state.resolution, new URL(job.manifest, api).href).href);
        } catch (error) {
            weatherStatus.textContent = error instanceof Error ? error.message : String(error);
        } finally {
            busy = false;
            prepare.enable();
        }
    } }, 'prepare').name('Prepare and load 3D weather');
}
