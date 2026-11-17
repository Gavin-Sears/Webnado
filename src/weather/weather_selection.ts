import type GUI from 'lil-gui';
import type { WeatherData } from './noaa_loader';

export async function buildWeatherSelection(gui: GUI, weather: WeatherData) {
    const folder = gui.addFolder('3D storm weather');
    const state = { status: 'Loading prepared cases', case: '', date: weather.date,
        hour: 18, forecast: 1, latitude: weather.site.latitude, longitude: weather.site.longitude };
    folder.add(state, 'status').name('Status').listen().disable();
    const open = (manifest: string) => {
        const url = new URL(location.href);
        for (const [key, value] of Object.entries({ case: 'hrrr', weather: manifest,
            domain: 'storm-focus', terrain: 'flat', follow: '1', state: '32', precision: '32', tke: 'deardorff' })) {
            url.searchParams.set(key, value);
        }
        location.assign(url.href);
    };
    const catalogUrl = new URL(`${import.meta.env.BASE_URL}weather/catalog.json`, location.origin);
    try {
        const response = await fetch(catalogUrl);
        if (!response.ok) throw new Error('Catalog unavailable');
        const cases = await response.json() as { name: string; manifest: string }[];
        if (!cases.length) throw new Error('Catalog empty');
        const choices = Object.fromEntries(cases.map(item => [item.name, new URL(item.manifest, catalogUrl).href]));
        state.case = Object.values(choices)[0];
        folder.add(state, 'case', choices).name('Prepared case');
        folder.add({ load: () => open(state.case) }, 'load').name('Load prepared weather');
        state.status = 'Prepared cases ready; weather does not guarantee a tornado';
    } catch {
        state.status = 'Prepared weather unavailable on this deployment';
    }
    const endpoint = import.meta.env.VITE_WEATHER_API_URL as string | undefined;
    if (!endpoint) return;
    const api = new URL(endpoint.endsWith('/') ? endpoint : endpoint + '/', location.href);
    const custom = folder.addFolder('Prepare another area');
    custom.add(state, 'latitude', 25, 49, 0.001).name('Latitude');
    custom.add(state, 'longitude', -124, -67, 0.001).name('Longitude');
    custom.add(state, 'date').name('Cycle date (UTC)');
    custom.add(state, 'hour', 0, 23, 1).name('Cycle hour (UTC)');
    custom.add(state, 'forecast', 1, 16, 1).name('First forecast hour');
    let busy = false;
    custom.add({ prepare: async () => {
        if (busy) return;
        busy = true;
        try {
            const response = await fetch(new URL('jobs', api), { method: 'POST',
                headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(state) });
            let job = await response.json();
            if (!response.ok) throw new Error(job.error ?? 'Weather request failed');
            while (job.status !== 'ready') {
                if (job.status === 'failed') throw new Error(job.message);
                state.status = job.message ?? 'Preparing weather';
                await new Promise(resolve => setTimeout(resolve, 2000));
                const poll = await fetch(new URL(`jobs/${job.id}`, api));
                job = await poll.json();
                if (!poll.ok) throw new Error(job.error ?? 'Weather worker unavailable');
            }
            open(new URL(job.manifest, api).href);
        } catch (error) {
            state.status = error instanceof Error ? error.message : String(error);
        } finally { busy = false; }
    } }, 'prepare').name('Prepare and load weather');
}
