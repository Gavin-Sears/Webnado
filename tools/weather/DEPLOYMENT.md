# Weather data and deployment

The solver consumes a prepared 3D weather manifest and binary, not raw GRIB2. HRRR is the supported importer, rather than a requirement of the equations. Choosing a tornado date or location does not guarantee a tornado. HRRR is a coarse weather model informed by observations, not an observed tornado recording.

## GitHub Pages

GitHub Pages runs the browser application and serves files; it cannot run Python. The deployment workflow prepares the two cases in `tools/weather/cases.json` on a GitHub Actions runner, caches the packed results, and includes them in the Pages artifact. No large binary files need to enter Git history. The existing Pages environment must be enabled in repository settings. The workflow has not been triggered by this change.

Visitors select **3D storm weather â†’ Prepared case â†’ Load prepared weather**. The app reloads with 3D weather shaders, FP32 state/velocity, the Deardorff preset, a flat comparison floor, and a following window. This loads initial weather and evolving lateral boundary conditions; the interior still evolves through the solver. Prepared cases retain the comparison floor heights used in the experiments. Preparation of new locations uses the native HRRR terrain height by default.

The catalog contains roughly 140 MiB for two cases. Pages currently allows a 1 GiB published site and has bandwidth limits; keep the catalog small. Dataset preparation downloads considerably more than the final packed files, needs at least 8 GiB of temporary free space, and may take several minutes per case. Temporary raw downloads are removed when a preparation job ends. A missing archive or failed preparation fails the deployment instead of publishing a broken HRRR option.

Local preparation:

```sh
python -m pip install -r tools/weather/requirements.txt
python tools/weather/prepare.py --output public/weather
npm install
npm run dev
```

`public/weather` is ignored. Individual extraction and packing tools are also restored for research use:

```sh
python tools/weather/export_hrrr.py --help
python tools/weather/pack_hrrr.py --help
```

The manifest records source URLs, forecast initialization/lead times, per-message checksums, units and conversions. The browser verifies the packed binary checksum before loading it. The current pipeline supports HRRR v4 CONUS cases from December 2020 onward, uses all 50 native levels and three consecutive forecasts from the same cycle, and retains ice input without evolving ice microphysics. Availability and coverage are checked during extraction. This is not a global weather importer.

## Arbitrary locations at runtime

For public visitors to request new coordinates and times, host a separate Python worker. A static site alone cannot provide this operation. Do not put a GitHub token or other processing credential in browser code.

```sh
python tools/weather/service.py --data tools/weather/data --origin http://localhost:5173
```

Set `VITE_WEATHER_API_URL=http://localhost:8080/` in local `.env`, then restart Vite. The **Prepare another area** controls appear only when this endpoint is configured. Enter coordinates, UTC model-cycle date/hour, and the first forecast lead. The worker extracts three forecast times, estimates the floor from source terrain, packs and caches the data, and the browser reloads when it is ready. Cached requests reuse prepared files.

The HTTP contract is `POST /jobs`, `GET /jobs/<id>`, and `GET /data/<id>/manifest.json` / `state.bin`. Jobs are asynchronous, processed one at a time, and limited to four queued/running requests, six new jobs per client address per hour, and 20 cached cases. Failed jobs report an error rather than silently substituting a sounding. The service binds to localhost by default.

For an external host, a Dockerfile is provided. Configure the real Pages origin, mount persistent `/data`, and run behind HTTPS plus an authenticated, rate-limited reverse proxy. CORS is not authentication. The worker trusts its direct client address; configure proxy-level limits for public use. The operator must remove unused cached case directories when the cache fills. This project worker is single-process; do not run several workers against the same cache. It is not a fully managed public weather service.

Set the repository Actions variable `VITE_WEATHER_API_URL` to that HTTPS endpoint to enable custom requests in the Pages build. Leaving it empty keeps the site entirely static with its prepared cases. No external worker has been deployed as part of this change.

## Checks and comparisons

`npm run test:solver` runs one GPU suite in `src/solver/solver.test.ts`, covering 16-bit and 32-bit fields. Install the test browser with `npx playwright install chromium`; to use installed Chrome, set `TEST_BROWSER_CHANNEL=chrome`. The fixtures use the low/local grid and `checkDivergence`. They cover rest, Jacobi and multigrid projection, solid boundaries, scalar transport, buoyancy/TKE/high-order integration, phase-change conservation, drag momentum exchange, and asynchronous tracking/reset. These numerical fixtures do not certify a historical tornado replay. The same suite can be opened manually at `/tests/solver.html?res=low&domain=local&state=32&precision=32&checkDivergence` during development.

Run weather-tool tests with `python -m unittest discover -s tools/weather -p 'test_*.py'`. These fast checks do not download HRRR. The existing real cases remain the integration data used for extraction/packing checks.

For edge clouds, use **Source weather comparison â†’ Show source slice**, choose a field/height, then **Snapshot at current time**. This read-only north-up slice interpolates the native model at the current window and time. Its blue/red range is labeled per snapshot. It does not overwrite simulated fields or show observational ground truth. Compare the source and simulation before attributing an edge feature to rendering, incoming weather, or boundary treatment.

GIS ground elevation now comes from Open-Meteo/Copernicus DEM when loading georeferenced tiles. Local demo/sample geometry keeps its local placement. Automatic height can be disabled for a manual override. DEM resolution and vertical datum can differ from Cesium terrain/ellipsoid heights, so this is an estimate rather than exact terrain contact. Failed lookups are visible; they do not silently use a new zero height. Re-bake visible collisions after tiles have loaded.

Sources: [HRRR description](https://rapidrefresh.noaa.gov/hrrr/), [NOAA archive](https://registry.opendata.aws/noaa-hrrr-pds/), [GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages), [Pages limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits), [elevation API](https://open-meteo.com/en/docs/elevation-api).

The ordinary CI job runs build and weather-tool checks. GPU checks need a WebGPU adapter: they passed locally on NVIDIA hardware, while a Windows software-adapter attempt was unavailable. For manual CI GPU runs, configure a trusted self-hosted Windows runner tagged `webgpu`, with Chrome installed, set repository variable `WEBNADO_GPU_RUNNER=true`, and dispatch the checks workflow. Do not expose this runner to untrusted pull-request code.

Prepared cases inherit the selected simulation resolution; choose High to match the research grid. Coarser previews can behave differently. Source temperature slices show absolute kelvin, whereas the solver temperature debug view shows potential-temperature perturbations, so their colors are not directly comparable.
