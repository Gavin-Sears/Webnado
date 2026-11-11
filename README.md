# Webnado

An interactive, browser-based implementation of the *Cyclogenesis* paper in WebGPU, with 3D tiles for terrain and buildings rendered using Three.js, and real weather data through Open-Meteo. Simulates micro (tornado) and synoptic (hurricane) scale cyclones.

## Getting Started

Requires Node.js 20.19+ or 22.12+ and a recent version of Chrome or Edge with WebGPU support.

```bash
npm install
npm run dev         # start the dev server and open http://localhost:5173
npm run build       # typecheck and build into dist/
npm run preview     # serve the production build at http://localhost:4173
npm run typecheck   # typecheck without building
```

When loading cesium tiles is implemented, set `VITE_CESIUM_ION_TOKEN` in the `.env` file. Never commit `.env`.

Right now you should see a gray test cloud above a green plane. `WebGPU init successful` will appear in the console.

This WebGPU setup follows a similar structure to our previous WebGPU assignments. `src/renderer.ts` contains `initWebGPU()` which sets up our device and context. There is also a `Renderer` base class, where `src/renderers/` (not committed yet) holds subclasses. We also have similar shader loading logic in `shaders.ts`, which prepends `common.wgsl`.

## URL Parameters

### General

- `res`: simulation grid size
    - `low` (64×32×64) 
    - `medium` (128×64×128, default)
    - `high` (256×128×256)
    - ex: http://localhost:5173/?res=high
- `quality`: raymarcher preset (`low`, `medium` or `high`). Without it, the preset is picked from the GPU
    - ex: http://localhost:5173/?quality=low

### Profiling

- `profile`: activates profiler. Displays a live readout of cpu, and total frame time (we will soon have three.js and raymarch time as well). Console log will also print a JSON object containing timing and device stats, which is also stored in the variable `window.profileResult`
- `warmup`: frames to skip before capture starts. The default is 120
- `frames`: frames to capture. The default is 600, and the maximum is 4096
- ex: http://localhost:5173/?res=high&profile&warmup=60&frames=2000 profiles the high resolution grid, skipping 60 frames and then capturing 2000 frames

For accurate numbers, launch Chrome without vsync or a frame rate limit. Developer features are only needed if the capture warns that GPU times look rounded. With git bash you can paste the following into your terminal:

`start "" chrome --disable-gpu-vsync --disable-frame-rate-limit --enable-webgpu-developer-features &`

## Credits and References

- [Cyclogenesis: Simulating Hurricanes and Tornadoes](https://dl.acm.org/doi/10.1145/3658149)
- [WebGPU spec](https://www.w3.org/TR/webgpu/) · [WGSL spec](https://www.w3.org/TR/WGSL/)
- [Three.js](https://threejs.org/docs/) · [3DTilesRendererJS](https://github.com/NASA-AMMOS/3DTilesRendererJS) · [Cesium ion](https://cesium.com/learn/ion/)
- Weather data by [Open-Meteo](https://open-meteo.com/), licensed under [CC BY 4.0](https://open-meteo.com/en/licence)
