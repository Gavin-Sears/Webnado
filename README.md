# Title Here

A real-time, browser-based implementation of the *Cyclogenesis* paper in WebGPU, running on live Cesium terrain and driven by real NOAA weather data.

## Getting Started

Requires Node.js 20+ and a WebGPU-capable browser (Chrome or Edge 113+).

```bash
npm install
npm run dev       # start the dev server at http://localhost:5173
npm run build     # typecheck and build into dist/
npm run preview   # serve the production build
```

If WebGPU works, you'll see a dark blue canvas and `WebGPU successfully initialized!` in the console.

## References

- [Cyclogenesis: Simulating Hurricanes and Tornadoes](https://dl.acm.org/doi/10.1145/3658149)
- [WebGPU spec](https://www.w3.org/TR/webgpu/) · [WGSL spec](https://www.w3.org/TR/WGSL/)
- [CesiumJS](https://cesium.com/platform/cesiumjs/)
- [NOAA NOMADS](https://nomads.ncep.noaa.gov/)
