import { GUI } from "lil-gui";

import { Fields } from "../stage/fields";
import { WeatherData } from "../weather/noaa_loader";

// The 3D Navier-Stokes solver on shared grid
// Input: fields.obstacleView, and weather.atmosphereBuffer for background air and wind
// Output: fields.velocity and fields.thermo, with the layouts in common.wgsl.
//   After step(), the raymarcher will draw the data in fields.thermo.read, so make sure "read"
// holds the newest data
//
// For now we are doing the tornado algorithm (algorithm 1), because it is in cartesian coordinates,
// and will be simpler to implement. If this goes well, we will be able to build off of it
// to make the hurricane model (which has a few more parameters and features, and is in cylindrical coords).
//
// Here is just a big info dump on features. Your buffer allocations will be in this file,
// since they will only be used here (list will mention something like "New in FluidSolver").
// Note that there are also a few references to equations and sections of the paper below.
// Ask me (Gavin) if you have questions about any of this stuff.
//
// M1 features (Cyclogenesis paper, Algorithm 1 only). Textures are 3D and grid sized unless noted.
//   - Advection: semi-Lagrangian, velocity and thermo, staggered velocity (common.wgsl)
//       Uses fields.velocity and fields.thermo (2 rgba16float each, in Fields). Nothing new.
//   - Pressure: divergence, Jacobi (40 iterations), subtract. Never clear pressure (warm start).
//       New in FluidSolver: pressure (2 r32float, ping-pong) and divergence (1 r32float)
//   - Open bits: per-cell "neighbor is open" bits from obstacles, used by Jacobi and boundaries
//       New in FluidSolver: 1 r32uint, rebuilt only when obstacles change
//   - Boundaries: no-slip ground via open bits, free-slip ceiling (5.1.2), side wall wind from atmosphereBuffer (5.2)
//       Uses the open bits and weather.atmosphereBuffer (storage buffer, in WeatherData). Nothing new.
//   - Forces: buoyancy (Eq. 7) from theta', qv', qc, qr, and standard vorticity confinement
//       New in FluidSolver: vorticity (1 rgba16float, curl in xyz)
//   - Setup: @workgroup_size(8, 8, 4), one cached bind group per PingPong state
//   - GUI: dt (0.5 s) and Jacobi iterations
//       New in FluidSolver: 1 uniform buffer for dt and other shader parameters
//   - Profiling: Add the timestampWrites "renderer.profiler?.passWrites("advect")" to each compute pass, to view with /?profile.
//       draw() already resolves and submits the frame, so don't call resolve() or submit() here. At most 16 passes
//       are timed per frame, so put all Jacobi iterations in one "jacobi" pass.
//   - Divergence check: behind a URL flag, copy the newest velocity (PingPong.textures) to the CPU and log RMS divergence
//       New in FluidSolver, only with the flag: 1 MAP_READ | COPY_DST buffer. RMS should stay under 5e-4.
// New in total: 5 textures, 1 uniform buffer, and readback buffer when checking
// After milestone 1, we will do microphysics, collisions (M2), better advection, faster pressure solvers (M3)
export class FluidSolver {
    constructor(readonly fields: Fields, readonly weather: WeatherData) {}

    buildGui(_gui: GUI) {
    }

    // TODO: forces, advect, then pressure projection. 
    // This will run once per frame in the frame's encoder, before raymarching.
    step(_encoder: GPUCommandEncoder) {
    }
}
