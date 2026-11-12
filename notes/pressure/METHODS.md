# Pressure methods

These notes compare ways to solve the pressure step. Faris's merged solver already does one of them on the real 3D grid. The scripts here are 2D checks on one building, run in NumPy on the CPU. They are not in the WebGPU frame, and they do not replace `src/solver/projection.wgsl`.

The question they answer is which algorithm is worth a second implementation once the domain gets large. On today's 128×64×128 grid the whole solver is already about 4.5 ms (Faris, RTX 4050, 720p, ~65 fps), so a faster pressure step does not change that frame.

## What is already in the solver

Weighted Jacobi, 40 iterations, pressure kept from the previous frame.

Each cell is updated from the previous pressure of its neighbors, then blended with the old value by 2/3. One compute dispatch touches every fluid cell. Solid cells stay zero. Two `r32float` textures hold the pressure. The iteration count is a GUI slider from 1 to 200.

Parallelism is the easy kind: no cell waits for another cell in the same dispatch. Accuracy is capped on purpose. Forty sweeps do not finish the Poisson equation once the box grows; they are cheap enough for the current grid. Keeping last frame's pressure is the useful warm start. A network in this slot did not beat it.

## Red-black SOR

`ml/compare_pressure.py`

Gauss–Seidel updates a cell from neighbors that were already updated in this sweep. That dependency is serial. Coloring the grid like a checkerboard removes it: red cells only touch black cells, then black cells only touch red cells. Two dispatches per sweep, same storage as Jacobi. The over-relaxation weight is 1.75. At weight 1 this is Gauss–Seidel.

On a 32×32 grid with one building, residual 10⁻³, eight held-out winds:

| method | sweeps | pressure error |
|---|---:|---:|
| undamped Jacobi from zero | 1665 | 4.8% |
| red-black SOR | 226 | 0.6% |
| conjugate gradients | 76 | 0.1% |

SOR needs less memory traffic than a coarse-grid hierarchy and maps onto WebGPU the same way Jacobi does. The sweep count still grows with the grid, so a 50 km box brings the same problem back.

This is not the kernel in the solver. That kernel is weighted Jacobi, which keeps the old value and updates every cell at once.

## Conjugate gradients

`ml/compare_pressure.py`, `ml/multigrid_pressure.py`

The search direction is built from every previous residual, not from a neighbor average. Each iteration is one stencil plus two sums over the whole grid. The sums are a global reduction. On WebGPU that reduction is a separate dispatch and a wait.

Sweeps to the same residual, one building, grid side n:

| n | Jacobi | conjugate gradients | multigrid cycles |
|---:|---:|---:|---:|
| 32 | 2020 | 78 | 5 |
| 64 | 4150 | 164 | 6 |
| 128 | 34845 | 338 | 5 |
| 256 | 42070 | 661 | 8 |

Extra memory is a few full grids (residual, direction, matrix-vector product). Accuracy at a tight tolerance matches the discrete equation. The reduction is why this loses to Jacobi on a GPU even when the sweep count looks better. It is a good reference solution, and it is the exact solve on the coarsest multigrid level (16×16).

## Geometric multigrid

`ml/multigrid_pressure.py`

Low-frequency error is what makes Jacobi slow. A coarse grid represents that error with far fewer cells. One W-cycle smooths the fine grid with red-black Gauss–Seidel (two parallel dispatches), restricts the residual, corrects the coarse grid twice, prolongs the correction, and smooths again. The building mask is injected onto even nodes. The 16×16 level is solved with conjugate gradients.

Work in the table above is counted in fine-grid sweeps. A coarse level costs a quarter of a fine sweep in 2D. The cycle count stays in a band of about 5 to 8 while Jacobi and conjugate gradients grow. Pressure error against a tight conjugate-gradient solution is about 0.01% or smaller.

Memory is the fine pressure plus the coarse levels. In 2D that extra storage is about one third of the fine grid. In 3D a full pyramid is about one seventh.

This is the method that stays cheap when the grid grows. It does not make the current 4.5 ms solver faster in any measurement we have, because it has not been run as a shader on that grid. The Jacobi numbers above are undamped Jacobi from zero, not Faris's 40 damped sweeps started from last frame.

## Fourier warm start

`ml/pressure_warmstart.py`, `ml/warmstart_multigrid.py`

For one fixed building the map from divergence to pressure is linear. The script learns that map on the low Fourier modes and starts the solver from the guess. The solver still runs to the end. The guess is a small matrix, a few hundred coefficients, fixed for that building.

Against undamped Jacobi on 32×32, 140 training winds, modes ±5: 1444 sweeps from zero, 519 from the guess, 150 from the true low modes. Mean relative error of the guess is 0.11. One of twelve test winds got worse (485 → 990).

Against the multigrid, modes ±7, 80 training winds:

| start | cycles, n=64 | cycles, n=128 |
|---|---:|---:|
| zero | 5.8 | 4.9 |
| learned guess | 4.8 | 4.0 |
| true low modes | 3.9 | 3.1 |

The guess alone is 14–16% off. One multigrid cycle from zero leaves 2.7% (n=64) or 1.3% (n=128). A low-frequency network cannot beat the true-low-mode row, and that row only removes about two cycles.

Parallelism is irrelevant: the map is tiny next to a sweep. It does not earn a buffer in the frame. Faris's separate trial of an ML warm start reached the same conclusion on the real solver.

## What these files leave out

Leapfrog flow maps change advection, not the pressure solve. Faris tried them beside an ML warm start. They did not make the step faster than the baseline.

Skipping raymarch pixels is already in the renderer. `resolutionScale` is 0.35, 0.5, or 0.75, and `composite.fs.wgsl` upsamples with a depth test. A learned filler would compete with that upsample, not with Jacobi.

An FFT pressure solve is one transform and ignores the building. On the 32×32 test its pressure error is 42%.

## Reproduce

```bash
python3.12 ml/compare_pressure.py
python3.12 ml/multigrid_pressure.py
python3.12 ml/pressure_warmstart.py
python3.12 ml/warmstart_multigrid.py
```

Needs NumPy. The first two scripts are the sweep counts above. The last two are the learned guess.

## Next

Weighted Jacobi stays in the WebGPU solver for this grid. The comparison is what we use to decide the next pressure step. A switch can sit next to the iteration slider and run Jacobi, red-black SOR, and multigrid on the same fields, timed with `?profile`. Conjugate gradients stay a reference. The Fourier guess does not go in that switch.
