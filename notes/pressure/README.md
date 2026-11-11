# WebGPU Tornado

Notes and CPU checks for the pressure step of Webnado. The browser solver lives in [Gavin's repo](https://github.com/Gavin-Sears/Webnado). Faris's merged solver uses weighted Jacobi there. This folder compares other ways to solve the same equation.

Read [METHODS.md](METHODS.md) for how each algorithm works and what it costs in parallelism, accuracy, speed, and memory.

```bash
python3.12 ml/compare_pressure.py       # Jacobi, red-black SOR, conjugate gradients
python3.12 ml/multigrid_pressure.py     # those two against a W-cycle, grids 32 to 256
python3.12 ml/pressure_warmstart.py     # low-frequency guess in front of Jacobi
python3.12 ml/warmstart_multigrid.py    # the same guess in front of the W-cycle
```

Needs NumPy. These runs are 2D and on the CPU. They are not frame times for the 128×64×128 WebGPU solver.
