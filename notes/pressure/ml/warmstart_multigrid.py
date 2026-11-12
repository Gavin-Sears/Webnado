"""Does a pressure guess still help once the solver is a W-cycle?

See ../METHODS.md.

The learned map is the same linear Fourier model as pressure_warmstart.py.
The oracle packs the true pressure into those same modes and unpacks it again.
Anything a low-frequency network can do sits between the cold start and that oracle.
"""

from __future__ import annotations

import numpy as np

import multigrid_pressure as mg

MODES = 8
TRAIN = 80
TEST = 8
RIDGE = 0.05
CHECK_AT = (0, 1, 2)


def y_indices(n: int) -> np.ndarray:
    return np.concatenate([np.arange(0, MODES), np.arange(n - MODES + 1, n)])


def x_indices() -> np.ndarray:
    return np.arange(0, MODES)


def pack_modes(field: np.ndarray) -> np.ndarray:
    n = field.shape[0]
    block = np.fft.rfft2(field)[np.ix_(y_indices(n), x_indices())]
    return np.concatenate([block.real.ravel(), block.imag.ravel()])


def unpack_modes(coeff: np.ndarray, n: int) -> np.ndarray:
    spec = np.zeros((n, n // 2 + 1), dtype=np.complex128)
    count = y_indices(n).size * x_indices().size
    block = coeff[:count].reshape(y_indices(n).size, x_indices().size)
    block = block + 1j * coeff[count:].reshape(y_indices(n).size, x_indices().size)
    spec[np.ix_(y_indices(n), x_indices())] = block
    return np.fft.irfft2(spec, s=(n, n))


def fit(xs: list[np.ndarray], ys: list[np.ndarray]):
    features = np.stack([pack_modes(x) for x in xs])
    targets = np.stack([pack_modes(y) for y in ys])
    mean = features.mean(axis=0)
    scale = features.std(axis=0) + 1e-6
    scaled = (features - mean) / scale
    gram = scaled.T @ scaled + RIDGE * np.eye(scaled.shape[1])
    weight = np.linalg.solve(gram, scaled.T @ targets)
    return weight, mean, scale


def predict(weight, mean, scale, rhs, mask):
    coeff = weight.T @ ((pack_modes(rhs) - mean) / scale)
    return unpack_modes(coeff, rhs.shape[0]) * mask


def cycles_from(p0: np.ndarray, rhs: np.ndarray, mask: np.ndarray, h: float) -> int:
    p = p0 * mask
    if mg.residual(p, rhs, mask, h) < mg.TOL:
        return 0
    for cycle in range(1, 40):
        p = mg.v_cycle(p, rhs, mask, h, [0.0, float(rhs.shape[0])])
        if mg.residual(p, rhs, mask, h) < mg.TOL:
            return cycle
    return 40


def field_after(p0: np.ndarray, rhs: np.ndarray, mask: np.ndarray, h: float, cycles: int) -> np.ndarray:
    p = p0 * mask
    for _ in range(cycles):
        p = mg.v_cycle(p, rhs, mask, h, [0.0, float(rhs.shape[0])])
    return p


def run_size(n: int, rng: np.random.Generator) -> None:
    mask = mg.make_terrain(n)
    h = 1.0 / n
    rights, pressures = [], []
    for _ in range(TRAIN):
        rhs = mg.make_rhs(n, mask, rng)
        _, truth = mg.cg_until(rhs, mask, h, mg.TIGHT)
        rights.append(rhs)
        pressures.append(truth)
    weight, mean, scale = fit(rights, pressures)

    cold, warm, oracle = [], [], []
    guess_err = []
    snapshots = {name: {k: [] for k in CHECK_AT} for name in ("cold", "warm")}
    for _ in range(TEST):
        rhs = mg.make_rhs(n, mask, rng)
        _, truth = mg.cg_until(rhs, mask, h, mg.TIGHT)
        guess = predict(weight, mean, scale, rhs, mask)
        perfect = unpack_modes(pack_modes(truth), n) * mask
        cold.append(cycles_from(np.zeros_like(rhs), rhs, mask, h))
        warm.append(cycles_from(guess, rhs, mask, h))
        oracle.append(cycles_from(perfect, rhs, mask, h))
        guess_err.append(mg.rel_error(guess, truth, mask))
        for k in CHECK_AT:
            snapshots["cold"][k].append(mg.rel_error(field_after(np.zeros_like(rhs), rhs, mask, h, k), truth, mask))
            snapshots["warm"][k].append(mg.rel_error(field_after(guess, rhs, mask, h, k), truth, mask))

    print(f"n={n}, modes ±{MODES - 1}, train {TRAIN}, test {TEST}")
    print(f"  cycles from zero:          {np.mean(cold):.1f}")
    print(f"  cycles from learned guess: {np.mean(warm):.1f}")
    print(f"  cycles from true low modes:{np.mean(oracle):.1f}")
    print(f"  error of the guess alone:  {np.mean(guess_err):.3f}")
    print("  pressure error after 0, 1, 2 cycles:")
    for name in ("cold", "warm"):
        values = "  ".join(f"{k}:{np.mean(snapshots[name][k]):.3f}" for k in CHECK_AT)
        print(f"    {name:<5} {values}")


def main() -> None:
    rng = np.random.default_rng(3)
    for n in (64, 128):
        run_size(n, rng)
        print()


if __name__ == "__main__":
    main()
