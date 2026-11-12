"""Pressure warm-start for one fixed terrain tile.

See ../METHODS.md. The guess only proposes the first pressure. Jacobi still
finishes the solve. On the multigrid this saves about one cycle, which is not
enough to put the model in the frame.

On an empty periodic grid the pressure step is one FFT. One fixed building
breaks that, and the divergence-to-pressure map stays linear. This script learns
it on the low Fourier modes, both positive and negative, and starts Jacobi from
the guess.
"""

from __future__ import annotations

import numpy as np

N = 32
MODES = 6
TRAIN = 140
TEST = 12
RIDGE = 0.05
TOL = 1e-3
MAX_ITERS = 4000


def lowpass() -> np.ndarray:
    ky = np.fft.fftfreq(N)[:, None]
    kx = np.fft.rfftfreq(N)[None, :]
    return np.exp(-40.0 * (kx * kx + ky * ky))


def y_indices() -> np.ndarray:
    # rfft2 stores negative y frequencies at the bottom of the array.
    if MODES == 1:
        return np.array([0])
    return np.concatenate([np.arange(0, MODES), np.arange(N - MODES + 1, N)])


def x_indices() -> np.ndarray:
    return np.arange(0, MODES)


def make_terrain(rng: np.random.Generator) -> np.ndarray:
    mask = np.ones((N, N), dtype=np.float64)
    x0, y0 = rng.integers(4, N // 2, size=2)
    width, height = rng.integers(4, 10, size=2)
    mask[y0 : y0 + height, x0 : x0 + width] = 0.0
    return mask


def random_divergence(rng: np.random.Generator, mask: np.ndarray) -> np.ndarray:
    noise = rng.normal(size=(N, N))
    rhs = np.fft.irfft2(np.fft.rfft2(noise) * lowpass())
    rhs = rhs * mask
    rhs -= rhs[mask > 0].mean()
    return rhs * mask


def laplacian(p: np.ndarray) -> np.ndarray:
    return (
        np.roll(p, 1, 0) + np.roll(p, -1, 0) + np.roll(p, 1, 1) + np.roll(p, -1, 1) - 4.0 * p
    )


def jacobi(rhs: np.ndarray, mask: np.ndarray, p0: np.ndarray) -> tuple[np.ndarray, int]:
    """5-point Poisson solve on fluid cells. Solid cells stay at pressure 0."""
    p = p0 * mask
    fluid = mask > 0
    scale = max(float(np.max(np.abs(rhs))), 1e-8)
    for step in range(1, MAX_ITERS + 1):
        neighbors = (
            np.roll(p, 1, 0) + np.roll(p, -1, 0) + np.roll(p, 1, 1) + np.roll(p, -1, 1)
        )
        p = np.where(fluid, (neighbors - rhs) / 4.0, 0.0)
        if step % 5 == 0:
            error = float(np.max(np.abs((laplacian(p) - rhs)[fluid]))) / scale
            if error < TOL:
                return p, step
    return p, MAX_ITERS


def pack_modes(field: np.ndarray) -> np.ndarray:
    block = np.fft.rfft2(field)[np.ix_(y_indices(), x_indices())]
    return np.concatenate([block.real.ravel(), block.imag.ravel()])


def unpack_modes(coeff: np.ndarray) -> np.ndarray:
    spec = np.zeros((N, N // 2 + 1), dtype=np.complex128)
    count = y_indices().size * x_indices().size
    block = coeff[:count].reshape(y_indices().size, x_indices().size)
    block = block + 1j * coeff[count:].reshape(y_indices().size, x_indices().size)
    spec[np.ix_(y_indices(), x_indices())] = block
    return np.fft.irfft2(spec, s=(N, N))


def fit(divergences: list[np.ndarray], pressures: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.stack([pack_modes(rhs) for rhs in divergences])
    targets = np.stack([pack_modes(pressure) for pressure in pressures])
    mean = features.mean(axis=0)
    scale = features.std(axis=0) + 1e-6
    scaled = (features - mean) / scale
    gram = scaled.T @ scaled + RIDGE * np.eye(scaled.shape[1])
    weight = np.linalg.solve(gram, scaled.T @ targets)
    return weight, mean, scale


def predict(
    weight: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    rhs: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    coeff = weight.T @ ((pack_modes(rhs) - mean) / scale)
    return unpack_modes(coeff) * mask


def main() -> None:
    rng = np.random.default_rng(0)
    mask = make_terrain(rng)
    divergences: list[np.ndarray] = []
    pressures: list[np.ndarray] = []
    for _ in range(TRAIN):
        rhs = random_divergence(rng, mask)
        pressure, _ = jacobi(rhs, mask, np.zeros_like(rhs))
        divergences.append(rhs)
        pressures.append(pressure)
    weight, mean, scale = fit(divergences, pressures)

    cold_iters: list[int] = []
    warm_iters: list[int] = []
    oracle_iters: list[int] = []
    errors: list[float] = []
    test_rng = np.random.default_rng(1)
    for _ in range(TEST):
        rhs = random_divergence(test_rng, mask)
        truth, cold = jacobi(rhs, mask, np.zeros_like(rhs))
        guess = predict(weight, mean, scale, rhs, mask)
        _, warm = jacobi(rhs, mask, guess)
        oracle = unpack_modes(pack_modes(truth)) * mask
        _, oracle_steps = jacobi(rhs, mask, oracle)
        cold_iters.append(cold)
        warm_iters.append(warm)
        oracle_iters.append(oracle_steps)
        errors.append(float(np.linalg.norm(guess - truth) / np.linalg.norm(truth)))

    print(f"grid {N}x{N}, one fixed building, Fourier modes ±{MODES - 1}")
    print(f"train {TRAIN}, test {TEST}")
    print(f"Jacobi from zero:              {np.mean(cold_iters):.0f} iterations (mean)")
    print(f"Jacobi from learned guess:     {np.mean(warm_iters):.0f} iterations (mean)")
    print(f"Jacobi from true low modes:    {np.mean(oracle_iters):.0f} iterations (mean)")
    print(f"relative error of the guess:   {np.mean(errors):.2f}")
    print("per test case, zero -> learned -> oracle:")
    for cold, warm, oracle_steps in zip(cold_iters, warm_iters, oracle_iters):
        print(f"  {cold:4d} -> {warm:4d} -> {oracle_steps:4d}")


if __name__ == "__main__":
    main()
