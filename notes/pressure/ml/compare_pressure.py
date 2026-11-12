"""Same building, same divergence, five ways to get the pressure.

See ../METHODS.md. Jacobi here is undamped and starts from zero. That is not
the weighted Jacobi in the WebGPU solver, which keeps the previous frame.

The reference is conjugate gradient run tight. Every other method is scored by
how many sweeps it needs to reach the same residual, and by the pressure error
it still has if it is stopped early. That is the speed/quality switch.
"""

from __future__ import annotations

import numpy as np

import pressure_warmstart as pw

BUDGETS = (0, 10, 40, 160, 640)
CASES = 8
OMEGA = 1.75
TIGHT = 1e-8


def residual(p: np.ndarray, rhs: np.ndarray, mask: np.ndarray) -> float:
    fluid = mask > 0
    scale = max(float(np.max(np.abs(rhs))), 1e-8)
    return float(np.max(np.abs((pw.laplacian(p) - rhs)[fluid]))) / scale


def rel_error(p: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    fluid = mask > 0
    return float(np.linalg.norm((p - truth)[fluid]) / np.linalg.norm(truth[fluid]))


def jacobi_steps(rhs: np.ndarray, mask: np.ndarray, steps: int, p0: np.ndarray | None = None) -> np.ndarray:
    p = np.zeros_like(rhs) if p0 is None else p0 * mask
    fluid = mask > 0
    for _ in range(steps):
        neighbors = (
            np.roll(p, 1, 0) + np.roll(p, -1, 0) + np.roll(p, 1, 1) + np.roll(p, -1, 1)
        )
        p = np.where(fluid, (neighbors - rhs) / 4.0, 0.0)
    return p


def sweeps_until(kind: str, rhs: np.ndarray, mask: np.ndarray, p0: np.ndarray | None = None) -> tuple[int, np.ndarray]:
    if kind == "jacobi":
        p, steps = pw.jacobi(rhs, mask, np.zeros_like(rhs) if p0 is None else p0)
        return steps, p
    if kind == "sor":
        return red_black_sor(rhs, mask, np.zeros_like(rhs) if p0 is None else p0)
    if kind == "cg":
        return conjugate_gradient(rhs, mask, np.zeros_like(rhs) if p0 is None else p0, pw.TOL)
    raise ValueError(kind)


def red_black_sor(rhs: np.ndarray, mask: np.ndarray, p0: np.ndarray) -> tuple[int, np.ndarray]:
    yy, xx = np.indices((pw.N, pw.N))
    colors = (
        ((yy + xx) % 2 == 0) & (mask > 0),
        ((yy + xx) % 2 == 1) & (mask > 0),
    )
    p = p0 * mask
    for step in range(1, pw.MAX_ITERS + 1):
        for color in colors:
            neighbors = (
                np.roll(p, 1, 0) + np.roll(p, -1, 0) + np.roll(p, 1, 1) + np.roll(p, -1, 1)
            )
            gauss = (neighbors - rhs) / 4.0
            p = np.where(color, (1.0 - OMEGA) * p + OMEGA * gauss, p)
        if step % 2 == 0 and residual(p, rhs, mask) < pw.TOL:
            return step, p
    return pw.MAX_ITERS, p


def apply_spd(p: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return -pw.laplacian(p * mask) * mask


def conjugate_gradient(
    rhs: np.ndarray, mask: np.ndarray, p0: np.ndarray, tol: float
) -> tuple[int, np.ndarray]:
    p = p0 * mask
    r = (-rhs * mask) - apply_spd(p, mask)
    d = r.copy()
    rr = float(np.sum(r * r))
    for step in range(1, pw.N * pw.N + 1):
        if residual(p, rhs, mask) < tol:
            return step - 1, p
        Ad = apply_spd(d, mask)
        denom = float(np.sum(d * Ad))
        if denom <= 1e-18 or rr <= 1e-30:
            return step - 1, p
        alpha = rr / denom
        p = (p + alpha * d) * mask
        r = (r - alpha * Ad) * mask
        rr_new = float(np.sum(r * r))
        d = (r + (rr_new / rr) * d) * mask
        rr = rr_new
    return pw.N * pw.N, p


def fft_pressure(rhs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Periodic Poisson with the building ignored. One transform."""
    ky = np.fft.fftfreq(pw.N)[:, None]
    kx = np.fft.rfftfreq(pw.N)[None, :]
    eig = 2.0 * np.cos(2.0 * np.pi * ky) + 2.0 * np.cos(2.0 * np.pi * kx) - 4.0
    eig[0, 0] = np.inf
    return np.fft.irfft2(np.fft.rfft2(rhs) / eig, s=(pw.N, pw.N)) * mask


def train(mask: np.ndarray, rng: np.random.Generator):
    divergences = []
    pressures = []
    for _ in range(pw.TRAIN):
        rhs = pw.random_divergence(rng, mask)
        _, pressure = conjugate_gradient(rhs, mask, np.zeros_like(rhs), pw.TOL)
        divergences.append(rhs)
        pressures.append(pressure)
    return pw.fit(divergences, pressures)


def main() -> None:
    rng = np.random.default_rng(0)
    mask = pw.make_terrain(rng)
    weight, mean, scale = train(mask, rng)

    test_rng = np.random.default_rng(1)
    to_tol: dict[str, list[int]] = {name: [] for name in ("jacobi", "sor", "cg", "warm+jacobi", "warm+cg")}
    at_tol_error: dict[str, list[float]] = {name: [] for name in to_tol}
    budgets: dict[str, dict[int, list[float]]] = {
        name: {k: [] for k in BUDGETS} for name in ("jacobi", "sor", "cg", "warm+jacobi")
    }
    guess_error: list[float] = []
    fft_error: list[float] = []

    for _ in range(CASES):
        rhs = pw.random_divergence(test_rng, mask)
        _, truth = conjugate_gradient(rhs, mask, np.zeros_like(rhs), TIGHT)
        guess = pw.predict(weight, mean, scale, rhs, mask)
        guess_error.append(rel_error(guess, truth, mask))
        fft_error.append(rel_error(fft_pressure(rhs, mask), truth, mask))

        starts = {
            "jacobi": None,
            "sor": None,
            "cg": None,
            "warm+jacobi": guess,
            "warm+cg": guess,
        }
        solved = {}
        for name, start in starts.items():
            kind = "jacobi" if "jacobi" in name else "sor" if name == "sor" else "cg"
            steps, field = sweeps_until(kind, rhs, mask, start)
            to_tol[name].append(steps)
            at_tol_error[name].append(rel_error(field, truth, mask))
            solved[name] = field

        for k in BUDGETS:
            budgets["jacobi"][k].append(rel_error(jacobi_steps(rhs, mask, k), truth, mask))
            budgets["warm+jacobi"][k].append(rel_error(jacobi_steps(rhs, mask, k, guess), truth, mask))
            budgets["sor"][k].append(rel_error(red_black_field(rhs, mask, k), truth, mask))
            budgets["cg"][k].append(rel_error(cg_field(rhs, mask, k), truth, mask))

    print(f"grid {pw.N}x{pw.N}, one fixed building, {CASES} held-out winds")
    print(f"target residual {pw.TOL:.0e} against a tight CG reference")
    print()
    print(f"{'method':<16} {'sweeps':>8} {'pressure error':>16}")
    for name in ("jacobi", "sor", "cg", "warm+jacobi", "warm+cg"):
        print(f"{name:<16} {np.mean(to_tol[name]):8.0f} {np.mean(at_tol_error[name]):16.3f}")
    print(f"{'guess only':<16} {'0':>8} {np.mean(guess_error):16.3f}")
    print(f"{'fft, no building':<16} {'1':>8} {np.mean(fft_error):16.3f}")
    print()
    header = f"{'sweeps':>8}" + "".join(f"{name:>14}" for name in ("jacobi", "sor", "cg", "warm+jacobi"))
    print("relative pressure error if stopped early")
    print(header)
    for k in BUDGETS:
        row = f"{k:8d}"
        for name in ("jacobi", "sor", "cg", "warm+jacobi"):
            row += f"{np.mean(budgets[name][k]):14.3f}"
        print(row)


def red_black_field(rhs: np.ndarray, mask: np.ndarray, steps: int) -> np.ndarray:
    yy, xx = np.indices((pw.N, pw.N))
    colors = (
        ((yy + xx) % 2 == 0) & (mask > 0),
        ((yy + xx) % 2 == 1) & (mask > 0),
    )
    p = np.zeros_like(rhs)
    for _ in range(steps):
        for color in colors:
            neighbors = (
                np.roll(p, 1, 0) + np.roll(p, -1, 0) + np.roll(p, 1, 1) + np.roll(p, -1, 1)
            )
            gauss = (neighbors - rhs) / 4.0
            p = np.where(color, (1.0 - OMEGA) * p + OMEGA * gauss, p)
    return p


def cg_field(rhs: np.ndarray, mask: np.ndarray, steps: int) -> np.ndarray:
    p = np.zeros_like(rhs)
    r = (-rhs * mask) - apply_spd(p, mask)
    d = r.copy()
    rr = float(np.sum(r * r))
    for _ in range(steps):
        Ad = apply_spd(d, mask)
        denom = float(np.sum(d * Ad))
        if denom <= 1e-18 or rr <= 1e-30:
            break
        alpha = rr / denom
        p = (p + alpha * d) * mask
        r = (r - alpha * Ad) * mask
        rr_new = float(np.sum(r * r))
        d = (r + (rr_new / rr) * d) * mask
        rr = rr_new
    return p


if __name__ == "__main__":
    main()
