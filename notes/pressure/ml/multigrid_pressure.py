"""Pressure Poisson on one building, from 32² to 256².

See ../METHODS.md. Jacobi, conjugate gradients, and a geometric W-cycle share
the same stencil.
Each red-black sweep is two parallel updates, the same shape as two WebGPU
dispatches. The cycle is a W-cycle: the coarse grid is corrected twice, which
is what kept the building from stalling the recursion. Work is counted in
fine-grid sweeps, so a coarser level costs a quarter of a fine sweep in 2D.
"""

from __future__ import annotations

import numpy as np

TOL = 1e-3
TIGHT = 1e-8
PRE = 2
POST = 2
SIZES = (32, 64, 128, 256)


def neighbors(p: np.ndarray) -> np.ndarray:
    return np.roll(p, 1, 0) + np.roll(p, -1, 0) + np.roll(p, 1, 1) + np.roll(p, -1, 1)


def apply(p: np.ndarray, h: float, mask: np.ndarray) -> np.ndarray:
    return ((neighbors(p) - 4.0 * p) / (h * h)) * mask


def residual(p: np.ndarray, rhs: np.ndarray, mask: np.ndarray, h: float) -> float:
    fluid = mask > 0
    scale = max(float(np.max(np.abs(rhs))), 1e-8)
    return float(np.max(np.abs((apply(p, h, mask) - rhs)[fluid]))) / scale


def make_terrain(n: int) -> np.ndarray:
    mask = np.ones((n, n), dtype=np.float64)
    y0, x0 = n // 3, n // 3
    height, width = n // 6, n // 5
    mask[y0 : y0 + height, x0 : x0 + width] = 0.0
    return mask


def make_rhs(n: int, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(size=(n, n))
    ky = np.fft.fftfreq(n)[:, None] * n
    kx = np.fft.rfftfreq(n)[None, :] * n
    low = np.exp(-0.2 * (kx * kx + ky * ky))
    rhs = np.fft.irfft2(np.fft.rfft2(noise) * low, s=(n, n))
    rhs *= mask
    rhs -= rhs[mask > 0].mean()
    return rhs * mask


def smooth(p: np.ndarray, rhs: np.ndarray, mask: np.ndarray, h: float, steps: int) -> np.ndarray:
    """Red-black Gauss–Seidel. Each color is one parallel dispatch."""
    yy, xx = np.indices(p.shape)
    colors = (((yy + xx) % 2 == 0) & (mask > 0), ((yy + xx) % 2 == 1) & (mask > 0))
    h2 = h * h
    for _ in range(steps):
        for color in colors:
            guess = (neighbors(p) - rhs * h2) / 4.0
            p = np.where(color, guess, p)
    return p


def jacobi_until(rhs: np.ndarray, mask: np.ndarray, h: float) -> tuple[int, np.ndarray]:
    p = np.zeros_like(rhs)
    fluid = mask > 0
    h2 = h * h
    for step in range(1, n_cap(rhs) + 1):
        guess = (neighbors(p) - rhs * h2) / 4.0
        p = np.where(fluid, guess, 0.0)
        if step % 5 == 0 and residual(p, rhs, mask, h) < TOL:
            return step, p
    return n_cap(rhs), p


def n_cap(rhs: np.ndarray) -> int:
    return 8 * rhs.shape[0] * rhs.shape[0]


def cg_until(rhs: np.ndarray, mask: np.ndarray, h: float, tol: float) -> tuple[int, np.ndarray]:
    p = np.zeros_like(rhs)
    r = (-rhs * mask) - (-apply(p, h, mask))
    d = r.copy()
    rr = float(np.sum(r * r))
    for step in range(1, rhs.shape[0] * rhs.shape[0] + 1):
        if residual(p, rhs, mask, h) < tol:
            return step - 1, p
        Ad = -apply(d, h, mask)
        denom = float(np.sum(d * Ad))
        if denom <= 1e-18 or rr <= 1e-30:
            return step - 1, p
        alpha = rr / denom
        p = (p + alpha * d) * mask
        r = (r - alpha * Ad) * mask
        rr_new = float(np.sum(r * r))
        d = (r + (rr_new / max(rr, 1e-30)) * d) * mask
        rr = rr_new
    return rhs.shape[0] * rhs.shape[0], p


def restrict(field: np.ndarray) -> np.ndarray:
    center = 4.0 * field
    edge = 2.0 * (
        np.roll(field, 1, 0) + np.roll(field, -1, 0) + np.roll(field, 1, 1) + np.roll(field, -1, 1)
    )
    corner = (
        np.roll(np.roll(field, 1, 0), 1, 1)
        + np.roll(np.roll(field, 1, 0), -1, 1)
        + np.roll(np.roll(field, -1, 0), 1, 1)
        + np.roll(np.roll(field, -1, 0), -1, 1)
    )
    return ((center + edge + corner) / 16.0)[::2, ::2]


def prolong(coarse: np.ndarray) -> np.ndarray:
    n = coarse.shape[0]
    fine = np.zeros((n * 2, n * 2), dtype=np.float64)
    shifted_y = np.roll(coarse, -1, 0)
    shifted_x = np.roll(coarse, -1, 1)
    shifted_xy = np.roll(shifted_y, -1, 1)
    fine[::2, ::2] = coarse
    fine[1::2, ::2] = 0.5 * (coarse + shifted_y)
    fine[::2, 1::2] = 0.5 * (coarse + shifted_x)
    fine[1::2, 1::2] = 0.25 * (coarse + shifted_y + shifted_x + shifted_xy)
    return fine


def coarsen_mask(mask: np.ndarray) -> np.ndarray:
    # Keep the even nodes. Fattening the building at every level stalled the cycle.
    return mask[::2, ::2].copy()


def v_cycle(p: np.ndarray, rhs: np.ndarray, mask: np.ndarray, h: float, work: list[float]) -> np.ndarray:
    n = p.shape[0]
    fine_n = work[1]
    level_cost = (n / fine_n) ** 2
    if n <= 16:
        solved_steps, solved = cg_until(rhs, mask, h, TIGHT)
        work[0] += solved_steps * level_cost
        return solved
    p = smooth(p, rhs, mask, h, PRE)
    work[0] += PRE * level_cost
    res = (rhs - apply(p, h, mask)) * mask
    mask_c = coarsen_mask(mask)
    rhs_c = restrict(res) * mask_c
    error = np.zeros_like(rhs_c)
    for _ in range(2):
        error = v_cycle(error, rhs_c, mask_c, h * 2, work)
    p = (p + prolong(error)) * mask
    p = smooth(p, rhs, mask, h, POST)
    work[0] += POST * level_cost
    return p


def multigrid_until(rhs: np.ndarray, mask: np.ndarray, h: float) -> tuple[int, float, np.ndarray]:
    p = np.zeros_like(rhs)
    spent = 0.0
    for cycle in range(1, 40):
        work = [0.0, float(rhs.shape[0])]
        p = v_cycle(p, rhs, mask, h, work)
        spent += work[0]
        if residual(p, rhs, mask, h) < TOL:
            return cycle, spent, p
    return 40, spent, p


def rel_error(p: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    fluid = mask > 0
    return float(np.linalg.norm((p - truth)[fluid]) / np.linalg.norm(truth[fluid]))


def main() -> None:
    rng = np.random.default_rng(2)
    print(f"{'n':>6} {'jacobi sweeps':>14} {'cg sweeps':>10} {'W-cycles':>9} {'MG fine-sweeps':>15} {'MG error':>10}")
    for n in SIZES:
        mask = make_terrain(n)
        rhs = make_rhs(n, mask, rng)
        h = 1.0 / n
        _, truth = cg_until(rhs, mask, h, TIGHT)
        jacobi_steps, _ = jacobi_until(rhs, mask, h)
        cg_steps, _ = cg_until(rhs, mask, h, TOL)
        cycles, fine_sweeps, field = multigrid_until(rhs, mask, h)
        print(
            f"{n:6d} {jacobi_steps:14d} {cg_steps:10d} {cycles:9d} {fine_sweeps:15.1f} {rel_error(field, truth, mask):10.4f}"
        )


if __name__ == "__main__":
    main()
