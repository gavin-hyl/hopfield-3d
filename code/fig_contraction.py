"""Contraction analysis: region where symmetric Jacobian is negative definite."""

import numpy as np
import matplotlib.pyplot as plt

from common import ASSETS, EQ0, P, jacobian


def check_contraction_identity(x):
    """Max eigenvalue of sym(Df(x)) — contracting if < 0."""
    j = jacobian(x)
    sym_j = (j + j.T) / 2
    return np.max(np.linalg.eigvalsh(sym_j))


def check_contraction_p(x):
    """Max eigenvalue of P Df + Df^T P — contracting if < 0."""
    j = jacobian(x)
    m = P @ j + j.T @ P
    return np.max(np.linalg.eigvalsh(m))


def find_contraction_radius(xstar, check_fn, n_dirs=5000):
    """Max radius where contraction holds in all sampled directions."""
    rng = np.random.RandomState(2)
    dirs = rng.randn(n_dirs, 3)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    radii = []
    for u in dirs:
        lo, hi = 0.0, 4.0
        for _ in range(50):
            mid = (lo + hi) / 2
            if check_fn(xstar + mid * u) < 0:
                lo = mid
            else:
                hi = mid
        radii.append(lo)
    return min(radii)


def main():
    r_id = find_contraction_radius(EQ0, check_contraction_identity)
    r_p = find_contraction_radius(EQ0, check_contraction_p)

    # Contraction rates at equilibrium
    j0 = jacobian(EQ0)
    sym_j0 = (j0 + j0.T) / 2
    rate_id = -np.max(np.linalg.eigvalsh(sym_j0))

    m0 = P @ j0 + j0.T @ P
    rate_p_raw = -np.max(np.linalg.eigvalsh(m0))
    rate_p = rate_p_raw / (2 * np.max(np.linalg.eigvalsh(P)))

    print(f"Identity metric: r = {r_id:.3f}, rate = {rate_id:.3f}")
    print(f"P-metric: r = {r_p:.3f}, rate = {rate_p:.3f}")

    # Heatmap: max eigenvalue of sym(Df) on a 2D slice through x*_0
    n_grid = 200
    offsets = np.linspace(-4, 4, n_grid)
    # Slice in the x1-x2 plane, x3 = EQ0[2]
    grid = np.zeros((n_grid, n_grid))
    for i, d1 in enumerate(offsets):
        for j, d2 in enumerate(offsets):
            x = EQ0 + np.array([d1, d2, 0.0])
            grid[j, i] = check_contraction_identity(x)

    fig, ax = plt.subplots(figsize=(6, 5))
    extent = [-4, 4, -4, 4]
    im = ax.imshow(
        grid, extent=extent, origin="lower", cmap="RdYlGn_r",
        vmin=-1.5, vmax=1.5, aspect="equal",
    )
    ax.contour(offsets, offsets, grid, levels=[0], colors="black", linewidths=2)

    # Mark contraction and barrier radii
    theta = np.linspace(0, 2 * np.pi, 200)
    for r, col, ls, label in [
        (r_id, "#d97706", "-.", f"Contraction (r={r_id:.2f})"),
        (2.947, "#dc2626", "--", "Barrier (r=2.95)"),
        (3.06, "#16a34a", "-", "RoA (r=3.06)"),
    ]:
        ax.plot(r * np.cos(theta), r * np.sin(theta), color=col, ls=ls, lw=2, label=label)

    ax.plot(0, 0, "k*", markersize=10)
    ax.set_xlabel("δx₁", fontsize=11)
    ax.set_ylabel("δx₂", fontsize=11)
    ax.set_title("max eig(sym(Df))  (x₁-x₂ slice at x*₀)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    plt.colorbar(im, ax=ax, label="max eigenvalue", shrink=0.8)
    fig.tight_layout()
    fig.savefig(f"{ASSETS}/contraction.png", dpi=180, facecolor="white")
    plt.close()


if __name__ == "__main__":
    main()
