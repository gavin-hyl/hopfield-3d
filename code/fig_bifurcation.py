"""Bifurcation diagram: equilibria vs coupling strength alpha."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

from common import ASSETS, B, TAU, W


def main():
    alphas = np.linspace(0.01, 2.0, 300)
    eq_data = []

    rng = np.random.RandomState(42)

    for alpha in alphas:
        wa = alpha * W

        def func(x, _wa=wa):
            return (-x + _wa @ np.tanh(x) + B) / TAU

        found = {}
        for x0 in rng.uniform(-4, 4, (80, 3)):
            sol, _info, ier, _msg = fsolve(func, x0, full_output=True)
            if ier == 1 and np.linalg.norm(func(sol)) < 1e-10:
                key = tuple(np.round(sol, 4))
                if key not in found:
                    j = (-np.eye(3) + wa @ np.diag(1 - np.tanh(sol) ** 2)) / TAU
                    eigs = np.linalg.eigvals(j)
                    stable = all(e.real < 0 for e in eigs)
                    found[key] = (sol, stable)

        for _key, (sol, stable) in found.items():
            eq_data.append((alpha, sol[0], stable))

    s_a = [d[0] for d in eq_data if d[2]]
    s_x = [d[1] for d in eq_data if d[2]]
    u_a = [d[0] for d in eq_data if not d[2]]
    u_x = [d[1] for d in eq_data if not d[2]]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(s_a, s_x, s=3, c="#16a34a", label="Stable", zorder=3)
    ax.scatter(u_a, u_x, s=3, c="#dc2626", label="Unstable", zorder=3)

    ax.axvline(0.55, color="#888", ls="--", alpha=0.5, lw=1)
    ax.axvline(1.20, color="#888", ls="--", alpha=0.5, lw=1)
    ax.axvline(1.0, color="#2563eb", ls=":", alpha=0.7, lw=1.5)

    ylim = ax.get_ylim()
    ax.text(0.55, ylim[1] * 0.92, "α ≈ 0.55", color="#555", fontsize=9, ha="center")
    ax.text(1.20, ylim[1] * 0.92, "α ≈ 1.20", color="#555", fontsize=9, ha="center")
    ax.text(1.0, ylim[0] * 0.85, "nominal", color="#2563eb", fontsize=9, ha="center")

    ax.set_xlabel("Coupling strength  α", fontsize=12)
    ax.set_ylabel("x₁  equilibrium value", fontsize=12)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{ASSETS}/bifurcation.png", dpi=180, facecolor="white")
    plt.close()


if __name__ == "__main__":
    main()
