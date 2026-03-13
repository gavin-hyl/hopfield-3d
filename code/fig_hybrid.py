"""Hybrid dynamics: associative memory with periodic resets."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from common import ASSETS, EQ0, EQ2, f

MEMORIES = [EQ0, EQ2]


def simulate_hybrid(x0, t_reset, k_gain, t_end=10):
    """Simulate hybrid system with periodic resets toward nearest memory."""
    ts_all, xs_all = [0.0], [x0.copy()]
    x, t = x0.copy(), 0.0
    while t < t_end:
        t_next = min(t + t_reset, t_end)
        sol = solve_ivp(lambda s, y: f(y), [t, t_next], x, max_step=0.02, dense_output=True)
        tt = np.linspace(t + 1e-8, t_next, max(2, int((t_next - t) * 50)))
        for ti in tt:
            ts_all.append(ti)
            xs_all.append(sol.sol(ti))
        x = sol.sol(t_next)
        dists = [np.linalg.norm(x - m) for m in MEMORIES]
        m_near = MEMORIES[np.argmin(dists)]
        x = x + k_gain * (m_near - x)
        t = t_next
    return np.array(ts_all), np.array(xs_all)


def dist_to_nearest(x):
    return min(np.linalg.norm(x - m) for m in MEMORIES)


def main():
    x0 = np.array([0.0, 0.0, 0.0])
    t_end = 10

    # Pure continuous
    sol_cont = solve_ivp(lambda t, x: f(x), [0, t_end], x0, max_step=0.02, dense_output=True)
    t_cont = np.linspace(0, t_end, 500)
    d_cont = [dist_to_nearest(sol_cont.sol(t)) for t in t_cont]

    # Hybrid configs
    configs = [
        (2.0, 0.3, "#d97706", "T=2.0, K=0.3"),
        (1.0, 0.5, "#16a34a", "T=1.0, K=0.5"),
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t_cont, d_cont, color="#6b7280", lw=2, label="Continuous only (7.05s)")

    for t_reset, k_gain, col, label in configs:
        ts, xs = simulate_hybrid(x0, t_reset, k_gain, t_end)
        d_h = [dist_to_nearest(x) for x in xs]
        ax.plot(ts, d_h, color=col, lw=1.8, label=label)

    ax.axhline(0.1, color="#dc2626", ls="--", lw=1, alpha=0.6)
    ax.text(9.5, 0.15, "ε = 0.1", color="#dc2626", fontsize=9, ha="right")

    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Distance to nearest memory", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.1)
    fig.tight_layout()
    fig.savefig(f"{ASSETS}/hybrid.png", dpi=180, facecolor="white")
    plt.close()


if __name__ == "__main__":
    main()
