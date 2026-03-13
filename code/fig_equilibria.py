"""3D phase portrait: trajectories converging to stable equilibria."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from common import ASSETS, EQ0, EQ1, EQ2, f


def main():
    np.random.seed(42)
    n_traj = 30
    t_span = (0, 15)
    t_eval = np.linspace(*t_span, 300)

    trajectories, attractors = [], []
    for _ in range(n_traj):
        x0 = np.random.uniform(-3.5, 3.5, 3)
        sol = solve_ivp(lambda t, x: f(x), t_span, x0, t_eval=t_eval, max_step=0.05)
        if sol.success and sol.y.shape[1] == len(t_eval):
            xf = sol.y[:, -1]
            d0, d2 = np.linalg.norm(xf - EQ0), np.linalg.norm(xf - EQ2)
            if min(d0, d2) < 0.5:
                trajectories.append(sol.y)
                attractors.append(0 if d0 < d2 else 2)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    for traj, att in zip(trajectories, attractors):
        col = "#3b82f6" if att == 0 else "#f97316"
        ax.plot(traj[0], traj[1], traj[2], color=col, alpha=0.4, lw=0.8)

    ax.scatter(*EQ0, s=150, c="#1d4ed8", marker="*", zorder=10, edgecolors="k", linewidths=0.5)
    ax.scatter(*EQ2, s=150, c="#c2410c", marker="*", zorder=10, edgecolors="k", linewidths=0.5)
    ax.scatter(*EQ1, s=100, c="#dc2626", marker="X", zorder=10, edgecolors="k", linewidths=0.5)

    ax.text(*EQ0 + [0.2, -0.5, 0.4], "x*₀ (stable)", color="#1d4ed8", fontsize=9)
    ax.text(*EQ2 + [0.2, 0.3, 0.4], "x*₂ (stable)", color="#c2410c", fontsize=9)
    ax.text(*EQ1 + [0.3, 0.3, 0.3], "x*₁ (saddle)", color="#dc2626", fontsize=9)

    ax.set_xlabel("x₁", fontsize=10)
    ax.set_ylabel("x₂", fontsize=10)
    ax.set_zlabel("x₃", fontsize=10)
    ax.view_init(elev=22, azim=-35)
    ax.tick_params(labelsize=8)

    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.set_facecolor("#f9fafb")
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"]["color"] = "#d1d5db"

    fig.tight_layout()
    fig.savefig(f"{ASSETS}/phase_portrait.png", dpi=180, facecolor="white")
    plt.close()


if __name__ == "__main__":
    main()
