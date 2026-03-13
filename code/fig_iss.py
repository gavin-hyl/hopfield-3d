"""ISS analysis: trajectory responses and gain scaling."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from common import ASSETS, EQ0, P_NORM, f, lyapunov_v


def compute_d_inf_factor():
    """sup_t ||[sin2t, cos3t, sint]||_2."""
    ts = np.linspace(0, 1000, 100000)
    norms_sq = np.sin(2 * ts) ** 2 + np.cos(3 * ts) ** 2 + np.sin(ts) ** 2
    return np.sqrt(np.max(norms_sq))


SUP_UNIT = compute_d_inf_factor()


def main():
    x0 = EQ0 + np.array([0.5, 0.5, 0.5])
    t_end = 30
    amplitudes = [0.0, 0.05, 0.10, 0.15, 0.30]
    colors = ["#16a34a", "#2563eb", "#d97706", "#dc2626", "#7c3aed"]

    # --- Trajectory plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    for amp, col in zip(amplitudes, colors):
        def rhs(t, x, a=amp):
            d = a * np.array([np.sin(2 * t), np.cos(3 * t), np.sin(t)])
            return f(x) + d

        sol = solve_ivp(rhs, [0, t_end], x0, max_step=0.05, dense_output=True)
        ts = np.linspace(0, t_end, 600)
        vs = np.array([lyapunov_v(sol.sol(t)) for t in ts])
        norms = np.array([np.linalg.norm(sol.sol(t) - EQ0) for t in ts])

        ax1.plot(ts, vs, color=col, lw=1.8, label=f"a = {amp}")
        ax2.plot(ts, norms, color=col, lw=1.8, label=f"a = {amp}")

    ax1.set_yscale("log")
    ax1.set_xlabel("Time", fontsize=11)
    ax1.set_ylabel("V(x)", fontsize=11)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Time", fontsize=11)
    ax2.set_ylabel("‖δx‖", fontsize=11)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{ASSETS}/iss_trajectories.png", dpi=180, facecolor="white")
    plt.close()

    # --- Gain scaling plot ---
    a_vals = np.array([0.01, 0.03, 0.05, 0.08, 0.10, 0.13, 0.15, 0.20, 0.25, 0.30, 0.35])
    ss_errors, rho_bounds = [], []

    for amp in a_vals:
        def rhs(t, x, a=amp):
            d = a * np.array([np.sin(2 * t), np.cos(3 * t), np.sin(t)])
            return f(x) + d

        sol = solve_ivp(rhs, [0, 40], x0, max_step=0.05, dense_output=True)
        ts = np.linspace(35, 40, 300)
        dx_norms = [np.linalg.norm(sol.sol(t) - EQ0) for t in ts]
        ss_errors.append(np.max(dx_norms))
        rho_bounds.append(4 * P_NORM * amp * SUP_UNIT)

    ss_errors = np.array(ss_errors)
    rho_bounds = np.array(rho_bounds)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(a_vals, rho_bounds, "--", color="#7c3aed", lw=1.2, alpha=0.5)
    ax.scatter(a_vals, rho_bounds, color="#7c3aed", s=50, zorder=5,
               edgecolors="white", linewidths=0.5, label="ρ(‖d‖∞) = 4‖P‖·‖d‖∞")
    ax.plot(a_vals, ss_errors, "--", color="#dc2626", lw=1.2, alpha=0.5)
    ax.scatter(a_vals, ss_errors, color="#dc2626", s=50, zorder=5,
               edgecolors="white", linewidths=0.5, label="ss ‖δx‖  (max over last 5s)")

    ax.text(0.98, 0.05, "x* = x*₀ = [−1.47, −2.61, −0.43]",
            transform=ax.transAxes, fontsize=9, color="#555555", ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="#ccc"))

    ax.set_xlabel("Disturbance amplitude  a", fontsize=12)
    ax.set_ylabel("Norm", fontsize=12)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.37)
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(f"{ASSETS}/iss_gain.png", dpi=180, facecolor="white")
    plt.close()


if __name__ == "__main__":
    main()
