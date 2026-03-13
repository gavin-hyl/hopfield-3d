"""Sensitivity analysis: IC perturbation and parameter perturbation."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from common import ASSETS, B, TAU, W

X0 = np.array([0.5, 0.3, 0.8])
T_SPAN = (0.0, 0.2)
T_EVAL = np.linspace(*T_SPAN, 300)

# Lipschitz constant (computed numerically)
L = 4.037


def f(x):
    return (-x + W @ np.tanh(x) + B) / TAU


def main():
    # Nominal trajectory
    sol_nom = solve_ivp(lambda t, x: f(x), T_SPAN, X0, t_eval=T_EVAL, rtol=1e-12)

    # --- IC perturbation ---
    delta_x0 = X0 / 10
    x0_pert = X0 + delta_x0
    sol_ic = solve_ivp(lambda t, x: f(x), T_SPAN, x0_pert, t_eval=T_EVAL, rtol=1e-12)

    diff_ic = np.linalg.norm(sol_nom.y.T - sol_ic.y.T, axis=1)
    bound_ic = np.linalg.norm(delta_x0) * np.exp(L * T_EVAL)

    # --- Parameter perturbation ---
    delta_b = B / 10
    mu = np.linalg.norm(delta_b) / TAU

    def f_pert(x):
        return (-x + W @ np.tanh(x) + B + delta_b) / TAU

    sol_param = solve_ivp(lambda t, x: f_pert(x), T_SPAN, X0, t_eval=T_EVAL, rtol=1e-12)

    diff_param = np.linalg.norm(sol_nom.y.T - sol_param.y.T, axis=1)
    bound_param = (mu / L) * (np.exp(L * T_EVAL) - 1)

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(T_EVAL, diff_ic, "b-", lw=2, label=r"$\|x(t) - z(t)\|$")
    ax1.plot(T_EVAL, bound_ic, "r--", lw=2, label=r"$\|\delta x_0\| e^{Lt}$")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Norm")
    ax1.set_title("IC Perturbation", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(T_EVAL, diff_param, "b-", lw=2, label=r"$\|x(t) - z(t)\|$")
    ax2.plot(T_EVAL, bound_param, "r--", lw=2, label=r"$(\mu/L)(e^{Lt} - 1)$")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Norm")
    ax2.set_title("Parameter Perturbation", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{ASSETS}/sensitivity.png", dpi=180, facecolor="white")
    plt.close()


if __name__ == "__main__":
    main()
