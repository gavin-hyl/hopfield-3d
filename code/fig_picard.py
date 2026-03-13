"""Picard iteration convergence vs RK45 ground truth."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from common import ASSETS, B, TAU, W

X0 = np.array([0.5, 0.3, 0.8])
T_SPAN = (0.0, 1.0)
T_EVAL = np.linspace(*T_SPAN, 200)
PICARD_ITERS = [0, 1, 2, 4, 6, 12]


def f(x):
    return (-x + W @ np.tanh(x) + B) / TAU


def picard_step(x_prev_func, t_eval):
    """One Picard iteration: x_{k+1}(t) = x0 + int_0^t f(x_k(s)) ds."""
    dt = t_eval[1] - t_eval[0]
    result = np.zeros((len(t_eval), 3))
    result[0] = X0
    for i in range(1, len(t_eval)):
        result[i] = result[i - 1] + dt * f(x_prev_func[i - 1])
    return result


def main():
    # Ground truth via RK45
    sol = solve_ivp(lambda t, x: f(x), T_SPAN, X0, t_eval=T_EVAL, rtol=1e-12)
    truth = sol.y.T

    # Picard iterates
    iterates = {0: np.tile(X0, (len(T_EVAL), 1))}
    for k in range(1, max(PICARD_ITERS) + 1):
        iterates[k] = picard_step(iterates[k - 1], T_EVAL)

    fig, axes = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
    colors = plt.cm.Blues(np.linspace(0.3, 0.85, len(PICARD_ITERS)))

    for j, ax in enumerate(axes):
        for ci, k in enumerate(PICARD_ITERS):
            ax.plot(
                T_EVAL, iterates[k][:, j],
                "--", color=colors[ci], lw=1, alpha=0.8, label=f"k={k}",
            )
        ax.plot(T_EVAL, truth[:, j], "k-", lw=2, label="RK45")
        ax.set_ylabel(f"$x_{j + 1}$", fontsize=11)
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8, ncol=4, loc="upper right")

    axes[-1].set_xlabel("Time", fontsize=11)
    fig.suptitle("Picard Iteration Convergence vs RK45", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{ASSETS}/picard.png", dpi=180, facecolor="white")
    plt.close()


if __name__ == "__main__":
    main()
