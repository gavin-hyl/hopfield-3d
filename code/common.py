"""Shared system parameters and utilities for the Hopfield network project."""

import numpy as np
from scipy.linalg import solve_continuous_lyapunov

# System parameters
TAU = 1.0
W = np.array([[0, 2, -1], [2, 0, 1.5], [-1, 1.5, 0]])
B = np.array([0.1, -0.2, 0.15])
N = 3

# Equilibria (from multi-start search)
EQ0 = np.array([-1.470, -2.613, -0.434])  # stable node
EQ1 = np.array([-0.148, 0.061, 0.388])  # saddle
EQ2 = np.array([1.448, 2.531, 0.736])  # stable node

# Jacobian at an equilibrium
A0 = (-np.eye(N) + W @ np.diag(1 - np.tanh(EQ0) ** 2)) / TAU

# CTLE solution: A^T P + P A = -I
P = solve_continuous_lyapunov(A0.T, -np.eye(N))
P_EIGVALS = np.linalg.eigvalsh(P)
P_NORM = P_EIGVALS.max()  # lambda_max(P)
P_MIN = P_EIGVALS.min()  # lambda_min(P)

ASSETS = "../assets"


def f(x):
    """Hopfield network dynamics f(x) = (1/tau)(-x + W tanh(x) + b)."""
    return (-x + W @ np.tanh(x) + B) / TAU


def jacobian(x):
    """Jacobian Df(x) = (1/tau)(-I + W diag(sech^2(x)))."""
    return (-np.eye(N) + W @ np.diag(1 - np.tanh(x) ** 2)) / TAU


def lyapunov_v(x, xstar=EQ0):
    """Quadratic Lyapunov function V(x) = dx^T P dx."""
    dx = x - xstar
    return dx @ P @ dx


def lyapunov_vdot(x, xstar=EQ0):
    """Time derivative of V along f."""
    dx = x - xstar
    return 2 * dx @ P @ f(x)
