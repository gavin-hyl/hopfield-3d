"""Ball barrier and Lyapunov sublevel barrier analysis."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from common import ASSETS, EQ0, EQ1, EQ2, f, lyapunov_v, lyapunov_vdot


def find_barrier_radius(xstar, n_dirs=10000):
    """Binary search for max r where h_dot >= 0 on ||x - x*|| = r."""
    rng = np.random.RandomState(0)
    dirs = rng.randn(n_dirs, 3)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    def hdot(x):
        return -2 * (x - xstar) @ f(x)

    radii = []
    for u in dirs:
        lo, hi = 0.0, 5.0
        for _ in range(50):
            mid = (lo + hi) / 2
            if hdot(xstar + mid * u) >= 0:
                lo = mid
            else:
                hi = mid
        radii.append(lo)
    return min(radii)


def find_lyapunov_sublevel(xstar, n_dirs=5000):
    """Max sublevel c where V_dot < 0 everywhere in {V <= c}.

    Returns (c, r_min, u_tight) where u_tight is the tightest direction.
    """
    rng = np.random.RandomState(1)
    dirs = rng.randn(n_dirs, 3)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    radii = []
    for u in dirs:
        lo, hi = 0.0, 6.0
        for _ in range(50):
            mid = (lo + hi) / 2
            x = xstar + mid * u
            if lyapunov_vdot(x, xstar) < 0:
                lo = mid
            else:
                hi = mid
        radii.append(lo)

    idx = int(np.argmin(radii))
    r_min = radii[idx]
    u_tight = dirs[idx]
    c = lyapunov_v(xstar + r_min * u_tight, xstar)
    return c, r_min, u_tight


def sphere_wireframe(center, r, n_lat=20, n_lon=20):
    u = np.linspace(0, 2 * np.pi, n_lon)
    v = np.linspace(0, np.pi, n_lat)
    x = center[0] + r * np.outer(np.cos(u), np.sin(v))
    y = center[1] + r * np.outer(np.sin(u), np.sin(v))
    z = center[2] + r * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def main():
    r0 = find_barrier_radius(EQ0)
    r2 = find_barrier_radius(EQ2)
    print(f"Ball barrier radius at x*_0: {r0:.3f}")
    print(f"Ball barrier radius at x*_2: {r2:.3f}")

    # Generate trajectories inside each ball
    rng = np.random.RandomState(7)
    trajs = {0: [], 2: []}
    for xstar, r, key in [(EQ0, r0, 0), (EQ2, r2, 2)]:
        for _ in range(12):
            u = rng.randn(3)
            u /= np.linalg.norm(u)
            x0 = xstar + r * rng.uniform(0.55, 0.92) * u
            sol = solve_ivp(lambda t, x: f(x), [0, 15], x0, max_step=0.05, dense_output=True)
            ts = np.linspace(0, 15, 300)
            trajs[key].append(np.array([sol.sol(t) for t in ts]))

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    for xs, ys, zs, col in [
        (*sphere_wireframe(EQ0, r0), "#2563eb"),
        (*sphere_wireframe(EQ2, r2), "#ea580c"),
    ]:
        ax.plot_surface(xs, ys, zs, alpha=0.06, color=col, shade=False)
        ax.plot_wireframe(xs, ys, zs, alpha=0.08, color=col, lw=0.3, rstride=3, cstride=3)

    for traj in trajs[0]:
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="#3b82f6", alpha=0.5, lw=0.9)
        ax.scatter(*traj[0], color="#3b82f6", s=12, alpha=0.7)
    for traj in trajs[2]:
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="#f97316", alpha=0.5, lw=0.9)
        ax.scatter(*traj[0], color="#f97316", s=12, alpha=0.7)

    ax.scatter(*EQ0, s=120, c="#1d4ed8", marker="*", zorder=10, edgecolors="k", linewidths=0.5)
    ax.scatter(*EQ2, s=120, c="#c2410c", marker="*", zorder=10, edgecolors="k", linewidths=0.5)
    ax.scatter(*EQ1, s=150, c="#dc2626", marker="X", zorder=10, edgecolors="k", linewidths=0.8)

    ax.text(*EQ0 + [-0.5, -0.5, 0.5], f"x*₀ (r={r0:.2f})", color="#1d4ed8", fontsize=10)
    ax.text(*EQ2 + [0.3, 0.3, 0.5], f"x*₂ (r={r2:.2f})", color="#c2410c", fontsize=10)
    ax.text(EQ1[0], EQ1[1], EQ1[2] + 0.6, "saddle", color="#dc2626", fontsize=10, ha="center")

    ax.set_xlabel("x₁", fontsize=10)
    ax.set_ylabel("x₂", fontsize=10)
    ax.set_zlabel("x₃", fontsize=10)
    ax.view_init(elev=18, azim=-30)
    ax.tick_params(labelsize=8)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.set_facecolor("#f9fafb")

    fig.tight_layout()
    fig.savefig(f"{ASSETS}/barrier.png", dpi=180, facecolor="white")
    plt.close()

    # Certified region comparison: ball barrier vs Lyapunov ellipsoid.
    # Left panel: 2D cross-section in the plane containing u_tight.
    # Right panel: 3D view of both surfaces.
    from matplotlib.patches import Circle
    from common import P, lyapunov_vdot

    c_lya, r_min, u_tight = find_lyapunov_sublevel(EQ0)
    barrier_r = r0

    # Build orthonormal basis {u_tight, u_perp} for the 2D slice.
    _, eigvecs_P = np.linalg.eigh(P)
    eigvals_P = np.linalg.eigvalsh(P)
    e_max_P = eigvecs_P[:, 2]
    u_perp = e_max_P - np.dot(e_max_P, u_tight) * u_tight
    u_perp /= np.linalg.norm(u_perp)

    # --- 2D data ---------------------------------------------------------
    PLOT_LIM = 8.0
    thetas = np.linspace(0, 2 * np.pi, 800)
    vdot_radii, ellipse_radii = [], []
    for theta in thetas:
        u = np.cos(theta) * u_tight + np.sin(theta) * u_perp
        lo, hi = 0.0, PLOT_LIM * 1.5
        for _ in range(64):
            mid = (lo + hi) / 2
            if lyapunov_vdot(EQ0 + mid * u) < 0:
                lo = mid
            else:
                hi = mid
        vdot_radii.append(lo)
        uPu = u @ P @ u
        ellipse_radii.append(np.sqrt(c_lya / uPu))

    # Clip V_dot=0 boundary to plot window before filling
    vdot_pts = np.array([[r * np.cos(t), r * np.sin(t)]
                          for r, t in zip(vdot_radii, thetas)])
    vdot_pts = np.clip(vdot_pts, -PLOT_LIM, PLOT_LIM)
    ell_x = [r * np.cos(t) for r, t in zip(ellipse_radii, thetas)]
    ell_y = [r * np.sin(t) for r, t in zip(ellipse_radii, thetas)]

    # --- 3D ellipsoid surface --------------------------------------------
    n_lat, n_lon = 40, 40
    u_s = np.linspace(0, 2 * np.pi, n_lon)
    v_s = np.linspace(0, np.pi, n_lat)
    # Unit sphere parametrisation
    Xs = np.outer(np.cos(u_s), np.sin(v_s))
    Ys = np.outer(np.sin(u_s), np.sin(v_s))
    Zs = np.outer(np.ones_like(u_s), np.cos(v_s))
    # Scale by semi-axes in eigenvector frame, then rotate to world
    semi_axes = np.sqrt(c_lya / eigvals_P)
    P_eig = np.array([semi_axes[0] * Xs, semi_axes[1] * Ys, semi_axes[2] * Zs])
    P_world = np.einsum("il,ljk->ijk", eigvecs_P, P_eig)
    Xe = EQ0[0] + P_world[0]
    Ye = EQ0[1] + P_world[1]
    Ze = EQ0[2] + P_world[2]

    # --- Figure ----------------------------------------------------------
    fig = plt.figure(figsize=(12, 5.5))
    ax2d = fig.add_subplot(1, 2, 1)
    ax3d = fig.add_subplot(1, 2, 2, projection="3d")

    # 2D panel
    ax2d.fill(vdot_pts[:, 0], vdot_pts[:, 1], color="#d1fae5", alpha=0.5, zorder=0)
    ax2d.plot(vdot_pts[:, 0], vdot_pts[:, 1], color="#16a34a", lw=1.2, ls="--",
              alpha=0.8, label=r"$\dot{V}=0$ boundary", zorder=1)
    ax2d.fill(ell_x, ell_y, color="#bfdbfe", alpha=0.55, zorder=2)
    ax2d.plot(ell_x, ell_y, color="#2563eb", lw=2,
              label=rf"$\Omega_c$ ($c={c_lya:.2f}$)", zorder=2)
    circ = Circle((0, 0), barrier_r, fill=True, facecolor="#fecaca", alpha=0.30,
                  edgecolor="#dc2626", lw=2, zorder=3,
                  label=rf"Ball barrier ($r={barrier_r:.2f}$)")
    ax2d.add_patch(circ)
    ax2d.plot(r_min, 0, "v", color="#16a34a", ms=8, zorder=5,
              label=f"Touch pt ($r={r_min:.2f}$)")
    ax2d.plot(0, 0, "ko", ms=6, zorder=10)
    ax2d.text(0.15, -0.3, r"$x_0^*$", fontsize=11, fontweight="bold")
    ax2d.set_xlim(-PLOT_LIM, PLOT_LIM)
    ax2d.set_ylim(-PLOT_LIM, PLOT_LIM)
    ax2d.set_aspect("equal")
    ax2d.legend(fontsize=8.5, loc="upper left")
    ax2d.grid(True, alpha=0.25)
    ax2d.set_xlabel(r"$\hat{u}_\mathrm{tight}$ direction", fontsize=10)
    ax2d.set_ylabel(r"orthogonal direction", fontsize=10)
    ax2d.set_title("2D cross-section\n(plane through tightest direction)", fontsize=10)  # noqa

    # 3D panel — ellipsoid
    ax3d.plot_surface(Xe, Ye, Ze, alpha=0.08, color="#2563eb", shade=False)
    ax3d.plot_wireframe(Xe, Ye, Ze, alpha=0.15, color="#2563eb",
                        lw=0.4, rstride=3, cstride=3)
    # 3D panel — sphere
    Xsph, Ysph, Zsph = sphere_wireframe(EQ0, barrier_r, n_lat=30, n_lon=30)
    ax3d.plot_surface(Xsph, Ysph, Zsph, alpha=0.08, color="#dc2626", shade=False)
    ax3d.plot_wireframe(Xsph, Ysph, Zsph, alpha=0.15, color="#dc2626",
                        lw=0.4, rstride=3, cstride=3)
    # Equilibrium star
    ax3d.scatter(*EQ0, s=120, c="#1d4ed8", marker="*",
                 zorder=10, edgecolors="k", linewidths=0.5)
    ax3d.text(*EQ0 + [-0.5, -0.5, 0.5], r"$x_0^*$", color="#1d4ed8", fontsize=9)
    # Proxy artists for legend
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], color="#2563eb", lw=2, label=rf"$\Omega_c$ ($c={c_lya:.2f}$)"),
        Line2D([0], [0], color="#dc2626", lw=2, label=rf"Ball barrier ($r={barrier_r:.2f}$)"),
    ]
    ax3d.legend(handles=legend_els, fontsize=8.5, loc="upper left")
    ax3d.set_xlabel("$x_1$", fontsize=9)
    ax3d.set_ylabel("$x_2$", fontsize=9)
    ax3d.set_zlabel("$x_3$", fontsize=9)
    ax3d.view_init(elev=20, azim=-40)
    ax3d.tick_params(labelsize=7)
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.set_facecolor("#f9fafb")
    ax3d.set_title("3D surfaces\n(world coordinates)", fontsize=10)

    fig.suptitle(
        rf"Certified invariant sets at $x_0^*$: ball barrier vs $\Omega_c$",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(f"{ASSETS}/roa_comparison.png", dpi=180,
                facecolor="white", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
