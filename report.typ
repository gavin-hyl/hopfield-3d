#set document(title: "Dynamics of a 3-Neuron Hopfield Network", author: "Gavin Hua")
#set page(margin: (x: 1in, y: 1in), numbering: "1")
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.55em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => { v(0.8em); text(size: 12pt, weight: "bold", it); v(0.4em) }
#show heading.where(level: 2): it => { v(0.6em); text(size: 11pt, weight: "bold", it); v(0.3em) }
#set math.equation(numbering: "(1)")
#show figure.caption: set text(size: 9pt)

#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [Dynamics of a 3-Neuron Hopfield Network],
  abstract: [
    We analyze a 3-neuron continuous-time Hopfield network using tools from nonlinear dynamical systems. The system has two stable equilibria (stored memories) and one saddle point. We establish well-posedness via global Lipschitz continuity, certify local exponential stability through linearization and quadratic Lyapunov functions, and prove input-to-state stability under bounded disturbances. Barrier functions certify disjoint forward-invariant sets around each memory, contained within the Lyapunov-based region of attraction. A hybrid system with periodic resets accelerates memory retrieval by up to $2.4 times$, and bifurcation analysis over coupling strength reveals two saddle-node bifurcations governing memory capacity.
  ],
  authors: (
    (
      name: "Gavin Hua",
      organization: [CDS 232 - Nonlinear Dynamics and Control],
      location: [Pasadena, CA],
      email: "ghua@caltech.edu"
    ),
  ),
)

= Introduction

The Hopfield network is a recurrent neural network with symmetric coupling that models associative memory. Introduced by Hopfield in 1982 as one of the first rigorous applications of Lyapunov-like energy arguments to neural computation, the architecture stores patterns as stable equilibria of a continuous dynamical system. 

We study the continuous-time Hopfield network:
$ tau dot(x) = -x + W sigma(x) + b $ <eq:hopfield>
where $x, b in RR^n$, $W in RR^(n times n)$ is symmetric with zero diagonal, and $sigma = tanh$ acts element-wise. The state $x$ represents neuron membrane potentials evolving under mutual excitation/inhibition (through $W sigma(x)$), a passive leak term ($-x$), and a tonic bias current ($b$). We fix $n = 3$, which generates rich dynamical behavior — multiple attractors, bifurcations, and nontrivial basin geometry — while retaining analytic tractability. Throughout this report, we set
$ tau = 1, quad W = mat(0, 2, -1; 2, 0, 1.5; -1, 1.5, 0), quad b = vec(0.1, -0.2, 0.15). $

This report summarizes the analyses performed over the course of the semester, applying the tools of nonlinear dynamics to characterize the Hopfield network's dynamics: existence and uniqueness of solutions, sensitivity to perturbations, equilibrium structure and linearized stability, Lyapunov stability and region of attraction estimates, input-to-state stability under disturbances, barrier functions for forward invariance, hybrid dynamics for accelerated memory retrieval, and bifurcation structure under parameter variation.

= Well-Posedness <sec:lipschitz>

We first establish that @eq:hopfield has unique solutions on all of $RR^n$ by verifying that the right-hand side is globally Lipschitz.

== Global Lipschitz Constant

Define $f(x) = tau^(-1)(-x + W sigma(x) + b)$. The Jacobian is
$ D f(x) = 1/tau (-I + W op("diag")(sigma'(x))). $
Since $sigma = tanh$, we have $|sigma'(x_i)| = |1 - tanh^2(x_i)| <= 1$ for all $x_i$, with equality at $x_i = 0$. Therefore
$ sup_x norm(D f(x)) = 1/tau norm(-I + W op("diag")(sigma'(x)))_2 \
<= 1/tau (1 + norm(W)_2). $
The operator norm is $norm(W)_2 = 3.037$ (the largest singular value of $W$). Since this bound is finite and independent of $x$, the system is globally Lipschitz with constant
$ L <= 1/tau (1 + norm(W)_2) = 4.037. $
Numerical optimization via `scipy.optimize` confirms $L approx 4.037$, achieved at $x = 0$ where all $sigma'(x_i) = 1$.

By the Picard-Lindelof theorem, the system admits a unique solution $x(t)$ on $[0, infinity)$ for any initial condition $x_0 in RR^3$, and by Gronwall's inequality, nearby solutions diverge at most exponentially: $norm(x(t) - z(t)) <= norm(x(0) - z(0)) e^(L t)$.

= Picard Iteration <sec:picard>

To numerically validate the existence and uniqueness theorem, we compare the Picard iterates
$ x^((k+1))(t) = x_0 + integral_0^t f(x^((k))(s)) d s, quad x^((0))(t) = x_0, $
against the RK45 numerical solution on $t in [0, 1]$ with $x_0 = (0.5, 0.3, 0.8)^top$.

#figure(
  image("assets/picard.png", width: 90%),
  caption: [Picard iteration convergence. Dashed lines show iterates $k = 0, 1, 2, 4, 6, 12$; solid black is the RK45 reference. By $k = 12$ the Picard iterate is visually indistinguishable from the true solution.],
) <fig:picard>

@fig:picard shows that by 12 iterations, the Picard iterate has converged to the true solution across all three components. The convergence rate is geometric, consistent with the contraction mapping proof of Picard's theorem.

= Sensitivity Analysis <sec:sensitivity>

We quantify the sensitivity of solutions to perturbations in both initial conditions and parameters, comparing the true trajectory deviation against the analytical bounds.

== Initial Condition Perturbation

We perturb the initial condition by $delta x_0 = x_0 slash 10$ and compare $norm(x(t) - z(t))$ against the Gronwall bound $norm(delta x_0) e^(L t)$ on $t in [0, 0.2]$.

== Parameter Perturbation

We perturb the bias by $delta b = b slash 10$. The perturbed system is $dot(z) = f(z) + g(z)$ where $g(z) = delta b slash tau$. This is a constant perturbation with $norm(g(z)) <= mu = norm(delta b) slash tau approx 0.027$. The bound from eq. 3.15 gives $norm(x(t) - z(t)) <= (mu slash L)(e^(L t) - 1)$.

#figure(
  image("assets/sensitivity.png", width: 95%),
  caption: [Sensitivity analysis on $t in [0, 0.2]$. Left: IC perturbation. Right: parameter perturbation. Blue: true difference. Red dashed: Gronwall bound. The bounds are conservative but correctly capture the exponential growth rate.],
) <fig:sensitivity>

In both cases (@fig:sensitivity), the analytical bounds correctly upper-bound the true deviation. The conservatism is moderate on this short time interval; on longer intervals the exponential bound becomes increasingly loose. Moreover, since we start in the region of attraction near a stable equilibrium, the IC-perturbed trajectory converges to that equilibrium, and therefore has decreasing distance to the unperturbed trajectory. This motivates the Lyapunov-based local analysis in subsequent sections.

= Equilibrium Analysis <sec:equilibria>

== Identifying Equilibria

Equilibria satisfy $f(x^*) = 0$, i.e., $x^* = W tanh(x^*) + b$. We perform multi-start root-finding: 80 random initial guesses in $[-4, 4]^3$ are passed to `scipy.optimize.fsolve`, keeping only solutions with $norm(f(x^*)) < 10^(-10)$, and deduplicating by rounding to 4 decimal places. This yields three equilibria:

#text(size: 8pt)[#table(
  columns: (auto, 1fr, 1fr, auto),
  stroke: 0.5pt,
  inset: 4pt,
  [*Equilibrium*], [*Value*], [*$lambda (D f(x^*))$*], [*Type*],
  [$x_0^*$], [$(-1.47, -2.61, -0.43)$], [$-1.51, -0.90, -0.59$], [Stable],
  [$x_1^*$], [$(-0.15, 0.06, 0.39)$], [$-3.90, 1.06, -0.15$], [Saddle],
  [$x_2^*$], [$(1.45, 2.53, 0.74)$], [$-1.46, -0.65, -0.89$], [Stable],
)]

The two stable nodes ($x_0^*$ and $x_2^*$) serve as stored memories of the Hopfield network. The saddle point $x_1^*$ lies on the basin boundary separating their domains of attraction.

== Linearization at $x_0^*$

At $x_0^*$, the Jacobian is
$ A &= D f(x_0^*) \
&= 1/tau (-I + W op("diag")(sigma'(x_0^*))) \
&= mat(-1.000, 0.382, -0.191; 0.043, -1.000, 0.032; -0.833, 1.249, -1.000). $
All eigenvalues ($lambda_1 = -1.505$, $lambda_2 = -0.901$, $lambda_3 = -0.594$) are real and strictly negative. Since all eigenvalues have nonzero real part, the Hartman--Grobman theorem applies: the nonlinear flow near $x_0^*$ is topologically conjugate to the linearized flow $delta dot(x) = A delta x$. By Theorem 8.3, $x_0^*$ is exponentially stable.

== Linearization at a Non-Equilibrium

At the origin $x_0 = (0, 0, 0)$, which is _not_ an equilibrium since $f(0) = b eq.not 0$, the first-order expansion gives $delta dot(x) approx A_0 delta x + C$ where $A_0 = -I + W$ and $C = f(0) = b$. Hartman--Grobman does not apply here since the origin is not an equilibrium point.

#figure(
  image("assets/phase_portrait.png", width: 65%),
  caption: [3D phase portrait showing trajectories from random initial conditions converging to the two stable equilibria $x_0^*$ (blue) and $x_2^*$ (orange). The saddle $x_1^*$ (red X) sits on the basin boundary.],
) <fig:phase>

= Lyapunov Stability and Region of Attraction <sec:lyapunov>

== Lyapunov Function Construction

We construct a quadratic Lyapunov function for $x_0^*$ by solving the continuous-time Lyapunov equation (CTLE):
$ A^top P + P A = -Q = -I_3 $ <eq:ctle>
where $A = D f(x_0^*)$. Since $A$ defines an globally exponentially stable system, @eq:ctle has a unique positive definite solution:
$ P = mat(0.575, 0.078, -0.236; 0.078, 0.513, 0.300; -0.236, 0.300, 1.071), $
with eigenvalues $lambda_min(P) = 0.277$ and $lambda_max(P) = 1.254$. The candidate Lyapunov function is
$ V(x) = delta x^top P delta x, quad delta x = x - x_0^*. $
Near $x_0^*$, the linearized dynamics give $dot(V) approx delta x^top (A^top P + P A) delta x = -norm(delta x)^2 < 0$, confirming local exponential stability per Theorem 11.1.

== Region of Attraction Estimate

The region of attraction (RoA) is the largest set from which trajectories converge to $x_0^*$. We approximate it by finding the largest Lyapunov sublevel set $Omega_c = {x : V(x) <= c}$ on which $dot(V) < 0$.

We sample 5000 directions uniformly on $S^2$ and, for each direction, perform a binary search for the distance from $x_0^*$ at which $dot(V)$ transitions from negative to non-negative. The minimum across all directions yields a radius of $r = 3.06$, corresponding to a sublevel value of $c = 6.13$.

By Proposition 7.1, $Omega_c$ is forward invariant when $dot(V) <= 0$ on $Omega_c$. By LaSalle's invariance principle (Theorem 10.2), since the only invariant set within ${x in Omega_c : dot(V)(x) = 0}$ is ${x_0^*}$ itself (verifiable by examining the structure of the Jacobian), trajectories converge to $x_0^*$.

= Input-to-State Stability <sec:iss>

We introduce an additive disturbance $d(t) in RR^3$ into the Hopfield network:
$ dot(x) = 1/tau (-x + W sigma(x) + b) + d(t), quad norm(d)_infinity < infinity. $ <eq:disturbed>
Here $norm(d)_infinity = op("ess sup")_(t >= 0) norm(d(t))$. Neurally, $d(t)$ represents unmodeled current injection — synaptic input from external neurons, sensory drive, or channel noise. We analyze ISS using the Lyapunov function $V(x) = delta x^top P delta x$ from @sec:lyapunov.

== ISS Certificate

The Lyapunov derivative along the disturbed flow is:
$ dot(V) = 2 delta x^top P [f(x) + d(t)]. $
Near $x_0^*$, the undisturbed component satisfies $2 delta x^top P f(x) approx -delta x^top Q delta x = -norm(delta x)^2$. The cross-term is bounded by Cauchy--Schwarz:
$ |2 delta x^top P d| <= 2 norm(P) norm(delta x) norm(d), $
where $norm(P) = lambda_max(P) = 1.254$. Combining:
$ dot(V) <= -norm(delta x)^2 + 2 norm(P) norm(delta x) norm(d). $ <eq:vdot-iss>

To obtain an ISS Lyapunov function per Theorem 11.2, we need $dot(V) <= -alpha_3(norm(delta x))$ when $norm(delta x) >= rho(norm(d))$, not merely $dot(V) < 0$. We split the decay term:
$ dot(V) <= -1/2 norm(delta x)^2 - 1/2 norm(delta x)^2 + 2 norm(P) norm(delta x) norm(d). $
Define $rho(s) = 4 norm(P) s$. When $norm(delta x) >= rho(norm(d))$:
$ 2 norm(P) norm(d) <= 1/2 norm(delta x) quad ==> quad -1/2 norm(delta x)^2 + 2 norm(P) norm(delta x) norm(d) <= 0. $
Therefore:
$ norm(delta x) >= rho(norm(d)) quad ==> quad dot(V) <= -1/2 norm(delta x)^2, $
establishing that $V$ is an ISS Lyapunov function with $alpha_3(r) = r^2 slash 2$ and $rho(s) = 4 norm(P) s approx 5.02 s$. By Corollary 11.1, the system is E-ISS.

== Numerical Verification

We simulate @eq:disturbed with sinusoidal disturbances $d(t) = a [sin 2t, cos 3t, sin t]^top$ for $a in {0, 0.05, 0.10, 0.15, 0.30}$. For this signal, $norm(d)_infinity = a dot sup_t norm([sin 2t, cos 3t, sin t])_2 = 1.586 a$ (computed numerically).

#figure(
  image("assets/iss_trajectories.png", width: 95%),
  caption: [ISS trajectories. Left: $V(t)$ on log scale. Right: $norm(delta x)$. Undisturbed trajectory ($a = 0$, green) converges to $x_0^*$. Disturbed trajectories oscillate in neighborhoods whose size scales with $norm(d)_infinity$.],
) <fig:iss-traj>

#figure(
  image("assets/iss_gain.png", width: 55%),
  caption: [ISS gain scaling. Purple: analytical bound $rho = 4 norm(P) dot norm(d)_infinity$. Red: actual steady-state $norm(delta x)$ (max over last 5 seconds of simulation). Both scale linearly with $a$, confirming the linear class $cal(K)$ gain. The analytical bound is approximately $11 times$ conservative.],
) <fig:iss-gain>

@fig:iss-traj and @fig:iss-gain confirm the ISS property. The steady-state error scales linearly as $approx 0.45 norm(d)_infinity$, much tighter than the analytical bound $rho(norm(d)_infinity) = 5.02 norm(d)_infinity$. The conservatism arises from two sources, aside from the theory providing an upper bound: the linearization approximation $2 delta x^top P f(x) approx -norm(delta x)^2$ and the half-splitting step used to extract $alpha_3$.

= Barrier Functions and Forward Invariance <sec:barrier>

We construct barrier functions to certify forward invariance of safe sets around the stable equilibria, following the framework of Lecture 12.

== Ball Barrier

Define $h(x) = r^2 - norm(x - x^*)^2$, so the safe set is $cal(S) = {x : h(x) >= 0}$, a ball of radius $r$ centered at $x^*$. This is the 0-superlevel set of $h$ in the sense of Definition 12.2. The barrier function condition (Definition 12.6) for forward invariance requires:
$ dot(h)(x) >= -alpha(h(x)), quad forall x in cal(S), quad alpha in cal(K)^e_infinity. $
On the boundary $partial cal(S)$ where $h = 0$, this reduces to the Nagumo condition $dot(h) >= 0$ (Theorem 12.2, Case II). Computing:
$ dot(h) = -2(x - x^*)^top f(x). $
We verify this numerically by sampling 10,000 unit directions $hat(u)$ uniformly on $S^2$. For each direction, binary search finds the largest $r$ such that $dot(h)(x^* + r hat(u)) >= 0$. The minimum across all directions gives the certified radius — the largest ball that is provably forward invariant.

Results: $r = 2.947$ at $x_0^*$ and $r = 2.865$ at $x_2^*$. The two balls are disjoint: $norm(x_0^* - x_2^*) = 6.03 > 2.947 + 2.865 = 5.81$, with the saddle $x_1^*$ sitting in the gap between them.

== Lyapunov Sublevel Barrier

By Proposition 12.1, the function $h(x) = c - V(x)$ defines the Lyapunov sublevel set $Omega_c = {x : V(x) <= c}$ as a barrier set. Since $dot(V) < 0$ inside $Omega_c$, we have $dot(h) = -dot(V) > 0$, which is strictly stronger than the barrier condition $dot(h) >= -alpha(h)$. The maximum sublevel where $dot(V)$ remains strictly negative is $c = 6.13$.

Both methods certify valid forward-invariant sets around $x_0^*$, but they characterize different regions -- neither strictly contains the other (@fig:roa-comparison). The ball barrier is a sphere of radius 2.95; $Omega_c$ is an ellipsoid whose semi-axes (2.21 to 4.70) are aligned with the eigenvectors of $P$. In the direction of $P$'s largest eigenvector (where the ellipsoid is narrowest), the ball extends somewhat further. The ellipsoid is larger in total volume by roughly $35%$ and covers more of state space in most directions.


#figure(
  image("assets/barrier.png", width: 80%),
  caption: [Ball barriers around both stable equilibria. Blue: $x_0^*$ ($r = 2.95$). Orange: $x_2^*$ ($r = 2.87$). Trajectories starting inside each ball remain inside and converge. The saddle (red X) sits in the gap between the two disjoint balls.],
) <fig:barrier>

#figure(
  image("assets/roa_comparison.png", width: 100%),
  caption: [Comparison of certified invariant sets at $x_0^*$. _Left:_ 2D cross-section in the plane containing the tightest direction $hat(u)$; the green shading shows ${dot(V) < 0}$, which extends well beyond the plot in most directions. The touch point (triangle) marks where $Omega_c$ just meets the $dot(V) = 0$ boundary. _Right:_ 3D view of the ball barrier (red, $r = 2.95$) and Lyapunov ellipsoid $Omega_c$ (blue, $c = 6.13$) in world coordinates.],
) <fig:roa-comparison>

= Hybrid Dynamics: Associative Memory with Resets <sec:hybrid>

Hopfield networks serve as content-addressable memories, where stable equilibria represent stored patterns. We augment the continuous dynamics with discrete resets to model an accelerated memory retrieval process, using the hybrid systems framework of Lecture 16.

== Hybrid System Formulation

Following Definition 16.1, the hybrid system $cal(H) = (cal(D), S, Delta, f)$ is defined as:
- *Domain:* $cal(D) = RR^3 times [0, T_"reset"]$
- *Guard (switching surface):* $S = RR^3 times {0}$
- *Continuous dynamics:* $dot(x) = f(x), space dot(tau) = -1$
- *Reset map:* $x^+ = x + K(m_"nearest" - x), space tau^+ = T_"reset"$

where $m_"nearest"$ is the nearest stored memory (stable equilibrium) to $x$, $K in (0, 1]$ is the reset gain, and $T_"reset"$ is the reset period. The timer $tau$ decreases linearly until it hits $S = {tau = 0}$, triggering a jump (eq. 16.1) that pushes the state toward the nearest attractor, after which $tau$ resets to $T_"reset"$ and continuous flow resumes.

== Results

Starting from the origin (approximately equidistant from both memories), convergence times to within $epsilon = 0.1$ of a memory are:

#align(center)[
#table(
  columns: 2,
  stroke: 0.5pt,
  inset: 6pt,
  [*Configuration*], [*Convergence time*],
  [continuous], [7.05 s],
  [Hybrid ($T_"reset" = 2.0$, $K = 0.3$)], [5.49 s],
  [Hybrid ($T_"reset" = 1.0$, $K = 0.5$)], [3.00 s],
)
]

#figure(
  image("assets/hybrid.png", width: 65%),
  caption: [Hybrid dynamics: distance to nearest memory over time. Gray: pure continuous dynamics. Colored: hybrid systems with periodic resets. More frequent, stronger resets accelerate convergence by up to $2.4 times$.],
) <fig:hybrid>

The periodic resets effectively nudge the trajectory toward the nearest attractor basin, accelerating convergence by up to $2.4 times$ (@fig:hybrid). 

= Bifurcation Analysis <sec:bifurcation>

We parameterize the weight matrix as $W -> alpha W$ and sweep $alpha in [0, 2]$ to study how the equilibrium structure changes with coupling strength.

== Bifurcation Structure

For each $alpha$ in a grid of 300 values, we repeat the multi-start root-finding procedure from @sec:equilibria (80 random initial guesses, fsolve, deduplication, eigenvalue-based stability classification).

For small $alpha$, the leak term $-x$ dominates and $g(x) = alpha W tanh(x) + b$ is a contraction. As $alpha$ increases, the feedback through $W$ becomes strong enough to sustain self-reinforcing saturated states:

- *$alpha approx 0.55$:* A saddle-node bifurcation creates two new equilibria (one stable, one saddle).The network can now store one extra memory.
- *$alpha approx 1.20$:* A second saddle-node bifurcation creates two more equilibria, bringing the total to 5 (2 stable, 3 unstable).

At the nominal parameter $alpha = 1$, the system has 3 equilibria (2 stable, 1 unstable), consistent with all analyses in previous sections.

#figure(
  image("assets/bifurcation.png", width: 70%),
  caption: [Bifurcation diagram. Green: stable equilibria. Red: unstable equilibria. Dashed lines mark the two saddle-node bifurcations. Blue dotted line: nominal $alpha = 1$.],
) <fig:bifurcation>

== Interpretation

The memory capacity of the network is directly controlled by the coupling strength $alpha$. The classical Hopfield result that an $n$-neuron network can store approximately $0.14 n$ patterns corresponds to the regime where coupling is strong enough to support multiple stable equilibria without destabilizing all of them. In our 3-neuron system with the given $W$ structure, the maximum memory capacity is 2 (achieved for $alpha > 1.20$). Stronger coupling creates more attractors but also more unstable equilibria (basin boundaries), and the basin geometry becomes increasingly complex.

= Conclusion

We have demonstrated this low-dimensional 3-neuron Hopfield network exhibits rich nonlinear phenomena. The key results are:

+ *Well-posedness:* Global Lipschitz continuity ($L = 4.037$) guarantees unique solutions for all time, validated by Picard iteration convergence.
+ *Equilibrium structure:* Three equilibria: two stable nodes (stored memories) and one saddle (basin boundary), with exponential stability certified by linearization and the Hartman--Grobman theorem.
+ *Lyapunov analysis:* A quadratic Lyapunov function from the CTLE certifies local exponential stability with an RoA radius of 3.06, and LaSalle's invariance principle (Theorem 10.2) upgrades stability to asymptotic convergence.
+ *ISS:* The system is E-ISS under bounded disturbances (Corollary 11.1). The analytical bound is approximately $11 times$ conservative compared to numerical steady-state behavior.
+ *Barrier functions:* Ball barriers (Theorem 12.2) certify forward-invariant spheres of radius 2.95 and 2.87 around the two memories. Lyapunov sublevel barriers (Proposition 12.1) certify an ellipsoidal set ($c = 6.13$, semi-axes 2.21–4.70). Both are valid but neither contains the other; the ellipsoid covers ~35% more volume while the ball extends further in $P$'s largest-eigenvector direction.
+ *Hybrid dynamics:* Periodic resets toward stored memories accelerate convergence by up to $2.4 times$.
+ *Bifurcation:* Two saddle-node bifurcations at $alpha approx 0.55$ and $alpha approx 1.20$ control memory capacity, connecting the network's computational function to its coupling strength.