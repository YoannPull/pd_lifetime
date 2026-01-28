#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lifetime PD extrapolation demo from sparse calibration points (1Y and 3Y).

We consider a portfolio segmented into *ordered* risk grades (1=best ... 5=worst).
For each grade g, we only observe:
- N_g : cohort size at t=0
- D_g(1) : cumulative defaults by 1 year
- D_g(3) : cumulative defaults by 3 years

Goal
----
Illustrate why fitting a *separate* parametric lifetime curve per grade (e.g., Weibull)
can yield *crossing* lifetime PD curves (violating the grade order) when extrapolated
to long horizons.

We compare:
(A) Weibull fitted independently by grade (may cross at long horizon)
(B) A "coherent" construction:
    - Proportional Hazards (PH): F_g(t) = 1 - exp(-exp(eta_g) * Lambda0(t))
    - Baseline cumulative hazard Lambda0(t) is piecewise-linear (piecewise-constant hazard)
      with knots at 1Y and 3Y (the only observed horizons).
    - Grade effects are constrained to be monotone (ordered): eta_1 <= eta_2 <= ... <= eta_5
      which implies no-crossing of F_g(t) for any t.
    - Tail regularization: we link the post-3Y hazard to the 1-3Y hazard as h3 = kappa*h2,
      and penalize deviations of kappa from a prior value kappa0 (log-normal style penalty).

Outputs
-------
Figures are saved to: outputs/figures/
- lifetime_full_0_50y.png : full horizon 0–50Y, Weibull vs coherent PH
- lifetime_zoom_0_10y.png : zoom horizon 0–10Y to better visualize early years

Dependencies
------------
- numpy
- scipy
- matplotlib
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# =============================================================================
# Utility functions
# =============================================================================
def softplus(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable softplus: log(1 + exp(x)).

    We use softplus to map unconstrained optimization variables (R) to strictly
    positive parameters (R_+), e.g., hazards and Weibull parameters.
    """
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def ensure_dir(path: str) -> None:
    """Create directory `path` if it does not exist."""
    os.makedirs(path, exist_ok=True)


def log_binom_pmf_wo_const(k: int, n: int, p: float) -> float:
    """
    Binomial log-PMF up to an additive constant.

    For MLE / MAP optimization, the combinatorial term log(n choose k) does not
    depend on the model parameter p, so we can safely drop it.

    Parameters
    ----------
    k : int
        Number of "successes" (defaults)
    n : int
        Number of trials (exposures)
    p : float
        Success probability

    Returns
    -------
    float
        k*log(p) + (n-k)*log(1-p) (with p clipped for stability)
    """
    p = float(np.clip(p, 1e-15, 1.0 - 1e-15))
    return k * np.log(p) + (n - k) * np.log(1.0 - p)


# =============================================================================
# Model primitives
# =============================================================================
def weibull_F(t: np.ndarray, k: float, alpha: float) -> np.ndarray:
    """
    Weibull cumulative distribution function.

    F(t) = 1 - exp(-(t/alpha)^k), for t >= 0.

    In credit terms, we interpret F(t) as the *cumulative default probability*
    up to horizon t (lifetime PD).

    Parameters
    ----------
    t : array
        Horizons (years)
    k : float
        Shape parameter (>0)
    alpha : float
        Scale parameter (>0)

    Returns
    -------
    array
        Cumulative PD F(t)
    """
    t = np.asarray(t, dtype=float)
    H = (t / alpha) ** k  # cumulative hazard under Weibull
    return 1.0 - np.exp(-H)


def piecewise_Lambda0(t: np.ndarray, h1: float, h2: float, h3: float) -> np.ndarray:
    """
    Baseline cumulative hazard Lambda0(t) under piecewise-constant hazard.

    We use a 3-segment hazard:
      - (0,1]  : hazard = h1
      - (1,3]  : hazard = h2
      - (3,∞)  : hazard = h3

    Then Lambda0(t) is piecewise-linear:
      - t <= 1:        Lambda0(t) = h1 * t
      - 1 < t <= 3:    Lambda0(t) = h1 + h2*(t-1)
      - t > 3:         Lambda0(t) = h1 + 2*h2 + h3*(t-3)

    Note:
    -----
    With sparse information (only 1Y and 3Y), this is a deliberately simple
    baseline: it ensures exact interpretability at anchors and provides a
    controlled tail behavior through h3.

    Parameters
    ----------
    t : array
        Horizons (years)
    h1, h2, h3 : float
        Segment hazards (>0)

    Returns
    -------
    array
        Baseline cumulative hazard Lambda0(t)
    """
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)

    # Segment 1: (0,1]
    m1 = (t <= 1.0)
    out[m1] = h1 * t[m1]

    # Segment 2: (1,3]
    m2 = (t > 1.0) & (t <= 3.0)
    out[m2] = h1 + h2 * (t[m2] - 1.0)

    # Segment 3: (3,∞)
    m3 = (t > 3.0)
    out[m3] = h1 + 2.0 * h2 + h3 * (t[m3] - 3.0)

    return out


def ph_F_from_Lambda0(eta: float, Lambda0_t: np.ndarray) -> np.ndarray:
    """
    Proportional Hazards (PH) cumulative default probability.

    In PH:
      H_g(t) = exp(eta_g) * Lambda0(t)
      F_g(t) = 1 - exp(-H_g(t))

    where eta_g is the (log) hazard multiplier for grade g
    and Lambda0(t) is the baseline cumulative hazard.

    Key property:
    -------------
    If eta_1 <= eta_2 <= ... <= eta_G, then for any t,
    F_1(t) <= F_2(t) <= ... <= F_G(t) (no crossing).

    Parameters
    ----------
    eta : float
        Grade effect (real)
    Lambda0_t : array
        Baseline cumulative hazard evaluated on t-grid

    Returns
    -------
    array
        Cumulative PD curve F_g(t)
    """
    H = np.exp(eta) * Lambda0_t
    return 1.0 - np.exp(-H)


# =============================================================================
# Data container
# =============================================================================
@dataclass(frozen=True)
class GradeCounts:
    """
    Minimal grade-level default count information at 1Y and 3Y.

    grade: 1=best ... 5=worst (assumed ordered)
    N:     cohort size at t=0
    D1:    cumulative defaults by 1 year
    D3:    cumulative defaults by 3 years
    """
    grade: int
    N: int
    D1: int
    D3: int

    @property
    def D13(self) -> int:
        """Defaults occurring in (1,3] (conditional on survival up to 1Y)."""
        return int(self.D3 - self.D1)

    @property
    def N_surv1(self) -> int:
        """Number of survivors after 1Y (used as denominator for conditional default in (1,3])."""
        return int(self.N - self.D1)

    @property
    def p1_hat(self) -> float:
        """Empirical cumulative PD at 1Y: D1/N."""
        return float(self.D1 / self.N)

    @property
    def p3_hat(self) -> float:
        """Empirical cumulative PD at 3Y: D3/N."""
        return float(self.D3 / self.N)


def validate_data(grades: Dict[int, GradeCounts]) -> None:
    """
    Basic integrity checks on the input.

    This avoids nonsensical input such as negative defaults or D3 < D1.
    """
    for g, gd in grades.items():
        if gd.D1 < 0 or gd.D3 < 0:
            raise ValueError(f"Negative defaults for grade {g}.")
        if gd.D1 > gd.N or gd.D3 > gd.N:
            raise ValueError(f"D>N for grade {g}.")
        if gd.D3 < gd.D1:
            raise ValueError(f"D3<D1 for grade {g}.")


# =============================================================================
# Likelihoods (objective functions)
# =============================================================================
def negloglik_weibull_per_grade(params: np.ndarray, data: GradeCounts) -> float:
    """
    Negative log-likelihood for a Weibull curve fitted to *one grade*.

    We use a coherent 2-point likelihood:
      - D1 ~ Binom(N, F(1))
      - D13 | survival to 1Y ~ Binom(N - D1, p13)
        where p13 is the conditional default probability in (1,3]:
            p13 = (F(3) - F(1)) / (1 - F(1))

    This is important: it respects the nesting of cumulative PDs and avoids
    treating D1 and D3 as independent binomials.

    Parameters
    ----------
    params : array
        Unconstrained parameters [u_k, u_alpha]
        mapped to (k, alpha) via softplus to ensure positivity.
    data : GradeCounts
        Grade-level counts

    Returns
    -------
    float
        Negative log-likelihood (up to an additive constant)
    """
    # Unconstrained -> positive
    u_k, u_a = float(params[0]), float(params[1])
    k = float(softplus(np.array([u_k]))[0] + 1e-6)
    alpha = float(softplus(np.array([u_a]))[0] + 1e-6)

    # Cumulative PD at 1Y and 3Y
    F1 = float(weibull_F(np.array([1.0]), k, alpha)[0])
    F3 = float(weibull_F(np.array([3.0]), k, alpha)[0])

    # Numerical safety + ensure F3 > F1
    F1 = float(np.clip(F1, 1e-12, 1.0 - 1e-12))
    F3 = float(np.clip(F3, F1 + 1e-12, 1.0 - 1e-12))

    # Conditional PD in (1,3] given survival at 1Y
    p13 = (F3 - F1) / (1.0 - F1)
    p13 = float(np.clip(p13, 1e-12, 1.0 - 1e-12))

    # Binomial log-likelihood (constants dropped)
    ll = 0.0
    ll += log_binom_pmf_wo_const(data.D1, data.N, F1)
    ll += log_binom_pmf_wo_const(data.D13, data.N_surv1, p13)

    return -ll


def negloglik_ph_ordered(
    x: np.ndarray,
    grades: Dict[int, GradeCounts],
    *,
    kappa0: float = 1.0,
    sigma_logkappa: float = 0.20,
    sigma_eta: float = 5.0,
) -> float:
    """
    Negative log-posterior (NLL + penalties) for the coherent PH model.

    Parameterization
    ----------------
    x = [u_h1, u_h2, u_kappa, z2, z3, z4, z5]

    Baseline hazards (positive via softplus):
      h1 = softplus(u_h1)
      h2 = softplus(u_h2)
      kappa = softplus(u_kappa)
      h3 = kappa * h2   (tail hazard tied to mid-term hazard)

    Ordered grade effects:
      deltas_j = softplus(z_j) for j=2..5 ensure deltas_j >= 0
      eta_1 = 0 (identification)
      eta_2 = delta2
      eta_3 = delta2 + delta3
      ...
      eta_5 = delta2 + delta3 + delta4 + delta5

    Likelihood
    ----------
    Same coherent 2-point structure as above:
      D1 ~ Binom(N, F(1))
      D13 | survival at 1 ~ Binom(N-D1, p13)

    Regularization ("priors")
    -------------------------
    1) Tail regularization on kappa (log scale):
         (log kappa - log kappa0)^2 / sigma_logkappa^2
       This controls long-horizon behavior with only 1Y/3Y information.

    2) Weak shrinkage on eta_g (g>=2):
         eta_g^2 / sigma_eta^2
       Prevents unrealistic separation when data is extremely sparse.

    Returns
    -------
    float
        Penalized negative log-likelihood
    """
    # --- Unpack parameters (unconstrained) ---
    u_h1, u_h2, u_kappa = float(x[0]), float(x[1]), float(x[2])
    z = np.asarray(x[3:7], dtype=float)

    # --- Map to positive hazards ---
    h1 = float(softplus(np.array([u_h1]))[0] + 1e-12)
    h2 = float(softplus(np.array([u_h2]))[0] + 1e-12)
    kappa = float(softplus(np.array([u_kappa]))[0] + 1e-12)
    h3 = kappa * h2  # tail hazard

    # --- Ordered etas via cumulative positive increments ---
    deltas = softplus(z)  # deltas >= 0
    eta = {
        1: 0.0,
        2: float(deltas[0]),
        3: float(deltas[0] + deltas[1]),
        4: float(deltas[0] + deltas[1] + deltas[2]),
        5: float(deltas[0] + deltas[1] + deltas[2] + deltas[3]),
    }

    # --- Baseline cumulative hazard at the anchor points ---
    Lam1 = float(piecewise_Lambda0(np.array([1.0]), h1, h2, h3)[0])
    Lam3 = float(piecewise_Lambda0(np.array([3.0]), h1, h2, h3)[0])

    # --- Likelihood contribution across grades ---
    nll = 0.0
    for g, gd in grades.items():
        # Cumulative PD at 1Y and 3Y under PH
        F1 = float(1.0 - np.exp(-np.exp(eta[g]) * Lam1))
        F3 = float(1.0 - np.exp(-np.exp(eta[g]) * Lam3))

        # Numerical safety + ensure F3 > F1
        F1 = float(np.clip(F1, 1e-12, 1.0 - 1e-12))
        F3 = float(np.clip(F3, F1 + 1e-12, 1.0 - 1e-12))

        # Conditional PD in (1,3]
        p13 = (F3 - F1) / (1.0 - F1)
        p13 = float(np.clip(p13, 1e-12, 1.0 - 1e-12))

        # Binomial log-likelihood (constants dropped)
        ll = 0.0
        ll += log_binom_pmf_wo_const(gd.D1, gd.N, F1)
        ll += log_binom_pmf_wo_const(gd.D13, gd.N_surv1, p13)

        nll -= ll

    # --- Tail regularization on log(kappa) ---
    logk = np.log(kappa)
    logk0 = np.log(kappa0)
    nll += 0.5 * ((logk - logk0) / sigma_logkappa) ** 2

    # --- Weak shrinkage on eta values for stability ---
    for g in range(2, 6):
        nll += 0.5 * (eta[g] / sigma_eta) ** 2

    return nll


# =============================================================================
# Estimation wrappers
# =============================================================================
def fit_weibull_grades(grades: Dict[int, GradeCounts]) -> Dict[int, Tuple[float, float]]:
    """
    Fit a Weibull curve independently for each grade.

    Returns
    -------
    dict
        grade -> (k, alpha)
    """
    out: Dict[int, Tuple[float, float]] = {}
    for g, gd in grades.items():
        # Simple initialization. (u_k ~ 0 => k ~ softplus(0) ~ 0.69)
        # (u_alpha ~ 2 => alpha ~ softplus(2) ~ 2.13)
        x0 = np.array([0.0, 2.0], dtype=float)

        res = minimize(
            negloglik_weibull_per_grade,
            x0,
            args=(gd,),
            method="L-BFGS-B",
        )

        u_k, u_a = float(res.x[0]), float(res.x[1])
        k = float(softplus(np.array([u_k]))[0] + 1e-6)
        alpha = float(softplus(np.array([u_a]))[0] + 1e-6)
        out[g] = (k, alpha)

    return out


def fit_ph_piecewise_ordered(
    grades: Dict[int, GradeCounts],
    *,
    kappa0: float = 1.0,
    sigma_logkappa: float = 0.20,
    sigma_eta: float = 5.0,
) -> Dict[str, float]:
    """
    Fit the coherent PH model with monotone grade effects and tail regularization.

    Important note about SciPy:
    ---------------------------
    scipy.optimize.minimize does not accept a `kwargs=...` argument.
    Therefore we build a closure `objective(x)` that captures hyperparameters.

    Returns
    -------
    dict
        Parameters: h1, h2, h3, kappa, and eta_1..eta_5.
    """
    # Parameter vector: [u_h1, u_h2, u_kappa, z2, z3, z4, z5]
    x0 = np.zeros(7, dtype=float)

    # Reasonable starting hazards:
    # u=-3 => softplus(-3) ~ 0.048 (small hazard consistent with small PDs)
    x0[0] = -3.0  # u_h1
    x0[1] = -3.0  # u_h2

    # Start kappa near 1 (softplus(0.3) ~ 0.85); optimizer can move it.
    x0[2] = 0.3

    # Small positive deltas to enforce gentle separation at start
    x0[3:] = 0.1

    def objective(x: np.ndarray) -> float:
        return negloglik_ph_ordered(
            x,
            grades,
            kappa0=kappa0,
            sigma_logkappa=sigma_logkappa,
            sigma_eta=sigma_eta,
        )

    res = minimize(objective, x0, method="L-BFGS-B")

    # Map back to interpretable parameters
    x = res.x
    u_h1, u_h2, u_kappa = float(x[0]), float(x[1]), float(x[2])
    z = np.asarray(x[3:7], dtype=float)

    h1 = float(softplus(np.array([u_h1]))[0] + 1e-12)
    h2 = float(softplus(np.array([u_h2]))[0] + 1e-12)
    kappa = float(softplus(np.array([u_kappa]))[0] + 1e-12)
    h3 = kappa * h2

    deltas = softplus(z)
    eta = {
        1: 0.0,
        2: float(deltas[0]),
        3: float(deltas[0] + deltas[1]),
        4: float(deltas[0] + deltas[1] + deltas[2]),
        5: float(deltas[0] + deltas[1] + deltas[2] + deltas[3]),
    }

    out: Dict[str, float] = {"h1": h1, "h2": h2, "h3": h3, "kappa": kappa}
    for g in range(1, 6):
        out[f"eta_{g}"] = float(eta[g])

    return out


# =============================================================================
# Demo data (replace with your actual portfolio grades)
# =============================================================================
def make_demo_data() -> Dict[int, GradeCounts]:
    """
    Build a toy dataset designed to be "problematic" for separate parametric fits:
    - grade order holds at 1Y and 3Y,
    - but independent Weibull fits may cross at long horizon.
    """
    P1 = np.array([0.001, 0.004, 0.008, 0.012, 0.020])  # cumulative PD at 1Y
    P3 = np.array([0.008, 0.010, 0.015, 0.020, 0.030])  # cumulative PD at 3Y
    N = np.array([20000, 20000, 20000, 20000, 20000])

    D1 = np.rint(N * P1).astype(int)
    D3 = np.rint(N * P3).astype(int)

    # Ensure monotone cumulative defaults within each grade
    D3 = np.maximum(D3, D1)

    grades: Dict[int, GradeCounts] = {}
    for i in range(5):
        g = i + 1
        grades[g] = GradeCounts(grade=g, N=int(N[i]), D1=int(D1[i]), D3=int(D3[i]))

    return grades


# =============================================================================
# Curve construction + diagnostics
# =============================================================================
def compute_curves_weibull(
    weibull_params_by_grade: Dict[int, Tuple[float, float]],
    tgrid: np.ndarray,
) -> Dict[int, np.ndarray]:
    """Compute Weibull lifetime PD curves F_g(t) for each grade on a common grid."""
    return {g: weibull_F(tgrid, k, a) for g, (k, a) in weibull_params_by_grade.items()}


def compute_curves_ph(params: Dict[str, float], tgrid: np.ndarray) -> Dict[int, np.ndarray]:
    """Compute PH lifetime PD curves F_g(t) for each grade on a common grid."""
    Lambda0 = piecewise_Lambda0(tgrid, params["h1"], params["h2"], params["h3"])
    return {g: ph_F_from_Lambda0(params[f"eta_{g}"], Lambda0) for g in range(1, 6)}


def check_crossings(curves: Dict[int, np.ndarray]) -> bool:
    """
    Check whether the grade ordering is violated anywhere on the grid.

    Expected: grade 1 <= grade 2 <= ... <= grade 5 for all t.
    """
    M = np.vstack([curves[g] for g in range(1, 6)])  # shape = (G, T)
    return bool(np.any(M[:-1, :] > M[1:, :] + 1e-12))


def _common_ylim_percent(
    curves_left: Dict[int, np.ndarray],
    curves_right: Dict[int, np.ndarray],
    ypad: float = 0.05,
) -> Tuple[float, float]:
    """
    Compute common y-axis limits (in %) for side-by-side plots.

    We set the top bound based on the maximum curve value across both methods,
    then add a small padding.
    """
    vals = []
    for g in range(1, 6):
        vals.append(curves_left[g])
        vals.append(curves_right[g])
    mx = float(np.max(np.vstack(vals)))
    mx = mx * (1.0 + ypad)
    return 0.0, 100.0 * mx


def report_fit_at_anchors(
    grades: Dict[int, GradeCounts],
    weibull_params: Dict[int, Tuple[float, float]],
    ph_params: Dict[str, float],
) -> None:
    """
    Print a concise table comparing observed (empirical) vs fitted PD at 1Y and 3Y.
    """
    Lam0_1 = piecewise_Lambda0(np.array([1.0]), ph_params["h1"], ph_params["h2"], ph_params["h3"])[0]
    Lam0_3 = piecewise_Lambda0(np.array([3.0]), ph_params["h1"], ph_params["h2"], ph_params["h3"])[0]

    print("\n" + "=" * 92)
    print("Observed vs fitted cumulative PD at anchors (in %)")
    print("=" * 92)
    hdr = (
        f"{'Grade':>5} | {'Obs 1Y':>8} {'Wbl 1Y':>8} {'PH 1Y':>8} || "
        f"{'Obs 3Y':>8} {'Wbl 3Y':>8} {'PH 3Y':>8}"
    )
    print(hdr)
    print("-" * 92)

    for g in range(1, 6):
        gd = grades[g]
        obs1 = 100.0 * gd.p1_hat
        obs3 = 100.0 * gd.p3_hat

        k, a = weibull_params[g]
        w1 = 100.0 * weibull_F(np.array([1.0]), k, a)[0]
        w3 = 100.0 * weibull_F(np.array([3.0]), k, a)[0]

        eta = ph_params[f"eta_{g}"]
        ph1 = 100.0 * (1.0 - np.exp(-np.exp(eta) * Lam0_1))
        ph3 = 100.0 * (1.0 - np.exp(-np.exp(eta) * Lam0_3))

        print(f"{g:>5} | {obs1:8.3f} {w1:8.3f} {ph1:8.3f} || {obs3:8.3f} {w3:8.3f} {ph3:8.3f}")

    print("-" * 92)
    print(
        f"Tail regularization: kappa={ph_params['kappa']:.4f}, "
        f"h3={ph_params['h3']:.6f}, h2={ph_params['h2']:.6f}"
    )
    print("=" * 92)


def print_table(
    curves_a: Dict[int, np.ndarray],
    curves_b: Dict[int, np.ndarray],
    tgrid: np.ndarray,
    horizons=(1, 3, 10, 20, 50),
    name_a="Weibull (grade-by-grade)",
    name_b="PH ordered + tail regularization",
) -> None:
    """
    Print lifetime PD (%) at selected horizons for both methods.
    """
    idx = {int(t): int(np.argmin(np.abs(tgrid - t))) for t in horizons}

    print("\n" + "=" * 90)
    print("Cumulative PD (%) at selected horizons")
    print("=" * 90)
    print(f"{'Grade':>5} | " + " | ".join([f"{t:>5}Y" for t in horizons]) + " || Method")
    print("-" * 90)

    for g in range(1, 6):
        row_a = [100.0 * curves_a[g][idx[t]] for t in horizons]
        row_b = [100.0 * curves_b[g][idx[t]] for t in horizons]
        print(f"{g:>5} | " + " | ".join([f"{v:5.2f}" for v in row_a]) + f" || {name_a}")
        print(f"{'':>5} | " + " | ".join([f"{v:5.2f}" for v in row_b]) + f" || {name_b}")
        print("-" * 90)


# =============================================================================
# Plotting (save to disk)
# =============================================================================
def plot_full_and_save(
    grades: Dict[int, GradeCounts],
    curves_weibull: Dict[int, np.ndarray],
    curves_ph: Dict[int, np.ndarray],
    tgrid: np.ndarray,
    ph_params: Dict[str, float],
    outdir: str,
    fname: str = "lifetime_full_0_50y.png",
) -> None:
    """
    Save the main 0–50Y figure (two panels) with observed anchors (1Y, 3Y).
    """
    ensure_dir(outdir)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Observed points at anchor horizons (empirical PDs)
    x_anchors = np.array([1.0, 3.0])
    obs_anchors = {g: np.array([grades[g].p1_hat, grades[g].p3_hat]) for g in range(1, 6)}

    # Use the same y-scale on both panels for an apples-to-apples comparison
    y0, y1 = _common_ylim_percent(curves_weibull, curves_ph, ypad=0.05)

    # --- Left panel: independent Weibull fits ---
    ax = axes[0]
    for g in range(1, 6):
        ax.plot(tgrid, 100.0 * curves_weibull[g], label=f"Grade {g}")
        ax.scatter(x_anchors, 100.0 * obs_anchors[g], s=30, marker="o")
    ax.set_title("Weibull fitted independently (may cross at long horizon)")
    ax.set_xlabel("Horizon (years)")
    ax.set_ylabel("Cumulative PD (%)")
    ax.set_ylim(y0, y1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # --- Right panel: coherent PH model (ordered grades + tail regularization) ---
    kappa = ph_params["kappa"]
    ax = axes[1]
    for g in range(1, 6):
        ax.plot(tgrid, 100.0 * curves_ph[g], label=f"Grade {g}")
        ax.scatter(x_anchors, 100.0 * obs_anchors[g], s=30, marker="o")
    ax.set_title(
        "PH + ordered grades + tail regularization (no crossing)\n"
        f"(kappa = {kappa:.3f})"
    )
    ax.set_xlabel("Horizon (years)")
    ax.set_ylim(y0, y1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    path = os.path.join(outdir, fname)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


def plot_zoom_and_save(
    grades: Dict[int, GradeCounts],
    curves_weibull: Dict[int, np.ndarray],
    curves_ph: Dict[int, np.ndarray],
    tgrid: np.ndarray,
    ph_params: Dict[str, float],
    outdir: str,
    zoom_years: float = 10.0,
    fname: str = "lifetime_zoom_0_10y.png",
) -> None:
    """
    Save a zoomed 0–zoom_years figure to better visualize early horizons.

    This is useful because with small PDs, differences at short horizons may be
    visually compressed in the 0–50Y plot.
    """
    ensure_dir(outdir)

    mask = tgrid <= zoom_years
    t = tgrid[mask]

    # Slice curves
    cw = {g: curves_weibull[g][mask] for g in range(1, 6)}
    cp = {g: curves_ph[g][mask] for g in range(1, 6)}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Keep only anchors within the zoomed horizon
    x_anchors = np.array([1.0, 3.0])
    x_anchors = x_anchors[x_anchors <= zoom_years]
    obs_anchors = {
        g: np.array([grades[g].p1_hat, grades[g].p3_hat])[: len(x_anchors)]
        for g in range(1, 6)
    }

    # Slightly larger padding in zoom view
    y0, y1 = _common_ylim_percent(cw, cp, ypad=0.10)

    # Left: Weibull
    ax = axes[0]
    for g in range(1, 6):
        ax.plot(t, 100.0 * cw[g], label=f"Grade {g}")
        ax.scatter(x_anchors, 100.0 * obs_anchors[g], s=30, marker="o")
    ax.set_title(f"Weibull — zoom 0–{int(zoom_years)} years")
    ax.set_xlabel("Horizon (years)")
    ax.set_ylabel("Cumulative PD (%)")
    ax.set_xlim(0.0, zoom_years)
    ax.set_ylim(y0, y1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Right: PH coherent
    kappa = ph_params["kappa"]
    ax = axes[1]
    for g in range(1, 6):
        ax.plot(t, 100.0 * cp[g], label=f"Grade {g}")
        ax.scatter(x_anchors, 100.0 * obs_anchors[g], s=30, marker="o")
    ax.set_title(
        f"PH coherent — zoom 0–{int(zoom_years)} years\n"
        f"(kappa = {kappa:.3f})"
    )
    ax.set_xlabel("Horizon (years)")
    ax.set_xlim(0.0, zoom_years)
    ax.set_ylim(y0, y1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    path = os.path.join(outdir, fname)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


# =============================================================================
# Main script
# =============================================================================
def main() -> None:
    """
    Run the demo end-to-end:
    1) Load data (here: a synthetic example)
    2) Fit Weibull independently per grade
    3) Fit coherent PH model with ordered grades + tail regularization
    4) Compute lifetime curves up to 50Y
    5) Save figures + print diagnostics
    """
    # Folder where plots will be saved
    outdir = os.path.join("outputs", "figures")
    ensure_dir(outdir)

    # Replace this with your real (N, D1, D3) by grade
    grades = make_demo_data()
    validate_data(grades)

    print("Input data (N, D1, D3):")
    for g in range(1, 6):
        gd = grades[g]
        print(f"  Grade {g}: N={gd.N}, D1={gd.D1}, D3={gd.D3} (D13={gd.D13})")

    # --- Fit models ---
    weibull_params_by_grade = fit_weibull_grades(grades)

    # Tail regularization controls long-horizon behavior with sparse anchor information.
    # - sigma_logkappa smaller => stronger pull toward kappa0
    # - sigma_logkappa larger  => freer tail
    ph_params = fit_ph_piecewise_ordered(
        grades,
        kappa0=1.0,
        sigma_logkappa=0.20,
        sigma_eta=5.0,
    )

    # --- Compute curves on a common grid ---
    TMAX = 50.0
    tgrid = np.linspace(0.0, TMAX, 501)
    # Avoid an exact 0 in Weibull computations (not required mathematically, but convenient)
    tgrid[0] = 1e-6

    curves_weibull = compute_curves_weibull(weibull_params_by_grade, tgrid)
    curves_ph = compute_curves_ph(ph_params, tgrid)

    # --- Check crossing property ---
    print("\nCrossing detected?")
    print(f"  Weibull grade-by-grade : {check_crossings(curves_weibull)}")
    print(f"  PH ordered + reg tail  : {check_crossings(curves_ph)}")

    # --- Diagnostics at anchor horizons ---
    report_fit_at_anchors(grades, weibull_params_by_grade, ph_params)
    print_table(curves_weibull, curves_ph, tgrid)

    # --- Save figures ---
    plot_full_and_save(
        grades,
        curves_weibull,
        curves_ph,
        tgrid,
        ph_params,
        outdir,
        fname="lifetime_full_0_50y.png",
    )
    plot_zoom_and_save(
        grades,
        curves_weibull,
        curves_ph,
        tgrid,
        ph_params,
        outdir,
        zoom_years=10.0,
        fname="lifetime_zoom_0_10y.png",
    )

    # --- Print estimated parameters ---
    print("\nFitted Weibull params (k, alpha):")
    for g in range(1, 6):
        k, a = weibull_params_by_grade[g]
        print(f"  Grade {g}: k={k:.4f}, alpha={a:.2f}")

    print("\nFitted PH parameters (tail regularized):")
    for key in ["h1", "h2", "h3", "kappa", "eta_1", "eta_2", "eta_3", "eta_4", "eta_5"]:
        print(f"  {key:>6} = {ph_params[key]:.6f}")


if __name__ == "__main__":
    main()
