"""LOR96 RBF integration experiment.

Adapted from the LOR63 plan in `results/LOR63/codes/integrate_compare.py`.
The differences are dictated by the dataset:

  * No analytic FOM RHS for LOR96 reduced coords. Ground truth is the
    strided FOM trajectory `alpha` from coords.npz; the "truth"
    continuation of an IC is the contiguous slice of `alpha` starting at
    the same index.
  * d_o = 35, so the short-window pointwise plot uses a mode grid rather
    than 3 stacked panels, and there is no 3D attractor projection.
  * The L63-specific (x, y, z) -> (-x, -y, z) symmetry diagnostic is
    dropped (LOR96 has translation symmetry on raw u, not on the
    reduced coords).

Outputs (`<out-dir>/`):
  short_window_ic0.png        mode-grid pointwise [0, T_short] s, IC 0
  short_window_rmse.png       RMSE(t) ensemble across n_ic ICs, with
                              Eq. 21 horizon marker and median horizon
  marginal_pdfs.png           per-mode marginal PDFs over the long
                              window (pooled across ICs) + Wasserstein-1
  autocorrelation.png         per-mode ACF over the long window
  vector_field_residual.png   ||f_rbf - alpha_dot_FOM|| on a 10k cloud,
                              vs distance to nearest centre
  step_halving_sanity.png     RK4 dt vs dt/2 max diff over [0, T_check]
  cloud_edge.png              d_nn(t) along each ROM trajectory vs FOM
                              cloud d_nn quantiles (extrapolation guard)
  stability.png               ROM-only d_nn(t) and |alpha(t)| over the
                              extended stability window (per IC); only
                              emitted when --stability-T > 0
  metrics.json                horizons, W1, ACF tau_corr, cloud_edge,
                              stability, etc.

Usage
-----
    python scripts/run_integrate_compare.py \\
        --model    /users/sbrw610/sharedscratch/RBF_ROM/results/LOR96/F8/fit_K1_n1200_raw/model.npz \\
        --out-dir  /users/sbrw610/sharedscratch/RBF_ROM/results/LOR96/F8/integration_K1_n1200_raw \\
        --T 50.0  --short-window-T 5.0  --long-window-t0 10.0 --n-ic 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from chord2.sindy import rbf_features_mahal  # noqa: E402


# LOR96 F=8: largest Lyapunov exponent ~ 1.68 nepers/unit (Lorenz & Emanuel
# 1998), T_lambda ~ 0.595 s. Override via --lyapunov-rate if needed for F=10.
LYAPUNOV_LOR96_F8 = 1.68
HORIZON_FRAC = 0.5


# ---------------------------------------------------------------------------
# Model wrapping
# ---------------------------------------------------------------------------

def _rhs_batch(A: np.ndarray, centers_std, Sigma_invs_std, mu_A, sigma_A,
               col_norms, xi) -> np.ndarray:
    """Vectorised RHS for shape (N, d_o)."""
    Phi = rbf_features_mahal(A, centers_std, Sigma_invs_std,
                             mu_A=mu_A, sigma_A=sigma_A)
    return (Phi / col_norms[None, :]) @ xi


def _make_rhs(centers_std, Sigma_invs_std, mu_A, sigma_A, col_norms, xi,
              corrector=None):
    """Build the integrator RHS. If `corrector(a)->da` is given it is added to
    the SINDy rate at every evaluation — used by the soft_reflect trust region
    to inject a smooth inward radial pull near the cover boundary."""
    def rhs(a: np.ndarray) -> np.ndarray:
        Phi = rbf_features_mahal(a[None, :], centers_std, Sigma_invs_std,
                                 mu_A=mu_A, sigma_A=sigma_A)
        da = (Phi[0] / col_norms) @ xi
        if corrector is not None:
            da = da + corrector(a)
        return da
    return rhs


def _rk4_integrate(rhs, x0: np.ndarray, dt: float, n_steps: int,
                   *, guard=None):
    """RK4 step loop.

    The guard, if given, must be callable as ``guard(x) -> (x_use, fired,
    terminate)``: ``x_use`` is the position to commit (possibly modified by
    reflect), ``fired`` tells the caller whether the event counted, and
    ``terminate`` ends the loop (stop mode) or lets it continue (reflect mode).

    Returns
    -------
    X : ndarray of shape (k+1, d)
        Trajectory truncated to k retained steps after x0.
    reason : str
        "complete"   - ran the full n_steps without incident
        "non_finite" - integrator produced NaN/Inf
        "guard"      - guard fired with terminate=True (stop mode); the
                       boundary-crossing step is included in X.
    n_events : int
        Number of guard activations along the trajectory (0 for stop unless
        it fired; can be many for reflect).
    first_event_step : int or None
        Step index (>=1) of the first guard activation, or None.
    """
    X = np.empty((n_steps + 1, x0.shape[0]), dtype=np.float64)
    X[0] = x0
    n_events = 0
    first_event_step: int | None = None
    for n in range(n_steps):
        a = X[n]
        k1 = rhs(a)
        k2 = rhs(a + 0.5 * dt * k1)
        k3 = rhs(a + 0.5 * dt * k2)
        k4 = rhs(a + dt * k3)
        nxt = a + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.all(np.isfinite(nxt)):
            return X[: n + 1], "non_finite", n_events, first_event_step
        if guard is not None:
            nxt, fired, terminate = guard(nxt)
            if fired:
                n_events += 1
                if first_event_step is None:
                    first_event_step = n + 1
                if terminate:
                    X[n + 1] = nxt
                    return X[: n + 2], "guard", n_events, first_event_step
        X[n + 1] = nxt
    return X, "complete", n_events, first_event_step


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _eq21_horizon(X_pred: np.ndarray, X_true: np.ndarray, dt: float,
                  std_climate: float) -> float:
    n_cmp = min(X_pred.shape[0], X_true.shape[0])
    err = X_pred[:n_cmp] - X_true[:n_cmp]
    rmse_t = np.sqrt((err * err).mean(axis=1))
    threshold = HORIZON_FRAC * std_climate
    crossed = np.flatnonzero(rmse_t > threshold)
    if crossed.size == 0:
        return float(dt * (n_cmp - 1))
    return float(crossed[0] * dt)


def _per_mode_acf(X: np.ndarray, n_lag: int) -> np.ndarray:
    """Biased autocorrelation per column up to `n_lag` (inclusive), via FFT."""
    M, d = X.shape
    Xc = X - X.mean(axis=0)
    var = (Xc * Xc).sum(axis=0)
    var = np.where(var == 0.0, 1.0, var)
    n_pad = 1 << int(np.ceil(np.log2(2 * M)))
    Fx = np.fft.rfft(Xc, n=n_pad, axis=0)
    acf_full = np.fft.irfft(Fx * np.conj(Fx), n=n_pad, axis=0)[:n_lag + 1]
    return acf_full / var[None, :]


def _grid_dims(d_o: int) -> tuple[int, int]:
    cols = int(np.ceil(np.sqrt(d_o)))
    rows = int(np.ceil(d_o / cols))
    return rows, cols


def _trajectory_d_nn(X: np.ndarray, mu_A: np.ndarray, sigma_A: np.ndarray,
                     tree: cKDTree) -> np.ndarray:
    """Distance to nearest RBF centre along a trajectory, in std coords."""
    X_std = (X - mu_A) / sigma_A
    d, _ = tree.query(X_std, k=1)
    return d


def _build_stop_guard(tree: cKDTree, mu_A: np.ndarray, sigma_A: np.ndarray,
                      threshold: float):
    """Stop guard: fires (terminate=True) when x leaves the trust region."""
    def guard(x: np.ndarray):
        x_std = (x - mu_A) / sigma_A
        d, _ = tree.query(x_std[None, :], k=1)
        if float(d[0]) > threshold:
            return x, True, True
        return x, False, False
    return guard


def _build_reflect_guard(tree: cKDTree, mu_A: np.ndarray, sigma_A: np.ndarray,
                         threshold: float):
    """Reflect guard: mirrors a proposed step that left the trust region back
    inside the nearest-centre Voronoi ball.

    With c = nearest centre to x_std, d = ||x_std - c||, u = (x_std - c) / d,
    the reflected point is c + (2*threshold - d) * u — an elastic bounce off
    the ball boundary. If the overshoot is so large that d > 2*threshold (the
    bounce would land on the wrong side of c), clip to the boundary instead.
    """
    centres_std = tree.data
    def guard(x: np.ndarray):
        x_std = (x - mu_A) / sigma_A
        d_arr, idx_arr = tree.query(x_std[None, :], k=1)
        d = float(d_arr[0])
        if d <= threshold:
            return x, False, False
        c = centres_std[int(idx_arr[0])]
        u = (x_std - c) / d
        new_d = 2.0 * threshold - d
        if new_d < 0.0:
            new_d = threshold  # clip rather than fold past the centre
        x_std_new = c + new_d * u
        x_new = x_std_new * sigma_A + mu_A
        return x_new, True, False
    return guard


def _build_soft_reflect_corrector(tree: cKDTree, mu_A: np.ndarray,
                                   sigma_A: np.ndarray, threshold: float,
                                   alpha: float, k_strength: float, p: float):
    """Soft-reflect RHS corrector: smooth inward radial pull active for
    d_nn > alpha*threshold.

    Returns ``corrector(a) -> da``, in raw-coord velocity units, evaluating to

        C(a) = -k * max(0, (d_nn - alpha*R) / (R*(1 - alpha)))^p * n_out  (in std coords)

    where ``n_out`` is the outward unit normal from the nearest centre in
    std coords. The result is converted to raw-coord rates via sigma_A so it
    composes correctly with the bare SINDy RHS inside the RK4 stages.

    Parameters
    ----------
    alpha
        On-ramp distance fraction; ramp starts at d_nn = alpha * R and reaches
        unit magnitude at d_nn = R.
    k_strength
        Peak corrector strength (std-coord units per second) when ramp = 1.
    p
        Ramp sharpness exponent.
    """
    centres_std = tree.data
    inv_denom = 1.0 / (threshold * (1.0 - alpha))
    def corrector(a: np.ndarray) -> np.ndarray:
        a_std = (a - mu_A) / sigma_A
        d_arr, idx_arr = tree.query(a_std[None, :], k=1)
        d = float(d_arr[0])
        x = (d - alpha * threshold) * inv_denom
        if x <= 0.0:
            return np.zeros_like(a)
        c = centres_std[int(idx_arr[0])]
        u_std = (a_std - c) / d
        # Saturate beyond the cover boundary so the corrector is bounded by
        # k_strength; otherwise an RK4 stage that overshoots into d >> R
        # generates an O(k * overshoot^p) inward impulse that flips the
        # trajectory and crashes the integrator (see KS_BUR job 53506).
        ramp = min(1.0, x) ** p
        return (-k_strength * ramp) * u_std * sigma_A
    return corrector


def _build_soft_reflect_corrector_per_cluster(
    global_tree: cKDTree,
    cluster_trees: list,
    parent_k: np.ndarray,
    mu_A: np.ndarray,
    sigma_A: np.ndarray,
    threshold,
    alpha: float,
    k_strength,
    p: float,
):
    """Per-cluster soft_reflect: the current cluster is decided by parent_k of
    the nearest *global* centre, and d_nn is measured against only that
    cluster's centres. Lets the trajectory transition between clusters
    without the universal-tree's bias toward the originating cluster, but
    still pulls back when it leaves the union of all clusters' covers.

    ``threshold`` and ``k_strength`` may be either a scalar (applied to all
    clusters) or a length-K array indexed by cluster ``k_now``. Per-cluster
    anchors come from the soft_reflect pre-flight diagnostic:
    ``R_k = Q_q(d_nn|k)`` for ``threshold_k`` and ``k_k = mult_k * P95+_emp``
    for ``k_strength_k``.
    """
    parent_k = np.asarray(parent_k, dtype=np.int64)
    cluster_centres = [t.data for t in cluster_trees]
    K_actual = len(cluster_trees)

    threshold_arr = np.broadcast_to(
        np.asarray(threshold, dtype=np.float64), (K_actual,)
    ).copy()
    k_strength_arr = np.broadcast_to(
        np.asarray(k_strength, dtype=np.float64), (K_actual,)
    ).copy()
    inv_denom_arr = 1.0 / (threshold_arr * (1.0 - alpha))

    def corrector(a: np.ndarray) -> np.ndarray:
        a_std = (a - mu_A) / sigma_A
        _, g_idx = global_tree.query(a_std[None, :], k=1)
        k_now = int(parent_k[int(g_idx[0])])
        tree_k = cluster_trees[k_now]
        d_arr, idx_arr = tree_k.query(a_std[None, :], k=1)
        d = float(d_arr[0])
        thr_k = threshold_arr[k_now]
        x = (d - alpha * thr_k) * inv_denom_arr[k_now]
        if x <= 0.0:
            return np.zeros_like(a)
        c = cluster_centres[k_now][int(idx_arr[0])]
        u_std = (a_std - c) / d
        ramp = min(1.0, x) ** p
        return (-k_strength_arr[k_now] * ramp) * u_std * sigma_A
    return corrector


def _build_passive_event_counter(tree: cKDTree, mu_A: np.ndarray,
                                  sigma_A: np.ndarray, threshold: float):
    """Passive guard: counts steps where d_nn > threshold but never modifies
    the state or terminates. Used with soft_reflect so its event-rate metric
    is directly comparable to the reflect guard's bounce count."""
    def guard(x: np.ndarray):
        x_std = (x - mu_A) / sigma_A
        d, _ = tree.query(x_std[None, :], k=1)
        if float(d[0]) > threshold:
            return x, True, False
        return x, False, False
    return guard


# ---------------------------------------------------------------------------
# Integrator construction (shared by the serial path and the parallel workers)
# ---------------------------------------------------------------------------

def build_integrator(model_path, *, escape_thresh, trust_region,
                     trust_region_mode, soft_alpha, soft_k, soft_p,
                     soft_threshold_mult, preflight_thr_per_k,
                     preflight_k_per_k):
    """Load the model and assemble the integrator RHS + kinematic guard.

    Takes only picklable inputs (a model file path, scalar guard knobs, and the
    pre-resolved per-cluster pre-flight arrays) so it can be called identically
    in the parent process and inside a ProcessPoolExecutor worker, where the
    cKDTree-capturing closures `rhs_int`/`guard` would not survive pickling.

    `escape_thresh` is computed once in the parent from the FOM cloud and passed
    in, so every worker sees the same trust-region radius. The guard-building
    block here is the verbatim logic that previously lived inline in `main`
    (none / stop / reflect / soft_reflect; universal / per_cluster; with or
    without a pre-flight JSON) — relocated unchanged so the serial and parallel
    paths cannot diverge.

    Returns
    -------
    bundle : dict
        Model arrays and derived objects the caller needs (`rhs`, `rhs_int`,
        `guard`, `tree`, `centers_std`, `mu_A`, `sigma_A`, `col_norms`, `xi`,
        `Sigma_invs_std`, `parent_k`, plus scalar metadata).
    """
    z = np.load(model_path, allow_pickle=True)
    centers_std = np.asarray(z["centers_std"], dtype=np.float64)
    Sigma_invs_std = np.asarray(z["Sigma_invs_std"], dtype=np.float64)
    parent_k = (np.asarray(z["parent_k"], dtype=np.int64)
                if "parent_k" in z.files else None)
    mu_A = np.asarray(z["mu_A"], dtype=np.float64)
    sigma_A = np.asarray(z["sigma_A"], dtype=np.float64)
    col_norms = np.asarray(z["col_norms_train"], dtype=np.float64)
    xi = np.asarray(z["xi"], dtype=np.float64)

    rhs = _make_rhs(centers_std, Sigma_invs_std, mu_A, sigma_A, col_norms, xi)
    tree = cKDTree(centers_std)

    rhs_int = rhs
    if trust_region == "stop":
        guard = _build_stop_guard(tree, mu_A, sigma_A, escape_thresh)
    elif trust_region == "reflect":
        guard = _build_reflect_guard(tree, mu_A, sigma_A, escape_thresh)
    elif trust_region == "soft_reflect":
        escape_thresh_sr = escape_thresh * soft_threshold_mult
        if trust_region_mode == "per_cluster":
            K_actual = int(parent_k.max()) + 1
            cluster_trees = []
            for k_ in range(K_actual):
                mask = parent_k == k_
                cluster_trees.append(cKDTree(centers_std[mask]))
            if preflight_thr_per_k is not None:
                thr_arg = preflight_thr_per_k
                k_arg = preflight_k_per_k
            else:
                thr_arg = escape_thresh_sr
                k_arg = soft_k
            corrector = _build_soft_reflect_corrector_per_cluster(
                tree, cluster_trees, parent_k, mu_A, sigma_A,
                thr_arg, alpha=soft_alpha, k_strength=k_arg, p=soft_p,
            )
        else:
            corrector = _build_soft_reflect_corrector(
                tree, mu_A, sigma_A, escape_thresh_sr,
                alpha=soft_alpha, k_strength=soft_k, p=soft_p,
            )
        rhs_int = _make_rhs(centers_std, Sigma_invs_std, mu_A, sigma_A,
                            col_norms, xi, corrector=corrector)
        guard = _build_passive_event_counter(tree, mu_A, sigma_A,
                                             escape_thresh)
    else:
        guard = None

    return {
        "rhs": rhs,
        "rhs_int": rhs_int,
        "guard": guard,
        "tree": tree,
        "centers_std": centers_std,
        "Sigma_invs_std": Sigma_invs_std,
        "mu_A": mu_A,
        "sigma_A": sigma_A,
        "col_norms": col_norms,
        "xi": xi,
        "parent_k": parent_k,
    }


# Worker globals: a ProcessPoolExecutor forks/spawns one process per worker and
# reuses it across tasks. Building the integrator (cKDTree construction) once
# per worker and caching it here avoids rebuilding it for every IC.
_WORKER_BUNDLE = None
_WORKER_KEY = None


def _integrate_one_ic(task):
    """Integrate exactly one IC in a worker process.

    `task` is a fully-picklable tuple. The integrator (rhs/guard closures) is
    rebuilt inside the worker from the model path + guard config via
    `build_integrator` and cached across tasks that share the same config.
    Returns the same per-IC products the serial loop collects.
    """
    global _WORKER_BUNDLE, _WORKER_KEY
    (cfg, x0, n_steps, dt) = task
    key = (cfg["model_path"], cfg["escape_thresh"], cfg["trust_region"],
           cfg["trust_region_mode"])
    if _WORKER_KEY != key:
        _WORKER_BUNDLE = build_integrator(
            cfg["model_path"],
            escape_thresh=cfg["escape_thresh"],
            trust_region=cfg["trust_region"],
            trust_region_mode=cfg["trust_region_mode"],
            soft_alpha=cfg["soft_alpha"],
            soft_k=cfg["soft_k"],
            soft_p=cfg["soft_p"],
            soft_threshold_mult=cfg["soft_threshold_mult"],
            preflight_thr_per_k=cfg["preflight_thr_per_k"],
            preflight_k_per_k=cfg["preflight_k_per_k"],
        )
        _WORKER_KEY = key
    X, reason, n_evt, first_evt = _rk4_integrate(
        _WORKER_BUNDLE["rhs_int"], np.asarray(x0, dtype=np.float64),
        dt, n_steps, guard=_WORKER_BUNDLE["guard"],
    )
    return X, reason, n_evt, first_evt


def _integrate_ics(ic_idx, A_full, n_steps, dt, rhs_int, guard, *,
                   n_workers, worker_cfg):
    """Integrate every IC, serially (n_workers==1) or in parallel.

    The serial branch is the exact loop body the script has always run; the
    parallel branch dispatches the same single-IC integration to worker
    processes and reassembles the results in IC order. Both return a list of
    `(X, reason, n_events, first_event_step)` aligned with `ic_idx`.
    """
    if n_workers <= 1:
        results = []
        for idx in ic_idx:
            results.append(_rk4_integrate(
                rhs_int, A_full[idx], dt, n_steps, guard=guard))
        return results
    tasks = [(worker_cfg, A_full[idx], n_steps, dt) for idx in ic_idx]
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        return list(ex.map(_integrate_one_ic, tasks))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--coords-path", type=Path, default=None,
                   help="override coords path stored in model.npz")
    p.add_argument("--n-ic", type=int, default=8)
    p.add_argument("--ic-rng-seed", type=int, default=0)
    p.add_argument("--T", type=float, default=50.0,
                   help="total integration window per IC (seconds)")
    p.add_argument("--short-window-T", type=float, default=5.0)
    p.add_argument("--long-window-t0", type=float, default=10.0)
    p.add_argument("--step-halving-T", type=float, default=2.0)
    p.add_argument("--n-cloud-residual", type=int, default=10_000)
    p.add_argument("--cloud-rng-seed", type=int, default=1)
    p.add_argument("--acf-T", type=float, default=10.0,
                   help="max lag for ACF (seconds)")
    p.add_argument("--lyapunov-rate", type=float, default=LYAPUNOV_LOR96_F8,
                   help="largest Lyapunov exponent (nepers/s) for the "
                        "T_lambda reporting; default is LOR96 F=8")
    p.add_argument("--ic-from", choices=["any", "test"], default="any",
                   help="IC sampling pool: any FOM snapshot, or only the "
                        "held-out test partition")
    p.add_argument("--cloud-escape-quantile", type=float, default=0.99,
                   help="quantile of FOM-cloud distance-to-nearest-centre "
                        "used as the cloud-edge escape threshold")
    p.add_argument("--stability-T", type=float, default=50.0,
                   help="if > 0, run an additional ROM-only integration to "
                        "this many seconds per IC (no FOM truth required); "
                        "emits stability.png and metrics.stability block; "
                        "set to 0 to disable")
    p.add_argument("--trust-region",
                   choices=["none", "stop", "reflect", "soft_reflect"],
                   default="none",
                   help="kinematic guard on the RBF integrator: 'none' is the "
                        "current behaviour; 'stop' terminates the trajectory "
                        "the first time d_nn exceeds the FOM-cloud escape "
                        "quantile; 'reflect' mirrors the proposed step back "
                        "inside the nearest-centre Voronoi ball and continues; "
                        "'soft_reflect' adds a smooth radial inward pull to "
                        "the RHS for d_nn > alpha*R (controlled by --soft-* "
                        "knobs); stop/reflect leave the RHS unchanged, "
                        "soft_reflect modifies it inside the RK4 stages")
    p.add_argument("--soft-alpha", type=float, default=0.85,
                   help="soft_reflect on-ramp fraction; the corrector starts "
                        "ramping at d_nn = alpha*R (default 0.85)")
    p.add_argument("--soft-k", type=float, default=10.0,
                   help="soft_reflect peak strength in std-coord units per "
                        "second when ramp = 1 (default 10.0)")
    p.add_argument("--soft-p", type=float, default=2.0,
                   help="soft_reflect ramp sharpness exponent (default 2.0)")
    p.add_argument("--soft-threshold-mult", type=float, default=1.0,
                   help="multiplier on the cloud-escape quantile used for the "
                        "soft_reflect trust-region radius R only. The raw "
                        "quantile still drives stop/reflect guards and the "
                        "cloud-edge diagnostic. Useful when the FOM cloud is "
                        "thinner than per-step motion and a wider buffer is "
                        "needed for stability (KS_BUR).")
    p.add_argument("--trust-region-mode",
                   choices=["universal", "per_cluster"], default="universal",
                   help="soft_reflect tree topology. 'universal' (default) "
                        "measures d_nn against all centres; 'per_cluster' "
                        "decides the current cluster k from parent_k of the "
                        "globally-nearest centre and measures d_nn against "
                        "only that cluster's centres. Per-cluster requires "
                        "model.npz to carry 'parent_k' (any K>=1 fit).")
    p.add_argument("--soft-preflight-json", type=Path, default=None,
                   help="path to soft_reflect pre-flight JSON "
                        "(scripts/run_soft_reflect_preflight.py). When given "
                        "and --trust-region-mode=per_cluster, overrides "
                        "scalar --soft-k and --soft-threshold-mult with "
                        "per-cluster anchors: threshold_k = per_cluster.R[k] "
                        "and k_k = per_cluster.mult[k] * "
                        "per_cluster.p95_pos_radial_empirical[k] (the "
                        "rom-specialist's recommendation given residual ~0 "
                        "in-sample). alpha and p stay universal.")
    p.add_argument("--n-workers", type=int, default=1,
                   help="CPU worker processes for the per-IC integration. 1 "
                        "(default) keeps the serial loop; >1 dispatches each "
                        "IC to a ProcessPoolExecutor worker. ICs are "
                        "independent, so results are byte-identical to the "
                        "serial path; only wall-time changes. The launcher "
                        "must keep BLAS single-threaded.")
    p.add_argument("--dt-override", type=float, default=None,
                   help="override dt_strided from model.npz for RK4 stepping. "
                        "Use when the coords file's dt_native does not match "
                        "the desired integration cadence (e.g. arc-resampled "
                        "coords where the stored dt_native is the FOM real-"
                        "time step but you want to step at the FOM step itself "
                        "rather than at FOM*stride)")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t_lambda = 1.0 / args.lyapunov_rate
    print(f"[cfg] T_lambda = 1/{args.lyapunov_rate:.3f} = {t_lambda:.3f} s",
          flush=True)

    # ---- load model ----
    print(f"[load] model {args.model}", flush=True)
    z = np.load(args.model, allow_pickle=True)
    centers_std = np.asarray(z["centers_std"], dtype=np.float64)
    Sigma_invs_std = np.asarray(z["Sigma_invs_std"], dtype=np.float64)
    parent_k = (np.asarray(z["parent_k"], dtype=np.int64)
                if "parent_k" in z.files else None)
    mu_A = np.asarray(z["mu_A"], dtype=np.float64)
    sigma_A = np.asarray(z["sigma_A"], dtype=np.float64)
    col_norms = np.asarray(z["col_norms_train"], dtype=np.float64)
    xi = np.asarray(z["xi"], dtype=np.float64)
    dt_fit = float(z["dt_strided"])
    dt = args.dt_override if args.dt_override is not None else dt_fit
    # The FOM ground truth A_full = alpha[::stride] only exists at the strided
    # cadence dt_strided = dt_native*stride. RK4 may step finer than that
    # (--dt-override < dt_strided), but every ROM-vs-FOM comparison must happen
    # at the strided cadence, otherwise ROM state at real time dt*k is scored
    # against FOM state at real time dt_strided*k -- a (dt_strided/dt)x time
    # misalignment that corrupts the Eq.21 horizon, W1 and autocorrelation.
    # `step_ratio` is the number of fine RK4 steps per strided comparison
    # sample; the ROM trajectory is sub-sampled by `step_ratio` before any
    # comparison, and `dt_strided` (NOT the fine `dt`) is the cadence between
    # compared samples used for every steps<->seconds conversion below.
    dt_strided = dt_fit
    step_ratio = int(round(dt_strided / dt))
    if step_ratio < 1:
        raise ValueError(
            f"--dt-override={dt} is coarser than dt_strided={dt_strided}; the "
            f"ROM cannot be sub-sampled to a cadence the FOM truth does not "
            f"support. Choose dt <= dt_strided with dt_strided / dt = integer."
        )
    if abs(step_ratio * dt - dt_strided) > 1e-9 * dt_strided:
        raise ValueError(
            f"--dt-override={dt} does not cleanly divide dt_strided="
            f"{dt_strided} (step_ratio={dt_strided / dt}); the comparison "
            f"cadence cannot align to the strided FOM truth. Choose a dt that "
            f"satisfies dt_strided / dt = integer."
        )
    stride = int(z["stride"])
    n_train = int(z["n_train"])
    K_shape = int(z["K_shape"]) if "K_shape" in z.files else 1
    n_rbf = int(z["n_rbf"]) if "n_rbf" in z.files else centers_std.shape[0]
    standardise = bool(z["standardise"]) if "standardise" in z.files else True
    coords_path = (args.coords_path if args.coords_path is not None
                   else Path(str(z["coords_path"])))
    if args.dt_override is not None:
        print(f"[load] K={K_shape}, n_rbf={n_rbf}, dt_strided(fit)={dt_fit:.4f}, "
              f"dt(integrate)={dt:.4f} [overridden], step_ratio={step_ratio} "
              f"(compare cadence stays at dt_strided={dt_strided:.4f}), "
              f"standardise={standardise}", flush=True)
    else:
        print(f"[load] K={K_shape}, n_rbf={n_rbf}, dt_strided={dt:.4f}, "
              f"standardise={standardise}", flush=True)

    # ---- load coords ----
    print(f"[load] coords {coords_path}", flush=True)
    with np.load(coords_path) as zc:
        alpha = np.asarray(zc["alpha"], dtype=np.float64)
        alpha_dot = np.asarray(zc["alpha_dot"], dtype=np.float64)
    A_full = alpha[::stride]
    Y_full = alpha_dot[::stride]
    M, d_o = A_full.shape
    print(f"[load] strided alpha: M={M}, d_o={d_o}", flush=True)

    # ---- climate scale, used for Eq.21 ----
    std_climate = float(A_full.std(axis=0).mean())
    print(f"[stat] climate std (mean of per-mode std) = {std_climate:.4f}",
          flush=True)

    # `rhs` (bare SINDy, no corrector) and `rhs_int` (with the soft_reflect
    # corrector, if any) are built below by `build_integrator` once the cloud
    # escape threshold is known. The cloud-residual diagnostic measures the
    # bare fit via `_rhs_batch`; step-halving uses `rhs`; the integrator sees
    # `rhs_int`.

    # ---- cloud sample + tree + escape threshold (used by section E plot
    # ----  and, when --trust-region stop, by the integrator guard) -----------
    rng_v = np.random.default_rng(args.cloud_rng_seed)
    n_cv = min(args.n_cloud_residual, A_full.shape[0])
    sel = rng_v.choice(A_full.shape[0], size=n_cv, replace=False)
    A_cloud = A_full[sel]
    Y_cloud = Y_full[sel]
    A_std_cloud = (A_cloud - mu_A) / sigma_A
    tree = cKDTree(centers_std)
    d_nn_cloud, _ = tree.query(A_std_cloud, k=1)
    fom_d_nn_q = {
        "p50": float(np.quantile(d_nn_cloud, 0.50)),
        "p95": float(np.quantile(d_nn_cloud, 0.95)),
        "p99": float(np.quantile(d_nn_cloud, 0.99)),
    }
    escape_thresh = float(np.quantile(d_nn_cloud, args.cloud_escape_quantile))
    print(f"[cloud] FOM d_nn quantiles (std coords): "
          f"p50={fom_d_nn_q['p50']:.3f}, p95={fom_d_nn_q['p95']:.3f}, "
          f"p99={fom_d_nn_q['p99']:.3f}; escape_thresh"
          f"(q={args.cloud_escape_quantile:.2f})={escape_thresh:.3f}",
          flush=True)

    # Resolve the trust-region config in the parent: validate args, load any
    # pre-flight JSON, and emit the diagnostic prints. The actual rhs/guard
    # closures are built by `build_integrator` so the serial path and the
    # workers share one builder (closures don't pickle, the config does).
    preflight_thr_per_k = None
    preflight_k_per_k = None
    preflight_source = None
    if args.trust_region == "stop":
        print(f"[guard] trust-region stop active (thresh={escape_thresh:.3f})",
              flush=True)
    elif args.trust_region == "reflect":
        print(f"[guard] trust-region reflect active "
              f"(thresh={escape_thresh:.3f})", flush=True)
    elif args.trust_region == "soft_reflect":
        escape_thresh_sr = escape_thresh * args.soft_threshold_mult
        if args.soft_preflight_json is not None:
            if args.trust_region_mode != "per_cluster":
                raise ValueError(
                    "--soft-preflight-json requires "
                    "--trust-region-mode=per_cluster"
                )
            pf = json.loads(args.soft_preflight_json.read_text())
            pc = pf["per_cluster"]
            preflight_thr_per_k = np.asarray(pc["R"], dtype=np.float64)
            preflight_k_per_k = (
                np.asarray(pc["mult"], dtype=np.float64)
                * np.asarray(pc["p95_pos_radial_empirical"],
                             dtype=np.float64)
            )
            preflight_source = str(args.soft_preflight_json)
            print(f"[guard] soft_reflect pre-flight loaded from "
                  f"{args.soft_preflight_json.name}: per-cluster R="
                  f"[{preflight_thr_per_k.min():.4f},"
                  f"{preflight_thr_per_k.max():.4f}], k="
                  f"[{preflight_k_per_k.min():.3f},"
                  f"{preflight_k_per_k.max():.3f}]", flush=True)
        if args.trust_region_mode == "per_cluster":
            if parent_k is None:
                raise ValueError(
                    "trust-region-mode=per_cluster requires model.npz to "
                    "carry 'parent_k'; this model does not."
                )
            K_actual = int(parent_k.max()) + 1
            sizes = [int((parent_k == k_).sum()) for k_ in range(K_actual)]
            if preflight_thr_per_k is not None:
                if preflight_thr_per_k.shape != (K_actual,):
                    raise ValueError(
                        f"pre-flight K mismatch: json carries "
                        f"{preflight_thr_per_k.shape[0]} clusters, model "
                        f"has K={K_actual}"
                    )
            print(f"[guard] per-cluster soft_reflect: K={K_actual} trees "
                  f"sizes={sizes}", flush=True)
            if preflight_thr_per_k is not None:
                for k_ in range(K_actual):
                    print(f"[guard]   k={k_}: R_k={preflight_thr_per_k[k_]:.4f}"
                          f", k_k={preflight_k_per_k[k_]:.3f}", flush=True)
        on_ramp_d = args.soft_alpha * escape_thresh_sr
        if preflight_thr_per_k is None:
            print(f"[guard] trust-region soft_reflect active "
                  f"(mode={args.trust_region_mode}, "
                  f"thresh_q={escape_thresh:.4f}, "
                  f"thresh_sr={escape_thresh_sr:.4f} "
                  f"[mult={args.soft_threshold_mult:.2f}], "
                  f"alpha={args.soft_alpha:.3f}, on-ramp d>{on_ramp_d:.4f}, "
                  f"k={args.soft_k:.3f}, p={args.soft_p:.2f})", flush=True)
        else:
            print(f"[guard] trust-region soft_reflect active "
                  f"(mode={args.trust_region_mode}, "
                  f"thresh=per-cluster R_k from pre-flight, "
                  f"alpha={args.soft_alpha:.3f}, "
                  f"k=per-cluster mult_k*P95+_emp, "
                  f"p={args.soft_p:.2f})", flush=True)
    # Build the integrator once for the parent (serial path / step-halving /
    # cloud-residual diagnostic). Reuses the model already loaded above; the
    # closures it returns are what the serial loop integrates.
    bundle = build_integrator(
        args.model, escape_thresh=escape_thresh,
        trust_region=args.trust_region,
        trust_region_mode=args.trust_region_mode,
        soft_alpha=args.soft_alpha, soft_k=args.soft_k, soft_p=args.soft_p,
        soft_threshold_mult=args.soft_threshold_mult,
        preflight_thr_per_k=preflight_thr_per_k,
        preflight_k_per_k=preflight_k_per_k,
    )
    rhs = bundle["rhs"]
    rhs_int = bundle["rhs_int"]
    guard = bundle["guard"]

    # Picklable config for the workers: they rebuild the same integrator from
    # the model path + escape_thresh + guard knobs (closures don't pickle).
    worker_cfg = {
        "model_path": str(args.model),
        "escape_thresh": escape_thresh,
        "trust_region": args.trust_region,
        "trust_region_mode": args.trust_region_mode,
        "soft_alpha": args.soft_alpha,
        "soft_k": args.soft_k,
        "soft_p": args.soft_p,
        "soft_threshold_mult": args.soft_threshold_mult,
        "preflight_thr_per_k": preflight_thr_per_k,
        "preflight_k_per_k": preflight_k_per_k,
    }

    # ---- IC sampling (done in the parent so the chosen ICs are identical
    # ----  regardless of --n-workers) ----
    # Coarse (strided-cadence) step count: this bounds the FOM-truth slice and
    # the IC pool. The fine RK4 integration runs n_steps_fine = n_steps *
    # step_ratio steps and is sub-sampled back to n_steps + 1 coarse samples.
    n_steps = int(round(args.T / dt_strided))
    n_steps_fine = n_steps * step_ratio
    if args.ic_from == "test":
        lo = n_train
    else:
        lo = 0
    hi = M - n_steps - 1
    if hi <= lo:
        raise ValueError(
            f"not enough strided snapshots for n_steps={n_steps} from "
            f"pool [{lo}, M-n_steps={hi}]"
        )
    rng_ic = np.random.default_rng(args.ic_rng_seed)
    ic_idx = np.sort(rng_ic.choice(np.arange(lo, hi),
                                   size=args.n_ic, replace=False))
    print(f"[ic ] n_ic={args.n_ic}, pool=[{lo}, {hi}), idx={ic_idx.tolist()}, "
          f"n_workers={args.n_workers}", flush=True)

    # ---- integrate ROM + slice FOM truth, per IC ----
    Xs_rom: list[np.ndarray] = []
    Xs_true: list[np.ndarray] = []
    horizons = []
    rom_reasons: list[str] = []
    rom_n_events: list[int] = []
    rom_first_event_seconds: list[float | None] = []
    t0 = time.time()
    rom_results = _integrate_ics(
        ic_idx, A_full, n_steps_fine, dt, rhs_int, guard,
        n_workers=args.n_workers, worker_cfg=worker_cfg,
    )
    for i, (idx, (X_rom_fine, reason, n_evt, first_evt)) in enumerate(
            zip(ic_idx, rom_results)):
        # Sub-sample the fine ROM trajectory to the strided comparison cadence;
        # X_rom is then directly comparable to the strided FOM truth, one row
        # per dt_strided. Everything downstream slices/scores X_rom in coarse
        # samples and converts to seconds with dt_strided.
        X_rom = X_rom_fine[::step_ratio]
        X_true = A_full[idx: idx + n_steps + 1]
        Xs_true.append(X_true)
        Xs_rom.append(X_rom)
        rom_reasons.append(reason)
        rom_n_events.append(n_evt)
        # first_evt is a fine-step index; convert to seconds with the fine dt.
        rom_first_event_seconds.append(
            float(first_evt * dt) if first_evt is not None else None
        )
        h = _eq21_horizon(X_rom, X_true, dt_strided, std_climate)
        horizons.append(h)
        evt_tag = (f" evt={n_evt}" if guard is not None else "")
        print(f"  IC {i} (idx={idx}): horizon={h:.2f}s "
              f"({h / t_lambda:.2f} t_lambda), n_steps_returned="
              f"{X_rom.shape[0] - 1}, term={reason}{evt_tag}", flush=True)
    wall = time.time() - t0
    h_med = float(np.median(horizons))
    print(f"[integrate] wall={wall:.1f}s | Eq.21 horizon median = "
          f"{h_med:.2f}s ({h_med / t_lambda:.2f} t_lambda)", flush=True)

    # ---- imports for plotting (deferred so import errors are obvious) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- (A) short-window pointwise IC 0, mode grid ----
    # All windows below count strided comparison samples (Xs_rom is already
    # sub-sampled to the dt_strided cadence) and convert to seconds with
    # dt_strided.
    n_short = int(round(args.short_window_T / dt_strided))
    n_short_eff = min(n_short, Xs_rom[0].shape[0] - 1, Xs_true[0].shape[0] - 1)
    t_short = np.arange(n_short_eff + 1) * dt_strided
    X_t = Xs_true[0][: n_short_eff + 1]
    X_r = Xs_rom[0][: n_short_eff + 1]

    rows, cols = _grid_dims(d_o)
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 1.8 * rows),
                             sharex=True)
    axes = np.atleast_2d(axes).ravel()
    for k in range(d_o):
        ax = axes[k]
        ax.plot(t_short, X_t[:, k], "k-", lw=0.9, label="FOM")
        ax.plot(t_short, X_r[:, k], "C3-", lw=0.9, alpha=0.85, label="ROM")
        ax.set_title(f"mode {k}", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(True, ls=":", alpha=0.4)
    for k in range(d_o, len(axes)):
        axes[k].set_visible(False)
    for k in range(d_o - cols, d_o):
        axes[k].set_xlabel("t [s]", fontsize=8)
    axes[0].legend(loc="upper right", fontsize=7)
    fig.suptitle(
        f"LOR96 RBF integration, IC 0 (idx={int(ic_idx[0])}), "
        f"K={K_shape}, n_rbf={n_rbf}, "
        f"{'raw' if not standardise else 'z-scored'}, "
        f"Eq.21 horizon = {horizons[0]:.2f}s "
        f"({horizons[0] / t_lambda:.2f} t_L)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out_dir / "short_window_ic0.png", dpi=140,
                bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {args.out_dir / 'short_window_ic0.png'}", flush=True)

    # ---- (B) short-window RMSE(t) ensemble ----
    n_short_b = n_short
    rmse_traces: list[np.ndarray] = []
    for X_r, X_t in zip(Xs_rom, Xs_true):
        n_eff = min(n_short_b, X_r.shape[0] - 1, X_t.shape[0] - 1)
        err = X_r[: n_eff + 1] - X_t[: n_eff + 1]
        rmse_traces.append(np.sqrt((err * err).mean(axis=1)))
    n_min = min(r.size for r in rmse_traces)
    rmse_arr = np.stack([r[:n_min] for r in rmse_traces], axis=0)
    t_b = np.arange(n_min) * dt_strided
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.5))
    for i, r in enumerate(rmse_traces):
        ax.plot(np.arange(r.size) * dt_strided, r, lw=0.7, alpha=0.5,
                color=plt.cm.tab10(i % 10),
                label=f"IC {i} (T_ph={horizons[i]:.2f}s)")
    ax.plot(t_b, np.median(rmse_arr, axis=0), "k-", lw=1.8,
            label="median across ICs")
    ax.axhline(HORIZON_FRAC * std_climate, color="C7", ls="--", lw=1.0,
               label=(f"Eq.21 threshold "
                      f"= {HORIZON_FRAC:.2f}*std = "
                      f"{HORIZON_FRAC * std_climate:.3f}"))
    ax.axvline(h_med, color="C2", ls=":", lw=1.0,
               label=f"median horizon = {h_med:.2f}s")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"RMSE$_t = \sqrt{\langle (\hat\alpha - \alpha)^2 \rangle_k}$")
    ax.set_yscale("log")
    ax.grid(True, ls=":", alpha=0.4, which="both")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.set_title(
        f"LOR96 RBF integration, RMSE(t) ensemble "
        f"(K={K_shape}, n_rbf={n_rbf}, "
        f"{'raw' if not standardise else 'z-scored'})"
    )
    fig.tight_layout()
    fig.savefig(args.out_dir / "short_window_rmse.png", dpi=140,
                bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {args.out_dir / 'short_window_rmse.png'}", flush=True)

    # ---- (C) marginal PDFs + Wasserstein-1 over long window ----
    n_lo = int(round(args.long_window_t0 / dt_strided))
    pool_true: list[np.ndarray] = []
    pool_rom: list[np.ndarray] = []
    for X_r, X_t in zip(Xs_rom, Xs_true):
        if X_r.shape[0] > n_lo:
            tail_r = X_r[n_lo:]
            tail_r = tail_r[np.all(np.isfinite(tail_r), axis=1)]
            if tail_r.shape[0] > 0:
                pool_rom.append(tail_r)
        if X_t.shape[0] > n_lo:
            pool_true.append(X_t[n_lo:])
    pool_true_arr = (np.concatenate(pool_true, axis=0)
                     if pool_true else np.zeros((0, d_o)))
    pool_rom_arr = (np.concatenate(pool_rom, axis=0)
                    if pool_rom else np.zeros((0, d_o)))
    print(f"[pdf ] pooled samples: true={pool_true_arr.shape[0]}, "
          f"rom={pool_rom_arr.shape[0]}", flush=True)

    W1 = np.full(d_o, np.nan)
    if pool_rom_arr.shape[0] > 1 and pool_true_arr.shape[0] > 1:
        for k in range(d_o):
            W1[k] = wasserstein_distance(pool_true_arr[:, k],
                                         pool_rom_arr[:, k])

    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.0 * rows))
    axes = np.atleast_2d(axes).ravel()
    for k in range(d_o):
        ax = axes[k]
        if pool_true_arr.size:
            lo = float(min(pool_true_arr[:, k].min(),
                           pool_rom_arr[:, k].min() if pool_rom_arr.size
                           else pool_true_arr[:, k].min()))
            hi = float(max(pool_true_arr[:, k].max(),
                           pool_rom_arr[:, k].max() if pool_rom_arr.size
                           else pool_true_arr[:, k].max()))
            bins = np.linspace(lo, hi, 60)
            ax.hist(pool_true_arr[:, k], bins=bins, density=True,
                    color="k", alpha=0.55, label="FOM")
            if pool_rom_arr.size:
                ax.hist(pool_rom_arr[:, k], bins=bins, density=True,
                        color="C3", alpha=0.55, label="ROM")
        ax.set_title(f"mode {k}, W1={W1[k]:.2f}", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(True, ls=":", alpha=0.4)
    for k in range(d_o, len(axes)):
        axes[k].set_visible(False)
    axes[0].legend(loc="upper right", fontsize=7)
    fig.suptitle(
        f"LOR96 RBF marginal PDFs over [{args.long_window_t0:g}, "
        f"{args.T:g}] s (K={K_shape}, n_rbf={n_rbf}, "
        f"{'raw' if not standardise else 'z-scored'})"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out_dir / "marginal_pdfs.png", dpi=140,
                bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {args.out_dir / 'marginal_pdfs.png'}", flush=True)

    # ---- (D) autocorrelation per mode ----
    n_lag = int(round(args.acf_T / dt_strided))
    if pool_true_arr.shape[0] > n_lag + 10:
        acf_true = _per_mode_acf(pool_true_arr, n_lag)
    else:
        acf_true = np.full((n_lag + 1, d_o), np.nan)
    if pool_rom_arr.shape[0] > n_lag + 10:
        acf_rom = _per_mode_acf(pool_rom_arr, n_lag)
    else:
        acf_rom = np.full((n_lag + 1, d_o), np.nan)
    tau_lag = np.arange(n_lag + 1) * dt_strided

    def _tau_corr(acf: np.ndarray, thresh: float = 1.0 / np.e) -> np.ndarray:
        out = np.full(d_o, np.nan)
        for k in range(d_o):
            ix = np.flatnonzero(acf[:, k] < thresh)
            if ix.size > 0:
                out[k] = ix[0] * dt_strided
        return out

    tau_true = _tau_corr(acf_true)
    tau_rom = _tau_corr(acf_rom)

    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 1.8 * rows),
                             sharex=True)
    axes = np.atleast_2d(axes).ravel()
    for k in range(d_o):
        ax = axes[k]
        ax.plot(tau_lag, acf_true[:, k], "k-", lw=0.9, label="FOM")
        ax.plot(tau_lag, acf_rom[:, k], "C3-", lw=0.9, alpha=0.85, label="ROM")
        ax.axhline(1.0 / np.e, color="C7", ls=":", lw=0.6)
        ax.set_title(
            f"mode {k}, $\\tau$={tau_true[k]:.2f}/{tau_rom[k]:.2f}s",
            fontsize=8,
        )
        ax.tick_params(axis="both", labelsize=7)
        ax.set_ylim(-0.5, 1.05)
        ax.grid(True, ls=":", alpha=0.4)
    for k in range(d_o, len(axes)):
        axes[k].set_visible(False)
    for k in range(d_o - cols, d_o):
        axes[k].set_xlabel("lag [s]", fontsize=8)
    axes[0].legend(loc="upper right", fontsize=7)
    fig.suptitle(
        f"LOR96 per-mode ACF over [{args.long_window_t0:g}, "
        f"{args.T:g}] s (title: tau_FOM / tau_ROM, 1/e)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out_dir / "autocorrelation.png", dpi=140,
                bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {args.out_dir / 'autocorrelation.png'}", flush=True)

    # ---- (E) vector-field residual vs distance to nearest centre ----
    # (cloud sample, tree, d_nn_cloud, fom_d_nn_q, escape_thresh were
    # precomputed before the per-IC integration loop so they are available
    # to the trust-region guard.)
    f_rom = _rhs_batch(A_cloud, centers_std, Sigma_invs_std,
                       mu_A, sigma_A, col_norms, xi)
    res = np.linalg.norm(Y_cloud - f_rom, axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.0))
    ax.scatter(d_nn_cloud, res, s=4, alpha=0.35, color="C3")
    ax.set_xlabel("distance to nearest centre (standardised coords)")
    ax.set_ylabel(r"$\Vert \dot\alpha_{FOM} - f_{RBF}(\alpha) \Vert$")
    ax.set_yscale("log")
    ax.grid(True, ls=":", alpha=0.4, which="both")
    ax.set_title(
        f"Vector-field residual on FOM cloud (median={np.median(res):.3e}, "
        f"p90={np.quantile(res, 0.9):.3e})"
    )
    fig.tight_layout()
    fig.savefig(args.out_dir / "vector_field_residual.png", dpi=140,
                bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {args.out_dir / 'vector_field_residual.png'}",
          flush=True)

    # ---- (F) step-halving sanity ----
    # This is a pure RK4 convergence check (dt vs dt/2) and operates on the
    # FINE integration dt, independent of the comparison cadence: integrate the
    # bare RHS at dt and at dt/2 from the same IC and compare. The full-dt run
    # here is a fresh fine integration (Xs_rom[0] is sub-sampled to dt_strided
    # and so is not at the fine cadence the check needs).
    n_check = int(round(args.step_halving_T / dt))
    dt_half = dt / 2.0
    X_full_check, _, _, _ = _rk4_integrate(
        rhs, A_full[ic_idx[0]], dt, n_check
    )
    X_half, _, _, _ = _rk4_integrate(
        rhs, A_full[ic_idx[0]], dt_half, 2 * n_check
    )
    if X_half.shape[0] >= 2 * n_check + 1 and X_full_check.shape[0] >= n_check + 1:
        diff = np.linalg.norm(X_half[::2][: n_check + 1]
                              - X_full_check[: n_check + 1], axis=1)
    else:
        diff = np.array([np.nan])
    t_check = np.arange(diff.size) * dt
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    ax.semilogy(t_check, np.maximum(diff, 1e-16), "-", color="C3",
                label=f"RBF: max={np.nanmax(diff):.2e}")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$\Vert X_{dt} - X_{dt/2} \Vert$")
    ax.set_title(
        f"RK4 step-halving sanity, IC 0  (dt={dt:g} vs {dt_half:g})"
    )
    ax.grid(True, ls=":", alpha=0.4, which="both")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out_dir / "step_halving_sanity.png", dpi=140,
                bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {args.out_dir / 'step_halving_sanity.png'}",
          flush=True)

    # ---- (G) cloud-edge guard along the comparison-window trajectories ----
    ce_per_ic: list[dict] = []
    ce_traces: list[np.ndarray] = []
    for i, X_r in enumerate(Xs_rom):
        finite_mask = np.all(np.isfinite(X_r), axis=1)
        n_fin = int(finite_mask.sum())
        if n_fin == 0:
            ce_traces.append(np.array([np.nan]))
            ce_per_ic.append({
                "max_d_nn": float("nan"),
                "frac_above_thresh": 1.0,
                "first_escape_seconds": 0.0,
                "blew_up_at_seconds": 0.0,
                "termination_reason": rom_reasons[i],
            })
            continue
        d_nn_t = _trajectory_d_nn(X_r[:n_fin], mu_A, sigma_A, tree)
        ce_traces.append(d_nn_t)
        above = d_nn_t > escape_thresh
        # X_r here is the comparison-window ROM, sub-sampled to dt_strided, so
        # its time axis and escape times are in dt_strided units.
        first_escape_s = (float(int(np.flatnonzero(above)[0]) * dt_strided)
                          if above.any() else None)
        blew_up_at = (float(n_fin * dt_strided)
                      if n_fin < X_r.shape[0] else None)
        ce_per_ic.append({
            "termination_reason": rom_reasons[i],
            "max_d_nn": float(d_nn_t.max()),
            "frac_above_thresh": float(above.mean()),
            "first_escape_seconds": first_escape_s,
            "blew_up_at_seconds": blew_up_at,
        })
    # FOM truth reference along IC 0
    truth_d_nn = _trajectory_d_nn(Xs_true[0], mu_A, sigma_A, tree)

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    for i, d_nn_t in enumerate(ce_traces):
        t_e = np.arange(d_nn_t.size) * dt_strided
        ax.plot(t_e, d_nn_t, lw=0.7, alpha=0.7,
                color=plt.cm.tab10(i % 10),
                label=(f"IC {i}" if i < 10 else None))
    ax.plot(np.arange(truth_d_nn.size) * dt_strided, truth_d_nn,
            "k--", lw=1.0, label="FOM truth (IC 0)")
    for q_name, q_val in fom_d_nn_q.items():
        ax.axhline(q_val, color="C7", ls=":", lw=0.6)
        ax.text(0.0, q_val, f" FOM {q_name}", fontsize=7,
                color="C7", va="bottom")
    ax.axhline(escape_thresh, color="C1", ls="--", lw=1.0,
               label=f"escape thresh (q={args.cloud_escape_quantile:.2f})")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("distance to nearest centre (std coords)")
    ax.set_yscale("log")
    ax.grid(True, ls=":", alpha=0.4, which="both")
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.set_title(
        f"Cloud-edge guard, [0, {args.T:g}]s "
        f"(K={K_shape}, n_rbf={n_rbf}, "
        f"{'raw' if not standardise else 'std'})"
    )
    fig.tight_layout()
    fig.savefig(args.out_dir / "cloud_edge.png", dpi=140,
                bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {args.out_dir / 'cloud_edge.png'}", flush=True)

    # ---- (H) ROM-only stability run, no FOM truth required ----
    stab_block = None
    if args.stability_T > 0.0:
        # ROM-only diagnostic: no FOM truth to align to, so it runs at the
        # FINE integration dt and its d_nn(t)/|alpha|(t) traces stay fine
        # (more informative for catching escape timing). All its seconds
        # conversions use the fine `dt`, which is still real seconds.
        n_stab = int(round(args.stability_T / dt))
        print(f"[stab] integrating {args.n_ic} IC(s) for {n_stab} steps "
              f"({args.stability_T:.1f}s)", flush=True)
        stab_dnn_traces: list[np.ndarray] = []
        stab_norm_traces: list[np.ndarray] = []
        stab_per_ic: list[dict] = []
        fom_norms = np.linalg.norm(A_full, axis=1)
        fom_norm_med = float(np.median(fom_norms))
        fom_norm_p99 = float(np.quantile(fom_norms, 0.99))
        stab_reasons: list[str] = []
        stab_n_events: list[int] = []
        stab_first_event_seconds: list[float | None] = []
        stab_results = _integrate_ics(
            ic_idx, A_full, n_stab, dt, rhs_int, guard,
            n_workers=args.n_workers, worker_cfg=worker_cfg,
        )
        for i, (idx, (X_s, reason, n_evt, first_evt)) in enumerate(
                zip(ic_idx, stab_results)):
            stab_reasons.append(reason)
            stab_n_events.append(n_evt)
            stab_first_event_seconds.append(
                float(first_evt * dt) if first_evt is not None else None
            )
            finite_mask = np.all(np.isfinite(X_s), axis=1)
            n_fin = int(finite_mask.sum())
            X_s_fin = X_s[:n_fin]
            d_nn_s = _trajectory_d_nn(X_s_fin, mu_A, sigma_A, tree)
            norms_s = np.linalg.norm(X_s_fin, axis=1)
            stab_dnn_traces.append(d_nn_s)
            stab_norm_traces.append(norms_s)
            above = d_nn_s > escape_thresh
            first_escape_s = (float(int(np.flatnonzero(above)[0]) * dt)
                              if above.any() else None)
            terminated_early = (n_fin != n_stab + 1)
            blew_up_at = (float(n_fin * dt) if terminated_early else None)
            stab_per_ic.append({
                "n_steps_finite": n_fin,
                "survived_full": bool(not terminated_early),
                "termination_reason": reason,
                "blew_up_at_seconds": blew_up_at,
                "max_d_nn": (float(d_nn_s.max()) if d_nn_s.size
                             else float("nan")),
                "frac_above_thresh": (float(above.mean()) if d_nn_s.size
                                      else 1.0),
                "first_escape_seconds": first_escape_s,
                "max_norm": (float(norms_s.max()) if norms_s.size
                             else float("nan")),
            })
            tag = ("survived" if blew_up_at is None
                   else f"{reason}@{blew_up_at:.1f}s")
            print(f"  stab IC {i} (idx={idx}): {tag}, "
                  f"max_d_nn={stab_per_ic[-1]['max_d_nn']:.2f}, "
                  f"frac_above={stab_per_ic[-1]['frac_above_thresh']:.3f}",
                  flush=True)

        fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
        for i, (d_s, n_s) in enumerate(zip(stab_dnn_traces, stab_norm_traces)):
            t_s = np.arange(d_s.size) * dt
            axes[0].plot(t_s, d_s, lw=0.7, alpha=0.75,
                         color=plt.cm.tab10(i % 10),
                         label=(f"IC {i}" if i < 10 else None))
            axes[1].plot(t_s, n_s, lw=0.7, alpha=0.75,
                         color=plt.cm.tab10(i % 10))
        for _, q_val in fom_d_nn_q.items():
            axes[0].axhline(q_val, color="C7", ls=":", lw=0.6)
        axes[0].axhline(escape_thresh, color="C1", ls="--", lw=1.0,
                        label=f"escape thresh "
                              f"(q={args.cloud_escape_quantile:.2f})")
        axes[0].set_yscale("log")
        axes[0].set_ylabel("d_nn (std)")
        axes[0].grid(True, ls=":", alpha=0.4, which="both")
        axes[0].legend(fontsize=7, ncol=2, loc="upper right")
        axes[0].set_title(
            f"Stability run, [0, {args.stability_T:g}]s "
            f"(K={K_shape}, n_rbf={n_rbf}, "
            f"{'raw' if not standardise else 'std'})"
        )
        axes[1].axhline(fom_norm_med, color="k", ls=":", lw=0.7,
                        label=f"FOM median |alpha|={fom_norm_med:.2f}")
        axes[1].axhline(fom_norm_p99, color="k", ls="--", lw=0.7,
                        label=f"FOM p99 |alpha|={fom_norm_p99:.2f}")
        axes[1].set_yscale("log")
        axes[1].set_ylabel("|alpha(t)|")
        axes[1].set_xlabel("t [s]")
        axes[1].grid(True, ls=":", alpha=0.4, which="both")
        axes[1].legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        fig.savefig(args.out_dir / "stability.png", dpi=140,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] wrote {args.out_dir / 'stability.png'}", flush=True)

        max_d_nn_list = [s["max_d_nn"] for s in stab_per_ic
                         if np.isfinite(s["max_d_nn"])]
        stab_block = {
            "stability_T": args.stability_T,
            "n_ic": args.n_ic,
            "n_ic_survived_full": int(sum(1 for s in stab_per_ic
                                          if s["survived_full"])),
            "median_max_d_nn": (float(np.median(max_d_nn_list))
                                if max_d_nn_list else float("nan")),
            "median_frac_above_thresh": float(np.median(
                [s["frac_above_thresh"] for s in stab_per_ic])),
            "fom_norm_median": fom_norm_med,
            "fom_norm_p99": fom_norm_p99,
            "per_ic": stab_per_ic,
        }

    # ---- metrics ----
    metrics = {
        "config": {
            "K_shape": K_shape,
            "n_rbf": n_rbf,
            "standardise": standardise,
            "stride": stride,
            "dt_strided": dt_strided,
            "dt_fine": dt,
            "dt_compare": dt_strided,
            "step_ratio": step_ratio,
            "n_train_strided": n_train,
            "M_strided": M,
            "T": args.T,
            "short_window_T": args.short_window_T,
            "long_window_t0": args.long_window_t0,
            "step_halving_T": args.step_halving_T,
            "n_ic": args.n_ic,
            "ic_rng_seed": args.ic_rng_seed,
            "ic_from": args.ic_from,
            "n_cloud_residual": n_cv,
            "lyapunov_rate": args.lyapunov_rate,
            "t_lambda": t_lambda,
            "horizon_frac": HORIZON_FRAC,
            "std_climate": std_climate,
            "cloud_escape_quantile": args.cloud_escape_quantile,
            "stability_T": args.stability_T,
            "trust_region": args.trust_region,
            "soft_alpha": args.soft_alpha if args.trust_region == "soft_reflect" else None,
            "soft_k":     args.soft_k     if args.trust_region == "soft_reflect" else None,
            "soft_p":     args.soft_p     if args.trust_region == "soft_reflect" else None,
            "soft_preflight_json": preflight_source if args.trust_region == "soft_reflect" else None,
            "soft_threshold_per_cluster": (preflight_thr_per_k.tolist()
                                            if preflight_thr_per_k is not None
                                            else None),
            "soft_k_per_cluster": (preflight_k_per_k.tolist()
                                    if preflight_k_per_k is not None
                                    else None),
        },
        "horizons_seconds": list(map(float, horizons)),
        "horizon_median_seconds": h_med,
        "horizon_median_lyapunov_times": h_med / t_lambda,
        "ic_idx": list(map(int, ic_idx)),
        "vector_field_residual_cloud": {
            "median": float(np.median(res)),
            "p90": float(np.quantile(res, 0.9)),
            "max": float(np.max(res)),
        },
        "wasserstein_1_per_mode": list(map(float, W1)),
        "wasserstein_1_median": float(np.nanmedian(W1)),
        "tau_corr_per_mode": {
            "fom": list(map(float, tau_true)),
            "rom": list(map(float, tau_rom)),
        },
        "step_halving_max_diff": float(np.nanmax(diff)),
        "cloud_edge": {
            "fom_d_nn_quantiles_std": fom_d_nn_q,
            "escape_threshold_std": escape_thresh,
            "per_ic": ce_per_ic,
            "median_max_d_nn": float(np.median(
                [c["max_d_nn"] for c in ce_per_ic
                 if np.isfinite(c["max_d_nn"])]
            )) if any(np.isfinite(c["max_d_nn"]) for c in ce_per_ic)
            else float("nan"),
            "median_frac_above_thresh": float(np.median(
                [c["frac_above_thresh"] for c in ce_per_ic])),
        },
    }
    if stab_block is not None:
        metrics["stability"] = stab_block
    if args.trust_region != "none":
        # Summarise guard events. "first_event_seconds" is the time of the
        # first guard activation as captured by the integrator; under "stop"
        # this is also the termination time, under "reflect" it is the first
        # bounce and trajectories continue. "n_events_per_ic" counts how many
        # times the guard fired along that trajectory (always <=1 for stop).
        comp_first = [t for t in rom_first_event_seconds if t is not None]
        guard_block = {
            "mode": args.trust_region,
            "escape_threshold_std": escape_thresh,
            "comparison_reasons": rom_reasons,
            "comparison_first_event_seconds_per_ic": rom_first_event_seconds,
            "comparison_median_first_event_seconds":
                float(np.median(comp_first)) if comp_first else None,
            "comparison_n_events_per_ic": rom_n_events,
            "comparison_total_events": int(sum(rom_n_events)),
        }
        if stab_block is not None:
            stab_first = [t for t in stab_first_event_seconds if t is not None]
            guard_block["stability_reasons"] = stab_reasons
            guard_block["stability_first_event_seconds_per_ic"] = \
                stab_first_event_seconds
            guard_block["stability_median_first_event_seconds"] = (
                float(np.median(stab_first)) if stab_first else None)
            guard_block["stability_n_events_per_ic"] = stab_n_events
            guard_block["stability_total_events"] = int(sum(stab_n_events))
            guard_block["stability_event_rate_per_second_per_ic"] = [
                (n / args.stability_T) if args.stability_T > 0 else 0.0
                for n in stab_n_events
            ]
        metrics["guard"] = guard_block
    out_json = args.out_dir / "metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2))
    print(f"[save] wrote {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
