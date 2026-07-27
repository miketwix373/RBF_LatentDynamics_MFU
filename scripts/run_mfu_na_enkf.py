"""EnKF ensemble-forecast harness for MFU_NA (d_o=45, alpha_tang=3.5).

Step 1 of the EnKF build after Ozalp, Novoa & Magri, "Real-time forecasting of
chaotic dynamics from sparse data and autoencoders", CMAME 450 (2026) 118600:
the ensemble FORECAST only, parallelised across the cores of one node.

Each of the N_ens members is a reduced-coordinate state a in R^{d_o}. The
forecast advances every member with the calibrated RBF vector field plus the
additive trust-region corrector (adot = Phi(a).xi + c(a), the model's
`soft_reflect` corrector) via RK4 over one observation interval -- the paper's
forecast step, psi_j^f = F(psi_j^a), Eq. (8a)/(13). Members are integrated
concurrently via ProcessPoolExecutor, reusing the picklable
`build_integrator` / `_integrate_one_ic` machinery from
`run_integrate_compare.py` so the serial and ensemble paths cannot diverge.

Ensemble initialisation follows the paper's step 2a (perturbed initial
conditions, or random snapshots sampled from the FOM record). Reduced
coordinates only; the wall-shear field is u = mu + V.a from coords_do45.npz.

The EnKF ANALYSIS step (perturbed-obs update, Eqs. 14-17; sparse wall-shear
operator H = M.V) is NOT implemented here yet -- see the seam at the end of
`main`. Consult rom-specialist before wiring it.

Usage
-----
    python scripts/run_mfu_na_enkf.py --n-ens 50 --interval-s 1.0 --seed 0
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_integrate_compare import (  # noqa: E402
    _make_rhs,
    _build_soft_reflect_corrector,
    _rk4_integrate,
)
from run_kschao_corrector_series import _load_model  # noqa: E402

SHARED_ROOT = (next(_p for _p in Path(__file__).resolve().parents if (_p / "pyproject.toml").exists()))
TPLUS_PER_S = 11.289   # wall-unit conversion: 1 s = 11.289 t+
DEFAULT_MODEL = SHARED_ROOT / "results/MFU_NA/do45/alpha3.5/model_n04800.npz"
DEFAULT_CALIB = SHARED_ROOT / "results/MFU_NA/do45/alpha3.5/corrector_calib.npz"
DEFAULT_COORDS = SHARED_ROOT / "results/MFU_NA/pod/coords_do45.npz"
DEFAULT_OUT = SHARED_ROOT / "results/MFU_NA/EnKF"


def _n_workers_default() -> int:
    """One process per allocated core; honour the SLURM allocation if present."""
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        if os.environ.get(var):
            return int(os.environ[var])
    return os.cpu_count() or 1


def _worker_cfg(model_path: Path, R: float, soft_alpha: float, soft_k: float,
                soft_p: float) -> dict:
    """Picklable per-member config. The RHS closures (cKDTree, corrector) do
    not survive pickling, so we ship only the model path + scalar corrector
    knobs and rebuild the integrator inside each worker (see `_build_rhs`).
    Mirrors the knobs `run_mfu_na_forecast_suite.py` passes for the free-running
    rollout, so an ensemble member forecasts identically to a suite IC."""
    return {
        "model_path": str(model_path),
        "escape_thresh": float(R),
        "soft_alpha": float(soft_alpha),
        "soft_k": float(soft_k),
        "soft_p": float(soft_p),
    }


# Worker globals: ProcessPoolExecutor reuses one process across tasks, so build
# the integrator (cKDTree + corrector) once per worker and cache it, keyed by
# the model + corrector config, exactly as run_integrate_compare does.
_WORKER_RHS = None
_WORKER_KEY = None


def _build_rhs(cfg: dict):
    """Load the model (expanding the compact Sigma form via `_load_model`) and
    assemble adot = Phi(a).xi + c(a) with the universal soft_reflect corrector
    at radius R. Same construction as run_mfu_na_forecast_suite.py."""
    m = _load_model(Path(cfg["model_path"]))
    tree = cKDTree(m["centers_std"])
    corr = _build_soft_reflect_corrector(
        tree, m["mu_A"], m["sigma_A"], cfg["escape_thresh"],
        alpha=cfg["soft_alpha"], k_strength=cfg["soft_k"], p=cfg["soft_p"])
    return _make_rhs(m["centers_std"], m["Sigma_invs_std"], m["mu_A"],
                     m["sigma_A"], m["col_norms"], m["xi"], corrector=corr)


def _forecast_one_member(task):
    """Integrate one ensemble member forward n_steps (RK4, no stopping guard).
    `task` is the picklable tuple (cfg, x0, n_steps, dt). Returns the same
    (X, reason, n_events, first_event) tuple as `_rk4_integrate`."""
    global _WORKER_RHS, _WORKER_KEY
    cfg, x0, n_steps, dt = task
    key = (cfg["model_path"], cfg["escape_thresh"], cfg["soft_alpha"],
           cfg["soft_k"], cfg["soft_p"])
    if _WORKER_KEY != key:
        _WORKER_RHS = _build_rhs(cfg)
        _WORKER_KEY = key
    return _rk4_integrate(_WORKER_RHS, np.asarray(x0, np.float64), dt, n_steps,
                          guard=None)


def init_ensemble(A_full: np.ndarray, n_train: int, n_ens: int, mode: str,
                  base_idx: int, spread: float, coord_std: np.ndarray,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Build the initial analysis ensemble A0 of shape (n_ens, d_o).

    Paper step 2a. Two modes:
      * "perturb"   -- one base snapshot A_full[base_idx] plus per-mode Gaussian
                       noise of std `spread * coord_std` (twin-experiment start:
                       members bracket a known truth IC). `coord_std` is the
                       climatological per-mode std of the FOM record, so spread
                       is a fraction of each mode's natural variability -- the
                       right scale whether or not the model standardises (for
                       MFU_NA sigma_A == 1, so raw-coord spread would be wrong).
      * "snapshots" -- n_ens states drawn at random from the training record
                       [0, n_train) (climatological / invariant-measure start).

    Returns (A0, meta_idx) where meta_idx are the source snapshot indices
    (base_idx repeated for "perturb", the drawn indices for "snapshots").
    """
    d_o = A_full.shape[1]
    if mode == "perturb":
        base = A_full[base_idx]
        A0 = base[None, :] + spread * coord_std[None, :] * rng.standard_normal(
            (n_ens, d_o))
        meta_idx = np.full(n_ens, base_idx, dtype=np.int64)
    elif mode == "snapshots":
        meta_idx = rng.choice(n_train, size=n_ens, replace=False).astype(np.int64)
        A0 = A_full[meta_idx].copy()
    else:
        raise ValueError(f"unknown init mode {mode!r}")
    return A0, meta_idx


def forecast_ensemble(A_ens: np.ndarray, cfg: dict, dt: float, n_steps: int,
                      n_workers: int, keep_traj: bool
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Advance every member n_steps via RK4 (+ corrector), across `n_workers`.

    Returns (A_fore, reasons, traj):
      A_fore : (n_ens, d_o)          member state at the assimilation time.
      reasons: (n_ens,)              per-member RK4 termination reason.
      traj   : (n_ens, n_steps+1, d_o) or None if `keep_traj` is False.
    A member that goes non-finite keeps its last finite state in A_fore and is
    flagged by its reason; the corrector is meant to keep every member bounded.
    """
    n_ens, d_o = A_ens.shape
    tasks = [(cfg, A_ens[j], n_steps, dt) for j in range(n_ens)]
    if n_workers <= 1:
        results = [_forecast_one_member(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(_forecast_one_member, tasks))

    A_fore = np.full((n_ens, d_o), np.nan, np.float64)
    reasons = np.empty(n_ens, dtype="<U12")
    traj = np.full((n_ens, n_steps + 1, d_o), np.nan, np.float64) if keep_traj \
        else None
    for j, (X, reason, _n_evt, _first) in enumerate(results):
        A_fore[j] = X[-1]
        reasons[j] = reason
        if keep_traj:
            traj[j, : X.shape[0]] = X
    return A_fore, reasons, traj


def cycled_reset_forecast(A_full: np.ndarray, base_idx: int,
                          coord_std: np.ndarray, spread: float, cfg: dict,
                          dt: float, n_total_steps: int, window_steps: int,
                          n_ens: int, rng: np.random.Generator, n_workers: int
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Perfect-analysis sawtooth for visualisation: free-run every member for
    `window_steps`, then reset the whole ensemble to the true state at that time
    plus a fresh perturbation (spread * coord_std), and repeat over the full
    horizon. The re-perturbation is required because the RBF field is
    deterministic -- resetting to the exact truth would collapse the ensemble to
    a single line within each window. This is a stand-in for the EnKF analysis
    (analysis == truth), not the analysis itself.

    Returns (traj, reset_steps): traj is (n_ens, n_total_steps+1, d_o); the
    step index that starts each window (where the reset happens) is in
    reset_steps.
    """
    d_o = A_full.shape[1]
    traj = np.full((n_ens, n_total_steps + 1, d_o), np.nan, np.float64)
    reset_steps = []
    s = 0
    while s < n_total_steps:
        truth_s = A_full[base_idx + s]
        A = truth_s[None, :] + spread * coord_std[None, :] * \
            rng.standard_normal((n_ens, d_o))
        traj[:, s] = A
        reset_steps.append(s)
        w = min(window_steps, n_total_steps - s)
        _, _, wtraj = forecast_ensemble(A, cfg, dt, w, n_workers, keep_traj=True)
        traj[:, s + 1: s + w + 1] = wtraj[:, 1:]
        s += w
    return traj, np.asarray(reset_steps, np.int64)


def build_obs_operator(V: np.ndarray, stride_x: int, stride_z: int,
                       components: str, ng: int = 64
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Sparse wall-shear observation operator H = M.V for a uniform (or
    anisotropic) stride mask on the ng x ng wall grid.

    State layout is [tau_x (ng*ng); tau_y (ng*ng)], each row-major (ng, ng).
    axis0 is taken as streamwise (x), axis1 as spanwise (z). `components` is
    'x', 'y' or 'both'. Returns (H, rows): H is (N_obs, d_o) = V[rows, :], the
    exact linear operator on the reduced coords (the decoder u = mu + V.a is
    linear, so no augmented state is needed -- see rom-specialist note); rows
    are the selected full-state indices.

    Because the observation is a projection onto the 45 POD modes, the mask's
    quality is the CONDITIONING of H (rank / sigma_min), not a Nyquist rule
    (fluid-dynamics-specialist): a mask that cannot separate two modes gives a
    rank-deficient H regardless of how it aliases the physical field.
    """
    csz = ng * ng
    a0 = np.arange(0, ng, stride_x)
    a1 = np.arange(0, ng, stride_z)
    sel = (a0[:, None] * ng + a1[None, :]).ravel()      # row-major (ng, ng)
    if components == "x":
        rows = sel
    elif components == "y":
        rows = csz + sel
    elif components == "both":
        rows = np.concatenate([sel, csz + sel])
    else:
        raise ValueError(f"unknown components {components!r}")
    return V[rows, :].copy(), rows


def obs_noise_std(V: np.ndarray, A_train: np.ndarray, rows: np.ndarray,
                  sigma: float) -> np.ndarray:
    """Per-sensor observation-noise std = sigma * rms(observable), i.e. a
    RELATIVE 10%(sigma) noise on the wall-shear FLUCTUATION each sensor sees
    (the centred field V[rows].a the ROM represents), so C_dd = std**2.

    This is the paper's diag(sigma*u_max) form adapted to a CENTRED POD: their
    fields are ~zero-mean so u_max tracks the signal scale, but here the mean
    skin friction dominates the full-field max and sigma*max_full would be ~70%
    of the fluctuation signal -- weak obs that let the filter diverge. Scaling
    by the observable's rms (rom-specialist recommendation) gives a clean, sane
    noise floor. mu drops out because V.a is already the centred fluctuation."""
    fluct = V[rows, :] @ A_train.T                      # (N_obs, n_train)
    return sigma * fluct.std(axis=1)


def enkf_update(A: np.ndarray, H: np.ndarray, y_obs: np.ndarray,
                cdd_diag: np.ndarray, inflation: float, coord_std: np.ndarray,
                additive_q: float, rng: np.random.Generator
                ) -> tuple[np.ndarray, dict]:
    """One perturbed-observation EnKF analysis on the reduced coords, using the
    EXACT known H (route b, rom-specialist): C_ay = C_aa H^T, C_yy = H C_aa H^T
    from the 45x45 ensemble covariance. Multiplicative prior inflation on the
    forecast deviations (mandatory in a perfect-model twin). y_obs is the
    centred noisy observation H a_true + eps (mu cancels between obs and
    prediction). Returns (A_analysis, diagnostics).

    Diagnostics (rom-specialist, wire from day one):
      chi2    -- normalised innovation (d^T S^-1 d); ~ N_obs if consistent,
                 >> N_obs means the filter is over-confident / collapsing.
      tr_Caa  -- total ensemble variance (monotone decay => collapse).
      spread  -- sqrt(tr_Caa).

    Spread maintenance in this perfect-model twin: multiplicative `inflation` on
    the forecast deviations, plus optional ADDITIVE inflation `additive_q` --
    N(0, (additive_q*coord_std)^2) injected into the analysis ensemble to floor
    the spread (the analysis contraction from 100+ strong obs is too severe for
    multiplicative inflation alone; without a floor Caa collapses, the gain
    vanishes, and the mean free-runs). additive_q is a fraction of each mode's
    climatological std.
    """
    n_ens, d_o = A.shape
    n_obs = H.shape[0]
    abar = A.mean(axis=0)
    Ap = A - abar
    if inflation != 1.0:
        Ap = inflation * Ap
        A = abar + Ap
    Caa = (Ap.T @ Ap) / (n_ens - 1)                     # (d_o, d_o)
    Cay = Caa @ H.T                                      # (d_o, n_obs)
    Cyy = H @ Cay                                        # (n_obs, n_obs)
    S = Cyy + np.diag(cdd_diag)
    K = Cay @ np.linalg.inv(S)                           # (d_o, n_obs)
    Y = A @ H.T                                          # predicted obs (centred)
    D = y_obs[None, :] + rng.standard_normal((n_ens, n_obs)) * \
        np.sqrt(cdd_diag)[None, :]                       # perturbed obs
    A_new = A + (D - Y) @ K.T
    if additive_q > 0.0:
        A_new = A_new + additive_q * coord_std[None, :] * \
            rng.standard_normal((n_ens, d_o))
    d_innov = y_obs - abar @ H.T
    chi2 = float(d_innov @ np.linalg.solve(S, d_innov))
    return A_new, {"chi2": chi2, "tr_Caa": float(np.trace(Caa)),
                   "spread": float(np.sqrt(np.trace(Caa)))}


def cycled_enkf_forecast(A_full: np.ndarray, base_idx: int,
                         coord_std: np.ndarray, spread: float, cfg: dict,
                         dt: float, n_total_steps: int, window_steps: int,
                         n_ens: int, rng: np.random.Generator, n_workers: int,
                         H: np.ndarray, cdd_std: np.ndarray, sigma: float,
                         inflation: float, additive_q: float,
                         truth: np.ndarray | None = None
                         ) -> tuple[np.ndarray, np.ndarray, list]:
    """Full EnKF assimilation cycle over the horizon: forecast every member one
    observation interval, then a perturbed-obs analysis against sparse
    wall-shear observations of the truth, repeat. Observations are
    y = H a_true + eps, eps ~ N(0, diag(cdd_std**2)). Returns (traj, obs_steps,
    diags) where the analysis (post-update) ensemble is stored at each obs step.

    Also records, per cycle, the per-mode FORECAST error (forecast mean - truth
    just before each analysis). With the previous analysis pinning the state
    near truth (full-rank obs), this is the model error accumulated over one
    cadence window per mode -- the raw material for the mode-resolved q*.

    `truth`, if given, replaces the FOM record as the observation source: a
    (n_total_steps+1, d_o) trajectory indexed by step (perfect-model twin --
    obs generated by the ROM itself, so the chi2/N=1 crossing is the pure
    DA-induced q* floor at this cadence).
    """
    d_o = A_full.shape[1]
    cdd_diag = cdd_std ** 2
    traj = np.full((n_ens, n_total_steps + 1, d_o), np.nan, np.float64)
    A = A_full[base_idx][None, :] + spread * coord_std[None, :] * \
        rng.standard_normal((n_ens, d_o))
    traj[:, 0] = A
    obs_steps, diags, ferr = [], [], []
    s = 0
    while s < n_total_steps:
        w = min(window_steps, n_total_steps - s)
        _, _, wtraj = forecast_ensemble(A, cfg, dt, w, n_workers, keep_traj=True)
        traj[:, s + 1: s + w + 1] = wtraj[:, 1:]
        A = wtraj[:, -1, :].copy()
        s += w
        a_true = truth[s] if truth is not None else A_full[base_idx + s]
        ferr.append(A.mean(0) - a_true)                 # per-mode forecast error
        y_obs = H @ a_true + rng.standard_normal(H.shape[0]) * cdd_std
        A, dg = enkf_update(A, H, y_obs, cdd_diag, inflation, coord_std,
                            additive_q, rng)
        traj[:, s] = A
        obs_steps.append(s)
        dg["err"] = float(np.linalg.norm(A.mean(0) - a_true))
        diags.append(dg)
    return (traj, np.asarray(obs_steps, np.int64), diags,
            np.asarray(ferr, np.float64))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--calib", type=Path, default=DEFAULT_CALIB,
                   help="corrector_calib.npz (R + soft_* params)")
    p.add_argument("--coords-path", type=Path, default=DEFAULT_COORDS)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--n-ens", type=int, default=50, help="ensemble size N_ens")
    p.add_argument("--interval-s", type=float, default=1.0,
                   help="observation interval (s); n_steps = interval / dt")
    p.add_argument("--init", choices=("perturb", "snapshots"), default="perturb")
    p.add_argument("--reset-tplus", type=float, default=0.0,
                   help="if >0, perfect-analysis sawtooth: reset the whole "
                        "ensemble to truth (+perturbation) every this many t+ "
                        "over the full --interval-s horizon (visualisation)")
    p.add_argument("--assimilate", action="store_true",
                   help="run the EnKF (perturbed-obs analysis vs sparse "
                        "wall-shear obs) instead of a plain forecast")
    p.add_argument("--twin", action="store_true",
                   help="perfect-model twin: observations come from a "
                        "deterministic ROM rollout (started at the base "
                        "snapshot) instead of the FOM record, giving the "
                        "DA-induced q* floor")
    p.add_argument("--cadence-tplus", type=float, default=5.0,
                   help="assimilation interval in t+ (below the ~9.5 t+ VPT)")
    p.add_argument("--sensor-stride", type=int, default=8,
                   help="uniform wall-grid sensor stride")
    p.add_argument("--sensor-stride-x", type=int, default=0,
                   help="streamwise stride override (0 => use --sensor-stride)")
    p.add_argument("--sensor-stride-z", type=int, default=0,
                   help="spanwise stride override (0 => use --sensor-stride)")
    p.add_argument("--components", choices=("x", "y", "both"), default="both")
    p.add_argument("--sigma-obs", type=float, default=0.1,
                   help="relative obs-noise level; std = sigma*u_max per sensor")
    p.add_argument("--inflation", type=float, default=1.05,
                   help="multiplicative prior inflation (perfect-model twin)")
    p.add_argument("--additive-q", type=float, default=0.0,
                   help="additive inflation: analysis noise std as a fraction "
                        "of each mode's climatological std (floors the spread)")
    p.add_argument("--corr-k-scale", type=float, default=1.0,
                   help="scale the corrector strength inside the assim window "
                        "(0 disables it; rom-specialist flags double-regular.)")
    p.add_argument("--base-idx", type=int, default=-1,
                   help="perturb-mode base snapshot; -1 => first test snapshot")
    p.add_argument("--spread", type=float, default=0.1,
                   help="perturb-mode noise std as a fraction of each mode's "
                        "climatological std (std of the FOM record)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=_n_workers_default())
    p.add_argument("--keep-traj", action="store_true",
                   help="also save the (n_ens, n_steps+1, d_o) forecast paths")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(args.model, allow_pickle=True)
    dt = float(z["dt_strided"])
    stride = int(z["stride"])
    n_train_strided = int(z["n_train"])

    c = np.load(args.calib, allow_pickle=True)
    R = float(c["escape_thresh"])
    soft_alpha = float(c["soft_alpha"])
    soft_k = float(c["soft_k"])
    soft_p = float(c["soft_p"])

    with np.load(args.coords_path) as zc:
        A_full = np.asarray(zc["alpha"], np.float64)[::stride]
    coord_std = A_full[:n_train_strided].std(axis=0)

    n_steps = int(round(args.interval_s / dt))
    base_idx = n_train_strided if args.base_idx < 0 else args.base_idx
    rng = np.random.default_rng(args.seed)

    A0, meta_idx = init_ensemble(A_full, n_train_strided, args.n_ens, args.init,
                                 base_idx, args.spread, coord_std, rng)

    reset_steps = np.empty(0, np.int64)
    mode = "free"
    extra = {}
    if args.assimilate:
        mode = "enkf"
        with np.load(args.coords_path) as zc:
            V = np.asarray(zc["V"], np.float64)
        sx = args.sensor_stride_x or args.sensor_stride
        sz = args.sensor_stride_z or args.sensor_stride
        H, rows = build_obs_operator(V, sx, sz, args.components)
        cdd_std = obs_noise_std(V, A_full[:n_train_strided], rows,
                                args.sigma_obs)
        svals = np.linalg.svd(H, compute_uv=False)
        rank = int((svals > svals[0] * 1e-10).sum())
        d_o = V.shape[1]
        print(f"[enkf] sensors: stride x={sx} z={sz} comp={args.components} "
              f"N_obs={H.shape[0]}  rank(H)={rank}/{d_o}  "
              f"sigma_min={svals[min(rank, d_o) - 1]:.2e} cond={svals[0]/svals[rank-1]:.1e}",
              flush=True)
        if rank < d_o:
            print(f"[enkf] WARNING: H rank-deficient ({rank}<{d_o}); {d_o - rank} "
                  f"modes unobserved -- coarsen the stride / add components",
                  flush=True)
        k_eff = soft_k * args.corr_k_scale
        cfg = _worker_cfg(args.model, R, soft_alpha, k_eff, soft_p)
        window_steps = max(1, int(round(
            (args.cadence_tplus / TPLUS_PER_S) / dt)))
        print(f"[enkf] N_ens={args.n_ens} d_o={d_o} ASSIMILATE every "
              f"{args.cadence_tplus:.1f} t+ ({window_steps} steps) over "
              f"{args.interval_s:.2f}s infl={args.inflation} "
              f"corr_k_scale={args.corr_k_scale} workers={args.n_workers}",
              flush=True)
        truth = None
        if args.twin:
            _, _, ttraj = forecast_ensemble(
                A_full[base_idx][None, :], cfg, dt, n_steps, 1, keep_traj=True)
            truth = ttraj[0]
            print(f"[enkf] TWIN: truth = deterministic ROM rollout from "
                  f"snapshot {base_idx} ({n_steps} steps)", flush=True)
        traj, reset_steps, diags, ferr = cycled_enkf_forecast(
            A_full, base_idx, coord_std, args.spread, cfg, dt, n_steps,
            window_steps, args.n_ens, rng, args.n_workers, H, cdd_std,
            args.sigma_obs, args.inflation, args.additive_q, truth=truth)
        A_fore = traj[:, -1]
        reasons = np.full(args.n_ens, "complete", dtype="<U12")
        traj_save = traj
        extra = dict(
            sensor_rows=rows.astype(np.int64),
            n_obs=np.int64(H.shape[0]), rank_H=np.int64(rank),
            svals_H=svals, sensor_stride_x=np.int64(sx),
            sensor_stride_z=np.int64(sz), components=str(args.components),
            sigma_obs=np.float64(args.sigma_obs), cdd_std=cdd_std,
            inflation=np.float64(args.inflation),
            additive_q=np.float64(args.additive_q),
            corr_k_scale=np.float64(args.corr_k_scale),
            cadence_tplus=np.float64(args.cadence_tplus),
            diag_chi2=np.array([d["chi2"] for d in diags]),
            diag_tr_Caa=np.array([d["tr_Caa"] for d in diags]),
            diag_spread=np.array([d["spread"] for d in diags]),
            diag_err=np.array([d["err"] for d in diags]),
            diag_ferr=ferr,
            coord_std=coord_std,
            twin=np.bool_(args.twin),
        )
        if truth is not None:
            extra["truth_traj"] = truth
        c2 = extra["diag_chi2"]
        print(f"[enkf] cycles={len(diags)}  mean chi2/N_obs="
              f"{c2.mean() / H.shape[0]:.2f} (~1 if consistent)  "
              f"final err={diags[-1]['err']:.4f} spread={diags[-1]['spread']:.4f}",
              flush=True)
    elif args.reset_tplus > 0.0:
        mode = "reset"
        cfg = _worker_cfg(args.model, R, soft_alpha, soft_k, soft_p)
        window_steps = max(1, int(round(
            (args.reset_tplus / TPLUS_PER_S) / dt)))
        print(f"[enkf] N_ens={args.n_ens} d_o={A0.shape[1]} RESET-cycled "
              f"every {args.reset_tplus:.1f} t+ ({window_steps} steps) over "
              f"{args.interval_s:.2f}s ({n_steps} steps, dt={dt}) "
              f"R={R:.4f} workers={args.n_workers}", flush=True)
        traj, reset_steps = cycled_reset_forecast(
            A_full, base_idx, coord_std, args.spread, cfg, dt, n_steps,
            window_steps, args.n_ens, rng, args.n_workers)
        A_fore = traj[:, -1]
        reasons = np.full(args.n_ens, "complete", dtype="<U12")
        traj_save = traj
    else:
        cfg = _worker_cfg(args.model, R, soft_alpha, soft_k, soft_p)
        print(f"[enkf] N_ens={args.n_ens} d_o={A0.shape[1]} init={args.init} "
              f"interval={args.interval_s:.2f}s ({n_steps} steps, dt={dt}) "
              f"R={R:.4f} workers={args.n_workers}", flush=True)
        A_fore, reasons, traj_save = forecast_ensemble(
            A0, cfg, dt, n_steps, args.n_workers, args.keep_traj)

    n_bad = int((reasons != "complete").sum())
    print(f"[enkf] {mode} complete; {n_bad}/{args.n_ens} non-finite; "
          f"ensemble-mean |a| {np.linalg.norm(np.nanmean(A_fore, 0)):.3f}",
          flush=True)

    out = args.out_dir / "ensemble_forecast.npz"
    reset_tplus_out = (args.cadence_tplus if mode == "enkf"
                       else args.reset_tplus)
    save = dict(
        A_init=A0, A_forecast=A_fore, reasons=reasons, meta_idx=meta_idx,
        dt=np.float64(dt), n_steps=np.int64(n_steps),
        interval_s=np.float64(args.interval_s), n_ens=np.int64(args.n_ens),
        init=str(args.init), base_idx=np.int64(base_idx),
        spread=np.float64(args.spread), seed=np.int64(args.seed),
        reset_tplus=np.float64(reset_tplus_out), reset_steps=reset_steps,
        mode=str(mode),
        escape_thresh=np.float64(R), soft_alpha=np.float64(soft_alpha),
        soft_k=np.float64(soft_k), soft_p=np.float64(soft_p),
        stride=np.int64(stride), n_train=np.int64(n_train_strided),
        model_path=str(args.model), coords_path=str(args.coords_path),
        **extra,
    )
    if traj_save is not None:
        save["traj"] = traj_save
    np.savez(out, **save)
    print(f"[done] wrote {out}", flush=True)

    # --- ANALYSIS STEP: next -------------------------------------------------
    # Perturbed-observation EnKF update (Ozalp et al. Eqs. 14-17) goes here:
    #   d_j ~ N(d, C_dd),  C_dd = diag(sigma * u_max)
    #   A <- A + C_a_hat.d (C_hat_d_hat_d + C_dd)^-1 (D - H A)
    # with the sparse wall-shear operator H = M.V (M selects sensor grid points
    # from the POD reconstruction u = mu + V.a). Left unwired pending
    # rom-specialist review of the augmented-state / observation-operator setup.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
