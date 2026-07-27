"""Integrate the KS_CHAO K=1 soft_reflect ROM across the full test horizon and
record, per step, how hard the trust-region corrector had to pull.

Same trajectory as scripts/plot_ks_chao_hovmoller.py / paper_images/fig_spacetime
(deterministic soft_reflect rollout from the first validation snapshot, same
escape threshold and soft params) but run to the end of the test set rather
than the 50 s window.

The soft_reflect corrector (run_integrate_compare._build_soft_reflect_corrector)
is a smooth inward radial pull active only when the nearest-centre distance in
standardised coordinates exceeds alpha*R:

    intensity(a) = k * min(1, x)^p ,   x = (d_nn - alpha*R) / (R*(1 - alpha))

i.e. the std-coord magnitude of the corrective velocity (u_out is a unit
vector). intensity = 0 means the corrector is dormant; "fired" == intensity > 0.
We evaluate it at each saved trajectory state and cache (t, d_nn, intensity)
for the plotting script.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_integrate_compare import (  # noqa: E402
    _make_rhs,
    _build_soft_reflect_corrector,
    _build_passive_event_counter,
    _rk4_integrate,
)
from chord2.sindy import rbf_features_mahal  # noqa: E402

SHARED_ROOT = Path("/users/sbrw610/sharedscratch/RBF_ROM")
MODEL_DIR = SHARED_ROOT / "results/KS_CHAO/integration_K1_n6000_softreflect"
DEFAULT_OUT = MODEL_DIR / "corrector_series.npz"


def _load_model(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    # Sigma_invs are stored either FULL (one (r, r) per centre, K=1 fits) or
    # COMPACT by the K-shaped sweeps (Sigma_invs_unique (K, r, r) plus
    # parent_compact indexing it). Expand the compact form here so the corrector
    # calibrates against the exact deployed representation (mirrors
    # run_integrate_rbf_model._load).
    if "Sigma_invs_std" in z.files:
        Sigma_invs_std = np.asarray(z["Sigma_invs_std"], dtype=np.float64)
    else:
        Sig_u = np.asarray(z["Sigma_invs_unique"], dtype=np.float64)
        parent_compact = np.asarray(z["parent_compact"], dtype=np.int64)
        Sigma_invs_std = Sig_u[parent_compact]
    return {
        "centers_std":    np.asarray(z["centers_std"], dtype=np.float64),
        "Sigma_invs_std": Sigma_invs_std,
        "mu_A":           np.asarray(z["mu_A"], dtype=np.float64),
        "sigma_A":        np.asarray(z["sigma_A"], dtype=np.float64),
        "col_norms":      np.asarray(z["col_norms_train"], dtype=np.float64),
        "xi":             np.asarray(z["xi"], dtype=np.float64),
        "dt":             float(z["dt_strided"]),
        "stride":         int(z["stride"]),
        "n_train":        int(z["n_train"]),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", type=Path,
                   default=MODEL_DIR / "fit_K1_n6000/model.npz")
    p.add_argument("--coords-path", type=Path,
                   default=SHARED_ROOT / "results/KS_CHAO/coords.npz")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--T-window", type=float, default=None,
                   help="rollout length [s]; default = full test horizon")
    p.add_argument("--soft-alpha", type=float, default=0.85)
    p.add_argument("--soft-k", type=float, default=10.0)
    p.add_argument("--soft-p", type=float, default=2.0)
    p.add_argument("--cloud-escape-quantile", type=float, default=0.99)
    p.add_argument("--escape-thresh", type=float, default=None,
                   help="use this trust radius R directly instead of "
                        "recomputing from the cloud quantile; set it to the "
                        "deployed model's guard.escape_threshold_std so the "
                        "rollout matches the integration behind the paper "
                        "figures (needed when q=1.0, whose max is draw-sensitive)")
    p.add_argument("--n-cloud", type=int, default=10000)
    p.add_argument("--cloud-rng-seed", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=500,
                   help="print rollout depth (step / n_steps, elapsed, rate, "
                        "ETA) every this many RK4 steps; 0 disables")
    args = p.parse_args()

    m = _load_model(args.model)
    with np.load(args.coords_path) as zc:
        alpha_coeff = np.asarray(zc["alpha"], dtype=np.float64)
        alpha_dot_coeff = np.asarray(zc["alpha_dot"], dtype=np.float64)

    A_full = alpha_coeff[::m["stride"]]
    n_train = m["n_train"]

    # Fixed training-climate reference: RMS of the a-priori standardised rate
    # ||alpha_dot / sigma_A|| over the TRAINING snapshots. Exogenous to the
    # rollout (IC-independent, uncontaminated by the corrector we are
    # measuring), and in std coords -- the space the trust region lives in.
    Y_train_std = (alpha_dot_coeff[::m["stride"]][:n_train]) / m["sigma_A"]
    rms_adot_std_train = float(np.sqrt(np.mean((Y_train_std ** 2).sum(axis=1))))
    max_steps = A_full.shape[0] - 1 - n_train
    if args.T_window is None:
        n_steps = max_steps
    else:
        n_steps = min(int(round(args.T_window / m["dt"])), max_steps)
    X_true = A_full[n_train: n_train + n_steps + 1]
    print(f"[setup] dt={m['dt']}s, n_train={n_train}, n_steps={n_steps} "
          f"({n_steps * m['dt']:.0f}s)", flush=True)

    tree = cKDTree(m["centers_std"])
    if args.escape_thresh is not None:
        R = float(args.escape_thresh)
        print(f"[guard] soft_reflect escape_thresh (override)={R:.4f}",
              flush=True)
    else:
        rng = np.random.default_rng(args.cloud_rng_seed)
        n_cv = min(args.n_cloud, A_full.shape[0])
        sel = rng.choice(A_full.shape[0], size=n_cv, replace=False)
        A_std_cloud = (A_full[sel] - m["mu_A"]) / m["sigma_A"]
        d_nn_cloud, _ = tree.query(A_std_cloud, k=1)
        R = float(np.quantile(d_nn_cloud, args.cloud_escape_quantile))
        print(f"[guard] soft_reflect escape_thresh"
              f"(q={args.cloud_escape_quantile})={R:.3f}", flush=True)

    corr = _build_soft_reflect_corrector(
        tree, m["mu_A"], m["sigma_A"], R,
        alpha=args.soft_alpha, k_strength=args.soft_k, p=args.soft_p,
    )
    rhs = _make_rhs(m["centers_std"], m["Sigma_invs_std"], m["mu_A"],
                    m["sigma_A"], m["col_norms"], m["xi"], corrector=corr)
    guard = _build_passive_event_counter(tree, m["mu_A"], m["sigma_A"], R)

    # Integrate in chunks so long rollouts report their depth. Chunking is
    # numerically identical to one _rk4_integrate call: the loop is
    # deterministic and each chunk restarts from the exact last committed
    # state (the returned trajectory's first row duplicates that state, so we
    # drop it when stitching).
    print(f"[integrate] {n_steps} RK4 steps from validation IC ...", flush=True)
    t_start = time.perf_counter()
    chunk = args.progress_every if args.progress_every > 0 else n_steps
    pieces = [X_true[0][None, :]]
    x0 = X_true[0]
    n_evt = 0
    reason = "complete"
    done = 0
    while done < n_steps:
        this = min(chunk, n_steps - done)
        Xc, reason, n_evt_c, _first = _rk4_integrate(
            rhs, x0, m["dt"], this, guard=guard,
        )
        pieces.append(Xc[1:])
        n_evt += n_evt_c
        done += Xc.shape[0] - 1
        x0 = Xc[-1]
        if args.progress_every > 0:
            el = time.perf_counter() - t_start
            rate = done / el if el > 0 else 0.0
            eta = (n_steps - done) / rate if rate > 0 else float("nan")
            print(f"[progress] step {done}/{n_steps} "
                  f"({100 * done / n_steps:5.1f}%)  t={done * m['dt']:.0f}s  "
                  f"elapsed={el:.0f}s  rate={rate:.1f} steps/s  "
                  f"eta={eta:.0f}s  hard_escapes={n_evt}", flush=True)
        if reason != "complete":
            print(f"[integrate] stopped early: reason={reason}", flush=True)
            break
    X_rom = np.concatenate(pieces, axis=0)
    print(f"[integrate] returned {X_rom.shape[0]} steps, reason={reason}, "
          f"hard_escape_events={n_evt}", flush=True)

    # Per-step corrector strength along the saved trajectory, measured as a
    # PERCENTAGE of the model's own rate:  100 * ||C(a)|| / ||alpha_dot(a)||.
    #   - alpha_dot = SINDy rate  (Phi/col_norms) @ xi        [raw-coord units]
    #   - C(a) = -k*min(1,x)^p * u_out (*) sigma_A,  x=(d_nn-alpha R)/(R(1-alpha))
    # Both vectors live in raw coords, so the ratio is how hard the corrector
    # pulls relative to the SINDy velocity it is correcting.
    A_std = (X_rom - m["mu_A"]) / m["sigma_A"]
    d_nn, idx = tree.query(A_std, k=1)
    x_ramp = (d_nn - args.soft_alpha * R) / (R * (1.0 - args.soft_alpha))
    ramp = np.clip(x_ramp, 0.0, 1.0) ** args.soft_p

    u_std = (A_std - m["centers_std"][idx]) / np.maximum(d_nn, 1e-300)[:, None]
    C = (-args.soft_k * ramp)[:, None] * u_std * m["sigma_A"][None, :]  # raw
    C_norm = np.linalg.norm(C, axis=1)

    Phi = rbf_features_mahal(X_rom, m["centers_std"], m["Sigma_invs_std"],
                             mu_A=m["mu_A"], sigma_A=m["sigma_A"])
    alpha_dot = (Phi / m["col_norms"][None, :]) @ m["xi"]              # raw
    adot_norm = np.linalg.norm(alpha_dot, axis=1)

    # Std-coord corrector magnitude = ||C / sigma_A|| = k*ramp (u_std is unit),
    # bounded in [0, k]. This is the corrector's native strength in the space
    # where the trust region is defined (rom-specialist rec, ranked best).
    C_norm_std = args.soft_k * ramp

    # Instantaneous raw ratio kept for reference; the reported metric is the
    # std-coord magnitude over the fixed training-climate RMS.
    pct_inst = 100.0 * C_norm / np.maximum(adot_norm, 1e-300)
    pct = 100.0 * C_norm_std / max(rms_adot_std_train, 1e-300)
    t = np.arange(X_rom.shape[0]) * m["dt"]

    fired = C_norm_std > 0.0
    hard = d_nn > R
    print(f"[fire] soft fires {int(fired.sum())}/{fired.size} "
          f"({100 * fired.mean():.2f}%), hard escapes {int(hard.sum())} "
          f"({100 * hard.mean():.2f}%)", flush=True)
    print(f"[metric] rms_adot_std_train={rms_adot_std_train:.4g}  "
          f"pct(std/climate) median-when-fired="
          f"{np.median(pct[fired]) if fired.any() else 0:.2f}%  "
          f"peak={pct.max():.2f}%", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        t=t, d_nn=d_nn, pct=pct, pct_inst=pct_inst,
        C_norm=C_norm, C_norm_std=C_norm_std, adot_norm=adot_norm,
        rms_adot_std_train=np.float64(rms_adot_std_train),
        fired=fired, hard=hard,
        escape_thresh=np.float64(R),
        soft_alpha=np.float64(args.soft_alpha),
        soft_k=np.float64(args.soft_k),
        soft_p=np.float64(args.soft_p),
        dt=np.float64(m["dt"]),
        n_fired=np.int64(fired.sum()), n_steps=np.int64(fired.size),
        n_hard=np.int64(hard.sum()), peak_pct=np.float64(pct.max()),
        reason=str(reason),
    )
    print(f"[done] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
