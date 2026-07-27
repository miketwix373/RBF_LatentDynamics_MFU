"""Forecast suite for one calibrated MFU_NA (d_o, alpha, N*) model + corrector.

Two products, both free-running (closed-loop) RK4 rollouts of adot = Phi(a).xi
plus the additive trust-region corrector at the model's stored trust radius R:

  1. Short-horizon ensemble: 100 ICs evenly spaced across the FULL coefficient
     record (train+test), each rolled out `horizon_s` seconds. Saved with the
     matching ground-truth windows so NRMSE(t)/VPT can be scored later.
  2. Long test: one rollout from the first testing-set snapshot to the last,
     for an invariant-measure / long-time-stability check. Saved with truth.

Reduced-coordinate trajectories only (POD-coeff space); reconstruct fields from
coords_do*.npz when needed. Writes forecast_ics.npz and long_test.npz next to
the model.
"""

from __future__ import annotations

import argparse
import sys
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


def _rollout(rhs, x0, dt, n_steps):
    """Full-length RK4 rollout (no stopping guard); pad with NaN if it goes
    non-finite so every ensemble member has the same shape."""
    X, reason, _, _ = _rk4_integrate(rhs, x0, dt, n_steps, guard=None)
    out = np.full((n_steps + 1, x0.shape[0]), np.nan, np.float64)
    out[: X.shape[0]] = X
    return out, reason


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--coords-path", type=Path, required=True)
    p.add_argument("--escape-thresh", type=float, required=True,
                   help="stored trust radius R (std coords)")
    p.add_argument("--soft-alpha", type=float, default=0.85)
    p.add_argument("--soft-k", type=float, default=10.0)
    p.add_argument("--soft-p", type=float, default=2.0)
    p.add_argument("--n-ics", type=int, default=100)
    p.add_argument("--horizon-s", type=float, default=30.0)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="default: the model's own directory")
    args = p.parse_args()

    out_dir = args.out_dir or args.model.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    m = _load_model(args.model)
    with np.load(args.coords_path) as zc:
        A_full = np.asarray(zc["alpha"], np.float64)[::m["stride"]]
    dt, n_train, d_o = m["dt"], m["n_train"], A_full.shape[1]
    n_tot = A_full.shape[0]

    tree = cKDTree(m["centers_std"])
    R = float(args.escape_thresh)
    corr = _build_soft_reflect_corrector(
        tree, m["mu_A"], m["sigma_A"], R,
        alpha=args.soft_alpha, k_strength=args.soft_k, p=args.soft_p)
    rhs = _make_rhs(m["centers_std"], m["Sigma_invs_std"], m["mu_A"],
                    m["sigma_A"], m["col_norms"], m["xi"], corrector=corr)

    # --- short-horizon ensemble -------------------------------------------
    H = int(round(args.horizon_s / dt))
    ic_idx = np.linspace(0, n_tot - 1 - H, args.n_ics).round().astype(np.int64)
    rom = np.full((args.n_ics, H + 1, d_o), np.nan, np.float64)
    truth = np.full((args.n_ics, H + 1, d_o), np.nan, np.float64)
    reasons = np.empty(args.n_ics, dtype="<U12")
    print(f"[short] {args.n_ics} ICs x {H} steps ({args.horizon_s:.0f}s), "
          f"d_o={d_o}, R={R:.4f}", flush=True)
    for j, i0 in enumerate(ic_idx):
        traj, reason = _rollout(rhs, A_full[i0], dt, H)
        rom[j] = traj
        truth[j] = A_full[i0: i0 + H + 1]
        reasons[j] = reason
        if (j + 1) % 20 == 0:
            print(f"[short] {j + 1}/{args.n_ics} done", flush=True)
    n_bad = int((reasons != "complete").sum())
    print(f"[short] complete; {n_bad}/{args.n_ics} went non-finite", flush=True)

    np.savez(
        out_dir / "forecast_ics.npz",
        ic_indices=ic_idx, rom=rom, truth=truth, reasons=reasons,
        dt=np.float64(dt), horizon_s=np.float64(args.horizon_s),
        H=np.int64(H), n_ics=np.int64(args.n_ics),
        escape_thresh=np.float64(R), soft_alpha=np.float64(args.soft_alpha),
        soft_k=np.float64(args.soft_k), soft_p=np.float64(args.soft_p),
        d_o=np.int64(d_o), stride=np.int64(m["stride"]),
        n_train=np.int64(n_train),
        model_path=str(args.model), coords_path=str(args.coords_path),
    )

    # --- long test: first -> last testing snapshot ------------------------
    n_long = n_tot - 1 - n_train
    print(f"[long] {n_long} steps ({n_long * dt:.0f}s) from first test "
          f"snapshot", flush=True)
    rom_long, reason_long = _rollout(rhs, A_full[n_train], dt, n_long)
    truth_long = A_full[n_train:]
    print(f"[long] reason={reason_long}", flush=True)

    np.savez(
        out_dir / "long_test.npz",
        ic_index=np.int64(n_train), rom=rom_long, truth=truth_long,
        n_steps=np.int64(n_long), reason=str(reason_long),
        dt=np.float64(dt), escape_thresh=np.float64(R),
        soft_alpha=np.float64(args.soft_alpha), soft_k=np.float64(args.soft_k),
        soft_p=np.float64(args.soft_p), d_o=np.int64(d_o),
        stride=np.int64(m["stride"]), n_train=np.int64(n_train),
        model_path=str(args.model), coords_path=str(args.coords_path),
    )
    print(f"[done] wrote {out_dir/'forecast_ics.npz'} and "
          f"{out_dir/'long_test.npz'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
