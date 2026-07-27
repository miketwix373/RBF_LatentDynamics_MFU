"""Per-rank VPT worker: reconstruction-based valid-prediction-time over ICs.

Each SLURM task handles a round-robin slice of the shared IC list. For each IC it
runs a FREE-running (no-corrector) rollout of length T from the true state and
records two valid-prediction times:
  (a) vpt_a = first t where e_rec(t) = ||a_ROM-a_true||/sigma_A exceeds --e-thresh
  (b) vpt_b = first t where ACC(t) = cos angle(a_ROM,a_true) drops below --acc-thresh
NaN if the criterion never triggers within T (right-censored). Non-finite ROM
states (blow-up) count as an immediate crossing. Writes part_<rank>.npz;
aggregate with plot_mfu_na_vpt_distribution.py.

Usage (under srun):
    srun python scripts/run_mfu_na_vpt_worker.py --model M.npz \\
        --coords-path C.npz --n-ics 96 --T 30 --out-dir DIR
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_integrate_compare import _make_rhs, _rk4_integrate  # noqa: E402


def _load_model(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    if "Sigma_invs_std" in z.files:
        Sig = np.asarray(z["Sigma_invs_std"], dtype=np.float64)
    else:
        Sig = np.asarray(z["Sigma_invs_unique"])[np.asarray(z["parent_compact"])]
    return dict(
        centers_std=np.asarray(z["centers_std"], dtype=np.float64),
        Sigma_invs_std=np.asarray(Sig, dtype=np.float64),
        mu_A=np.asarray(z["mu_A"], dtype=np.float64),
        sigma_A=np.asarray(z["sigma_A"], dtype=np.float64),
        col_norms=np.asarray(z["col_norms_train"], dtype=np.float64),
        xi=np.asarray(z["xi"], dtype=np.float64),
        dt=float(z["dt_strided"]), stride=int(z["stride"]),
        n_train=int(z["n_train"]),
    )


def _first_cross(t, sig, thr, above):
    bad = sig > thr if above else sig < thr
    if not bad.any():
        return np.nan          # right-censored: never violated within T
    return float(t[int(np.argmax(bad))])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--coords-path", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-ics", type=int, default=96)
    p.add_argument("--T", type=float, default=30.0)
    p.add_argument("--e-thresh", type=float, default=0.25)
    p.add_argument("--acc-thresh", type=float, default=0.5)
    args = p.parse_args()

    rank = int(os.environ.get("SLURM_PROCID", "0"))
    ntasks = int(os.environ.get("SLURM_NTASKS", "1"))

    m = _load_model(args.model)
    with np.load(args.coords_path) as z:
        A = np.asarray(z["alpha"], dtype=np.float64)[::m["stride"]]
    sigma_A = float(np.sqrt(A.var(0).sum()))
    M = A.shape[0]
    nsteps = int(round(args.T / m["dt"]))
    # shared, deterministic IC list spread across the held-out test region
    ics = np.linspace(m["n_train"], M - nsteps - 1, args.n_ics).astype(np.int64)
    my = ics[rank::ntasks]

    rhs = _make_rhs(m["centers_std"], m["Sigma_invs_std"], m["mu_A"],
                    m["sigma_A"], m["col_norms"], m["xi"])   # NO corrector
    out = []
    for ic in my:
        A_true = A[ic:ic + nsteps + 1]
        X, _reason, _n, _f = _rk4_integrate(rhs, A_true[0], m["dt"], nsteps)
        n = min(X.shape[0], A_true.shape[0])
        X, At = X[:n], A_true[:n]
        t = np.arange(n) * m["dt"]
        e = np.linalg.norm(X - At, axis=1) / sigma_A
        e = np.where(np.isfinite(e), e, np.inf)
        nrm = np.linalg.norm(X, axis=1) * np.linalg.norm(At, axis=1)
        acc = np.where(nrm > 0, (X * At).sum(1) / nrm, 0.0)
        acc = np.where(np.isfinite(acc), acc, -np.inf)
        out.append((int(ic),
                    _first_cross(t, e, args.e_thresh, True),
                    _first_cross(t, acc, args.acc_thresh, False)))

    out = np.array(out, dtype=np.float64).reshape(-1, 3)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / f"part_{rank:03d}.npz",
             ic=out[:, 0], vpt_a=out[:, 1], vpt_b=out[:, 2],
             T=args.T, e_thresh=args.e_thresh, acc_thresh=args.acc_thresh,
             dt=m["dt"])
    print(f"[rank {rank}/{ntasks}] {my.size} ICs done", flush=True)


if __name__ == "__main__":
    main()
