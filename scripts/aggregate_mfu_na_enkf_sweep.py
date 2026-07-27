"""Aggregate the alpha8 EnKF parametric sweep (sweep_a8) into one summary.

Walks results/MFU_NA/EnKF/sweep_a8/<group>/<tag>/ensemble_forecast.npz, pulls
the innovation-consistency and error diagnostics of every run, interpolates
the chi2/N_obs = 1 crossing (q*) within each mini q-grid family, and writes
summary.json next to the groups. The q* construction follows
analyze_mfu_na_enkf_qstar.py: chi2/N decreases monotonically with additive q,
so the crossing is a linear interpolation between the bracketing grid points.

Metrics per run:
  chi2_N      -- mean over cycles of chi2 / N_obs (1 if consistent)
  err_mean    -- mean over cycles of |analysis mean - truth|
  err_norm    -- err_mean / |coord_std| (fraction of climatological amplitude)
  spread_mean -- mean over cycles of sqrt(tr Caa)
  spread_skill-- spread_mean / err_mean (~1 for a calibrated ensemble)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SHARED_ROOT = (next(_p for _p in Path(__file__).resolve().parents if (_p / "pyproject.toml").exists()))
SWEEP = SHARED_ROOT / "results/MFU_NA/EnKF/sweep_a8"


def run_metrics(npz: Path) -> dict:
    z = np.load(npz, allow_pickle=True)
    n_obs = int(z["n_obs"])
    chi2 = np.asarray(z["diag_chi2"], np.float64)
    err = np.asarray(z["diag_err"], np.float64)
    spread = np.asarray(z["diag_spread"], np.float64)
    coord_std = np.asarray(z["coord_std"], np.float64)
    return dict(
        additive_q=float(z["additive_q"]), inflation=float(z["inflation"]),
        cadence_tplus=float(z["cadence_tplus"]), n_ens=int(z["n_ens"]),
        sigma_obs=float(z["sigma_obs"]),
        stride_x=int(z["sensor_stride_x"]), stride_z=int(z["sensor_stride_z"]),
        components=str(z["components"]), n_obs=n_obs, rank_H=int(z["rank_H"]),
        twin=bool(z["twin"]) if "twin" in z.files else False,
        n_cycles=int(chi2.size),
        chi2_N=float(chi2.mean() / n_obs),
        err_mean=float(err.mean()),
        err_norm=float(err.mean() / np.linalg.norm(coord_std)),
        spread_mean=float(spread.mean()),
        spread_skill=float(spread.mean() / err.mean()),
    )


def qstar_from_family(rows: list[dict]) -> dict:
    """chi2/N = 1 crossing over the additive-q grid of one family (one value
    of the swept nuisance parameter). Returns the crossing plus its bracket;
    qstar is None when the whole grid sits on one side of 1."""
    rows = sorted(rows, key=lambda r: r["additive_q"])
    q = np.array([r["additive_q"] for r in rows])
    chi = np.array([r["chi2_N"] for r in rows])
    out = dict(q_grid=q.tolist(), chi2_N=chi.tolist(), qstar=None)
    below = np.where(chi <= 1.0)[0]
    if below.size and below[0] > 0:
        j = below[0]
        out["qstar"] = float(q[j - 1] + (q[j] - q[j - 1]) *
                             (chi[j - 1] - 1.0) / (chi[j - 1] - chi[j]))
    elif below.size and below[0] == 0:
        out["qstar"] = 0.0          # consistent already at the smallest q
    return out


# family key = the nuisance-parameter value a mini q-grid shares
FAMILY_KEY = {
    "qstar": lambda r: "base",
    "cadence": lambda r: r["cadence_tplus"],
    "cadence_twin": lambda r: r["cadence_tplus"],
    "sigma": lambda r: r["sigma_obs"],
    "nens": lambda r: r["n_ens"],
    "stride_q": lambda r: r["stride_x"],
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sweep-root", type=Path, default=SWEEP)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    out_path = args.out or args.sweep_root / "summary.json"

    runs = defaultdict(dict)
    for npz in sorted(args.sweep_root.glob("*/*/ensemble_forecast.npz")):
        group, tag = npz.parent.parent.name, npz.parent.name
        runs[group][tag] = run_metrics(npz)
        r = runs[group][tag]
        print(f"{group:13s} {tag:14s} chi2/N={r['chi2_N']:6.3f} "
              f"err={r['err_mean']:.4f} sk={r['spread_skill']:.2f} "
              f"rank={r['rank_H']}/{45} cyc={r['n_cycles']}")

    qstar = {}
    for group, key_fn in FAMILY_KEY.items():
        if group not in runs:
            continue
        fams = defaultdict(list)
        for r in runs[group].values():
            fams[key_fn(r)].append(r)
        qstar[group] = {str(k): qstar_from_family(v)
                        for k, v in sorted(fams.items())}
        for k, v in qstar[group].items():
            print(f"q*[{group}][{k}] = {v['qstar']}")

    out_path.write_text(json.dumps(
        {"runs": {g: dict(t) for g, t in runs.items()}, "qstar": qstar},
        indent=1))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
