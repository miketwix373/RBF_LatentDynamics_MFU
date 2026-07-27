"""Package the MFU_OPO WSS arrays into the CHORD2 stats.npz schema.

Maps the wall-shear pair onto the vector2d_uv layout consumed by
chord2/baselines/svd_basis.py: u = wss_x, v = wss_y, each (T, 64, 64).

Unlike MFU_NA, the OPO record has a single uniform cadence
(Delta-step = 10, dt_sim = 0.005 => dt_snap = 0.05) throughout, so there
is no cadence break to reserve a holdout segment. We take an 80/20 time
split: the first 80% goes to stats.npz for training, the last 20% to
stats_holdout.npz for invariant-measure validation. Both share the same
dtStats = 0.05, matching MFU_NA segment A so the 5-point derivative in
precompute_coords.py carries over unchanged.

Written with np.savez (uncompressed) so the svd_basis memmap opener
can stream u.npy / v.npy inside the archive.

Geometry (dx, L) is unknown pending metadata from the user; those keys
are stored as NaN.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DT_SIM = 0.005  # solver time step (user, 2026-07-10)
DSTEP = 10      # uniform snapshot cadence (steps between saved frames)
SPLIT = 0.8     # fraction of the record used for training (stats.npz)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path,
                    default=(next(_p for _p in Path(__file__).resolve().parents if (_p / "pyproject.toml").exists()) / "data/MFU_OPO"))
    args = ap.parse_args()

    steps = np.load(args.data_dir / "wss_steps.npy")
    wx = np.load(args.data_dir / "wss_x_all.npy", mmap_mode="r")
    wy = np.load(args.data_dir / "wss_y_all.npy", mmap_mode="r")

    d = np.diff(steps)
    assert np.all(d == DSTEP), "expected a single uniform cadence"

    n = len(steps)
    brk = int(round(SPLIT * n))

    common = dict(
        nx=np.int64(wx.shape[1]), ny=np.int64(wx.shape[2]),
        dx=np.float64(np.nan), L=np.float64(np.nan),
        dt_sim=np.float64(DT_SIM), tInit=np.float64(steps[0] * DT_SIM),
        bc_type="periodic",
    )
    dt_stats = np.float64(DSTEP * DT_SIM)

    for name, sl in [("stats.npz", slice(0, brk)),
                     ("stats_holdout.npz", slice(brk, None))]:
        s = steps[sl]
        out = args.data_dir / name
        np.savez(out,
                 u=np.asarray(wx[sl]), v=np.asarray(wy[sl]),
                 t=s * DT_SIM, steps=s,
                 dtStats=dt_stats, **common)
        print(f"{out}: {len(s)} snapshots, t=[{s[0]*DT_SIM}, {s[-1]*DT_SIM}], "
              f"dtStats={dt_stats}")


if __name__ == "__main__":
    main()
