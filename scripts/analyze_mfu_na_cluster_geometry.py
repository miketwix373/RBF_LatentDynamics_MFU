"""Does the local-flow geometry of each cluster (spiral / node / saddle,
source / sink) correlate with which quadrant of the raw (A_streak, sigma_y)
plane the cluster centroid lives in?

For each K=16 cluster: linearise the global RBF field at the centroid (J_k),
classify the local dynamics two ways --
  (1) leading full-spectrum eigenvalue lam1 = arg max Re: spiral if Im!=0 else
      node; source if Re>0 else sink; n_unstable = #(Re>0).
  (2) the 2x2 flow actually drawn in the small-multiple portraits: J_k projected
      onto its leading eigen-plane, classified by trace/determinant
      (saddle / spiral / node, source / sink).
-- and tabulate against the centroid quadrant (x=A_streak, y=sigma_y):
   Q1=(+,+) mature, Q2=(-,+) breakdown, Q3=(-,-) quiescent, Q4=(+,-) post-lift-up.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_mfu_na_phase_circulation import (  # noqa: E402
    _signals, _streak_amp, SHARED_ROOT)
from compute_mfu_na_cluster_jacobians import (  # noqa: E402
    _load_model, _phi_and_grad, TPLUS_PER_S as JT)

QLAB = {1: "Q1 mature", 2: "Q2 breakdown", 3: "Q3 quiescent", 4: "Q4 post-liftup"}


def _quadrant(x, y):
    if x >= 0 and y >= 0:
        return 1
    if x < 0 and y >= 0:
        return 2
    if x < 0 and y < 0:
        return 3
    return 4


def _classify_2d(J2):
    tr, det = np.trace(J2), np.linalg.det(J2)
    if det < 0:
        return "saddle"
    kind = "spiral" if tr * tr - 4 * det < 0 else "node"
    return f"{kind}-{'source' if tr > 0 else 'sink'}"


def _leading_plane(J):
    w, Vv = np.linalg.eig(J)
    order = np.argsort(-w.real)
    top = order[0]
    if abs(w[top].imag) > 1e-9:
        br, bi = Vv[:, top].real.copy(), Vv[:, top].imag.copy()
    else:
        br, bi = Vv[:, order[0]].real.copy(), Vv[:, order[1]].real.copy()
    br /= np.linalg.norm(br)
    bi -= (bi @ br) * br
    bi /= np.linalg.norm(bi)
    B = np.column_stack([br, bi])
    return B.T @ J @ B


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--coords", type=Path,
                   default=SHARED_ROOT / "results/MFU_NA/pod/coords_do45.npz")
    p.add_argument("--labels", type=Path, default=SHARED_ROOT /
                   "results/MFU_NA/cluster_sweep/do45/K=16.npz")
    p.add_argument("--model", type=Path, default=SHARED_ROOT /
                   "results/MFU_NA/do45/alpha3.5/model_n04800.npz")
    args = p.parse_args()

    with np.load(args.coords) as z:
        alpha = np.asarray(z["alpha"], np.float64)
        V = np.asarray(z["V"], np.float64)
    _, sy_sig = _signals(alpha, V)
    streak_sig = _streak_amp(alpha, V)
    xr = (streak_sig - streak_sig.mean()) / streak_sig.std()
    yr = (sy_sig - sy_sig.mean()) / sy_sig.std()

    clab = np.asarray(np.load(args.labels)["labels"], np.int64)
    step = len(alpha) // len(clab)
    Kc = int(clab.max()) + 1
    aL = xr[::step][:len(clab)]; sL = yr[::step][:len(clab)]
    cx = np.array([aL[clab == k].mean() for k in range(Kc)])
    cy = np.array([sL[clab == k].mean() for k in range(Kc)])
    occ = np.array([(clab == k).mean() for k in range(Kc)])
    cent = np.array([alpha[::step][:len(clab)][clab == k].mean(0) for k in range(Kc)])

    mdl = _load_model(args.model)
    print(f"{'k':>2} {'quad':>14} {'cx':>6} {'cy':>6} "
          f"{'Reλ1':>8} {'Imλ1':>8} {'lead-type':>12} {'#unstab':>7} {'portrait':>14}")
    rows = []
    for k in range(Kc):
        _, Jk = _phi_and_grad(cent[k], mdl)
        lam = np.linalg.eigvals(Jk)
        i1 = np.argmax(lam.real)
        l1 = lam[i1]
        lead = ("spiral" if abs(l1.imag) > 1e-9 else "node") + \
               ("-source" if l1.real > 0 else "-sink")
        n_unstab = int((lam.real > 0).sum())
        port = _classify_2d(_leading_plane(Jk))
        q = _quadrant(cx[k], cy[k])
        rows.append((k, q, lead, port, l1.real / JT, abs(l1.imag) / JT, n_unstab))
        print(f"{k:>2} {QLAB[q]:>14} {cx[k]:>6.2f} {cy[k]:>6.2f} "
              f"{l1.real / JT:>8.4f} {abs(l1.imag) / JT:>8.4f} {lead:>12} "
              f"{n_unstab:>7} {port:>14}")

    print("\n--- by quadrant ---")
    for q in (1, 2, 3, 4):
        grp = [r for r in rows if r[1] == q]
        if not grp:
            print(f"{QLAB[q]:>14}: (empty)")
            continue
        ks = [r[0] for r in grp]
        ports = [r[3] for r in grp]
        nun = np.mean([r[6] for r in grp])
        absc = np.mean([r[4] for r in grp])
        rot = np.mean([r[5] for r in grp])
        spir = sum("spiral" in r[3] for r in grp)
        print(f"{QLAB[q]:>14}: clusters {ks}")
        print(f"{'':>14}  portraits: {ports}")
        print(f"{'':>14}  spiral {spir}/{len(grp)}, mean #unstable {nun:.1f}, "
              f"mean Reλ1 {absc:+.4f}/t+, mean |Imλ1| {rot:.4f}/t+")


if __name__ == "__main__":
    main()
