"""Phase-resolved phase-volume contraction of the MFU_NA RBF field.

tr J(a) (1/s) -- the local phase-volume contraction rate of the fitted
45-dimensional wall-shear dynamics -- evaluated with the analytic Jacobian at
~3000 states along the strided record; per-K=16-cluster means and an
(A_streak, sigma_y) map. Where does the wall dynamics contract hardest
(dissipative quiescent decay?) and least (bursts?).

Outputs under results/MFU_NA/field_analysis/interpretables/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from find_mfu_na_upo import RBFField, MFU, TPLUS_PER_S  # noqa: E402
from plot_mfu_na_phase_circulation import _signals, _streak_amp  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--coords", type=Path, default=MFU / "pod/coords_do45.npz")
    p.add_argument("--model", type=Path,
                   default=MFU / "do45/alpha8.0/model_n00960.npz")
    p.add_argument("--labels", type=Path,
                   default=MFU / "cluster_sweep/do45/K=16.npz")
    p.add_argument("--n-states", type=int, default=3000)
    p.add_argument("--out-dir", type=Path,
                   default=MFU / "field_analysis/interpretables")
    args = p.parse_args()

    with np.load(args.coords) as z:
        alpha = np.asarray(z["alpha"], np.float64)
        V = np.asarray(z["V"], np.float64)
    A = alpha[::2]
    field = RBFField(args.model)
    cent = np.asarray(np.load(args.labels)["centroids"], np.float64)

    idx = np.linspace(0, len(A) - 1, args.n_states).astype(int)
    X = A[idx]
    lab = np.argmin(((X[:, None, :] - cent[None]) ** 2).sum(-1), axis=1)
    tr = np.empty(args.n_states)
    for b0 in range(0, args.n_states, 500):
        sl = slice(b0, min(b0 + 500, args.n_states))
        _, J = field.f_and_J(X[sl])
        tr[sl] = np.trace(J, axis1=1, axis2=2)

    rows = []
    for k in range(16):
        m = lab == k
        rows.append(dict(k=k, n=int(m.sum()),
                         trJ_per_s=float(tr[m].mean()),
                         trJ_per_tplus=float(tr[m].mean() / TPLUS_PER_S)))
        print(f"k={k:2d} n={m.sum():4d}  trJ = {rows[-1]['trJ_per_s']:+8.3f} /s"
              f"  = {rows[-1]['trJ_per_tplus']:+7.4f} /t+")
    print(f"overall: {tr.mean():+.3f} /s  ({tr.mean() / TPLUS_PER_S:+.4f} /t+)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / "trace_contraction.npz", idx=idx, labels=lab,
             trJ=tr, model_path=str(args.model))
    (args.out_dir / "trace_contraction.json").write_text(
        json.dumps(dict(overall_per_s=float(tr.mean()), rows=rows), indent=2))

    import shutil
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    use_tex = shutil.which("latex") and shutil.which("dvipng")
    plt.rcParams.update({"text.usetex": bool(use_tex), "font.family": "serif",
                         "mathtext.fontset": "cm"})
    _, sy_d = _signals(alpha, V)
    ax_d = _streak_amp(alpha, V)
    xs = (ax_d[::2][idx] - ax_d.mean()) / ax_d.std()
    ys = (sy_d[::2][idx] - sy_d.mean()) / sy_d.std()
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    hb = ax.hexbin(xs, ys, C=tr, gridsize=28, cmap="RdBu_r",
                   reduce_C_function=np.mean, mincnt=3)
    plt.colorbar(hb, ax=ax, label=r"$\mathrm{tr}\,J$ (1/s)")
    ax.set_xlabel(r"$A_x$ (std.)")
    ax.set_ylabel(r"$\sigma_y$ (std.)")
    ax.set_title(r"phase-volume contraction $\mathrm{tr}\,J$")
    fig.tight_layout()
    fig.savefig(args.out_dir / "trace_contraction.png", dpi=170)
    print(f"[done] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
