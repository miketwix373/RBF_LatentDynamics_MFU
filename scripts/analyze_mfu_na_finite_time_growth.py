"""Finite-time transient growth of the MFU_NA wall-shear RBF field.

At N states sampled along the attractor, integrate the variational flow
Phi' = J(a) Phi over horizons tau+ = 5 and 10 and record the maximum
finite-time amplification G_tau = sigma_1(Phi) with its optimal input
direction (leading right singular vector). Companion controls: the numerical
abscissa max eig((J+J^T)/2) (instantaneous non-normal growth), the local
speed ||f||, and the alignment cos(v1, f) -- to establish whether transient
growth is an independent signal or speed in disguise. Eigenvalues at cluster
centroids carry no such signal (corr -0.07 with forecast error); the
finite-time norm is the correct measure for streak-breakdown physics
(Schoppa & Hussain, J. Fluid Mech. 453, 2002).

Outputs under results/MFU_NA/field_analysis/ftg/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from find_mfu_na_upo import RBFField, flow, MFU, TPLUS_PER_S  # noqa: E402
from plot_mfu_na_phase_circulation import _signals, _streak_amp  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--coords", type=Path, default=MFU / "pod/coords_do45.npz")
    p.add_argument("--model", type=Path,
                   default=MFU / "do45/alpha8.0/model_n00960.npz")
    p.add_argument("--labels", type=Path,
                   default=MFU / "cluster_sweep/do45/K=16.npz")
    p.add_argument("--n-states", type=int, default=3000)
    p.add_argument("--batch", type=int, default=500)
    p.add_argument("--dt", type=float, default=0.025)
    p.add_argument("--out-dir", type=Path, default=MFU / "field_analysis/ftg")
    args = p.parse_args()

    with np.load(args.coords) as z:
        alpha = np.asarray(z["alpha"], np.float64)
        V = np.asarray(z["V"], np.float64)
    A = alpha[::2]
    field = RBFField(args.model)
    cent = np.asarray(np.load(args.labels)["centroids"], np.float64)
    T5 = 5.0 / TPLUS_PER_S                      # 0.443 s

    idx = np.linspace(0, len(A) - 1, args.n_states).astype(int)
    X = A[idx]
    lab = np.argmin(((X[:, None, :] - cent[None]) ** 2).sum(-1), axis=1)

    G5 = np.empty(args.n_states)
    G10 = np.empty(args.n_states)
    V1 = np.empty((args.n_states, 45))
    absc = np.empty(args.n_states)
    spd = np.empty(args.n_states)
    cosvf = np.empty(args.n_states)
    cov_end = np.empty(args.n_states)
    for b0 in range(0, args.n_states, args.batch):
        sl = slice(b0, min(b0 + args.batch, args.n_states))
        Xb = X[sl]
        f, J = field.f_and_J(Xb)
        spd[sl] = np.linalg.norm(f, axis=1)
        absc[sl] = np.linalg.eigvalsh(0.5 * (J + J.transpose(0, 2, 1)))[:, -1]
        X5, P5 = flow(field, Xb, T5, args.dt, variational=True)
        X10, PB = flow(field, X5, T5, args.dt, variational=True)
        P10 = np.einsum("bij,bjk->bik", PB, P5)
        G5[sl] = np.linalg.norm(P5, ord=2, axis=(1, 2))
        U, S, Vt = np.linalg.svd(P10)
        G10[sl] = S[:, 0]
        V1[sl] = Vt[:, 0, :]
        cosvf[sl] = np.abs((Vt[:, 0, :] * (f / spd[sl][:, None])).sum(1))
        cov_end[sl] = field.coverage(X10)
        print(f"[batch] {sl.stop}/{args.n_states}", flush=True)

    cov_p5 = float(np.percentile(
        field.coverage(A[np.random.default_rng(0).choice(len(A), 2000)]), 5))

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    from scipy.stats import spearmanr
    stats = dict(
        corr_G10_speed=corr(G10, spd),
        corr_G10_absc=corr(G10, absc),
        spearman_G10_speed=float(spearmanr(G10, spd).statistic),
        spearman_G10_absc=float(spearmanr(G10, absc).statistic),
        mean_cos_v1_f=float(cosvf.mean()),
        G5_mean=float(G5.mean()), G5_p90=float(np.percentile(G5, 90)),
        G10_mean=float(G10.mean()), G10_p90=float(np.percentile(G10, 90)),
        cov_end_below_p5_frac=float((cov_end < cov_p5).mean()))
    per_k = []
    for k in range(16):
        m = lab == k
        if not m.any():
            per_k.append(dict(k=k, n=0))
            continue
        per_k.append(dict(k=k, n=int(m.sum()),
                          G10_mean=float(G10[m].mean()),
                          G10_p90=float(np.percentile(G10[m], 90)),
                          G5_mean=float(G5[m].mean()),
                          absc_mean=float(absc[m].mean()),
                          speed_mean=float(spd[m].mean())))
        print(f"k={k:2d} n={m.sum():4d}  G10={per_k[-1]['G10_mean']:.2f} "
              f"(p90 {per_k[-1]['G10_p90']:.2f})  G5={per_k[-1]['G5_mean']:.2f} "
              f"absc={per_k[-1]['absc_mean']:+.2f}  |f|={per_k[-1]['speed_mean']:.4f}")
    print("[stats]", json.dumps(stats, indent=1))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / "ftg.npz", idx=idx, labels=lab, G5=G5, G10=G10,
             v1=V1, abscissa=absc, speed=spd, cos_v1_f=cosvf,
             cov_end=cov_end, cov_p5=cov_p5, model_path=str(args.model))
    (args.out_dir / "summary.json").write_text(
        json.dumps(dict(stats=stats, per_cluster=per_k), indent=2))

    # figures
    import shutil
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    use_tex = shutil.which("latex") and shutil.which("dvipng")
    plt.rcParams.update({"text.usetex": bool(use_tex), "font.family": "serif",
                         "mathtext.fontset": "cm"})

    _, sy_d = _signals(alpha, V)
    ax_d = _streak_amp(alpha, V)
    mx, sx, my, sy_s = ax_d.mean(), ax_d.std(), sy_d.mean(), sy_d.std()
    xs = (ax_d[::2][idx] - mx) / sx
    ys = (sy_d[::2][idx] - my) / sy_s

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    hb = axs[0].hexbin(xs, ys, C=G10, gridsize=28, cmap="Reds",
                       reduce_C_function=np.mean, mincnt=3)
    plt.colorbar(hb, ax=axs[0], label=r"$\langle G_{10}\rangle$")
    axs[0].set_xlabel(r"$A_x$ (std.)"); axs[0].set_ylabel(r"$\sigma_y$ (std.)")
    axs[0].set_title(r"finite-time growth $G_{\tau^+=10}$ on the SSP plane")
    ks = np.arange(16)
    axs[1].bar(ks, [r.get("G10_mean", 0.0) for r in per_k],
               color="#D55E00", alpha=0.85)
    axs[1].set_xticks(ks)
    axs[1].set_xlabel(r"cluster $k$")
    axs[1].set_ylabel(r"$\langle G_{10}\rangle$")
    axs[1].grid(axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(args.out_dir / "ftg_map.png", dpi=170)

    top = np.argsort(-G10)[:5]
    fig, axs = plt.subplots(1, 5, figsize=(18, 3.4))
    for c, i in enumerate(top):
        w = (V[:4096] @ V1[i]).reshape(64, 64)
        vv = np.abs(w).max()
        axs[c].imshow(w, origin="lower", cmap="RdBu_r", vmin=-vv, vmax=vv)
        axs[c].set_title(f"G10={G10[i]:.1f}, k={lab[i]}", fontsize=11)
        axs[c].set_xticks([]); axs[c].set_yticks([])
    fig.suptitle(r"optimal perturbation wall patterns ($\tau_x$ part)")
    fig.tight_layout()
    fig.savefig(args.out_dir / "ftg_optimal_perturbations.png", dpi=170)
    print(f"[done] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
