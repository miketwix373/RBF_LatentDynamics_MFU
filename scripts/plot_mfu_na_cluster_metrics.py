"""MFU_NA cluster-selection criteria, one figure per d_o, in the exact
format of the KS_CHAO paper figure (scripts/plot_kschao_cluster_metrics.py):

  (a) BIC marginal variation  Delta BIC / Delta K  (elbow = its minimum).
  (b) geometric (shape) distinctness over Voronoi-adjacent pairs: the largest
      principal angle sin(theta_max) and the participation-ratio gap |P_i-P_j|.
  (c) dynamical (motion) distinctness: the cluster-speed gap
      |v_i - v_j| / (v_i + v_j), v_k = <||alpha_dot||> over snapshots in k.

Reads the per-K shards under results/MFU_NA/cluster_sweep/do<d_o>/ and the
matching coords_do<d_o>.npz; the BIC dimension r is the coords' d_o. The
selected K is the Delta-BIC elbow unless --selected-K overrides it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT / "scripts"))

from plot_lor63_fig1_clustering import bic_kmeans, neighbour_distinctness  # noqa: E402

TITLE_FS, LABEL_FS, TICK_FS, LEG_FS, TAG_FS = 26, 24, 22, 20, 26


def load_metrics(sweep_dir: Path, coords: Path, stride: int,
                 metric_K: list[int]) -> dict:
    with np.load(coords) as z:
        speed = np.linalg.norm(np.asarray(z["alpha_dot"])[::stride], axis=1)
        r = int(z["d_o"])
    M = speed.shape[0]

    Kc, bic, sin_t, dpr, sgap = [], [], [], [], []
    for K in metric_K:
        with np.load(sweep_dir / f"K={K:02d}.npz") as s:
            if np.asarray(s["labels"]).shape[0] != M:
                raise SystemExit(
                    f"K={K}: label count {np.asarray(s['labels']).shape[0]} != "
                    f"{M}; check --stride")
            bic.append(bic_kmeans(float(s["inertia"]), s["n_k"], M, r, K))
            d = neighbour_distinctness(dict(s), speed, K)
        Kc.append(K)
        sin_t.append(d["sin_theta_max"])
        dpr.append(d["dPR_max"])
        sgap.append(d["speed_gap_max"])
    Kc = np.array(Kc)
    bic = np.array(bic)
    dbic = np.diff(bic)
    return {"M": M, "K": Kc, "K_mid": Kc[1:], "bic": bic, "dbic": dbic,
            "sin_t": np.array(sin_t), "dpr": np.array(dpr),
            "sgap": np.array(sgap)}


def make_figure(d_o: int, sweep_dir: Path, coords: Path, stride: int,
                metric_K: list[int], selected_K: int | None, out: Path,
                plt) -> None:
    m = load_metrics(sweep_dir, coords, stride, metric_K)
    Kc, K_mid, bic, dbic = m["K"], m["K_mid"], m["bic"], m["dbic"]
    sin_t, dpr, sgap = m["sin_t"], m["dpr"], m["sgap"]
    K_star = (selected_K if selected_K is not None
              else int(K_mid[np.argmin(dbic)]))
    print(f"[MFU_NA d_o={d_o}] M={m['M']}, Delta-BIC elbow at K={K_star}",
          flush=True)
    for K, b, st, dp, sg in zip(Kc, bic, sin_t, dpr, sgap):
        print(f"  K={K:2d}  BIC={b:.3g}  sin_theta_max={st:.3f}  "
              f"|dP|={dp:.3f}  speed_gap={sg:.3f}", flush=True)

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(21, 6.5))

    axA.plot(K_mid, dbic, "o-", color="k", lw=2.4, ms=9)
    axA.scatter([K_star], [dbic[K_mid == K_star][0]], s=200, facecolors="none",
                edgecolors="k", linewidths=2.2, zorder=5)
    axA.axvline(K_star, color="k", lw=1.2, ls="--",
                label=rf"elbow, $K={K_star}$")
    axA.axhline(0.0, color="k", lw=0.8, ls=":")
    axA.set_xlabel(r"$K$", fontsize=LABEL_FS)
    axA.set_title(r"$\Delta\mathrm{BIC}/\Delta K$", fontsize=TITLE_FS)
    axA.tick_params(labelsize=TICK_FS)
    axA.grid(alpha=0.3)
    axA.legend(fontsize=LEG_FS, loc="best")
    axA.text(-0.06, 1.02, "(a)", transform=axA.transAxes, fontsize=TAG_FS,
             fontweight="bold", va="bottom", ha="left")

    axB.plot(Kc, sin_t, "o-", color="k", lw=2.4, ms=9,
             label=r"$\sin\theta_{\max}$")
    axB.plot(Kc, dpr, "^--", color="k", lw=2.4, ms=9, label=r"$|P_i-P_j|$")
    axB.axvline(K_star, color="k", lw=1.2, ls="--")
    axB.set_xlabel(r"$K$", fontsize=LABEL_FS)
    axB.set_ylim(bottom=-0.02)
    axB.set_title("distinctness (adjacent pairs)", fontsize=TITLE_FS)
    axB.tick_params(labelsize=TICK_FS)
    axB.grid(alpha=0.3)
    axB.legend(fontsize=LEG_FS, loc="best")
    axB.text(-0.06, 1.02, "(b)", transform=axB.transAxes, fontsize=TAG_FS,
             fontweight="bold", va="bottom", ha="left")

    axC.plot(Kc, sgap, "s-", color="k", lw=2.4, ms=9)
    axC.axvline(K_star, color="k", lw=1.2, ls="--")
    axC.set_xlabel(r"$K$", fontsize=LABEL_FS)
    sgap_arr = np.asarray(sgap)
    lo, hi = float(np.nanmin(sgap_arr)), float(np.nanmax(sgap_arr))
    pad = 0.05 * (hi - lo) if hi > lo else 0.02
    axC.set_ylim(lo - pad, hi + pad)
    axC.set_title(r"$|v_i-v_j|/(v_i+v_j)$", fontsize=TITLE_FS)
    axC.tick_params(labelsize=TICK_FS)
    axC.grid(alpha=0.3)
    axC.text(-0.06, 1.02, "(c)", transform=axC.transAxes, fontsize=TAG_FS,
             fontweight="bold", va="bottom", ha="left")

    fig.tight_layout(w_pad=4.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_kw = dict(dpi=200, bbox_inches="tight", pad_inches=0.05)
    try:
        fig.savefig(out, **save_kw)
        fig.savefig(out.with_suffix(".png"), **save_kw)
    except RuntimeError as exc:
        print(f"[warn] LaTeX failed ({exc}); mathtext fallback", flush=True)
        plt.rcParams["text.usetex"] = False
        fig.savefig(out, **save_kw)
        fig.savefig(out.with_suffix(".png"), **save_kw)
    plt.close(fig)
    print(f"[done] wrote {out}", flush=True)


def make_combined_figure(d_os: list[int], root: Path, stride: int,
                         metric_K: list[int], out: Path, plt,
                         selected_K: int | None = None) -> None:
    """All d_o overlaid, one colour per d_o. Three panels, one criterion each:
    (a) Delta-BIC elbow, (b) |P_i-P_j|, (c) speed gap. The near-flat
    sin(theta_max) criterion is dropped as non-discriminating. Okabe-Ito
    CVD-safe colours (no yellow)."""
    colours = {45: "#0072B2", 75: "#D55E00", 170: "#009E73",  # blue/vermillion/green
               76: "#0072B2", 128: "#D55E00"}                 # MFU_OPO 90%/95%
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(21, 6.5))

    for d_o in d_os:
        m = load_metrics(root / f"cluster_sweep/do{d_o}",
                         root / f"pod/coords_do{d_o}.npz", stride, metric_K)
        col = colours.get(d_o, None)
        lab = rf"$d_o={d_o}$"
        axA.plot(m["K_mid"], m["dbic"], "o-", color=col, lw=2.0, ms=7,
                 label=lab)
        axB.plot(m["K"], m["dpr"], "^--", color=col, lw=2.0, ms=7, label=lab)
        axC.plot(m["K"], m["sgap"], "s-", color=col, lw=2.0, ms=7, label=lab)

    if selected_K is not None:
        for ax in (axA, axB, axC):
            ax.axvline(selected_K, color="0.35", lw=1.8, ls="--", zorder=0)

    axA.axhline(0.0, color="k", lw=0.8, ls=":")
    axA.set_xlabel(r"$K$", fontsize=LABEL_FS)
    axA.set_title(r"$\Delta\mathrm{BIC}/\Delta K$", fontsize=TITLE_FS)
    axA.legend(fontsize=LEG_FS, loc="best")
    axB.set_xlabel(r"$K$", fontsize=LABEL_FS)
    axB.set_ylim(bottom=-0.02)
    axB.set_title(r"$|P_i-P_j|$", fontsize=TITLE_FS)
    axB.legend(fontsize=LEG_FS, loc="best")
    axC.set_xlabel(r"$K$", fontsize=LABEL_FS)
    axC.set_title(r"$|v_i-v_j|/(v_i+v_j)$", fontsize=TITLE_FS)
    axC.legend(fontsize=LEG_FS, loc="best")
    for ax, tag in zip((axA, axB, axC), ("(a)", "(b)", "(c)")):
        ax.tick_params(labelsize=TICK_FS)
        ax.grid(alpha=0.3)
        ax.text(-0.06, 1.02, tag, transform=ax.transAxes, fontsize=TAG_FS,
                fontweight="bold", va="bottom", ha="left")

    fig.tight_layout(w_pad=4.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_kw = dict(dpi=200, bbox_inches="tight", pad_inches=0.05)
    try:
        fig.savefig(out, **save_kw)
        fig.savefig(out.with_suffix(".png"), **save_kw)
    except RuntimeError as exc:
        print(f"[warn] LaTeX failed ({exc}); mathtext fallback", flush=True)
        plt.rcParams["text.usetex"] = False
        fig.savefig(out, **save_kw)
        fig.savefig(out.with_suffix(".png"), **save_kw)
    plt.close(fig)
    print(f"[done] wrote {out}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--d-o", type=int, nargs="+", default=[45, 75, 170])
    p.add_argument("--root", type=Path, default=Path("results/MFU_NA"))
    p.add_argument("--stride", type=int, default=2,
                   help="snapshot stride; must match the sweep's stride")
    p.add_argument("--metric-K", type=int, nargs="+",
                   default=list(range(1, 31)))
    p.add_argument("--selected-K", type=int, default=None,
                   help="K to mark; default = the Delta-BIC elbow")
    p.add_argument("--per-do", action="store_true",
                   help="also write the per-d_o single-figure panels")
    args = p.parse_args()

    import shutil
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    use_tex = shutil.which("latex") is not None and shutil.which("dvipng") is not None
    plt.rcParams.update({
        "text.usetex": use_tex, "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
    })

    if args.per_do:
        for d_o in args.d_o:
            make_figure(
                d_o,
                sweep_dir=args.root / f"cluster_sweep/do{d_o}",
                coords=args.root / f"pod/coords_do{d_o}.npz",
                stride=args.stride,
                metric_K=args.metric_K,
                selected_K=args.selected_K,
                out=args.root / f"paper_images/cluster_metrics_do{d_o}.pdf",
                plt=plt,
            )

    if len(args.d_o) > 1:
        make_combined_figure(
            args.d_o, args.root, args.stride, args.metric_K,
            out=args.root / "paper_images/cluster_metrics_all.pdf",
            plt=plt, selected_K=args.selected_K,
        )


if __name__ == "__main__":
    main()
