"""FTC interpretability figure 15: the dynamical anatomy of the wall cycle.

Three panels, each per-cluster and ordered along the SSP cycle phase theta_k
(from the phase-circulation cluster-role plane, stored as cyc_order in the
lift-up npz):

  (a) PROVISIONAL non-normal transient growth and phase-volume contraction.
      Finite-time growth G(tau=10 t+) per cluster from
      analyze_mfu_na_finite_time_growth.py (single fit, single 10 t+ horizon --
      hence provisional, and the horizon is stated on the axis). Overlaid, on a
      twin axis, the phase-volume contraction tr J per cluster
      (analyze_mfu_na_trace_contraction.py) as a band across the four sibling
      fits: the SIGN of tr J is fit-dependent, but the ORDERING is not --
      burst-onset clusters are least-contracting, the deep-burst cluster k=12
      contracts hardest in every fit (sibling battery, 2026-07-20). Burst onset
      (max G) and deep burst (min tr J) are marked.

  (b) Directed lift-up gains per cluster from analyze_mfu_na_liftup_matrix.py:
      g_sv (vortex->streak) vs g_vs (streak->vortex), the phase-conditioned
      observable-projected 2x2 (M2_k_tot). g_sv is positive in all 16 clusters
      and ~5x larger than g_vs; the sibling spread (min/max across the four
      fits' cycle-averaged g_sv) is shown as a band.

  (c) Per-cluster closure deficit rel_err from analyze_mfu_na_markovianity.py
      (a-priori residual ||adot - f_RBF|| / ||adot||): flat near ~0.30 across
      clusters (phase-uniform), with the run mean line and a derivative-
      estimator band (3-point FD vs 5-point-stored vs Savitzky-Golay).

Reads on-disk analysis outputs only; recomputes nothing. Style mirrors
plot_mfu_na_enkf_paper.py (Okabe-Ito, Computer Modern serif, save() -> pdf+png).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

SHARED_ROOT = Path("/users/sbrw610/sharedscratch/RBF_ROM")
MFU = SHARED_ROOT / "results/MFU_NA"
FA = MFU / "field_analysis"

TPLUS_PER_S = 11.289
FTG_HORIZON_TPLUS = 10.0

C_TRUTH = "black"
C_ROM = "#D55E00"
C_BLUE = "#0072B2"
C_GREEN = "#009E73"
C_PURPLE = "#CC79A7"
C_ORANGE = "#E69F00"

LABEL_FS, TICK_FS, LEG_FS, ANN_FS = 13, 10, 10, 11

# tr J across the four sibling fits (per-cluster contraction ordering band)
TRACE_FITS = [
    FA / "interpretables/trace_contraction.json",       # a8, n=960 (primary)
    FA / "siblings/a8_n1920/trace/trace_contraction.json",
    FA / "siblings/a8_n4800/trace/trace_contraction.json",
    FA / "siblings/a35_n4800/trace/trace_contraction.json",
]
# lift-up cycle-averaged g_sv across the four fits (sibling spread for panel b)
LIFTUP_FITS = [
    FA / "liftup/summary.json",
    FA / "siblings/a8_n1920/liftup/summary.json",
    FA / "siblings/a8_n4800/liftup/summary.json",
    FA / "siblings/a35_n4800/liftup/summary.json",
]


def style() -> bool:
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
    return use_tex


def save(fig, out_dirs: list[Path], stem: str) -> None:
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(d / f"{stem}.png", dpi=200, bbox_inches="tight")
        print(f"[done] wrote {d / stem}.pdf", flush=True)


def per_cluster(json_path: Path, key: str, rows_key: str = "rows") -> np.ndarray:
    d = json.loads(json_path.read_text())
    rows = d[rows_key] if isinstance(d, dict) else d
    rows = sorted(rows, key=lambda r: r["k"])
    return np.array([r[key] for r in rows])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", type=Path,
                   default=SHARED_ROOT / "sn-article-template/images")
    p.add_argument("--also", type=Path, default=MFU / "paper_images")
    args = p.parse_args()

    style()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # cycle order + per-cluster lift-up gains from the primary lift-up npz
    zl = np.load(FA / "liftup/liftup_matrix.npz", allow_pickle=True)
    cyc = np.asarray(zl["cyc_order"], np.int64)
    K = cyc.size
    g_sv = np.asarray(zl["M2_k_tot"])[:, 0, 1]          # vortex->streak
    g_vs = np.asarray(zl["M2_k_tot"])[:, 1, 0]          # streak->vortex
    g_sv_se = np.asarray(zl["M2_se_tot"])[:, 0, 1]
    g_vs_se = np.asarray(zl["M2_se_tot"])[:, 1, 0]

    # (a) finite-time growth (primary) + tr J band across fits
    ftg = json.loads((FA / "ftg/summary.json").read_text())
    G10 = np.array([r["G10_mean"] for r in
                    sorted(ftg["per_cluster"], key=lambda r: r["k"])])
    G10_p90 = np.array([r["G10_p90"] for r in
                        sorted(ftg["per_cluster"], key=lambda r: r["k"])])
    G5 = np.array([r["G5_mean"] for r in
                   sorted(ftg["per_cluster"], key=lambda r: r["k"])])
    trJ = np.stack([per_cluster(f, "trJ_per_tplus") for f in TRACE_FITS])
    trJ_lo, trJ_hi = trJ.min(0), trJ.max(0)
    trJ_mid = 0.5 * (trJ_lo + trJ_hi)

    # (b) sibling spread of cycle-averaged g_sv / g_vs
    gsv_sib = np.array([json.loads(f.read_text())["variants"]["tot"]["g_sv"]
                        for f in LIFTUP_FITS])
    gvs_sib = np.array([json.loads(f.read_text())["variants"]["tot"]["g_vs"]
                        for f in LIFTUP_FITS])

    # (c) closure deficit + estimator band
    mk = json.loads((MFU / "markovianity/summary.json").read_text())
    mk = sorted(mk, key=lambda r: r["k"])
    rel = np.array([r["rel_err"] for r in mk])
    rel_lo = np.array([min(r["rel_err"], r["rel_err_fd3"], r["rel_err_sg"])
                       for r in mk])
    rel_hi = np.array([max(r["rel_err"], r["rel_err_fd3"], r["rel_err_sg"])
                       for r in mk])

    x = np.arange(K)
    xlab = [str(k) for k in cyc]

    print(f"[anatomy] burst onset (max G10) cluster {cyc[np.argmax(G10[cyc])]} "
          f"G10={G10.max():.2f}; deep burst (min trJ) cluster "
          f"{int(np.argmin(trJ_mid))} trJ_mid={trJ_mid.min():.3f}/t+")
    print(f"[anatomy] g_sv>0 in {int((g_sv > 0).sum())}/{K} clusters; "
          f"cycle-avg g_sv/g_vs = {gsv_sib.mean():.4f}/{gvs_sib.mean():.4f} "
          f"~ {gsv_sib.mean()/max(abs(gvs_sib.mean()),1e-9):.1f}x")
    print(f"[anatomy] closure deficit mean {rel.mean():.3f} "
          f"range [{rel.min():.3f}, {rel.max():.3f}]")

    fig, (a1, a3) = plt.subplots(1, 2, figsize=(10.8, 4.3))

    # (a) growth vs contraction, one point per cluster ----------------------
    Gc, Tc = G10[cyc], trJ_mid[cyc]
    phase = np.arange(K)
    sc = a1.scatter(Tc, Gc, c=phase, cmap="coolwarm_r", s=95, edgecolors="k",
                    linewidths=0.8, zorder=3)
    a1.set_xlim(Tc.min() - 0.008, 0.006)
    a1.set_ylim(min(Gc.min(), 2.12) - 0.03, Gc.max() + 0.05)
    a1.axvline(0, color="0.5", ls=":", lw=1.0, zorder=0)
    cbax = a1.inset_axes([0.15, -0.42, 0.7, 0.045])
    cb = fig.colorbar(sc, cax=cbax, orientation="horizontal")
    cb.set_label("SSP cycle phase", fontsize=LABEL_FS + 4)
    cb.set_ticks([])

    io, idp = int(np.argmax(Gc)), int(np.argmin(Tc))
    a1.annotate("burst onset", (Tc[io], Gc[io]), textcoords="data",
                xytext=(-0.06, 2.51), ha="right", va="center",
                fontsize=ANN_FS + 2, arrowprops=dict(arrowstyle="->", lw=0.9))
    a1.annotate("deep burst", (Tc[idp], Gc[idp]), textcoords="data",
                xytext=(-0.09, 2.12), ha="left", va="bottom",
                fontsize=ANN_FS + 2, arrowprops=dict(arrowstyle="->", lw=0.9))

    # proposed cycle: quiescent -> burst onset -> deep burst -> quiescent,
    # drawn behind the points, with arrowheads along each leg
    right = Tc > np.median(Tc)
    iq = int(np.where(right)[0][np.argmin(Gc[right])])           # RHS low
    loop = [(Tc[iq], Gc[iq]), (Tc[io], Gc[io]), (Tc[idp], Gc[idp])]

    def flow_arrow(pa, pb, curv=0.14, n_heads=3):
        to_ax = lambda p: a1.transAxes.inverted().transform(
            a1.transData.transform(p))
        A, B = np.array(to_ax(pa)), np.array(to_ax(pb))
        mid, d = 0.5 * (A + B), B - A
        C = mid + curv * np.array([-d[1], d[0]])
        t = np.linspace(0, 1, 120)[:, None]
        crv = (1 - t) ** 2 * A + 2 * (1 - t) * t * C + t ** 2 * B
        a1.plot(crv[:, 0], crv[:, 1], transform=a1.transAxes, color="0.45",
                lw=1.5, alpha=0.7, zorder=1, solid_capstyle="round")
        for tf in np.linspace(0.28, 0.92, n_heads):
            b = (1 - tf) ** 2 * A + 2 * (1 - tf) * tf * C + tf ** 2 * B
            db = 2 * (1 - tf) * (C - A) + 2 * tf * (B - C)
            db = db / (np.linalg.norm(db) + 1e-9)
            a1.annotate("", xy=b, xytext=b - 0.035 * db, xycoords="axes fraction",
                        textcoords="axes fraction", zorder=1,
                        annotation_clip=False,
                        arrowprops=dict(arrowstyle="-|>", color="0.45", lw=1.5,
                                        alpha=0.8, mutation_scale=15))

    for pa, pb in zip(loop, loop[1:] + loop[:1]):
        flow_arrow(pa, pb)

    a1.set_xlabel(r"$\mathrm{tr}\,J$ $[1/t^+]$", fontsize=LABEL_FS + 4)
    a1.set_ylabel(rf"$G\,(\tau{{=}}{FTG_HORIZON_TPLUS:g}\,t^+)$",
                  fontsize=LABEL_FS + 4)
    a1.text(0.03, 0.97, "(a)", transform=a1.transAxes, fontsize=LABEL_FS + 6,
            ha="left", va="top")
    a1.grid(True, ls=":", alpha=0.4, zorder=0)
    a1.tick_params(labelsize=TICK_FS + 4)

    # (b) closure deficit ---------------------------------------------------
    a3.bar(x, rel[cyc], width=0.7, color=C_GREEN, alpha=0.85, zorder=2)
    a3.axhline(rel.mean(), color=C_TRUTH, ls=":", lw=1.3, zorder=3)
    a3.set_ylim(0, max(0.45, rel_hi.max() * 1.1))
    a3.set_xticks(x)
    a3.set_xticklabels(xlab, fontsize=TICK_FS + 4)
    a3.tick_params(axis="y", labelsize=TICK_FS + 4)
    a3.set_xlabel(r"cluster $k$ (SSP cycle order)", fontsize=LABEL_FS + 4)
    a3.set_ylabel(r"$\|\dot a - f_{\mathrm{RBF}}\|/\|\dot a\|$",
                  fontsize=LABEL_FS + 4)
    a3.text(0.03, 0.97, "(b)", transform=a3.transAxes, fontsize=LABEL_FS + 6,
            ha="left", va="top")
    a3.grid(True, axis="y", ls=":", alpha=0.4, zorder=0)

    fig.tight_layout()
    save(fig, [args.out_dir, args.also], "fig15_anatomy")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
