"""Three-panel MFU_NA near-wall cycle figure with the cycle and the clusters
drawn in their correct coordinate frames.

The single-panel plot_mfu_na_phase_circulation.py conflates two things that live
in different frames: the directed lift-up loop only exists in the high-pass
(fluctuation) plane (D = -0.20, vs raw D ~ 0), whereas the cluster centroids are
K-means partitions of the RAW amplitude space -- in high-pass coords they
collapse to the origin. This driver separates them:

  (a) RAW plane: the 16 cluster centroids (size = occupancy) with switching
      arrows -- cluster amplitude roles (Q2 = coherent streak collapsed but
      broadband QSV high = breakdown). Transitions are near-symmetric (detailed
      balance): the raw plane carries no directed cycle.
  (b) FILTERED plane: residence density + binned mean phase-velocity streamlines.
      This is where the weak, surrogate-significant clockwise lift-up circulation
      (sigma_y leads A_x) is real and visible -- justifying the cycle only after
      the slow amplitude drift is removed.
  (c) per-cluster LOCAL DYNAMICS: the fitted global RBF field linearised at each
      centroid, plotted as f_k vs Re(lambda_k), coloured by phase coherence R.

Axes as in plot_mfu_na_phase_circulation.py: x = A_streak (k_x=0, k_y=1 of
tau_x'), y = total sigma_y. Wall time t+ = t u_tau^2/nu.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_mfu_na_phase_circulation import (  # noqa: E402
    _signals, _streak_amp, _highpass, _directedness,
    TPLUS_PER_T, SHARED_ROOT, STREAM_C, TAU_W, LABEL_FS, TICK_FS, ANN_FS)
from compute_mfu_na_cluster_jacobians import (  # noqa: E402
    _load_model, _phi_and_grad)
from analyze_mfu_na_cluster_geometry import (  # noqa: E402
    _leading_plane, _quadrant)


def _standardise(v):
    return (v - v.mean()) / v.std()


def _stream_field(ax, xx, yy, lim, nbp=28):
    from scipy.ndimage import gaussian_filter
    dxx, dyy = np.gradient(xx), np.gradient(yy)
    eg = np.linspace(-lim, lim, nbp + 1); ce = 0.5 * (eg[:-1] + eg[1:])
    mm = (np.abs(xx) < lim) & (np.abs(yy) < lim)
    jx, jy = np.digitize(xx[mm], eg) - 1, np.digitize(yy[mm], eg) - 1
    SX = np.zeros((nbp, nbp)); SY = np.zeros((nbp, nbp)); CN = np.zeros((nbp, nbp))
    np.add.at(SX, (jy, jx), dxx[mm]); np.add.at(SY, (jy, jx), dyy[mm])
    np.add.at(CN, (jy, jx), 1.0)
    SXs, SYs, CNs = gaussian_filter(SX, 1.1), gaussian_filter(SY, 1.1), gaussian_filter(CN, 1.1)
    okk = CNs >= 8.0
    Uu = np.where(okk, SXs / np.maximum(CNs, 1e-9), np.nan)
    Vw = np.where(okk, SYs / np.maximum(CNs, 1e-9), np.nan)
    sp = np.hypot(Uu, Vw)
    lww = 0.8 + 2.4 * np.nan_to_num(sp / np.nanmax(sp))
    ax.streamplot(ce, ce, np.ma.masked_invalid(Uu), np.ma.masked_invalid(Vw),
                  color=STREAM_C, density=1.5, linewidth=lww, arrowsize=1.3, zorder=3)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--coords", type=Path,
                   default=SHARED_ROOT / "results/MFU_NA/pod/coords_do45.npz")
    p.add_argument("--detrend-tplus", type=float, default=100.0)
    p.add_argument("--labels", type=Path, default=SHARED_ROOT /
                   "results/MFU_NA/cluster_sweep/do45/K=16.npz")
    p.add_argument("--model", type=Path, default=SHARED_ROOT /
                   "results/MFU_NA/do45/alpha3.5/model_n04800.npz")
    p.add_argument("--out", type=Path, default=SHARED_ROOT /
                   "results/MFU_NA/paper_images/phase_cycle_split_K16.pdf")
    args = p.parse_args()

    import shutil
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as mcm
    from matplotlib.patches import FancyArrowPatch

    use_tex = shutil.which("latex") is not None and shutil.which("dvipng") is not None
    plt.rcParams.update({
        "text.usetex": use_tex, "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
    })

    with np.load(args.coords) as z:
        alpha = np.asarray(z["alpha"], np.float64)
        V = np.asarray(z["V"], np.float64)
        t = np.asarray(z["t"], np.float64)
    dt = float(np.median(np.diff(t)))
    tpf = dt * TPLUS_PER_T
    w = int(round(args.detrend_tplus / tpf))

    ax_sig, sy_sig = _signals(alpha, V)
    streak_sig = _streak_amp(alpha, V)

    # filtered (fluctuation) plane -- where the directed loop lives
    xf = _standardise(_highpass(streak_sig, w))
    yf = _standardise(_highpass(sy_sig, w))
    _, D, _ = _directedness(xf, yf)
    sense = "clockwise" if D < 0 else "counter-clockwise"

    # raw amplitude plane -- where the clusters live
    xr = _standardise(streak_sig)
    yr = _standardise(sy_sig)

    clab = np.asarray(np.load(args.labels)["labels"], np.int64)
    step = len(xr) // len(clab)
    Kc = int(clab.max()) + 1
    aL = xr[::step][:len(clab)]; sL = yr[::step][:len(clab)]
    cx_k = np.array([aL[clab == k].mean() for k in range(Kc)])
    cy_k = np.array([sL[clab == k].mean() for k in range(Kc)])
    occ_k = np.array([(clab == k).mean() for k in range(Kc)])
    C = np.zeros((Kc, Kc)); np.add.at(C, (clab[:-1], clab[1:]), 1.0)
    np.fill_diagonal(C, 0.0)
    Pr = C.sum(1, keepdims=True)
    Pij = np.divide(C, Pr, out=np.zeros_like(C), where=Pr > 0)
    cmax = float(C.max())

    # per-cluster centroids + quadrant/radius for the panel-(c) portraits
    cent_k = np.array([alpha[::step][:len(clab)][clab == k].mean(0) for k in range(Kc)])
    mdl = _load_model(args.model)
    quad_k = np.array([_quadrant(cx_k[k], cy_k[k]) for k in range(Kc)])
    rad_k = np.hypot(cx_k, cy_k)

    # link col-(c) portraits to their cluster: a distinct CVD-safe border colour
    # per selected cluster, shared between the column-(a) circle edge and the
    # column-(c) tile spine. Selection = 2 furthest-from-origin clusters/quadrant.
    PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7"]
    sel = [sorted([k for k in range(Kc) if quad_k[k] == q],
                  key=lambda k: -rad_k[k])[:2] for q in (1, 2, 3)]
    cl_color = {}
    ii = 0
    for cc in range(3):
        for rr in range(2):
            cl_color[sel[cc][rr]] = PALETTE[ii]; ii += 1
    sel_ks = [k for k in range(Kc) if k in cl_color]
    other_ks = [k for k in range(Kc) if k not in cl_color]

    LABEL_FS, TICK_FS, ANN_FS = 30, 24, 25
    LIM = 2.4
    # explicit placement (inches). Top row: a2/a1 equal squares of side S, panel
    # (c) a 2x3 grid of S/2 tiles. Bottom row: three tau_x wall-shear fields at
    # the fig2 wall-domain aspect (LX_PLUS:LZ_PLUS), one per quadrant, aligned
    # under the three top-row columns.
    Win, Hin = 24.3, 13.8
    LXP, LZP = 480.6, 144.0
    fig = plt.figure(figsize=(Win, Hin))

    def _ax(l, b, w, h):
        return fig.add_axes([l / Win, b / Hin, w / Win, h / Hin])

    S = 5.92                                  # top-row square side (in)
    yb = 7.49                                 # top-row bottom (in)
    a2 = _ax(0.95, yb, S, S)
    a1 = _ax(8.82, yb, S, S)
    ct = S / 2                                # panel-(c) tile side (in)
    cl0 = 15.19                               # panel-(c) left (in)
    tgap = 0.13                               # gap so both border colours show
    cg = np.empty((2, 3), dtype=object)
    for rr in range(2):
        for cc in range(3):
            cg[rr, cc] = _ax(cl0 + cc * ct + tgap / 2,
                             yb + (1 - rr) * ct + tgap / 2, ct - tgap, ct - tgap)
    tw2 = 6.74                                # row-2 panel width (in); larger => tighter gaps
    tH = tw2 * 2.8 / 5.92                     # keep the row-2 aspect ratio
    tyb = 2.0                                 # row-2 bottom (in), leaving room for colorbar
    txl = [0.95, 9.14, 17.33]                 # uniform, tight column spacing across row 2
    tax = [_ax(l, tyb, tw2, tH) for l in txl]

    # (b) FILTERED plane: density + directed streamlines
    a1.hexbin(xf, yf, gridsize=40, cmap="Greys", bins="log", mincnt=1, zorder=1)
    _stream_field(a1, xf, yf, lim=3.2)
    a1.axhline(0, color="0.6", lw=1.0, ls=":"); a1.axvline(0, color="0.6", lw=1.0, ls=":")
    a1.set_xlabel(r"coherent streak $A_x$ ($k_y{=}1$, high-pass)", fontsize=LABEL_FS - 3)
    a1.set_ylabel(r"total spanwise shear $\sigma_y$ (high-pass)", fontsize=LABEL_FS - 3)
    a1.set_xlim(-LIM, LIM); a1.set_ylim(-LIM, LIM); a1.set_aspect("equal")
    a1.tick_params(labelsize=TICK_FS)
    a1.text(-0.02, 1.02, "(b)", transform=a1.transAxes, fontsize=ANN_FS,
            fontweight="bold", va="bottom", ha="right")

    # (a) RAW plane: cluster centroids + switching arrows
    a2.hexbin(xr, yr, gridsize=40, cmap="Greys", bins="log", mincnt=1, zorder=1)
    keep = list(range(Kc))
    occ_ref = occ_k[keep].max()
    for i in keep:
        for jj in keep:
            if i == jj or Pij[i, jj] < 0.12:
                continue
            wln = 1.0 + 5.0 * (C[i, jj] / cmax) ** 0.7
            a2.add_patch(FancyArrowPatch(
                (cx_k[i], cy_k[i]), (cx_k[jj], cy_k[jj]),
                connectionstyle="arc3,rad=0.16", arrowstyle="-|>",
                mutation_scale=11, lw=wln, color="0.35",
                alpha=0.3 + 0.5 * (C[i, jj] / cmax), zorder=4))
    a2.scatter(cx_k[other_ks], cy_k[other_ks],
               s=140 + 1500 * (occ_k[other_ks] / occ_ref),
               facecolors="white", edgecolors="k", linewidths=1.3, zorder=5)
    a2.scatter(cx_k[sel_ks], cy_k[sel_ks],
               s=140 + 1500 * (occ_k[sel_ks] / occ_ref),
               facecolors=[cl_color[k] for k in sel_ks],
               edgecolors=[cl_color[k] for k in sel_ks], linewidths=2.8, zorder=6)
    a2.axhline(0, color="0.6", lw=1.0, ls=":"); a2.axvline(0, color="0.6", lw=1.0, ls=":")
    a2.set_xlabel(r"coherent streak $A_x$ ($k_y{=}1$, std.)", fontsize=LABEL_FS - 3)
    a2.set_ylabel(r"total spanwise shear $\sigma_y$ (std.)", fontsize=LABEL_FS - 3)
    a2.set_xlim(-LIM, LIM); a2.set_ylim(-LIM, LIM); a2.set_aspect("equal")
    a2.tick_params(labelsize=TICK_FS)
    a2.text(-0.02, 1.02, "(a)", transform=a2.transAxes, fontsize=ANN_FS,
            fontweight="bold", va="bottom", ha="right")

    # (c) per-cluster local phase portraits: J_k projected onto its leading
    # eigen-plane. Columns = raw-plane quadrants, rows = the two clusters
    # furthest from the origin in each. Q1 (mature) spirals; Q2/Q3 nodes.
    gg = np.linspace(-1, 1, 24); GX, GY = np.meshgrid(gg, gg)
    for cc, q in enumerate((1, 2, 3)):
        for rr in range(2):
            k = sel[cc][rr]
            ax = cg[rr, cc]
            _, Jk = _phi_and_grad(cent_k[k], mdl)
            J2 = _leading_plane(Jk)
            U = J2[0, 0] * GX + J2[0, 1] * GY
            Wv = J2[1, 0] * GX + J2[1, 1] * GY
            ax.streamplot(gg, gg, U, Wv, color=np.hypot(U, Wv), cmap="viridis",
                          density=0.7, linewidth=0.9, arrowsize=0.7)
            ax.set_xticks([]); ax.set_yticks([]); ax.margins(0)
            for sp in ax.spines.values():
                sp.set_color(cl_color[k]); sp.set_linewidth(3.4)
            if rr == 0:
                ax.set_title(rf"Q{q}", fontsize=ANN_FS + 2, fontweight="bold")
    fig.text(cl0 / Win - 0.004, (yb + 1.02 * S) / Hin, "(c)", fontsize=ANN_FS,
             fontweight="bold", va="bottom", ha="right")

    # (d) second row: average tau_x' field of each quadrant's representative
    # (row-1, furthest-from-origin) cluster, reconstructed from its POD centroid
    # V_x @ cent_k. Border colour links to that cluster in panels (a)/(c). Same
    # wall-domain dimensions as fig2 (x+ streamwise, y+ spanwise).
    Vx = V[:64 * 64]
    rep = [sel[0][0], sel[1][0], sel[2][1]]   # Q1/Q2 row-1 cluster; Q3 the pink one
    fields = [(Vx @ cent_k[rep[cc]]).reshape(64, 64) / TAU_W for cc in range(3)]
    vv = max(float(np.abs(f).max()) for f in fields)     # common symmetric clim
    xg = np.linspace(0, LXP, 64); yg = np.linspace(0, LZP, 64)
    Xg, Yg = np.meshgrid(xg, yg)
    im = None
    for cc in range(3):
        k = rep[cc]; ax = tax[cc]
        im = ax.imshow(fields[cc], origin="lower", extent=(0, LXP, 0, LZP),
                       cmap="RdBu_r", vmin=-vv, vmax=vv,
                       interpolation="bilinear", aspect="auto")
        ax.contour(Xg, Yg, fields[cc], levels=np.linspace(-vv, vv, 13),
                   colors="k", linewidths=1.3, alpha=0.7)
        ax.set_xlim(0, LXP); ax.set_ylim(0, LZP)
        for sp in ax.spines.values():
            sp.set_color(cl_color[k]); sp.set_linewidth(5.0)
        ax.set_xticks([0, 160, 320, 480]); ax.set_yticks([0, 72, 144])
        ax.tick_params(labelsize=TICK_FS + 4)
        ax.set_xlabel(r"$x^+$", fontsize=LABEL_FS - 2)
        ax.set_title(rf"$\overline{{\tau}}_x$, Q{cc + 1}", fontsize=ANN_FS + 4, pad=16)
        if cc == 0:
            ax.set_ylabel(r"$y^+$", fontsize=LABEL_FS - 2)
        else:
            ax.set_yticklabels([])
        fig.text(txl[cc] / Win - 0.004, (tyb + tH + 0.35) / Hin,
                 f"({chr(100 + cc)})", fontsize=ANN_FS + 4, fontweight="bold",
                 va="bottom", ha="right")
    cax = _ax(6.65, 0.45, 12.0, 0.28)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(r"$\tau_x / u_\tau^2$", fontsize=LABEL_FS - 2)
    cb.ax.tick_params(labelsize=TICK_FS + 1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_kw = dict(dpi=200, bbox_inches="tight", pad_inches=0.05)
    try:
        fig.savefig(args.out, **save_kw)
        fig.savefig(args.out.with_suffix(".png"), **save_kw)
    except RuntimeError as exc:
        print(f"[warn] LaTeX failed ({exc}); mathtext fallback", flush=True)
        plt.rcParams["text.usetex"] = False
        fig.savefig(args.out, **save_kw)
        fig.savefig(args.out.with_suffix(".png"), **save_kw)
    plt.close(fig)
    print(f"[done] D={D:+.3f} ({sense}); wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
