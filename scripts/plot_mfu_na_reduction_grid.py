"""Paper figure (MFU_NA, minimal-flow-unit wall-shear stress): dimensionality-
reduction grid, in the layout of the KOL20 reduction figure
(plot_kol20_reduction_grid) and the Computer-Modern style of Lorenz-96 Fig 1.

Two rows, four columns:

  col 1  SVD energy retention: cumulative kept fluctuation energy vs number of
         POD modes, spanning both rows. The three kept truncations d_o = 45, 75,
         170 (90/95/99% energy) are flagged with nested, progressively darker
         blue bands; beyond the last the discarded band is orange.
  cols 2-4  one leading POD mode each, split by wall-shear component: the
         streamwise stress tau_x' (top) and the spanwise stress tau_y' (bottom).
         Columns show modes 0, 2, 4. The state is the stacked [tau_x | tau_y]
         fluctuation on the 64x64 wall plane, so V[:CSZ, m] is the tau_x field
         and V[CSZ:, m] the tau_y field (C-order reshape, matches
         plot_mfu_na_wss_spectrum.py).

Wall-unit geometry: Re_tau = u_tau/nu = 180 (u_tau = 180/2870, nu = 1/2870), box
(Lx, Lz) = (2.67, 0.8) -> (Lx+, Lz+) = (480.6, 144.0); field axes are x+ (stream-
wise, horizontal) and y+ (spanwise, vertical), drawn to true aspect ratio.

Reads results/MFU_NA/pod/svd_basis.npz (keys ``V``, ``energy_frac``).
Outputs PDF + PNG.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SHARED_ROOT = Path("/users/sbrw610/sharedscratch/RBF_ROM")
DEFAULT_SVD = SHARED_ROOT / "results/MFU_NA/pod/svd_basis.npz"
DEFAULT_OUT = SHARED_ROOT / "results/MFU_NA/paper_images/reduction_grid.pdf"

NG = 64                           # wall plane is 64x64
CSZ = NG * NG                     # 4096: length of one stress component
X_MAX = 200                       # energy has reached 99% by d_o=170
MODES = (0, 2, 4)                 # POD modes shown, one per column
MODE_C = ("#D55E00", "#0072B2", "#009E73")   # Okabe-Ito, no yellow

RE_TAU = 180.0                    # u_tau/nu = (180/2870)/(1/2870)
LX_PLUS = 2.67 * RE_TAU           # streamwise box length in wall units = 480.6
LZ_PLUS = 0.8 * RE_TAU            # spanwise box length in wall units  = 144.0

D_KEPT = (45, 75, 170)                        # 90 / 95 / 99% kept truncations
BAND_ALPHA = (0.12, 0.22, 0.34)               # nested bands: darker each step
LABEL_BLUE = ("#4a90c4", "#2171b5", "#08306b")

TITLE_FS, LABEL_FS, TICK_FS, LEG_FS, TAG_FS = 30, 28, 24, 24, 29


def _component(V, m, half):
    """Streamwise (half=0) or spanwise (half=1) stress field of POD mode m,
    signed so the largest-magnitude cell is positive."""
    f = V[half * CSZ:(half + 1) * CSZ, m].reshape(NG, NG)
    if f.flat[np.argmax(np.abs(f))] < 0:
        f = -f
    return f


def _draw_energy(ax, ef):
    """Col 1: cumulative POD fluctuation energy vs number of modes, the three
    kept truncations flagged with nested darker-blue bands."""
    if not (np.all(np.diff(ef) >= -1e-9) and ef[-1] <= 1.0001):
        ef = np.cumsum(ef)
    n = np.arange(1, ef.size + 1)

    edges = [0, *D_KEPT]
    for lo, hi, a in zip(edges[:-1], edges[1:], BAND_ALPHA):
        ax.axvspan(lo, hi, color="#0072B2", alpha=a, zorder=0)
    ax.axvspan(D_KEPT[-1], X_MAX, color="#E69F00", alpha=0.18, zorder=0)

    ax.plot(n, ef, color="black", lw=4.6, zorder=2)
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(0, 1.04)
    ax.grid(True, ls="-", lw=0.6, color="0.85")

    for d, cl in zip(D_KEPT, LABEL_BLUE):
        y = float(ef[d - 1])
        ax.vlines(d, 0, y, color="black", ls="--", lw=2.0, zorder=1)
        ax.plot([d], [y], marker="o", color=cl, ms=12, mec="k", mew=1.3,
                zorder=3)

    from matplotlib.ticker import FormatStrFormatter
    ax.set_xticks([0, 45, 75, 170])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_xlabel(r"Number of POD modes", fontsize=LABEL_FS)
    ax.set_ylabel(r"Cumulative energy", fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)


def _draw_field(ax, V, m, half, *, show_x, show_y):
    """One field panel: the tau_x (half=0) or tau_y (half=1) part of mode m,
    scaled to its own min/max for full per-panel contrast."""
    f = _component(V, m, half)
    ax.imshow(f, origin="lower", extent=(0, LX_PLUS, 0, LZ_PLUS),
              cmap="RdBu_r", vmin=float(f.min()), vmax=float(f.max()),
              interpolation="bilinear", aspect="auto")
    ax.set_xticks([0, 160, 320, 480])
    ax.set_yticks([0, 72, 144])
    if show_x:
        ax.set_xlabel(r"$x^+$", fontsize=LABEL_FS)
    else:
        ax.set_xticklabels([])
    if show_y:
        ax.set_ylabel(r"$y^+$", fontsize=LABEL_FS)
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="both", labelsize=TICK_FS)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--svd", type=Path, default=DEFAULT_SVD)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
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

    with np.load(args.svd) as z:
        V = np.asarray(z["V"], dtype=np.float64)
        ef = np.asarray(z["energy_frac"], dtype=np.float64)

    wr = [0.95, 0.16, 1.0, 1.0, 1.0]
    fig = plt.figure(figsize=(26.0, 6.0))
    gs = fig.add_gridspec(2, 5, width_ratios=wr, hspace=0.42, wspace=0.16)

    ax_e = fig.add_subplot(gs[:, 0])
    _draw_energy(ax_e, ef)
    ax_e.text(-0.24, 1.0, "(a)", transform=ax_e.transAxes, fontsize=TAG_FS,
              fontweight="bold", va="top", ha="right")

    row_title = (r"$\tau_x$", r"$\tau_y$")
    tags = iter("bcd")
    for j, m in enumerate(MODES):
        col = j + 2                       # cols 2,3,4; col 1 is the spacer
        for half in (0, 1):
            ax = fig.add_subplot(gs[half, col])
            _draw_field(ax, V, m, half, show_x=(half == 1), show_y=(j == 0))
            if j == 1:                    # single centred title per row
                ax.set_title(row_title[half], fontsize=TITLE_FS, pad=18)
            if half == 0:
                ax.text(-0.02, 1.30, f"({next(tags)})", transform=ax.transAxes,
                        fontsize=TAG_FS, fontweight="bold", va="top", ha="right")

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
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
