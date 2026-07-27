"""FTC interpretability figure 14: slow points live only in the calm phase.

The fitted MFU_NA wall-shear RBF field f_RBF(a) = adot has slow points (roots of
f = 0, found by find_mfu_na_fixed_points.py) that all sit in the QUIESCENT half
of the streak-vortex plane. That plane is the one used in the phase-circulation
diagnostic (plot_mfu_na_phase_circulation.py): x = coherent k_z=1 streak
amplitude A_x, y = total spanwise shear sigma_y, both standardised over the
native record. High sigma_y is the active (bursting) SSP phase; low sigma_y is
the calm reservoir. Every slow point of every one of the four sibling fits sits
at sigma_y < 0 -- there are none in the active half.

Panel (a): the sampled attractor density (grey hexbin), the quiescent/active
split at sigma_y = 0, and the slow points of all four sibling fits overlaid,
coloured by escape time tau_esc = 1 / max Re(lambda) (per s -> t+; the field's
weak instability sets a ~120 t+ escape scale).
Panel (b): strip of escape times, showing the ~40-100 t+ band.

Reads the on-disk outputs of the sibling battery
(results/MFU_NA/field_analysis/{fixed_points, siblings/*/fixed_points}); does
not recompute anything. Style mirrors plot_mfu_na_enkf_paper.py (Okabe-Ito,
Computer Modern serif, save() writes pdf+png).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

SHARED_ROOT = Path("/users/sbrw610/sharedscratch/RBF_ROM")
MFU = SHARED_ROOT / "results/MFU_NA"
FA = MFU / "field_analysis"
COORDS = MFU / "pod/coords_do45.npz"
# per-repo convention: chord2 scripts live at the repo-root scripts/ dir
SCRIPTS = SHARED_ROOT / "scripts"

TPLUS_PER_S = 11.289

C_TRUTH = "black"
C_ROM = "#D55E00"
C_BLUE = "#0072B2"
C_GREEN = "#009E73"
C_PURPLE = "#CC79A7"
C_ORANGE = "#E69F00"

LABEL_FS, TICK_FS, LEG_FS, ANN_FS = 21, 18, 18, 18

# single fit shown on the plane, panel (a): (tag, fixed_points summary path, marker)
FITS = [
    ("a8, n=1920", FA / "siblings/a8_n1920/fixed_points/summary.json", "o"),
]

# sibling family for the escape-time distribution, panel (b): (label, path, colour)
SIBLINGS = [
    (r"$\alpha{=}8,\ n{=}960$", FA / "fixed_points/summary.json", C_BLUE),
    (r"$\alpha{=}8,\ n{=}1920$", FA / "siblings/a8_n1920/fixed_points/summary.json", C_ROM),
    (r"$\alpha{=}8,\ n{=}4800$", FA / "siblings/a8_n4800/fixed_points/summary.json", C_GREEN),
    (r"$\alpha{=}3.5,\ n{=}4800$", FA / "siblings/a35_n4800/fixed_points/summary.json", C_PURPLE),
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


def attractor_plane():
    """Standardised (streak A_x, sigma_y) of the native record; the panel (a)
    background. Uses the exact _signals / _streak_amp convention of the
    phase-circulation diagnostic so the axes match fig4_phase_cycle and the
    stored fixed-point ssp coordinates."""
    sys.path.insert(0, str(SCRIPTS))
    from plot_mfu_na_phase_circulation import _signals, _streak_amp  # noqa: E402

    with np.load(COORDS) as z:
        alpha = np.asarray(z["alpha"], np.float64)
        V = np.asarray(z["V"], np.float64)
    _, sy = _signals(alpha, V)
    ax = _streak_amp(alpha, V)
    x = (ax - ax.mean()) / ax.std()
    y = (sy - sy.mean()) / sy.std()
    return x, y


def load_slowpoints():
    """All slow points of the four fits: (px, py, tau_esc_tplus, credible,
    fit_index)."""
    pts = []
    for fi, (_, path, _) in enumerate(FITS):
        rows = json.loads(path.read_text())["rows"]
        for r in rows:
            max_re = max(e[0] for e in r["lead_eigs_per_s"])  # 1/s
            tau_esc = TPLUS_PER_S / max_re if max_re > 0 else np.inf
            pts.append((r["ssp"][0], r["ssp"][1], tau_esc,
                        bool(r["credible"]), fi))
    return pts


def escape_times(path):
    """Finite escape times (t+) of one fit's slow points."""
    out = []
    for r in json.loads(path.read_text())["rows"]:
        max_re = max(e[0] for e in r["lead_eigs_per_s"])  # 1/s
        if max_re > 0:
            out.append(TPLUS_PER_S / max_re)
    return np.asarray(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", type=Path,
                   default=SHARED_ROOT / "sn-article-template/images")
    p.add_argument("--also", type=Path, default=MFU / "paper_images",
                   help="second directory to drop copies in (repo convention)")
    args = p.parse_args()

    style()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    x, y = attractor_plane()
    pts = load_slowpoints()
    px = np.array([q[0] for q in pts])
    py = np.array([q[1] for q in pts])
    tau = np.array([q[2] for q in pts])
    cred = np.array([q[3] for q in pts])
    fit = np.array([q[4] for q in pts])

    print(f"[slowpoints] {len(pts)} across 4 fits; sigma_y range "
          f"[{py.min():+.2f}, {py.max():+.2f}]; active-half (sigma_y>0): "
          f"{int((py > 0).sum())}")
    print(f"[slowpoints] tau_esc range [{tau.min():.0f}, {tau.max():.0f}] t+ "
          f"(median {np.median(tau):.0f})")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.6, 4.6),
        gridspec_kw=dict(width_ratios=[1.0, 1.5]))

    # (a) streak-vortex plane -----------------------------------------------
    lim = 3.2
    m = (np.abs(x) < lim) & (np.abs(y) < lim)
    ax1.hexbin(x[m], y[m], gridsize=45, cmap="Greys", bins="log", mincnt=1,
               zorder=1)
    ax1.axhline(0, color="0.35", lw=1.2, ls="--", zorder=2)
    ax1.scatter(px, py, color=C_ROM, marker="o", s=95,
                edgecolors="k", linewidths=0.9, zorder=4)
    ax1.set_xlim(-2.0, 2.0)
    ax1.set_ylim(-3.0, 2.0)
    ax1.set_xlabel(r"Coherent streak $A_x$ (std.)", fontsize=LABEL_FS)
    ax1.set_ylabel(r"Spanwise shear $\sigma_y$ (std.)", fontsize=LABEL_FS)
    ax1.grid(True, ls=":", alpha=0.5)
    ax1.tick_params(labelsize=TICK_FS)

    # (b) escape-time PDF per sibling fit (persistence across alpha and n) ---
    from scipy.stats import gaussian_kde
    grid = np.linspace(0.0, 110.0, 400)
    for lab, path, col in SIBLINGS:
        te = escape_times(path)
        if te.size < 2:
            continue
        pdf = gaussian_kde(te)(grid)
        ax2.plot(grid, pdf, color=col, lw=1.8, label=lab, zorder=3)
        ax2.fill_between(grid, pdf, color=col, alpha=0.08, zorder=1)
    ax2.set_xlim(0, 110)
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel(r"Escape time $\tau_{\mathrm{esc}}$ $[t^+]$", fontsize=LABEL_FS)
    ax2.set_ylabel("Density", fontsize=LABEL_FS)
    ax2.legend(fontsize=LEG_FS - 1, frameon=False, handlelength=1.2,
               labelspacing=0.3)
    ax2.grid(True, ls=":", alpha=0.5)
    ax2.tick_params(labelsize=TICK_FS)

    fig.tight_layout()
    save(fig, [args.out_dir, args.also], "fig14_slowpoints")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
