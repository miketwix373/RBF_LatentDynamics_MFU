"""MFU_NA: tangent-anisotropy alpha study, one row per observable dimension
(d_o=45 top, d_o=75 bottom). Left column: mode-averaged test NRMSE vs number of
RBFs, one line per alpha. Right column: the floor of each curve (min over N)
against alpha, exposing the width basin. Curves and markers are coloured by
alpha (viridis) on a shared scale; a single colorbar replaces the legend.

The mode-averaged test NRMSE is the mean over the retained POD modes of
`nrmse_test_per_mode` from each `alpha<A>/sweep.npz`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SHARED_ROOT = (next(_p for _p in Path(__file__).resolve().parents if (_p / "pyproject.toml").exists()))
MFU = SHARED_ROOT / "results/MFU_NA"
# Panels are (d_o, directory). Every plain least-squares alpha*/sweep.npz under
# the directory is picked up automatically; ridge-regularised sweeps (name
# contains "ridge") are a separate study and excluded.
PANELS = [(45, MFU / "do45"), (75, MFU / "do75")]
LABEL_FS, TICK_FS = 22, 18


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=MFU /
                   "paper_images/alpha_nrmse_do45_do75.pdf")
    args = p.parse_args()

    import shutil
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    use_tex = shutil.which("latex") is not None and shutil.which("dvipng") is not None
    plt.rcParams.update({
        "text.usetex": use_tex, "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
    })

    # Load every curve first so the colour scale spans all alphas in both panels.
    panel_curves = []
    all_alphas = set()
    for do, root in PANELS:
        curves = []
        for sweep in sorted(root.glob("alpha*/sweep.npz")):
            if "ridge" in sweep.parent.name:
                continue
            z = np.load(sweep)
            n_rbf = np.asarray(z["n_rbf"], np.int64)
            nrmse = np.asarray(z["nrmse_test_per_mode"], np.float64).mean(axis=1)
            a = float(z["alpha_tangent"])
            curves.append((a, n_rbf, nrmse))
            all_alphas.add(a)
        curves.sort(key=lambda c: c[0])
        panel_curves.append((do, curves))

    alphas_sorted = sorted(all_alphas)
    norm = Normalize(vmin=min(alphas_sorted), vmax=max(alphas_sorted))
    cmap = cm.viridis

    markers = {45: "o", 75: "s"}
    lstyles = {45: "-", 75: "--"}

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), sharey=True)
    for col, (do, curves) in enumerate(panel_curves):
        axN = axes[col]
        for a, n_rbf, nrmse in curves:
            axN.plot(n_rbf, nrmse, "-o", color=cmap(norm(a)), lw=2.0, ms=6)
        axN.set_xscale("log")
        axN.tick_params(labelsize=TICK_FS)
        axN.grid(True, which="both", alpha=0.3)
        axN.set_title(rf"$d_o={do}$", fontsize=LABEL_FS)
        axN.set_xlabel(r"number of RBFs $N$", fontsize=LABEL_FS)
    axes[0].set_ylabel(r"test $\overline{\mathrm{NRMSE}}$", fontsize=LABEL_FS)

    axF = axes[2]
    FLOOR_C = "black"
    for do, curves in panel_curves:
        alpha_arr = np.array([c[0] for c in curves])
        floor = np.array([c[2].min() for c in curves])
        axF.plot(alpha_arr, floor, lstyles[do], color=FLOOR_C, lw=2.0,
                 marker=markers[do], ms=7, label=rf"$d_o={do}$")
    axF.set_xscale("log")
    axF.tick_params(labelsize=TICK_FS)
    axF.grid(True, which="both", alpha=0.3)
    axF.set_xlabel(r"$\alpha_{\mathrm{tan}}$", fontsize=LABEL_FS)
    axF.set_ylabel(r"min test $\overline{\mathrm{NRMSE}}$", fontsize=LABEL_FS)
    axF.tick_params(labelleft=True)
    axF.legend(fontsize=TICK_FS)

    fig.subplots_adjust(bottom=0.30, wspace=0.12)
    p2 = axes[2].get_position()
    axes[2].set_position([p2.x0 + 0.05, p2.y0, p2.width, p2.height])
    p0 = axes[0].get_position()
    p1 = axes[1].get_position()
    cax = fig.add_axes([p0.x0, 0.09, p1.x1 - p0.x0, 0.035])
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label(r"$\alpha_{\mathrm{tan}}$", fontsize=LABEL_FS)
    cbar.ax.tick_params(labelsize=TICK_FS)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {args.out} and {args.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
