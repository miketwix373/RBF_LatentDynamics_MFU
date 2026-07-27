"""Mean skin-friction time history vs FOM across the alpha_tangent family,
d_o=45 and d_o=75 overlaid.

One stacked row per alpha_tangent (present in both d_o): the free-running ROM
reconstruction of the global skin friction <tau_x>(t) = <mu_x> + a(t).<V_k>_x
over the long test, with its marginal PDF (invariant measure) as a sidebar,
against the shared FOM ground truth (full-field spatial mean from stats.npz).
Time axis in wall units t+. Colours: FOM black, d_o=45 vermillion (#D55E00),
d_o=75 blue (#0072B2).

Same reconstruction and climate-comparison framing as
plot_mfu_na_skin_friction.py; see that file's docstring.

Usage
-----
    python scripts/plot_mfu_na_skin_friction_alpha_grid.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

SHARED_ROOT = (next(_p for _p in Path(__file__).resolve().parents if (_p / "pyproject.toml").exists()))
MFU = SHARED_ROOT / "results/MFU_NA"
PANELS = [(45, "#D55E00"), (75, "#0072B2")]
FOM_C = "black"
LABEL_FS, TICK_FS = 22, 18
TPLUS_PER_S = 11.289  # t+ = t u_tau^2/nu, u_tau=180/2870, nu=1/2870


def _drag_series(d_o: int, alphas: list[float]) -> tuple[dict, np.ndarray, float]:
    """Per-alpha ROM <tau_x>(t) for this d_o, plus the FOM trace and dt."""
    zc = np.load(MFU / "pod" / f"coords_do{d_o}.npz", allow_pickle=True)
    V = np.asarray(zc["V"], np.float64)
    mu = np.asarray(zc["mu"], np.float64)
    nx = mu.shape[0] // 2
    Vx, mux = V[:nx].mean(0), mu[:nx].mean()
    u = np.load(Path(str(zc["source_stats"])))["u"]
    u_flat = u.reshape(u.shape[0], -1)

    cf, cf_fom, dt = {}, None, None
    for a in alphas:
        zl = np.load(MFU / f"do{d_o}" / f"alpha{a:.1f}" / "long_test.npz")
        rom = np.asarray(zl["rom"], np.float64)
        ntr, stride, dt = int(zl["n_train"]), int(zl["stride"]), float(zl["dt"])
        cf[a] = mux + rom @ Vx
        if cf_fom is None:
            idx = stride * (ntr + np.arange(rom.shape[0]))
            cf_fom = u_flat[idx].mean(1)
    return cf, cf_fom, dt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--alpha-min", type=float, default=2.5)
    p.add_argument("--alpha-max", type=float, default=12.0)
    p.add_argument("--out-path", type=Path, default=MFU /
                   "paper_images/skin_friction_alpha_grid_do45_do75.pdf")
    args = p.parse_args()

    pat = re.compile(r"alpha([0-9.]+)$")
    per_do = []
    for d_o, _ in PANELS:
        per_do.append({float(m.group(1)) for d in (MFU / f"do{d_o}").iterdir()
                       if (m := pat.match(d.name))
                       and (d / "long_test.npz").exists()})
    alphas = sorted(a for a in set.intersection(*per_do)
                    if args.alpha_min <= a <= args.alpha_max)

    series = {d_o: _drag_series(d_o, alphas) for d_o, _ in PANELS}
    cf_fom, dt = series[PANELS[0][0]][1], series[PANELS[0][0]][2]
    tplus = np.arange(cf_fom.shape[0]) * dt * TPLUS_PER_S

    import shutil
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    use_tex = shutil.which("latex") is not None and shutil.which("dvipng") is not None
    plt.rcParams.update({
        "text.usetex": use_tex, "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
    })

    n = len(alphas)
    fig, axes = plt.subplots(
        n, 2, figsize=(16, 2.6 * n), squeeze=False,
        gridspec_kw=dict(width_ratios=[4, 1], wspace=0.03, hspace=0.45))

    for r, a in enumerate(alphas):
        ax, axh = axes[r, 0], axes[r, 1]
        ax.plot(tplus, cf_fom, color=FOM_C, lw=0.8)
        ax.axhline(cf_fom.mean(), color=FOM_C, ls="--", lw=0.9)
        for d_o, colour in PANELS:
            cf = series[d_o][0][a]
            ax.plot(tplus, cf, color=colour, lw=0.8, alpha=0.8)
            ax.axhline(cf.mean(), color=colour, ls="--", lw=0.9)
        ax.set_xlim(tplus[0], tplus[-1])
        ax.set_ylabel(r"$\langle\tau_x\rangle$", fontsize=LABEL_FS)
        ax.tick_params(labelsize=TICK_FS)
        ax.grid(True, ls=":", alpha=0.4)
        ax.set_title(rf"$\alpha_{{\mathrm{{tan}}}}={a:g}$",
                     fontsize=LABEL_FS, loc="left")
        if r < n - 1:
            ax.set_xticklabels([])

        allcf = [cf_fom] + [series[d_o][0][a] for d_o, _ in PANELS]
        lo = min(c.min() for c in allcf)
        hi = max(c.max() for c in allcf)
        bins = np.linspace(lo, hi, 50)
        axh.hist(cf_fom, bins=bins, orientation="horizontal", color=FOM_C,
                 histtype="step", density=True, lw=1.6)
        for d_o, colour in PANELS:
            axh.hist(series[d_o][0][a], bins=bins, orientation="horizontal",
                     color=colour, histtype="step", density=True, lw=1.6)
        axh.set_ylim(ax.get_ylim())
        axh.set_yticklabels([])
        axh.set_xticks([])
        axh.grid(True, ls=":", alpha=0.4)
    axes[-1, 0].set_xlabel(r"$t^+$", fontsize=LABEL_FS)

    handles = [Line2D([], [], color=FOM_C, lw=2.0, label="FOM")]
    handles += [Line2D([], [], color=colour, lw=2.0, label=rf"$d_o={d_o}$")
                for d_o, colour in PANELS]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=LABEL_FS, bbox_to_anchor=(0.5, -0.015))

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_path, bbox_inches="tight")
    fig.savefig(args.out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[cf] FOM mean={cf_fom.mean():.4e} std={cf_fom.std():.4e}")
    for a in alphas:
        for d_o, _ in PANELS:
            cf = series[d_o][0][a]
            print(f"  a{a:<5g} d_o={d_o}  mean={cf.mean():.4e} "
                  f"({cf.mean()/cf_fom.mean()-1:+.1%})  std={cf.std():.4e} "
                  f"({cf.std()/cf_fom.std()-1:+.1%})")
    print(f"[plot] wrote {args.out_path} (+ .png)")


if __name__ == "__main__":
    main()
