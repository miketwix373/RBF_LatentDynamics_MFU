"""MFU_NA: forecast horizon and climate errors across the alpha_tangent
family. Three panels sharing a log alpha_tangent axis, one line per
observable dimension (d_o=45 solid circles, d_o=75 dashed squares):

(a) Ensemble-median VPT in wall units t+ (e_rec<thresh from vpt_vs_alpha.json).
(b) Marginal invariant measure of the drag: W1(<tau_x>_ROM, <tau_x>_FOM) /
    sigma_FOM from long_test.npz.
(c) Temporal spectrum of the drag: total-variation distance between the
    area-normalised premultiplied spectra f*PSD of <tau_x>(t) below ALIAS_HZ.

Metrics identical to plot_mfu_na_vpt_error_twinaxis.py; only the framing
changes.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

SHARED_ROOT = Path("/users/sbrw610/sharedscratch/RBF_ROM")
MFU = SHARED_ROOT / "results/MFU_NA"
LABEL_FS, TICK_FS = 26, 24
TPLUS_PER_S = 11.289  # t+ = t u_tau^2/nu, u_tau=180/2870, nu=1/2870
ALIAS_HZ, NPERSEG = 2.0, 1024
D_OS = (45, 75)
MARKERS = {45: "o", 75: "s"}
LSTYLES = {45: "-", 75: "--"}


def _climate_metrics(d_o: int) -> tuple[dict, dict]:
    """Per-alpha (W1(<tau_x>)/sigma_FOM, spectral-shape TV) for this d_o."""
    from scipy.stats import wasserstein_distance
    from scipy.signal import welch
    zc = np.load(MFU / "pod" / f"coords_do{d_o}.npz", allow_pickle=True)
    V = np.asarray(zc["V"], np.float64)
    mu = np.asarray(zc["mu"], np.float64)
    nx = mu.shape[0] // 2
    Vx, mux = V[:nx].mean(0), mu[:nx].mean()
    u = np.load(Path(str(zc["source_stats"])))["u"]
    u_flat = u.reshape(u.shape[0], -1)
    pat = re.compile(r"alpha([0-9.]+)$")
    dirs = sorted([(float(m.group(1)), d) for d in (MFU / f"do{d_o}").iterdir()
                   if (m := pat.match(d.name)) and (d / "long_test.npz").exists()])

    def _premult(sig, fs):
        f, P = welch(sig, fs=fs, nperseg=min(NPERSEG, sig.shape[0]))
        return f, f * P

    w1, tv, cf_fom, sig, fp_fom, f = {}, {}, None, None, None, None
    for alpha, d in dirs:
        zl = np.load(d / "long_test.npz")
        rom = np.asarray(zl["rom"], np.float64)
        ntr, stride, dt = int(zl["n_train"]), int(zl["stride"]), float(zl["dt"])
        cf = mux + rom @ Vx
        if cf_fom is None:
            idx = stride * (ntr + np.arange(rom.shape[0]))
            cf_fom = u_flat[idx].mean(1)
            sig = cf_fom.std()
            f, fp_fom = _premult(cf_fom, 1.0 / dt)
        w1[alpha] = wasserstein_distance(cf, cf_fom) / sig
        _, fp = _premult(cf, 1.0 / dt)
        b = (f > 0) & (f <= ALIAS_HZ)
        dr, dt_ = fp[b] / fp[b].sum(), fp_fom[b] / fp_fom[b].sum()
        tv[alpha] = 0.5 * float(np.abs(dr - dt_).sum())
    return w1, tv


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--vpt-json", type=Path,
                   default=MFU / "paper_images/vpt_vs_alpha.json")
    p.add_argument("--criterion", choices=["e", "acc"], default="e")
    p.add_argument("--out", type=Path,
                   default=MFU / "paper_images/vpt_climate_vs_nrmse.pdf")
    args = p.parse_args()

    blob = json.load(open(args.vpt_json))
    recs = blob["records"]
    pre = "e_" if args.criterion == "e" else "a_"
    thr = blob["e_thresh"] if args.criterion == "e" else blob["acc_thresh"]
    crit = (rf"$e_{{rec}}<{thr:g}$" if args.criterion == "e"
            else rf"$ACC>{thr:g}$")
    vptd = {d_o: {r["alpha"]: r[pre + "med"] * TPLUS_PER_S
                  for r in recs if r["d_o"] == d_o} for d_o in D_OS}
    metrics = {d_o: _climate_metrics(d_o) for d_o in D_OS}
    w1 = {d_o: metrics[d_o][0] for d_o in D_OS}
    tv = {d_o: metrics[d_o][1] for d_o in D_OS}
    alphas = {d_o: sorted(set(vptd[d_o]) & set(w1[d_o])) for d_o in D_OS}

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

    panels = [
        (vptd, rf"VPT $[t^+]$  ({crit})"),
        (w1, r"$W_1(\langle\tau_x\rangle)/\sigma_{\mathrm{FOM}}$"),
        (tv, r"spectral TV of $f{\cdot}$PSD"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    for ax, (metric, ylabel) in zip(axes, panels):
        for d_o in D_OS:
            al = alphas[d_o]
            ax.plot(al, [metric[d_o][a] for a in al], LSTYLES[d_o],
                    color="black", lw=2.0, marker=MARKERS[d_o], ms=7,
                    label=rf"$d_o={d_o}$")
        ax.set_xscale("log")
        ax.set_xticks([2, 4, 8, 16, 32])
        ax.set_xticklabels(["2", "4", "8", "16", "32"])
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlabel(r"$\alpha_{\mathrm{tan}}$", fontsize=LABEL_FS)
        ax.set_ylabel(ylabel, fontsize=LABEL_FS)
        ax.tick_params(labelsize=TICK_FS)
        ax.grid(True, which="both", ls=":", alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               fontsize=LABEL_FS, bbox_to_anchor=(0.5, -0.18),
               frameon=False)

    fig.tight_layout(w_pad=3.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    for d_o in D_OS:
        print(f"d_o={d_o}: " + "  ".join(
            f"a{a:g} vpt={vptd[d_o][a]:.1f} w1={w1[d_o][a]:.2f} "
            f"tv={tv[d_o][a]:.2f}" for a in alphas[d_o]))
    print(f"wrote {args.out} and {args.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
