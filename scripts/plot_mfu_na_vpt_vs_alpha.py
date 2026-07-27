"""VPT vs tangent-inflation alpha for the calibrated MFU_NA forecast suite.

Reduces the 100-IC short-horizon ensembles written by
run_mfu_na_forecast_suite.py into a valid-prediction-time (VPT) curve over the
(d_o, alpha_tangent) grid, under the two reconstruction-based skill criteria of
plot_mfu_na_vpt_criteria.py:

  (a) relative reconstruction error  e_rec(t) = ||a_ROM - a_true|| / sigma_A;
      VPT = first t with e_rec > e_thresh (default 0.25).
  (b) anomaly/pattern correlation     ACC(t) = cos angle(a_ROM, a_true);
      VPT = first t with ACC < acc_thresh (default 0.5).

sigma_A = sqrt(sum var(A)) is the attractor RMS amplitude from the full strided
coefficient record (per d_o). VPT is computed per IC and reduced to the ensemble
median (band = inter-quartile range). Never-crossing members are right-censored
at the rollout horizon.

Two panels (one per criterion), VPT vs alpha, one line per d_o (45 and 75).

Usage
-----
    python scripts/plot_mfu_na_vpt_vs_alpha.py \\
        --manifest results/MFU_NA/model_selection/corrector_calibration.json \\
        --out-path results/MFU_NA/model_selection/vpt_vs_alpha.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _sigma_A(coords_path: str, stride: int) -> float:
    with np.load(coords_path) as z:
        A = np.asarray(z["alpha"], np.float64)[::stride]
    return float(np.sqrt(A.var(0).sum()))


def _vpt_per_ic(rom, truth, dt, sigma_A, e_thresh, acc_thresh):
    """Return (vpt_e, vpt_acc, censored_e, censored_acc) arrays over ICs.

    rom/truth: (n_ics, H+1, d_o). VPT is the time of first criterion violation;
    if never violated within the horizon it is censored at the horizon time."""
    n_ics, nT, _ = rom.shape
    t = np.arange(nT) * dt
    horizon = t[-1]
    d = rom - truth
    e_rec = np.linalg.norm(d, axis=2) / sigma_A                       # (n_ics,nT)
    num = (rom * truth).sum(2)
    acc = num / (np.linalg.norm(rom, axis=2) * np.linalg.norm(truth, axis=2))

    def _first(bad):                                                  # (n_ics,nT)
        ever = bad.any(1)
        idx = np.argmax(bad, axis=1)
        vpt = np.where(ever, t[idx], horizon)
        return vpt, ~ever

    vpt_e, cens_e = _first(e_rec > e_thresh)
    vpt_a, cens_a = _first(acc < acc_thresh)
    return vpt_e, vpt_a, cens_e, cens_a


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifest", type=Path,
                   default=Path("results/MFU_NA/model_selection/"
                                "corrector_calibration.json"))
    p.add_argument("--out-path", type=Path,
                   default=Path("results/MFU_NA/model_selection/"
                                "vpt_vs_alpha.png"))
    p.add_argument("--e-thresh", type=float, default=0.25)
    p.add_argument("--acc-thresh", type=float, default=0.5)
    args = p.parse_args()

    rows = json.load(open(args.manifest))
    sig_cache: dict[str, float] = {}
    recs = []
    for r in rows:
        fpath = Path(r["model_path"]).parent / "forecast_ics.npz"
        if not fpath.exists():
            print(f"[skip] missing {fpath}", flush=True)
            continue
        cp = r["coords_path"]
        if cp not in sig_cache:
            sig_cache[cp] = _sigma_A(cp, int(r["stride"]))
        z = np.load(fpath)
        vpt_e, vpt_a, cens_e, cens_a = _vpt_per_ic(
            np.asarray(z["rom"], np.float64), np.asarray(z["truth"], np.float64),
            float(z["dt"]), sig_cache[cp], args.e_thresh, args.acc_thresh)
        recs.append(dict(
            d_o=int(r["d_o"]), alpha=float(r["alpha_tangent"]),
            n_rbf=int(r["best_n_rbf"]),
            e_med=float(np.median(vpt_e)), e_min=float(vpt_e.min()),
            e_max=float(vpt_e.max()), e_cens=int(cens_e.sum()),
            a_med=float(np.median(vpt_a)), a_min=float(vpt_a.min()),
            a_max=float(vpt_a.max()), a_cens=int(cens_a.sum()),
        ))
        print(f"[{r['d_o']:>3} a{r['alpha_tangent']:<5}] "
              f"VPT_e={recs[-1]['e_med']:.2f}s (cens {recs[-1]['e_cens']}) "
              f"VPT_acc={recs[-1]['a_med']:.2f}s (cens {recs[-1]['a_cens']})",
              flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {45: "#0072B2", 75: "#D55E00"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
    panels = [("e_", args.e_thresh, r"(a) $e_{rec}<%.2f$" % args.e_thresh),
              ("a_", args.acc_thresh, r"(b) $ACC>%.2f$" % args.acc_thresh)]
    for axi, (pre, thr, title) in zip(ax, panels):
        for d_o in (45, 75):
            g = sorted([r for r in recs if r["d_o"] == d_o],
                       key=lambda r: r["alpha"])
            if not g:
                continue
            al = [r["alpha"] for r in g]
            med = [r[pre + "med"] for r in g]
            lo = [r[pre + "min"] for r in g]
            hi = [r[pre + "max"] for r in g]
            axi.plot(al, med, "o-", color=colours[d_o], lw=2, ms=6,
                     label=fr"$d_o={d_o}$")
            axi.plot(al, lo, "--", color=colours[d_o], lw=1, alpha=0.7)
            axi.plot(al, hi, "--", color=colours[d_o], lw=1, alpha=0.7)
        axi.set_xlabel(r"$\alpha_{tangent}$")
        axi.set_title(title)
        axi.grid(True, ls=":", alpha=0.4)
        axi.legend(fontsize=10)
        axi.set_ylabel("VPT [s]  (median solid, min/max dashed)")
    fig.suptitle("MFU_NA — valid prediction time vs tangent inflation "
                 "(100-IC short-horizon ensemble)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_path, dpi=140, bbox_inches="tight")
    fig.savefig(args.out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    jout = args.out_path.with_suffix(".json")
    json.dump({"e_thresh": args.e_thresh, "acc_thresh": args.acc_thresh,
               "records": recs}, open(jout, "w"), indent=2)
    print(f"[plot] wrote {args.out_path}\n[json] wrote {jout}", flush=True)


if __name__ == "__main__":
    main()
