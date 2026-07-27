"""Per-mode premultiplied power spectral density of a long ROM rollout vs truth.

Reads a long_test.npz written by run_mfu_na_forecast_suite.py (closed-loop RK4
rollout of adot = Phi(a).xi + corrector, over the full test horizon) and its
matching ground-truth POD-coefficient record. For each of the first `n_modes`
POD coordinates it estimates the PSD by Welch's method and plots the
PREMULTIPLIED spectrum f*PSD vs log-frequency (the wall-turbulence convention:
equal areas carry equal energy, so the energy-containing band is area-true and a
flat broadband noise floor shows as a rising ramp rather than masquerading as a
match). ROM overlaid on FOM.

The ground-truth WSS record is decimated by stride 2 with no anti-alias filter,
so the top decade below the 5 Hz strided Nyquist is unreliable; the band above
`alias_hz` (default 2 Hz, where the FOM carries ~0% variance) is shaded and must
not be interpreted. The residual ROM excess there is closed-loop RK4/corrector
step noise, not resolved dynamics -- see the aniso PSD attribution note.

Colours follow the paper palette: FOM black, RBF-ROM vermillion (#D55E00).

Usage
-----
    python scripts/plot_mfu_na_psd_modes.py \\
        --long-test results/MFU_NA/do45/alpha3.5/long_test.npz \\
        --out-path results/MFU_NA/paper_images/psd_modes_do45_a3.5.png \\
        --n-modes 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import welch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--long-test", type=Path, required=True)
    p.add_argument("--out-path", type=Path, required=True)
    p.add_argument("--n-modes", type=int, default=10)
    p.add_argument("--nperseg", type=int, default=1024)
    p.add_argument("--alias-hz", type=float, default=2.0,
                   help="shade/greyed band above this frequency (untrustworthy)")
    args = p.parse_args()

    z = np.load(args.long_test)
    rom = np.asarray(z["rom"], np.float64)
    truth = np.asarray(z["truth"], np.float64)
    dt = float(z["dt"])
    fs = 1.0 / dt
    d_o = rom.shape[1]
    n = min(args.n_modes, d_o)

    # finite prefix only (a diverged rollout is NaN-padded; here it is complete)
    good = np.isfinite(rom).all(1)
    rom, truth = rom[good], truth[good]
    nps = min(args.nperseg, rom.shape[0])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FOM, ROM = "black", "#D55E00"
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.7 * nrow),
                             sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for m in range(n):
        ax = axes[m]
        f, P_t = welch(truth[:, m], fs=fs, nperseg=nps)
        _, P_r = welch(rom[:, m], fs=fs, nperseg=nps)
        ax.semilogx(f[1:], f[1:] * P_t[1:], color=FOM, lw=1.6, label="FOM")
        ax.semilogx(f[1:], f[1:] * P_r[1:], color=ROM, lw=1.4, label="RBF-ROM")
        ax.axvspan(args.alias_hz, fs / 2, color="0.85", alpha=0.6, zorder=0)
        ax.set_title(fr"mode {m + 1}", fontsize=10)
        ax.grid(True, which="both", ls=":", alpha=0.35)
        if m == 0:
            ax.legend(fontsize=8, loc="upper left")
    for m in range(n, len(axes)):
        axes[m].set_visible(False)
    for m in range(n):
        if m // ncol == nrow - 1 or m + ncol >= n:
            axes[m].set_xlabel("frequency [Hz]")
    for r in range(nrow):
        axes[r * ncol].set_ylabel(r"$f\cdot$PSD")

    tag = args.long_test.parent.parent.name + "/" + args.long_test.parent.name
    fig.suptitle(f"MFU_NA {tag} — premultiplied per-mode spectrum, "
                 f"{rom.shape[0] * dt:.0f}s long rollout vs FOM (first {n} POD "
                 f"modes; shaded >{args.alias_hz:.0f} Hz unreliable)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_path, dpi=140, bbox_inches="tight")
    fig.savefig(args.out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[psd] {n} modes, {rom.shape[0]} samples, nperseg={nps}, fs={fs:.2f}Hz")
    print(f"[plot] wrote {args.out_path} (+ .pdf)")


if __name__ == "__main__":
    main()
