"""MFU_NA EnKF paper figures from the alpha8 sweep (sweep_a8) diagnostics.

Four figures into results/MFU_NA/paper_images, matching the fig_* house style
(Computer Modern serif, Okabe-Ito palette, truth black / RBF-ROM vermillion):

  fig9_enkf_modes        -- ensemble tracking of the leading modes (showcase):
                            members, EnKF mean, FOM truth, free-running ROM.
  fig10_enkf_qstar       -- innovation consistency chi2/N vs additive q for the
                            imperfect model and the perfect-model twin; the
                            chi2/N = 1 crossing defines q*.
  fig11_enkf_robustness  -- q* is a model property: chi2/N(q) collapses across
                            observation noise (a) and sensor stride (b); the
                            price of sparse sensing is analysis error (c).
  fig12_enkf_cadence_bursts -- per-cycle chi2/N of the 443-cycle long run
                            showing intermittent inconsistency bursts (a) and
                            the raw-plane K=16 cluster map with faces coloured
                            by per-cluster mean chi2/N, mapping that error onto
                            the near-wall cycle's event types (b).
  fig13_enkf_cluster_chi2 -- per-cycle chi2/N of the full-test-window run
                            (resolved_full/long9000, 1990 cycles, all 16
                            clusters occupied) conditioned on the K=16 cluster
                            the truth occupies at analysis time: model error is
                            phase-dependent (~4x across clusters), tracking
                            spanwise-shear (QSV) activity.

Inputs: sweep_a8/summary.json (from aggregate_mfu_na_enkf_sweep.py) plus the
ensemble_forecast.npz bundles of the showcase and long runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

SHARED_ROOT = Path("/users/sbrw610/sharedscratch/RBF_ROM")
ENKF = SHARED_ROOT / "results/MFU_NA/EnKF"
COORDS = SHARED_ROOT / "results/MFU_NA/pod/coords_do45.npz"
OUT_DIR = SHARED_ROOT / "results/MFU_NA/paper_images"

TPLUS_PER_S = 11.289   # wall-unit conversion: 1 s = 11.289 t+

# Okabe-Ito roles (paper palette): truth black, RBF-ROM vermillion.
C_TRUTH = "black"
C_ROM = "#D55E00"
C_BLUE = "#0072B2"
C_GREEN = "#009E73"
C_PURPLE = "#CC79A7"
C_ORANGE = "#E69F00"
C_MEMBER = "#9ecae1"   # light blue ensemble members (matches enkf_ensemble_modes)
C_MEAN = "#08519c"     # dark blue EnKF mean

LABEL_FS, TICK_FS, LEG_FS = 13, 10, 10


def style() -> bool:
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
    return use_tex


def save(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
    print(f"[done] wrote {out_dir / stem}.pdf", flush=True)


def qgrid(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    rows = sorted(rows, key=lambda r: r["additive_q"])
    return (np.array([r["additive_q"] for r in rows]),
            np.array([r["chi2_N"] for r in rows]))


def fig_modes(out_dir: Path, n_modes: int) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    z = np.load(ENKF / "alpha8.0/ensemble_forecast.npz", allow_pickle=True)
    traj = np.asarray(z["traj"], np.float64)             # (n_ens, nT, d_o)
    dt, stride, base_idx = float(z["dt"]), int(z["stride"]), int(z["base_idx"])
    n_ens, nT, _ = traj.shape
    t = np.arange(nT) * dt * TPLUS_PER_S

    with np.load(COORDS) as zc:
        A = np.asarray(zc["alpha"], np.float64)[::stride]
    truth = A[base_idx: base_idx + nT]
    t_truth = np.arange(truth.shape[0]) * dt * TPLUS_PER_S

    zf = np.load(ENKF / "alpha8.0/freerun/ensemble_forecast.npz", allow_pickle=True)
    free = np.nanmean(np.asarray(zf["traj"], np.float64), axis=0)
    t_free = np.arange(free.shape[0]) * dt * TPLUS_PER_S

    t_obs = np.asarray(z["reset_steps"], np.int64) * dt * TPLUS_PER_S
    mean = np.nanmean(traj, axis=0)

    ncol = 2
    nrow = int(np.ceil(n_modes / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 1.9 * nrow + 0.7),
                             sharex=True, squeeze=False)
    axes = axes.ravel()
    for i in range(n_modes):
        ax = axes[i]
        for tr in t_obs[1:]:
            ax.axvline(tr, color="0.75", lw=0.6, ls=":", zorder=0)
        ax.plot(t, traj[:, :, i].T, color=C_MEMBER, lw=0.7, alpha=0.5, zorder=1)
        ax.plot(t_free, free[:, i], color=C_ROM, lw=1.6, zorder=2)
        ax.plot(t, mean[:, i], color=C_MEAN, lw=2.0, zorder=3)
        ax.plot(t_truth, truth[:, i], color=C_TRUTH, lw=1.6, zorder=4)
        ax.set_ylabel(rf"$a_{{{i + 1}}}$", fontsize=LABEL_FS)
        ax.grid(True, ls=":", alpha=0.5)
        ax.tick_params(labelsize=TICK_FS)
        ax.set_xlim(t[0], t[-1])
    for ax in axes[n_modes - ncol: n_modes]:
        ax.set_xlabel(r"$t^+$", fontsize=LABEL_FS)

    handles = [Line2D([], [], color=C_MEMBER, lw=1.5, alpha=0.7,
                      label=f"members ($N_e={n_ens}$)"),
               Line2D([], [], color=C_MEAN, lw=2.0, label="EnKF mean"),
               Line2D([], [], color=C_ROM, lw=1.6, label="RBF-ROM (free run)"),
               Line2D([], [], color=C_TRUTH, lw=1.6, label="FOM")]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               fontsize=LEG_FS + 1, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out_dir, "fig9_enkf_modes")
    plt.close(fig)


def fig_qstar(runs: dict, qstar: dict, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    q_imp, chi_imp = qgrid(list(runs["qstar"].values()))
    twin = [r for r in runs["cadence_twin"].values() if r["cadence_tplus"] == 5.0]
    q_tw, chi_tw = qgrid(twin)
    qs = qstar["qstar"]["base"]["qstar"]

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.axhline(1.0, color="0.4", ls="--", lw=1.0, zorder=1)
    ax.semilogy(q_imp, chi_imp, "o-", color=C_ROM, lw=1.8, ms=5,
                label="imperfect model (FOM truth)", zorder=3)
    ax.semilogy(q_tw, chi_tw, "s-", color=C_TRUTH, lw=1.8, ms=5,
                label="perfect-model twin", zorder=3)
    ax.axvline(qs, color=C_ROM, ls=":", lw=1.2, zorder=2)
    ax.annotate(rf"$q^\ast = {qs:.2f}$", xy=(qs, 1.0), xytext=(qs + 0.02, 2.6),
                fontsize=LABEL_FS, color=C_ROM)
    ax.set_xlabel(r"additive inflation $q$", fontsize=LABEL_FS)
    ax.set_ylabel(r"$\chi^2 / N_{\mathrm{obs}}$", fontsize=LABEL_FS)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.tick_params(labelsize=TICK_FS)
    ax.legend(fontsize=LEG_FS, frameon=False)
    fig.tight_layout()
    save(fig, out_dir, "fig10_enkf_qstar")
    plt.close(fig)


def fig_robustness(runs: dict, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7))

    # (a) chi2/N(q) across observation noise: sigma group + base sigma=0.1.
    ax = axes[0]
    fams = {0.1: list(runs["qstar"].values())}
    for r in runs["sigma"].values():
        fams.setdefault(r["sigma_obs"], []).append(r)
    colours = {0.05: C_BLUE, 0.1: C_ROM, 0.2: C_GREEN, 0.4: C_PURPLE}
    for sig in sorted(fams):
        q, chi = qgrid(fams[sig])
        ax.semilogy(q, chi, "o-", color=colours[sig], lw=1.6, ms=4,
                    label=rf"$\sigma_{{\mathrm{{obs}}}}={sig:g}$")
    ax.set_title("(a)", fontsize=LABEL_FS)
    ax.set_ylabel(r"$\chi^2 / N_{\mathrm{obs}}$", fontsize=LABEL_FS)

    # (b) chi2/N(q) across sensor stride: stride_q group + base stride=8.
    ax = axes[1]
    fams = {8: list(runs["qstar"].values())}
    for r in runs["stride_q"].values():
        fams.setdefault(r["stride_x"], []).append(r)
    colours = {4: C_BLUE, 8: C_ROM, 16: C_GREEN, 32: C_PURPLE}
    for st in sorted(fams):
        q, chi = qgrid(fams[st])
        rank = fams[st][0]["rank_H"]
        ax.semilogy(q, chi, "o-", color=colours[st], lw=1.6, ms=4,
                    label=rf"$\Delta={st}$ (rank {rank})")
    ax.set_title("(b)", fontsize=LABEL_FS)

    for ax in axes[:2]:
        ax.axhline(1.0, color="0.4", ls="--", lw=1.0)
        ax.set_xlabel(r"additive inflation $q$", fontsize=LABEL_FS)
        ax.legend(fontsize=LEG_FS - 1, frameon=False)

    # (c) analysis error vs sensor count at fixed q: obs group, tau_x / tau_y / both.
    ax = axes[2]
    comp_style = {"both": (C_ROM, "o", "both components"),
                  "x": (C_BLUE, "s", r"$\tau_x$ only"),
                  "y": (C_GREEN, "^", r"$\tau_y$ only")}
    for comp, (col, mk, lab) in comp_style.items():
        rows = sorted((r for tag, r in runs["obs"].items()
                       if tag.endswith(f"_{comp}")), key=lambda r: r["n_obs"])
        n = [r["n_obs"] for r in rows]
        e = [r["err_norm"] for r in rows]
        ax.loglog(n, e, mk + "-", color=col, lw=1.6, ms=5, label=lab)
    ax.set_title("(c)", fontsize=LABEL_FS)
    ax.set_xlabel(r"$N_{\mathrm{obs}}$", fontsize=LABEL_FS)
    ax.set_ylabel(r"analysis error / climatology", fontsize=LABEL_FS)
    ax.legend(fontsize=LEG_FS - 1, frameon=False)

    for ax in axes:
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.tick_params(labelsize=TICK_FS)
    fig.tight_layout()
    save(fig, out_dir, "fig11_enkf_robustness")
    plt.close(fig)


def fig_cadence_bursts(runs: dict, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.cm as mcm
    from plot_mfu_na_phase_cycle_chi2 import draw_phase_cycle_chi2

    label_fs, tick_fs = 24, 20

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2),
                             gridspec_kw=dict(width_ratios=[1.25, 1.0]))

    # (a) per-cycle chi2/N of the 443-cycle long run: bursts above the mean level.
    ax = axes[0]
    z = np.load(ENKF / "sweep_a8/resolved/long2000/ensemble_forecast.npz",
                allow_pickle=True)
    chi = np.asarray(z["diag_chi2"], np.float64) / int(z["n_obs"])
    steps = np.asarray(z["reset_steps"], np.int64)
    steps = steps[-chi.size:]                            # cycles with an innovation
    t = steps * float(z["dt"]) * TPLUS_PER_S
    ax.semilogy(t, chi, color=C_ROM, lw=1.0)
    ax.axhline(1.0, color="0.4", ls="--", lw=1.0)
    ax.axhline(chi.mean(), color=C_TRUTH, ls=":", lw=1.2)
    ax.set_xlabel(r"$t^+$", fontsize=label_fs)
    ax.set_ylabel(r"$\chi^2 / N_{\mathrm{obs}}$ per cycle", fontsize=label_fs)
    ax.set_title("(a)", fontsize=label_fs)
    ax.set_xlim(t[0], t[-1])
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.tick_params(labelsize=tick_fs)

    # (b) raw-plane K=16 cluster map, faces coloured by per-cluster mean chi2/N:
    #     the same statistic as (a) mapped onto the near-wall cycle's event types.
    ax = axes[1]
    norm, cmap = draw_phase_cycle_chi2(ax, label_fs=label_fs, tick_fs=tick_fs,
                                       label_clusters=False, mark_unvisited=False,
                                       equal_size=True)
    ax.set_title("(b)", fontsize=label_fs)
    cb = fig.colorbar(mcm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                      fraction=0.046, pad=0.03)
    cb.set_label(r"$\langle\chi^2 / N_{\mathrm{obs}}\rangle$ per cycle",
                 fontsize=label_fs)
    cb.ax.tick_params(labelsize=tick_fs)

    fig.tight_layout()
    save(fig, out_dir, "fig12_enkf_cadence_bursts")
    plt.close(fig)


def fig_cluster_chi2(out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    z = np.load(ENKF / "sweep_a8/resolved_full/long9000/ensemble_forecast.npz",
                allow_pickle=True)
    chi = np.asarray(z["diag_chi2"], np.float64) / int(z["n_obs"])
    steps = np.asarray(z["reset_steps"], np.int64)[-chi.size:]
    stride, base_idx = int(z["stride"]), int(z["base_idx"])

    with np.load(COORDS) as zc:
        A = np.asarray(zc["alpha"], np.float64)[::stride]
    truth = A[base_idx + steps]                          # (n_cycles, d_o)

    cent = np.asarray(np.load(SHARED_ROOT /
                              "results/MFU_NA/cluster_sweep/do45/K=16.npz")
                      ["centroids"], np.float64)
    lab = np.argmin(((truth[:, None, :] - cent[None]) ** 2).sum(-1), axis=1)

    ks = np.array(sorted(k for k in range(cent.shape[0]) if (lab == k).any()))
    mean = np.array([chi[lab == k].mean() for k in ks])
    sem = np.array([chi[lab == k].std(ddof=1) / np.sqrt((lab == k).sum())
                    for k in ks])
    occ = np.array([(lab == k).mean() for k in ks])

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    x = np.arange(ks.size)
    ax.bar(x, mean, yerr=sem, color=C_ROM, alpha=0.85, width=0.7,
           error_kw=dict(ecolor="black", lw=1.0, capsize=2), zorder=2)
    ax.axhline(1.0, color="0.4", ls="--", lw=1.0, zorder=1)
    ax.axhline(chi.mean(), color=C_TRUTH, ls=":", lw=1.2, zorder=1)
    ax.text(-0.6, chi.mean(), "run mean", fontsize=TICK_FS,
            color=C_TRUTH, ha="left", va="bottom")
    ax.text(x[-1] + 0.45, 1.0, "consistent", fontsize=TICK_FS,
            color="0.35", ha="right", va="bottom")
    for xi, m, s, o in zip(x, mean, sem, occ):
        ax.text(xi, m + s + 0.15, rf"{100 * o:.0f}\%"
                if plt.rcParams["text.usetex"] else f"{100 * o:.0f}%",
                fontsize=TICK_FS - 1, ha="center", color="0.3")
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel(r"cluster $k$", fontsize=LABEL_FS)
    ax.set_ylabel(r"$\langle\chi^2 / N_{\mathrm{obs}}\rangle$ per cycle",
                  fontsize=LABEL_FS)
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    ax.tick_params(labelsize=TICK_FS)
    ax.set_xlim(-0.7, x[-1] + 0.7)
    fig.tight_layout()
    save(fig, out_dir, "fig13_enkf_cluster_chi2")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--summary", type=Path,
                   default=ENKF / "sweep_a8/summary.json")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--n-modes", type=int, default=4)
    args = p.parse_args()

    style()
    s = json.loads(args.summary.read_text())
    runs, qstar = s["runs"], s["qstar"]

    fig_modes(args.out_dir, args.n_modes)
    fig_qstar(runs, qstar, args.out_dir)
    fig_robustness(runs, args.out_dir)
    fig_cadence_bursts(runs, args.out_dir)
    fig_cluster_chi2(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
