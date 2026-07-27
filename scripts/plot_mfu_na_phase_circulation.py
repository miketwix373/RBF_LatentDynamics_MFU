"""Phase-plane circulation diagnostic for the MFU_NA near-wall cycle.

Rationale (fluid-dynamics-specialist, 2026-07-13): the discrete cluster-switching
matrix cannot expose a directed SSP loop -- box averaging destroys the spatial
phase and the ~10 t+ lift-up lag is far shorter than cluster residence, so the
quadrant circulation is ~0 (detailed balance). The directed information survives
only in the CONTINUOUS signals. This diagnostic tests the loop hypothesis where
it actually lives: form the phase angle of the (streak, QSV) trajectory and
measure its net winding.

  A_x(t)   = sqrt( alpha_t^T (Vx^T Vx) alpha_t / Nxy ) / tau_w   (streak)
  sig_y(t) = sqrt( alpha_t^T (Vy^T Vy) alpha_t / Nxy ) / tau_w   (QSV footprint)
High-pass each (moving-average, default ~ one SSP period), standardise, then
  theta(t) = atan2( sig_y_hp, A_x_hp )
A genuine limit cycle gives a sign-definite mean angular velocity omega =
<d theta / dt>; jitter gives zero. Reported scalars:
  omega           mean angular velocity (rad / t+), period = 2 pi / |omega|
  D               directedness = mean(d theta) / mean(|d theta|)  in [-1, 1]
  A_dot           enclosed signed-area rate (1/2T) oint (x dy - y dx)
Significance from circular-shift surrogates that keep each spectrum but destroy
the cross-phase (null D distribution). Sign convention: omega < 0 (clockwise in
the A_x-sig_y plane) corresponds to sig_y leading A_x, i.e. vortex -> streak
(lift-up), matching the lead-lag.

Wall time t+ = t u_tau^2 / nu, u_tau = 180/2870, nu = 1/2870 => 0.5645 t+/frame
at dt = 0.05. Training coords are one continuous uniform segment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_mfu_na_cluster_jacobians import (  # noqa: E402
    _load_model, _phi_and_grad, TPLUS_PER_S as JAC_TPLUS_PER_S)

SHARED_ROOT = (next(_p for _p in Path(__file__).resolve().parents if (_p / "pyproject.toml").exists()))
CSZ = 64 * 64
U_TAU = 180.0 / 2870.0
TAU_W = U_TAU ** 2
NU = 1.0 / 2870.0
TPLUS_PER_T = U_TAU ** 2 / NU
LABEL_FS, TICK_FS, ANN_FS = 22, 18, 20
STREAM_C = "#E8000B"


NGRID = 64


def _signals(alpha, V):
    Vx, Vy = V[:CSZ], V[CSZ:]
    Gx, Gy = Vx.T @ Vx, Vy.T @ Vy
    ax = np.sqrt(np.einsum("tm,mn,tn->t", alpha, Gx, alpha) / CSZ) / TAU_W
    sy = np.sqrt(np.einsum("tm,mn,tn->t", alpha, Gy, alpha) / CSZ) / TAU_W
    return ax, sy


def _streak_amp(alpha, V, kz=1, chunk=5000):
    # spanwise-coherent low-speed-streak amplitude: the k_x=0 (streamwise-
    # averaged), fundamental-k_z Fourier amplitude of the reconstructed tau_x'.
    # Unlike the total sigma_x it is a strict subset of the streamwise energy,
    # so it breaks the A_x^2+sigma_y^2=E^2 identity and lets a genuine
    # streak<->QSV anti-correlation appear (fluid-dynamics-specialist,
    # 2026-07-17). Snapshot reshapes C-order to (nz, nx): axis 2 is streamwise
    # and axis 1 spanwise -- verified empirically (streamwise-averaging
    # preserves streak variance 5:1; spanwise |FFT| peaks at k_z=1).
    Vx = V[:CSZ]
    out = np.empty(len(alpha))
    for i0 in range(0, len(alpha), chunk):
        A = alpha[i0:i0 + chunk]
        tx = (A @ Vx.T).reshape(-1, NGRID, NGRID)          # (T, nz, nx)
        txm = tx.mean(axis=2)                              # k_x=0 streamwise avg
        out[i0:i0 + len(A)] = np.abs(
            np.fft.rfft(txm, axis=1)[:, kz]) / NGRID / TAU_W
    return out


def _highpass(x, w):
    if w <= 1:
        return x - x.mean()
    k = np.ones(w) / w
    return x - np.convolve(x, k, mode="same")


def _directedness(x, y):
    th = np.unwrap(np.arctan2(y, x))
    d = np.diff(th)
    return th, d.mean() / np.abs(d).mean(), d


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--coords", type=Path,
                   default=SHARED_ROOT / "results/MFU_NA/pod/coords_do45.npz")
    p.add_argument("--detrend-tplus", type=float, default=100.0)
    p.add_argument("--nsurr", type=int, default=400)
    p.add_argument("--field", choices=["stream", "quiver"], default="stream",
                   help="phase-velocity render: streamlines or normalised arrows")
    p.add_argument("--labels", type=Path, default=SHARED_ROOT /
                   "results/MFU_NA/cluster_sweep/do45/K=16.npz",
                   help="K-means labels for the cluster<->cycle link, panel (c)")
    p.add_argument("--model", type=Path, default=SHARED_ROOT /
                   "results/MFU_NA/do45/alpha3.5/model_n04800.npz",
                   help="fitted RBF field, linearised per cluster for panel (c)")
    p.add_argument("--out", type=Path, default=SHARED_ROOT /
                   "results/MFU_NA/paper_images/phase_circulation_K16.pdf")
    args = p.parse_args()

    import shutil
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
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
    x = _highpass(streak_sig, w); y = _highpass(sy_sig, w)
    x = (x - x.mean()) / x.std(); y = (y - y.mean()) / y.std()

    th, D, dth = _directedness(x, y)
    dur_tp = (len(x) - 1) * tpf
    omega = (th[-1] - th[0]) / dur_tp                    # rad / t+
    period = 2 * np.pi / abs(omega) if omega != 0 else np.inf
    A_dot = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) / dur_tp

    rng = np.random.default_rng(0)
    Dn = np.empty(args.nsurr)
    for s in range(args.nsurr):
        L = int(rng.integers(w, len(y) - w))
        _, Dn[s], _ = _directedness(x, np.roll(y, L))
    pval = float((np.abs(Dn) >= abs(D)).mean())
    sense = "clockwise" if omega < 0 else "counter-clockwise"
    limb = "sig_y leads A_x (vortex->streak, lift-up)" if omega < 0 \
        else "A_x leads sig_y (streak->vortex, regen.)"
    print(f"[phase] dt={dt:.3f} ({tpf:.3f} t+/frame), detrend w={w} "
          f"({args.detrend_tplus:.0f} t+), N={len(x)}")
    print(f"[phase] omega = {omega:+.4e} rad/t+  => period {period:.0f} t+  ({sense})")
    print(f"[phase] directedness D = {D:+.3f}   (surrogate |D| mean {np.abs(Dn).mean():.3f}, "
          f"p = {pval:.3f})")
    print(f"[phase] enclosed-area rate A_dot = {A_dot:+.3e} /t+")
    print(f"[phase] sense: {limb}")

    # cluster <-> cycle link: per-cluster residence vs local phase speed. The
    # K-means labels index the coords at a coarser stride than the continuous
    # signal, so align by that stride.
    clab = np.asarray(np.load(args.labels)["labels"], np.int64)
    step = len(x) // len(clab)
    Kc = int(clab.max()) + 1
    dthL = dth[::step][:len(clab)]
    syL = sy_sig[::step][:len(clab)]
    dt_lab = step * tpf
    runs = []
    i = 0
    while i < len(clab):
        j = i
        while j + 1 < len(clab) and clab[j + 1] == clab[i]:
            j += 1
        runs.append((int(clab[i]), (j - i + 1) * dt_lab))
        i = j + 1
    spd_k = np.array([np.abs(dthL[clab == k]).mean() / tpf for k in range(Kc)])
    dwell_k = np.array([np.mean([d for q, d in runs if q == k])
                        for k in range(Kc)])
    sy_k = np.array([syL[clab == k].mean() for k in range(Kc)])
    rlink = float(np.corrcoef(spd_k, dwell_k)[0, 1])
    print(f"[phase] cluster dwell vs local phase-speed corr = {rlink:+.3f}")

    # per-cluster phase coherence R = |<exp(i theta)>| on the cycle: high R =
    # cluster parked at a definite phase ("cycle" cluster), low R = phase-blind
    # ("burst" cluster). A continuum of roles, not two families.
    thwL = np.arctan2(y, x)[::step][:len(clab)]
    R_k = np.array([np.abs(np.mean(np.exp(1j * thwL[clab == k])))
                    for k in range(Kc)])
    print(f"[phase] cluster phase-coherence R in [{R_k.min():.2f}, "
          f"{R_k.max():.2f}], mean {R_k.mean():.2f}")

    # cluster-role plane. The streak leg is the spanwise-coherent streak
    # amplitude A_streak (k_x=0, k_z=1 of tau_x'), NOT the total sigma_x used in
    # panel (a): the total obeys A_x^2+sigma_y^2=E^2 exactly, forcing every
    # centroid onto the activity diagonal (Q1/Q3). A_streak is a strict subset
    # of the streamwise energy, so the identity is broken and the streak<->QSV
    # decoupling can put streak-collapsed/QSV-elevated clusters in Q2.
    axr = (streak_sig - streak_sig.mean()) / streak_sig.std()
    syr = (sy_sig - sy_sig.mean()) / sy_sig.std()
    aL2 = axr[::step][:len(clab)]; sL2 = syr[::step][:len(clab)]
    cx_k = np.array([aL2[clab == k].mean() for k in range(Kc)])
    cy_k = np.array([sL2[clab == k].mean() for k in range(Kc)])
    occ_k = np.array([(clab == k).mean() for k in range(Kc)])
    q2 = [k for k in range(Kc) if cx_k[k] < 0 and cy_k[k] > 0]
    q4 = [k for k in range(Kc) if cx_k[k] > 0 and cy_k[k] < 0]
    print(f"[phase] coherent-streak plane: centroid corr(A_streak,sig_y) = "
          f"{np.corrcoef(cx_k, cy_k)[0, 1]:+.3f}; Q2 (streak-,QSV+) clusters "
          f"{q2}; Q4 (streak+,QSV-) clusters {q4}")
    C = np.zeros((Kc, Kc)); np.add.at(C, (clab[:-1], clab[1:]), 1.0)
    np.fill_diagonal(C, 0.0)
    Pr = C.sum(1, keepdims=True)
    Pij = np.divide(C, Pr, out=np.zeros_like(C), where=Pr > 0)
    cmax = float(C.max())

    # per-cluster LOCAL DYNAMICS: linearise the fitted global RBF field at each
    # cluster centroid (in the full d_o reduced space) and take its leading
    # growing oscillatory eigenpair. rom-specialist 2026-07-15: this is the
    # honest, model-native "the local dynamics differ" statement -- a 2x2 fit in
    # the (A_x, sigma_y) plane is biased to antisymmetry by A_x^2+sigma_y^2=E^2
    # and is partly circular (clusters are defined by partitioning the state).
    # The production ROM is one global field (K shapes widths only), so what
    # differs per cluster is J_k = d adot/da at the centroid: each cluster is a
    # distinct local oscillator. Analytic Jacobian FD-verified (rel.err 3e-9).
    cent_k = np.array([alpha[::step][:len(clab)][clab == k].mean(0)
                       for k in range(Kc)])
    mdl = _load_model(args.model)
    fq_k = np.zeros(Kc); gr_k = np.zeros(Kc)
    for k in range(Kc):
        _, Jk = _phi_and_grad(cent_k[k], mdl)
        lam = np.linalg.eigvals(Jk)
        osc = lam[np.abs(lam.imag) > 1e-9]
        lead = osc[np.argmax(osc.real)] if osc.size else lam[np.argmax(lam.real)]
        fq_k[k] = abs(lead.imag) / (2 * np.pi) / JAC_TPLUS_PER_S   # [1/t+]
        gr_k[k] = lead.real / JAC_TPLUS_PER_S                      # [1/t+]
    per_k = np.where(fq_k > 0, 1.0 / np.where(fq_k > 0, fq_k, np.nan), np.inf)
    print(f"[phase] per-cluster local growth Re(lam) in "
          f"[{gr_k.min():.3e}, {gr_k.max():.3e}] /t+")
    print(f"[phase] per-cluster local period in "
          f"[{per_k[np.isfinite(per_k)].min():.0f}, "
          f"{per_k[np.isfinite(per_k)].max():.0f}] t+")

    def draw_portrait(ax, xx, yy, filt, arc=True):
        # residence density (grey) + binned mean phase-velocity (current) field
        ax.hexbin(xx, yy, gridsize=40, cmap="Greys", bins="log", mincnt=1,
                  zorder=1)
        dxx, dyy = np.gradient(xx), np.gradient(yy)
        lim = 3.2
        nbp = 28 if args.field == "stream" else 15
        eg = np.linspace(-lim, lim, nbp + 1); ce = 0.5 * (eg[:-1] + eg[1:])
        mm = (np.abs(xx) < lim) & (np.abs(yy) < lim)
        jx, jy = np.digitize(xx[mm], eg) - 1, np.digitize(yy[mm], eg) - 1
        SX = np.zeros((nbp, nbp)); SY = np.zeros((nbp, nbp))
        CN = np.zeros((nbp, nbp))
        np.add.at(SX, (jy, jx), dxx[mm]); np.add.at(SY, (jy, jx), dyy[mm])
        np.add.at(CN, (jy, jx), 1.0)
        if args.field == "stream":
            from scipy.ndimage import gaussian_filter
            SXs, SYs = gaussian_filter(SX, 1.1), gaussian_filter(SY, 1.1)
            CNs = gaussian_filter(CN, 1.1); okk = CNs >= 8.0
            Uu = np.where(okk, SXs / np.maximum(CNs, 1e-9), np.nan)
            Vw = np.where(okk, SYs / np.maximum(CNs, 1e-9), np.nan)
            sp = np.hypot(Uu, Vw)
            lww = 0.8 + 2.4 * np.nan_to_num(sp / np.nanmax(sp))
            st = ax.streamplot(ce, ce, np.ma.masked_invalid(Uu),
                               np.ma.masked_invalid(Vw), color=STREAM_C,
                               density=1.5, linewidth=lww, arrowsize=1.3,
                               zorder=3)
            mp = None
        else:
            gd = CN >= 40; mg = np.hypot(SX, SY); nzz = gd & (mg > 0)
            Uu = np.where(nzz, SX / np.where(mg > 0, mg, 1), np.nan)
            Vw = np.where(nzz, SY / np.where(mg > 0, mg, 1), np.nan)
            sp = np.where(gd, mg / np.maximum(CN, 1), np.nan)
            GXp, GYp = np.meshgrid(ce, ce)
            mp = ax.quiver(GXp, GYp, Uu, Vw, sp, cmap="plasma", angles="xy",
                           scale_units="xy", scale=2.4, width=0.006,
                           pivot="mid", zorder=3)
        if args.field == "quiver":
            cbp = fig.colorbar(mp, ax=ax, fraction=0.046, pad=0.02)
            cbp.set_label(r"phase speed $|\dot\phi|$", fontsize=LABEL_FS - 8)
            cbp.ax.tick_params(labelsize=TICK_FS - 4)
        else:
            import matplotlib.cm as mcm
            sm = mcm.ScalarMappable(cmap="plasma"); sm.set_array([])
            fig.colorbar(sm, ax=ax, fraction=0.046,
                         pad=0.02).ax.set_visible(False)
        if arc:
            # net rotation sense (signed-area rate) as a perimeter arc
            omg = np.mean(xx[:-1] * yy[1:] - xx[1:] * yy[:-1])
            aa = np.linspace(0.25 * np.pi, 1.75 * np.pi, 40)
            if omg < 0:
                aa = aa[::-1]
            ax.plot(1.8 * np.cos(aa), 1.8 * np.sin(aa), color="#009E73",
                    lw=2.5, zorder=4)
            ax.annotate("", xy=(1.8 * np.cos(aa[-1]), 1.8 * np.sin(aa[-1])),
                        xytext=(1.8 * np.cos(aa[-3]), 1.8 * np.sin(aa[-3])),
                        arrowprops=dict(arrowstyle="-|>", color="#009E73",
                                        lw=2.5), zorder=4)
        ax.axhline(0, color="0.6", lw=1.0, ls=":")
        ax.axvline(0, color="0.6", lw=1.0, ls=":")
        ax.set_xlabel(rf"coherent streak $A_x$ ($k_z{{=}}1$, {filt})",
                      fontsize=LABEL_FS - 3)
        ax.set_ylabel(rf"total spanwise shear $\sigma_y$ ({filt})",
                      fontsize=LABEL_FS - 3)
        ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_aspect("equal")
        ax.tick_params(labelsize=TICK_FS)

    fig, (a1, a3) = plt.subplots(1, 2, figsize=(16.5, 7.4))

    # (a) combined phase portrait + quantization roles in the raw
    # (A_streak, sigma_y) plane. The axes are ASYMMETRIC BY DESIGN: x is the
    # coherent (k_z=1) streak amplitude, y is the TOTAL spanwise-shear activity
    # sigma_y. This asymmetry is deliberate -- it is what populates Q2 (coherent
    # streak collapsed but broadband spanwise activity high = breakdown, e.g.
    # clusters 11/12/15). A coherence-matched k_z=1 tau_y axis collapses every
    # centroid onto the Q1/Q3 activity diagonal (corr +0.96) because the
    # coherent streak and coherent vortex are lift-up-bound (fluid consult
    # 2026-07-17). Streamlines show amplitude breathing; the DIRECTED lift-up
    # loop lives only in the high-pass plane (D = -0.20, vs raw D ~ 0) and is a
    # caption statistic, not drawn. Cluster centroids overlaid (size =
    # occupancy); cluster 0 (far low-activity reservoir) omitted to focus the
    # box. Arrows = switching frequency between centroids.
    draw_portrait(a1, axr, syr, "std.", arc=False)
    keep = [k for k in range(Kc) if k != 0]
    occ_ref = occ_k[keep].max()
    for i in keep:
        for jj in keep:
            if i == jj or Pij[i, jj] < 0.12:
                continue
            wln = 1.0 + 5.0 * (C[i, jj] / cmax) ** 0.7
            a1.add_patch(FancyArrowPatch(
                (cx_k[i], cy_k[i]), (cx_k[jj], cy_k[jj]),
                connectionstyle="arc3,rad=0.16", arrowstyle="-|>",
                mutation_scale=11, lw=wln, color="0.35",
                alpha=0.3 + 0.5 * (C[i, jj] / cmax), zorder=4))
    a1.scatter(cx_k[keep], cy_k[keep],
               s=140 + 1500 * (occ_k[keep] / occ_ref),
               facecolors="white", edgecolors="k", linewidths=1.3, zorder=5)
    a1.set_xlim(-1.9, 1.9); a1.set_ylim(-1.6, 1.4)
    a1.text(-0.02, 1.02, "(a)", transform=a1.transAxes, fontsize=ANN_FS,
            fontweight="bold", va="bottom", ha="right")

    # (c) the local dynamics differ per cluster: the fitted global RBF field,
    # linearised at each cluster centroid, gives a distinct leading eigenpair.
    # Plot each cluster as a local oscillator -- instantaneous frequency f_k vs
    # growth rate Re(lam_k) -- coloured by phase coherence R (same key as (b))
    # and sized by occupancy. The wide spread (period 5-445 t+, growth 2-5x) is
    # the model-native statement that each cluster carries its own dynamics.
    sc3 = a3.scatter(fq_k, gr_k, s=140 + 1500 * (occ_k / occ_k.max()),
                     c=R_k, cmap="plasma", edgecolors="k", linewidths=1.1,
                     zorder=3)
    for k in range(Kc):
        a3.annotate(rf"${k}$", (fq_k[k], gr_k[k]), fontsize=ANN_FS - 10,
                    ha="center", va="center", zorder=4, color="w")
    cb3 = fig.colorbar(sc3, ax=a3, fraction=0.046, pad=0.02)
    cb3.set_label(r"phase coherence $R$", fontsize=LABEL_FS - 8)
    cb3.ax.tick_params(labelsize=TICK_FS - 4)
    a3.set_xlabel(r"local frequency $f_k$ ($1/t^+$)", fontsize=LABEL_FS - 3)
    a3.set_ylabel(r"local growth rate $\mathrm{Re}\,\lambda_k$ ($1/t^+$)",
                  fontsize=LABEL_FS - 3)
    a3.set_xlim(left=0.0)
    a3.margins(x=0.08, y=0.12)
    a3.tick_params(labelsize=TICK_FS)
    a3.text(-0.02, 1.02, "(b)", transform=a3.transAxes, fontsize=ANN_FS,
            fontweight="bold", va="bottom", ha="right")

    fig.tight_layout()
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
