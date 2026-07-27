"""Directed lift-up coupling in the fitted MFU_NA wall-shear RBF field.

Tests whether the learned vector field adot = f_RBF(a) (45 centred-POD
wall-shear coordinates) has internalised the vortex-drives-streak leg of the
self-sustaining process (Hamilton, Kim & Waleffe, JFM 287, 1995; lift-up:
Landahl, JFM 98, 1980). Empirical target from the July lead-lag study:
sigma_y (QSV footprint) LEADS A_x (coherent k_z=1 streak amplitude) by
~10 t+, SSP period ~120 t+.

Probes, per the fluid-dynamics-specialist assessment (2026-07-20):

1. Phase-conditioned observable-projected Jacobians. At ~2000 data states,
   J(a) = df/da (analytic, RBFField.f_and_J) is projected onto the per-state
   observable-gradient plane N = [grad A_x, grad sigma_y]:
     - grad A_x tracks the instantaneous spanwise streak phase
       (A_x = |c^T a|, c fixed complex k_z=1 coefficient vector);
     - TWO vortex observables are carried through in parallel: 'tot'
       (total spanwise-shear rms, the observable of the empirical lead-lag
       study — the apples-to-apples comparison) and 'inc' (the k_x=0,
       k_z=1 coherent component of tau_y projected out — decontaminated of
       streak meander, but the data show NO robust lead-lag for it:
       removing the 12% streak-locked coherent tau_y energy drops the
       data-side corr(sigma_y, A_x) peak from r=0.24 to r=0.05, so the
       "+10 t+ empirical lead" belongs to 'tot' only);
     - the linearised observable dynamics keeps the Hessian transport term:
       d(grad h . da)/dt = (grad h . J + f . H_h) da (ratio reported);
     - the 2x2 is the Gram-corrected oblique projection M2 = R N (N^T N)^-1,
       standardised by the high-passed observable scales (the lead angle,
       unlike the eigenvalues, is not rescaling-invariant), averaged per
       K=16 cluster at data states (not centroids), then occupancy-averaged.
   Interpretation frame: the empirical 10 t+ lead over a 120 t+ period
   (~30 deg) is a FORCED-RESPONSE lag, not a quarter-cycle oscillation.
   Lift-up as fast slaved forcing predicts g_ss < 0, g_sv > 0,
   |g_sv| > |g_vs|; the transfer function g_sv/(i w - g_ss) at
   w = 2 pi / 120 t+ gives the model's implied lead. A clean complex pair
   with ~90 deg lead would CONTRADICT the data.

2. Timescale in the full 45-d: complex eigenpairs of the per-cluster and
   occupancy-averaged mean J (caveat: an averaged J carries no Floquet
   guarantee; order-of-magnitude agreement with 120 t+ is the bar).

3. A-priori check with no linearisation: the model-implied coherent-streak
   growth rate dA_x/dt = grad A_x . f(a_t) along the full record,
   cross-correlated against each sigma_y variant, versus the finite-
   difference-from-data version of the same curve.

4. Gold standard: converged UPOs of this exact field at the SSP tier
   (T ~ 99-140 t+, results/MFU_NA/upo/trav_*) — the fundamental-harmonic
   phase lead of sigma_y over A_x along the orbit. Orbits whose observables
   barely modulate (pure travelling waves: A_x is spanwise-phase-invariant,
   so a drifting streak leaves it constant) are gated out — their harmonic
   phase is meaningless.

The prescribed static-mode-weight construction (w_v from tau_y energy
fractions e_y(m), w_s from coherent-streak content s(m), top quartiles) is
reported alongside, magnitudes-first: POD mode signs are arbitrary, so its
signed gains and eigenvalues are convention-dependent (flipping one mode's
sign flips cross terms while the sign-blind weights do not respond).

Note: J is the bare RBF field; the additive trust-region corrector is
inactive at interior data states (hinge off), so its absence is immaterial
under the coverage gate.

Outputs under results/MFU_NA/field_analysis/liftup/:
liftup_matrix.png, liftup_prescribed_weights.png, liftup_matrix.npz,
summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from find_mfu_na_upo import RBFField  # noqa: E402
from plot_mfu_na_phase_circulation import (  # noqa: E402
    _highpass, _signals, _streak_amp, CSZ, NGRID, TAU_W)

SHARED_ROOT = (next(_p for _p in Path(__file__).resolve().parents if (_p / "pyproject.toml").exists()))
MFU = SHARED_ROOT / "results/MFU_NA"
TPLUS_PER_S = 11.289
DT_STRIDED = 0.1                       # s per strided frame
TPF = DT_STRIDED * TPLUS_PER_S         # t+ per strided frame
LEAD_EMP, PERIOD_EMP = 10.0, 120.0     # t+ (July lead-lag study, 'tot')
# Okabe-Ito, no yellow
C_SV, C_VS, C_FOM, C_UPO, C_AUX = ("#D55E00", "#0072B2", "black",
                                   "#CC79A7", "#009E73")
LABEL_FS, TICK_FS = 13, 11


def coherent_kz1_part(Vhalf: np.ndarray) -> np.ndarray:
    """(k_x=0, k_z=1) component of each column of a (4096, m) block.

    Mirrors _streak_amp's convention: C-order reshape to (nz, nx), axis 1
    streamwise (averaged), axis 0 spanwise (FFT)."""
    n = Vhalf.shape[1]
    out = np.empty_like(Vhalf)
    z = np.arange(NGRID)
    for m in range(n):
        f2 = Vhalf[:, m].reshape(NGRID, NGRID)
        X1 = np.fft.rfft(f2.mean(axis=1))[1]
        coh_z = 2.0 * np.real(X1 * np.exp(2j * np.pi * z / NGRID)) / NGRID
        out[:, m] = np.repeat(coh_z, NGRID)
    return out


def streak_coeff_vector(Vx: np.ndarray) -> np.ndarray:
    """Complex c with A_x(a) = |c^T a|, matching _streak_amp exactly."""
    c = np.empty(Vx.shape[1], complex)
    for m in range(Vx.shape[1]):
        txm = Vx[:, m].reshape(NGRID, NGRID).mean(axis=1)
        c[m] = np.fft.rfft(txm)[1]
    return c / NGRID / TAU_W


def quad_observable(A: np.ndarray, M: np.ndarray):
    """h = sqrt(a^T M a), grad h = M a / h, batched. Returns h (b,), grad (b,d)."""
    G = A @ M
    h = np.sqrt(np.einsum("bd,bd->b", G, A))
    return h, G / h[:, None]


def fourier_lead(x: np.ndarray, y: np.ndarray, T_tplus: float) -> float:
    """Lead of y over x (t+) from the fundamental harmonic of periodic series."""
    w = 2 * np.pi / T_tplus
    phx, phy = np.angle(np.fft.rfft(x - x.mean())[1]), \
        np.angle(np.fft.rfft(y - y.mean())[1])
    dphi = (phy - phx + np.pi) % (2 * np.pi) - np.pi
    return dphi / w


def h1_frac(x: np.ndarray) -> tuple[float, float]:
    """Fundamental-harmonic variance fraction and modulation depth std/mean."""
    E = np.abs(np.fft.rfft(x - x.mean()))**2
    return float(E[1] / E.sum()), float(x.std() / x.mean())


def xcorr(x: np.ndarray, y: np.ndarray, max_lag: int):
    """c(l) = corr(x(t), y(t+l)) for l in [-max_lag, max_lag]."""
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    lags = np.arange(-max_lag, max_lag + 1)
    c = np.array([np.mean(x[:len(x) - l] * y[l:]) if l >= 0
                  else np.mean(x[-l:] * y[:len(y) + l]) for l in lags])
    return lags, c


def observable_rows(As, fs, Js, M):
    """Per-state row (grad h . J + f . H_h) and gradient for h = sqrt(a^T M a).

    Returns h (b,), grad (b,d), row (b,d), hess/J row-norm ratio (b,)."""
    h, g = quad_observable(As, M)
    row_J = np.einsum("bd,bde->be", g, Js)
    fM = fs @ M
    row_H = (fM - np.einsum("bd,bd->b", fs, g)[:, None] * g) / h[:, None]
    ratio = np.linalg.norm(row_H, axis=1) / np.linalg.norm(row_J, axis=1)
    return h, g, row_J + row_H, ratio


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--coords", type=Path, default=MFU / "pod/coords_do45.npz")
    p.add_argument("--model", type=Path,
                   default=MFU / "do45/alpha8.0/model_n00960.npz")
    p.add_argument("--clusters", type=Path,
                   default=MFU / "cluster_sweep/do45/K=16.npz")
    p.add_argument("--upo-root", type=Path, default=MFU / "upo")
    p.add_argument("--n-states", type=int, default=2000)
    p.add_argument("--detrend-tplus", type=float, default=100.0)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--h1-gate", type=float, default=0.3,
                   help="min fundamental-harmonic variance fraction for a "
                        "UPO observable phase to count")
    p.add_argument("--out", type=Path, default=MFU / "field_analysis/liftup")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    with np.load(args.coords) as z:
        A = np.asarray(z["alpha"], np.float64)[::2]
        V = np.asarray(z["V"], np.float64)
    with np.load(args.clusters) as z:
        labels = np.asarray(z["labels"])
    K = 16
    field = RBFField(args.model)
    Vx, Vy = V[:CSZ], V[CSZ:]
    nT, d = A.shape

    # ---- observable machinery -------------------------------------------
    c = streak_coeff_vector(Vx)
    Ms = np.outer(c.real, c.real) + np.outer(c.imag, c.imag)   # A_x^2 = a^T Ms a
    Mv_tot = Vy.T @ Vy / (CSZ * TAU_W**2)
    Vy_inc = Vy - coherent_kz1_part(Vy)
    Mv_inc = Vy_inc.T @ Vy_inc / (CSZ * TAU_W**2)
    frac_coh_y = 1.0 - np.trace(Mv_inc) / np.trace(Mv_tot)
    variants = {"tot": Mv_tot, "inc": Mv_inc}

    # sanity: c reproduces _streak_amp; analytic J reproduces FD
    assert np.allclose(np.abs(A[:50] @ c), _streak_amp(A[:50], V), rtol=1e-10)
    a0, eps = A[1000], 1e-5
    _, J0 = field.f_and_J(a0[None])
    e7 = np.zeros(d)
    e7[7] = eps
    fd = (field.f((a0 + e7)[None]) - field.f((a0 - e7)[None]))[0] / (2 * eps)
    print(f"[sanity] J vs FD col-7 rel err "
          f"{np.abs(J0[0][:, 7] - fd).max() / np.abs(fd).max():.1e}")

    # ---- record-level series --------------------------------------------
    st_sig = np.abs(A @ c)
    sy = {v: np.sqrt(np.einsum("bd,bd->b", A @ Mv, A))
          for v, Mv in variants.items()}
    w = max(1, int(round(args.detrend_tplus / TPF)))
    x_hp = _highpass(st_sig, w)
    y_hp = {v: _highpass(sy[v], w) for v in variants}
    # cycle ordering from the 'tot' plane (July cluster-role convention)
    xs = x_hp / x_hp.std()
    ys = y_hp["tot"] / y_hp["tot"].std()
    theta_k = np.array([np.arctan2(ys[labels == j].mean(),
                                   xs[labels == j].mean()) for j in range(K)])
    cyc_order = np.argsort(-theta_k)   # clockwise = sig_y leads A_x

    # ---- Jacobians at data states ---------------------------------------
    idx = np.arange(0, nT, max(1, nT // args.n_states))
    cov = np.concatenate([field.coverage(A[idx[i:i + 500]])
                          for i in range(0, len(idx), 500)])
    gate = cov >= np.percentile(cov, 1.0)
    idx = idx[gate]
    print(f"[states] {len(idx)} gated states (dropped {np.sum(~gate)} "
          f"below coverage p1)")
    lab_s = labels[idx]
    occ = np.bincount(lab_s, minlength=K) / len(lab_s)

    fs = np.empty((len(idx), d))
    Js = np.empty((len(idx), d, d))
    for i in range(0, len(idx), 250):
        fs[i:i + 250], Js[i:i + 250] = field.f_and_J(A[idx[i:i + 250]])
    As = A[idx]
    J_k = np.stack([Js[lab_s == j].mean(0) for j in range(K)])
    J_glob = Js.mean(0)

    # ---- per-variant oblique observable-projected 2x2 -------------------
    _, gs, row_s, ratio_s = observable_rows(As, fs, Js, Ms)
    res: dict[str, dict] = {}
    for v, Mv in variants.items():
        _, gv, row_v, ratio_v = observable_rows(As, fs, Js, Mv)
        N = np.stack([gs, gv], axis=2)                    # (b, d, 2)
        R = np.stack([row_s, row_v], axis=1)              # (b, 2, d)
        Gram = np.einsum("bdi,bdj->bij", N, N)
        RN = np.einsum("bid,bdj->bij", R, N)
        M2 = np.transpose(np.linalg.solve(Gram, np.transpose(RN, (0, 2, 1))),
                          (0, 2, 1))                      # R N Gram^-1
        D = np.diag([x_hp.std(), y_hp[v].std()])
        M2_std = np.einsum("ij,bjk,kl->bil", np.linalg.inv(D), M2,
                           D) / TPLUS_PER_S               # 1/t+
        M2_k = np.stack([M2_std[lab_s == j].mean(0) for j in range(K)])
        M2_se = np.empty((K, 2, 2))
        for j in range(K):
            mem = np.where(lab_s == j)[0]
            boot = np.stack([M2_std[rng.choice(mem, len(mem))].mean(0)
                             for _ in range(args.n_boot)])
            M2_se[j] = boot.std(0)
        M2_cyc = np.einsum("k,kij->ij", occ, M2_k)
        g_ss, g_sv = M2_cyc[0]
        g_vs, g_vv = M2_cyc[1]
        w_ssp = 2 * np.pi / PERIOD_EMP
        H_tf = g_sv / (1j * w_ssp - g_ss)
        lam2, vec2 = np.linalg.eig(M2_cyc)
        if np.abs(lam2.imag).max() > 1e-12:
            i2 = int(np.argmax(lam2.imag))
            period_2x2 = 2 * np.pi / lam2[i2].imag
            v2 = vec2[:, i2]
            dphi = (np.angle(v2[1]) - np.angle(v2[0]) + np.pi) \
                % (2 * np.pi) - np.pi
            lead_2x2 = dphi / lam2[i2].imag
        else:
            period_2x2, lead_2x2 = np.inf, np.nan
        n_v_bar = gv.mean(0)
        res[v] = dict(M2_k=M2_k, M2_se=M2_se, M2_cyc=M2_cyc,
                      eig_k=np.linalg.eigvals(M2_k),
                      hess_ratio_v=ratio_v, H_tf=H_tf,
                      lead_forced=-np.angle(H_tf) / w_ssp,
                      tau_relax=1.0 / abs(g_ss),
                      period_2x2=period_2x2, lead_2x2=lead_2x2,
                      n_v_bar=n_v_bar / np.linalg.norm(n_v_bar))
    n_s_bar = gs.mean(0)
    n_s_bar /= np.linalg.norm(n_s_bar)

    # ---- prescribed static-weight construction --------------------------
    e_y = (Vy**2).sum(0)
    s_m = np.array([2 * np.abs(np.fft.rfft(
        Vx[:, m].reshape(NGRID, NGRID).mean(1))[1])**2 for m in range(d)])
    w_v = np.where(e_y >= np.quantile(e_y, 0.75), e_y, 0.0)
    w_s = np.where(s_m >= np.quantile(s_m, 0.75), s_m, 0.0)
    w_v /= np.linalg.norm(w_v)
    w_s /= np.linalg.norm(w_s)
    gA_sv = np.einsum("i,kij,j->k", w_s, J_k, w_v) / TPLUS_PER_S
    gA_vs = np.einsum("i,kij,j->k", w_v, J_k, w_s) / TPLUS_PER_S

    # ---- full-45d eigenpairs --------------------------------------------
    def complex_pairs(J):
        lam, vec = np.linalg.eig(J / TPLUS_PER_S)   # 1/t+
        sel = np.where(lam.imag > 1e-9)[0]
        per = 2 * np.pi / lam.imag[sel]
        leads = {v: np.array([
            ((np.angle(res[v]["n_v_bar"] @ vec[:, i])
              - np.angle(n_s_bar @ vec[:, i]) + np.pi) % (2 * np.pi) - np.pi)
            / lam.imag[i] for i in sel]) for v in variants}
        return per, lam.real[sel], leads

    per_g, dec_g, lead_g = complex_pairs(J_glob)
    per_k_all, dec_k_all, occ_pairs = [], [], []
    for j in range(K):
        pj, dj, _ = complex_pairs(J_k[j])
        per_k_all.append(pj)
        dec_k_all.append(dj)
        occ_pairs.append(np.full(len(pj), occ[j]))
    per_k_all = np.concatenate(per_k_all)
    dec_k_all = np.concatenate(dec_k_all)
    occ_pairs = np.concatenate(occ_pairs)

    # ---- a-priori check: model dA_x/dt vs data dA_x/dt ------------------
    f_full = np.concatenate([field.f(A[i:i + 4000])
                             for i in range(0, nT, 4000)])
    _, gs_full = quad_observable(A, Ms)
    dax_model = np.einsum("bd,bd->b", gs_full, f_full) / TPLUS_PER_S  # per t+
    dax_data = np.gradient(st_sig, TPF)
    max_lag = int(round(80.0 / TPF))
    dax_m_hp = _highpass(dax_model, w)
    dax_d_hp = _highpass(dax_data, w)
    corr_mod_dat = np.corrcoef(dax_m_hp, dax_d_hp)[0, 1]
    lags, _ = xcorr(dax_d_hp, dax_d_hp, max_lag)
    lag_t = lags * TPF
    xc = {}
    for v in variants:
        _, c_mod = xcorr(y_hp[v], dax_m_hp, max_lag)
        _, c_dat = xcorr(y_hp[v], dax_d_hp, max_lag)
        _, c_ref = xcorr(y_hp[v], x_hp, max_lag)
        xc[v] = dict(model=c_mod, data=c_dat, ref=c_ref)

    # ---- UPO gold standard ----------------------------------------------
    upo_rows = []
    for dd in sorted(args.upo_root.iterdir()):
        f = dd / "upo.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        T_tp = float(z["T"]) * TPLUS_PER_S
        if not (bool(z["converged"]) and 60 < T_tp < 250):
            continue
        orb = np.asarray(z["orbit"], np.float64)
        ax_o = np.abs(orb @ c)
        f1_ax, mod_ax = h1_frac(ax_o)
        row = dict(name=dd.name, T=T_tp, f1_ax=f1_ax, mod_ax=mod_ax)
        for v, Mv in variants.items():
            sy_o = np.sqrt(np.einsum("bd,bd->b", orb @ Mv, orb))
            f1_sy, _ = h1_frac(sy_o)
            row[f"lead_{v}"] = fourier_lead(ax_o, sy_o, T_tp)
            row[f"f1_sy_{v}"] = f1_sy
            row[f"ok_{v}"] = (f1_ax >= args.h1_gate and f1_sy >= args.h1_gate
                              and mod_ax >= 0.02)
        upo_rows.append(row)
        print(f"[upo] {row['name']}: T={T_tp:.1f} t+  A_x mod={mod_ax:.3f} "
              f"(h1 {f1_ax:.2f})  lead_tot={row['lead_tot']:+.1f} t+ "
              f"(h1 {row['f1_sy_tot']:.2f}, "
              f"{'PASS' if row['ok_tot'] else 'gated out'})")

    # ---- report ----------------------------------------------------------
    print(f"\n[weights] coherent kz=1 fraction of tau_y energy: "
          f"{frac_coh_y:.3f}; hess/J row-norm ratio median: "
          f"streak {np.median(ratio_s):.2f}, "
          f"vortex tot {np.median(res['tot']['hess_ratio_v']):.2f}, "
          f"inc {np.median(res['inc']['hess_ratio_v']):.2f}")
    for v in variants:
        r = res[v]
        (g_ss, g_sv), (g_vs, g_vv) = r["M2_cyc"]
        print(f"\n[2x2 cycle-avg '{v}', std., 1/t+] g_ss={g_ss:+.4f} "
              f"g_sv={g_sv:+.4f} g_vs={g_vs:+.4f} g_vv={g_vv:+.4f}")
        print(f"  streak relaxation 1/|g_ss| = {r['tau_relax']:.1f} t+; "
              f"transfer at 120 t+: |H|={np.abs(r['H_tf']):.2f}, vortex lead "
              f"= {r['lead_forced']:+.1f} t+")
        print(f"  eigenpair: period {r['period_2x2']:.0f} t+, vortex-lead "
              f"{r['lead_2x2']:+.1f} t+")
    print("\n cluster (cycle order) | theta    g_sv tot  g_vs tot  "
          "g_sv inc  g_vs inc   occ")
    for j in cyc_order:
        print(f"   {j:2d}                | {np.degrees(theta_k[j]):+7.1f}  "
              f"{res['tot']['M2_k'][j, 0, 1]:+.4f}   "
              f"{res['tot']['M2_k'][j, 1, 0]:+.4f}   "
              f"{res['inc']['M2_k'][j, 0, 1]:+.4f}   "
              f"{res['inc']['M2_k'][j, 1, 0]:+.4f}   {occ[j]:.3f}")
    near = np.abs(per_g - PERIOD_EMP).argsort()[:4]
    print("\n[full-45d global mean J] complex pairs nearest 120 t+:")
    for i in near:
        print(f"   period {per_g[i]:7.1f} t+  decay {dec_g[i]:+.4f} /t+  "
              f"vortex-lead tot {lead_g['tot'][i]:+6.1f} / "
              f"inc {lead_g['inc'][i]:+6.1f} t+")
    print(f"\n[a-priori] corr(model dA_x/dt, data dA_x/dt) = "
          f"{corr_mod_dat:.3f}")
    for v in variants:
        cm, cd, cr = xc[v]["model"], xc[v]["data"], xc[v]["ref"]
        print(f"  '{v}': peak corr(sig_y, dA_x/dt) model "
              f"{lag_t[cm.argmax()]:+.1f} t+ (r={cm.max():.2f}), data "
              f"{lag_t[cd.argmax()]:+.1f} t+ (r={cd.max():.2f}); "
              f"ref corr(sig_y, A_x) peak {lag_t[cr.argmax()]:+.1f} t+ "
              f"(r={cr.max():.2f})")

    # ---- figures ---------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    ordk = cyc_order
    xpos = np.arange(K)
    ax = axs[0, 0]
    ax.bar(xpos - 0.2, res["tot"]["M2_k"][ordk, 0, 1], 0.38,
           yerr=res["tot"]["M2_se"][ordk, 0, 1], color=C_SV,
           label=r"$g_{sv}$ (vortex$\to$streak)", capsize=2)
    ax.bar(xpos + 0.2, res["tot"]["M2_k"][ordk, 1, 0], 0.38,
           yerr=res["tot"]["M2_se"][ordk, 1, 0], color=C_VS,
           label=r"$g_{vs}$ (streak$\to$vortex)", capsize=2)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(ordk, fontsize=TICK_FS - 1)
    ax.set_xlabel(r"cluster (ordered by SSP cycle phase $\theta_k$)",
                  fontsize=LABEL_FS)
    ax.set_ylabel(r"standardised gain $[1/t^+]$", fontsize=LABEL_FS)
    ax.legend(fontsize=TICK_FS)
    ax.set_title(r"(a) phase-conditioned directed gains ($\sigma_y$ total)",
                 fontsize=LABEL_FS)

    ax = axs[0, 1]
    ax.plot(lag_t, xc["tot"]["data"], color=C_FOM, lw=2,
            label=r"data FD $\dot A_x$ vs $\sigma_y$")
    ax.plot(lag_t, xc["tot"]["model"], color=C_SV, lw=2,
            label=r"model $\nabla A_x\!\cdot\!f(a)$ vs $\sigma_y$")
    ax.plot(lag_t, xc["tot"]["ref"], color="0.65", lw=1.2, ls="--",
            label=r"ref: $\sigma_y$ vs $A_x$")
    ax.plot(lag_t, xc["inc"]["data"], color=C_FOM, lw=1, ls=":",
            label=r"data, $\sigma_y^{inc}$")
    ax.plot(lag_t, xc["inc"]["model"], color=C_SV, lw=1, ls=":",
            label=r"model, $\sigma_y^{inc}$")
    for cc, col in ((xc["tot"]["data"], C_FOM), (xc["tot"]["model"], C_SV)):
        ax.axvline(lag_t[cc.argmax()], color=col, lw=0.8, ls=":")
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel(r"lag $[t^+]$ ($\sigma_y$ earlier $\to$ positive)",
                  fontsize=LABEL_FS)
    ax.set_ylabel(r"corr$(\sigma_y(t),\ \cdot\,(t+\ell))$", fontsize=LABEL_FS)
    ax.legend(fontsize=TICK_FS - 2)
    ax.set_title(f"(b) a-priori lift-up content of $f$ "
                 f"(model vs data r={corr_mod_dat:.2f})", fontsize=LABEL_FS)

    ax = axs[1, 0]
    ax.scatter(per_k_all, dec_k_all, s=400 * occ_pairs, c=C_VS, alpha=0.55,
               label="per-cluster mean $J$ (size = occupancy)")
    ax.scatter(per_g, dec_g, marker="*", s=90, c=C_SV, zorder=3,
               label="occupancy-avg $J$")
    ax.axvline(PERIOD_EMP, color=C_AUX, lw=1.5, ls="--",
               label=r"empirical $T_{SSP}=120\,t^+$")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlim(3, 2000)
    ax.set_xlabel(r"eigenpair period $2\pi/\mathrm{Im}\,\lambda$ $[t^+]$",
                  fontsize=LABEL_FS)
    ax.set_ylabel(r"$\mathrm{Re}\,\lambda$ $[1/t^+]$", fontsize=LABEL_FS)
    ax.legend(fontsize=TICK_FS - 1, loc="lower right")
    ax.set_title("(c) full-45d Jacobian eigenpairs", fontsize=LABEL_FS)

    ax = axs[1, 1]
    Tf = np.linspace(30, 400, 200)
    wf = 2 * np.pi / Tf
    for v, ls in (("tot", "-"), ("inc", ":")):
        (g_ss, g_sv) = res[v]["M2_cyc"][0]
        lead_curve = -np.angle(g_sv / (1j * wf - g_ss)) / wf
        ax.plot(Tf, lead_curve, color=C_SV, lw=2, ls=ls,
                label=rf"2x2 forced response ('{v}')")
    ax.scatter([PERIOD_EMP], [LEAD_EMP], marker="s", s=70, c=C_FOM, zorder=3,
               label=r"empirical (120, 10) $t^+$ ['tot']")
    for row in upo_rows:
        if not row["ok_tot"]:
            continue
        ax.scatter([row["T"]], [row["lead_tot"]], marker="D", s=55, c=C_UPO,
                   zorder=3)
        ax.annotate(row["name"].replace("trav_t", "UPO "),
                    (row["T"], row["lead_tot"]), textcoords="offset points",
                    xytext=(6, 4), fontsize=TICK_FS - 2)
    ax.scatter([], [], marker="D", s=55, c=C_UPO,
               label="UPOs of fitted field (exact, gated)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel(r"period $[t^+]$", fontsize=LABEL_FS)
    ax.set_ylabel(r"$\sigma_y$ lead of $A_x$ $[t^+]$", fontsize=LABEL_FS)
    ax.legend(fontsize=TICK_FS - 2, loc="upper left")
    ax.set_title(r"(d) implied lead vs empirical; $1/|g_{ss}|$ = "
                 f"{res['tot']['tau_relax']:.0f} $t^+$ ('tot')",
                 fontsize=LABEL_FS)
    for a in axs.flat:
        a.tick_params(labelsize=TICK_FS)
    fig.tight_layout()
    fig.savefig(args.out / "liftup_matrix.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(xpos - 0.2, gA_sv[ordk], 0.38, color=C_SV,
           label=r"$w_s^T \bar J_k w_v$")
    ax.bar(xpos + 0.2, gA_vs[ordk], 0.38, color=C_VS,
           label=r"$w_v^T \bar J_k w_s$")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(ordk, fontsize=TICK_FS - 1)
    ax.set_xlabel(r"cluster (cycle order)", fontsize=LABEL_FS)
    ax.set_ylabel(r"gain $[1/t^+]$", fontsize=LABEL_FS)
    ax.legend(fontsize=TICK_FS)
    ax.set_title("prescribed static mode weights — signs are POD-sign-"
                 "convention-dependent; magnitudes only", fontsize=TICK_FS)
    fig.tight_layout()
    fig.savefig(args.out / "liftup_prescribed_weights.png", dpi=200)
    plt.close(fig)

    # ---- persist ---------------------------------------------------------
    np.savez(args.out / "liftup_matrix.npz",
             idx=idx, labels=lab_s, occ=occ, theta_k=theta_k,
             cyc_order=cyc_order,
             M2_k_tot=res["tot"]["M2_k"], M2_se_tot=res["tot"]["M2_se"],
             M2_cyc_tot=res["tot"]["M2_cyc"], eig_k_tot=res["tot"]["eig_k"],
             M2_k_inc=res["inc"]["M2_k"], M2_se_inc=res["inc"]["M2_se"],
             M2_cyc_inc=res["inc"]["M2_cyc"], eig_k_inc=res["inc"]["eig_k"],
             hess_ratio_s=ratio_s,
             hess_ratio_v_tot=res["tot"]["hess_ratio_v"],
             hess_ratio_v_inc=res["inc"]["hess_ratio_v"],
             gA_sv=gA_sv, gA_vs=gA_vs, w_s=w_s, w_v=w_v, e_y=e_y, s_m=s_m,
             per_glob=per_g, dec_glob=dec_g,
             lead_glob_tot=lead_g["tot"], lead_glob_inc=lead_g["inc"],
             per_k=per_k_all, dec_k=dec_k_all, occ_pairs=occ_pairs,
             lag_tplus=lag_t,
             xc_model_tot=xc["tot"]["model"], xc_data_tot=xc["tot"]["data"],
             xc_ref_tot=xc["tot"]["ref"],
             xc_model_inc=xc["inc"]["model"], xc_data_inc=xc["inc"]["data"],
             xc_ref_inc=xc["inc"]["ref"],
             upo_names=np.array([r["name"] for r in upo_rows]),
             upo_T=np.array([r["T"] for r in upo_rows]),
             upo_lead_tot=np.array([r["lead_tot"] for r in upo_rows]),
             upo_lead_inc=np.array([r["lead_inc"] for r in upo_rows]),
             upo_ok_tot=np.array([r["ok_tot"] for r in upo_rows]),
             n_s_bar=n_s_bar, n_v_bar_tot=res["tot"]["n_v_bar"],
             n_v_bar_inc=res["inc"]["n_v_bar"],
             model_path=str(args.model))
    summary = dict(
        variants={v: dict(
            g_ss=float(res[v]["M2_cyc"][0, 0]),
            g_sv=float(res[v]["M2_cyc"][0, 1]),
            g_vs=float(res[v]["M2_cyc"][1, 0]),
            g_vv=float(res[v]["M2_cyc"][1, 1]),
            tau_relax_tplus=float(res[v]["tau_relax"]),
            lead_forced_120=float(res[v]["lead_forced"]),
            period_2x2=float(res[v]["period_2x2"]),
            lead_2x2=float(res[v]["lead_2x2"]),
            peak_lag_model=float(lag_t[xc[v]["model"].argmax()]),
            peak_lag_data=float(lag_t[xc[v]["data"].argmax()]),
            peak_r_model=float(xc[v]["model"].max()),
            peak_r_data=float(xc[v]["data"].max()),
            peak_lag_ref_ax=float(lag_t[xc[v]["ref"].argmax()]),
            peak_r_ref_ax=float(xc[v]["ref"].max()),
            hess_ratio_v_median=float(np.median(res[v]["hess_ratio_v"])))
            for v in variants},
        corr_model_data_daxdt=float(corr_mod_dat),
        hess_ratio_s_median=float(np.median(ratio_s)),
        frac_coherent_tauy=float(frac_coh_y),
        upo=[{k: (float(v) if isinstance(v, (int, float, np.floating))
                  else bool(v) if isinstance(v, (bool, np.bool_)) else v)
              for k, v in r.items()} for r in upo_rows],
        n_states=int(len(idx)),
        empirical=dict(lead=LEAD_EMP, period=PERIOD_EMP, observable="tot"))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done] wrote {args.out}/liftup_matrix.png, "
          f"liftup_prescribed_weights.png, liftup_matrix.npz, summary.json")


if __name__ == "__main__":
    main()
