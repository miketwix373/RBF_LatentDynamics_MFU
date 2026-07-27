"""Per-cluster local Jacobian of the fitted RBF field for MFU_NA.

The production MFU_NA ROM is a single GLOBAL RBF vector field (K only shapes the
kernel widths; the a-priori fit is K-independent -- see memory
`quantized-local-switching-not-implemented`). "The local dynamics differ per
cluster" is therefore the honest statement that the one nonlinear field,
linearised at each cluster centroid, has a distinct Jacobian: the K=16 clusters
occupy dynamically distinct neighbourhoods of the field.

rom-specialist (2026-07-15) rejected re-regressing a 2x2 model in the
(A_x, sigma_y) observable plane: the exact identity A_x^2 + sigma_y^2 = E^2 forces
any constant-energy motion to be rotational, so such a fit is biased toward
antisymmetry regardless of the true dynamics, and is partly circular (clusters
are defined by partitioning the state). The defensible diagnostic linearises the
ACTUAL fitted field:

    adot(a) = ( Phi(a) / col_norms ) . xi ,
    Phi_j(a) = exp( -1/2 u_j^T P_j u_j ),   u_j = (a - mu)/sigma - c_j .

Analytic Jacobian (P_j = Sigma_invs_std[j], symmetric):
    d Phi_j / d a = -Phi_j * ( P_j u_j ) / sigma            (chain rule)
    J[i, m] = d adot_i / d a_m
            = sum_j (xi[j, i] / col_norms_j) * dPhi_j/da_m .

Evaluated at each cluster centroid c_k = mean_{t in k} a(t) (original reduced
coords). The soft-reflect corrector is a boundary-only radial pull and is
inactive at centroids, so J_k is exactly the in-cloud linearisation the ROM
integrates. Eigenvalues lambda [1/s] give the local growth rate Re(lambda) and
oscillation frequency Im(lambda)/2pi; wall units via 1 s = 11.289 t+.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SHARED_ROOT = Path("/users/sbrw610/sharedscratch/RBF_ROM")
TPLUS_PER_S = 11.289


def _load_model(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    if "Sigma_invs_std" in z.files:
        Sig = np.asarray(z["Sigma_invs_std"], np.float64)
    else:
        Sig_u = np.asarray(z["Sigma_invs_unique"], np.float64)
        pc = np.asarray(z["parent_compact"], np.int64)
        Sig = Sig_u[pc]
    return dict(
        centers_std=np.asarray(z["centers_std"], np.float64),
        Sigma=Sig,
        mu_A=np.asarray(z["mu_A"], np.float64),
        sigma_A=np.asarray(z["sigma_A"], np.float64),
        col_norms=np.asarray(z["col_norms_train"], np.float64),
        xi=np.asarray(z["xi"], np.float64),
    )


def _phi_and_grad(a, m):
    """Return (adot, J) at point a: adot (r,), J = d adot / d a (r, r)."""
    a_std = (a - m["mu_A"]) / m["sigma_A"]
    U = a_std[None, :] - m["centers_std"]                 # (n, r) u_j
    g = np.einsum("nij,nj->ni", m["Sigma"], U)            # (n, r) P_j u_j
    d2 = np.einsum("ni,ni->n", U, g)                      # (n,)
    Phi = np.exp(-0.5 * d2)                               # (n,)
    w = Phi / m["col_norms"]                              # (n,)
    adot = w @ m["xi"]                                    # (r,)
    dPhi = -(Phi[:, None] * g) / m["sigma_A"][None, :]    # (n, r) dPhi_j/da
    J = (m["xi"] / m["col_norms"][:, None]).T @ dPhi      # (r, r)
    return adot, J


def _rhs(a, m):
    a_std = (a - m["mu_A"]) / m["sigma_A"]
    U = a_std[None, :] - m["centers_std"]
    d2 = np.einsum("nij,nj,ni->n", m["Sigma"], U, U)
    Phi = np.exp(-0.5 * d2)
    return (Phi / m["col_norms"]) @ m["xi"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--coords", type=Path,
                   default=SHARED_ROOT / "results/MFU_NA/pod/coords_do45.npz")
    p.add_argument("--labels", type=Path, default=SHARED_ROOT /
                   "results/MFU_NA/cluster_sweep/do45/K=16.npz")
    p.add_argument("--model", type=Path, default=SHARED_ROOT /
                   "results/MFU_NA/do45/alpha3.5/model_n04800.npz")
    p.add_argument("--out", type=Path, default=SHARED_ROOT /
                   "results/MFU_NA/do45/cluster_jacobians_K16.npz")
    args = p.parse_args()

    with np.load(args.coords) as z:
        alpha = np.asarray(z["alpha"], np.float64)        # (T, r)
    clab = np.asarray(np.load(args.labels)["labels"], np.int64)
    m = _load_model(args.model)
    r = alpha.shape[1]
    Kc = int(clab.max()) + 1

    # align labels to the coords stride (labels sampled coarser than alpha)
    step = len(alpha) // len(clab)
    aL = alpha[::step][:len(clab)]                        # (n_lab, r)
    cent = np.array([aL[clab == k].mean(0) for k in range(Kc)])   # (K, r)
    occ = np.array([(clab == k).mean() for k in range(Kc)])

    # FD check of the analytic Jacobian at cluster 0's centroid
    a0 = cent[0]
    _, Ja = _phi_and_grad(a0, m)
    eps = 1e-6
    Jfd = np.empty((r, r))
    for mm in range(r):
        e = np.zeros(r); e[mm] = eps
        Jfd[:, mm] = (_rhs(a0 + e, m) - _rhs(a0 - e, m)) / (2 * eps)
    rel = np.linalg.norm(Ja - Jfd) / np.linalg.norm(Jfd)
    print(f"[check] analytic vs FD Jacobian rel.err = {rel:.2e}  (want < 1e-5)")

    # per-cluster spectra
    eig = np.zeros((Kc, r), complex)
    lead_osc = np.zeros(Kc, complex)     # least-damped oscillatory eigenpair
    lead_real = np.zeros(Kc)             # largest real part overall
    for k in range(Kc):
        _, Jk = _phi_and_grad(cent[k], m)
        lam = np.linalg.eigvals(Jk)
        eig[k] = lam
        lead_real[k] = lam.real.max()
        osc = lam[np.abs(lam.imag) > 1e-9]
        if osc.size:
            lead_osc[k] = osc[np.argmax(osc.real)]        # least-damped osc
        else:
            lead_osc[k] = lam[np.argmax(lam.real)]

    f_k = np.abs(lead_osc.imag) / (2 * np.pi) / TPLUS_PER_S   # [1/t+]
    per_k = np.where(f_k > 0, 1.0 / np.where(f_k > 0, f_k, np.nan), np.inf)
    gr_k = lead_osc.real / TPLUS_PER_S                        # [1/t+]

    print(f"[jac] K={Kc}  r={r}  model={args.model.name}")
    print(f"{'k':>3} {'occ%':>6} {'Re(lam)/t+':>11} {'f[1/t+]':>9} "
          f"{'period t+':>10} {'|osc| pairs':>11}")
    for k in range(Kc):
        npair = int((np.abs(eig[k].imag) > 1e-9).sum() // 2)
        print(f"{k:>3} {100*occ[k]:6.1f} {gr_k[k]:11.2e} {f_k[k]:9.4f} "
              f"{per_k[k]:10.0f} {npair:11d}")
    print(f"[jac] growth Re in [{gr_k.min():.2e}, {gr_k.max():.2e}] /t+")
    print(f"[jac] local period in [{per_k[np.isfinite(per_k)].min():.0f}, "
          f"{per_k[np.isfinite(per_k)].max():.0f}] t+ "
          f"(global cycle ~120 t+)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, eig=eig, lead_osc=lead_osc, lead_real=lead_real,
             f_k=f_k, growth_k=gr_k, period_k=per_k, cent=cent, occ=occ,
             tplus_per_s=TPLUS_PER_S, model=str(args.model))
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
