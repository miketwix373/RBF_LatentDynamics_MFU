"""Hunt unstable periodic orbits of the fitted MFU_NA wall-shear RBF field.

The explicit field adot = f_RBF(a) (45 centred-POD wall-shear coords) is cheap
and differentiable, so periodic orbits -- the wall-observable skeleton of the
near-wall regeneration cycle (cf. Kawahara & Kida, J. Fluid Mech. 449, 2001)
-- can be sought by multiple shooting with the analytic Jacobian:

  1. seeds: near-recurrences of the recorded trajectory, min_T ||a(t+T)-a(t)||
     (three period tiers: ~10 t+ fast oscillation, ~40 t+, ~120 t+ SSP cycle);
  2. multiple shooting, M segments: unknowns {x_0..x_{M-1}, T}; residuals
     x_{i+1} - phi_{T/M}(x_i) (cyclic) + phase anchor
     <f(x_0^seed), x_0 - x_0^seed> = 0; damped Gauss-Newton;
  3. flow and its state-transition matrix by RK4 on the variational equation
     Phi' = J(a) Phi, batched over segments;
  4. coverage gate: sum-of-kernel activation along the orbit must stay within
     the on-attractor range (the field is unsupported off the data cloud);
  5. Floquet multipliers from the ordered segment-monodromy product.

Outputs under results/MFU_NA/upo/<tag>/: upo.npz (orbit, T, Floquet,
coverage), convergence log.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

SHARED_ROOT = (next(_p for _p in Path(__file__).resolve().parents if (_p / "pyproject.toml").exists()))
MFU = SHARED_ROOT / "results/MFU_NA"
TPLUS_PER_S = 11.289


class RBFField:
    """Batched f, J, and kernel coverage for a saved anisotropic-RBF model."""

    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=True)
        self.C = np.asarray(z["centers_std"], np.float64)
        self.Sig_u = np.asarray(z["Sigma_invs_unique"], np.float64)
        self.pc = np.asarray(z["parent_compact"], np.int64)
        self.mu = np.asarray(z["mu_A"], np.float64)
        self.sd = np.asarray(z["sigma_A"], np.float64)
        self.cn = np.asarray(z["col_norms_train"], np.float64)
        self.xi = np.asarray(z["xi"], np.float64)
        self.L = [np.linalg.cholesky(S) for S in self.Sig_u]
        self.groups = [np.where(self.pc == q)[0] for q in range(len(self.L))]

    def phi(self, X: np.ndarray) -> np.ndarray:
        """Kernel matrix (b, n) at batch X (b, d)."""
        Xs = (X - self.mu) / self.sd
        Phi = np.empty((X.shape[0], self.C.shape[0]))
        for q, idx in enumerate(self.groups):
            Y, Cq = Xs @ self.L[q], self.C[idx] @ self.L[q]
            d2 = ((Y**2).sum(1)[:, None] + (Cq**2).sum(1)[None, :]
                  - 2.0 * Y @ Cq.T)
            Phi[:, idx] = np.exp(-0.5 * np.maximum(d2, 0.0))
        return Phi

    def f(self, X: np.ndarray) -> np.ndarray:
        return (self.phi(X) / self.cn) @ self.xi

    def f_and_J(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """f (b, d) and Jacobian (b, d, d) at batch X."""
        Xs = (X - self.mu) / self.sd
        b, d = X.shape
        Phi = self.phi(X)
        W = Phi / self.cn                       # (b, n)
        f = W @ self.xi
        # dPhi_j/da = -Phi_j * (Sigma_j u_j) / sd ; assemble J per unique metric
        J = np.zeros((b, d, d))
        for q, idx in enumerate(self.groups):
            U = Xs[:, None, :] - self.C[idx][None, :, :]        # (b, nq, d)
            G = np.einsum("ij,bnj->bni", self.Sig_u[q], U)      # (b, nq, d)
            WX = W[:, idx, None] * G / self.sd[None, None, :]   # (b, nq, d)
            J -= np.einsum("bnd,ne->bed", WX, self.xi[idx])
        return f, J

    def coverage(self, X: np.ndarray) -> np.ndarray:
        return self.phi(X).sum(1)


def flow(field: RBFField, X0: np.ndarray, T: float, dt: float,
         variational: bool):
    """RK4 flow over time T for a batch of starts; optionally the
    state-transition matrices via the variational equation."""
    n_steps = max(1, int(round(T / dt)))
    h = T / n_steps
    X = X0.copy()
    b, d = X.shape
    Phi = np.tile(np.eye(d), (b, 1, 1)) if variational else None
    for _ in range(n_steps):
        if variational:
            k1, J1 = field.f_and_J(X)
            P1 = np.einsum("bij,bjk->bik", J1, Phi)
            k2, J2 = field.f_and_J(X + 0.5 * h * k1)
            P2 = np.einsum("bij,bjk->bik", J2, Phi + 0.5 * h * P1)
            k3, J3 = field.f_and_J(X + 0.5 * h * k2)
            P3 = np.einsum("bij,bjk->bik", J3, Phi + 0.5 * h * P2)
            k4, J4 = field.f_and_J(X + h * k3)
            P4 = np.einsum("bij,bjk->bik", J4, Phi + h * P3)
            Phi = Phi + (h / 6) * (P1 + 2 * P2 + 2 * P3 + P4)
        else:
            k1 = field.f(X)
            k2 = field.f(X + 0.5 * h * k1)
            k3 = field.f(X + 0.5 * h * k2)
            k4 = field.f(X + h * k3)
        X = X + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return X, Phi


def shoot(field: RBFField, X: np.ndarray, T: float, dt: float,
          f_anchor: np.ndarray, x_anchor: np.ndarray, max_iter: int,
          tol_rel: float, scale: float, fix_T_iters: int = 0, log=print):
    """LM multiple shooting. X: (M, d) segment starts.
    Two stages: the first fix_T_iters iterations hold T fixed (the trivial
    T -> 0 family -- all segments collapsing onto one point -- is a solution
    manifold of the free-T closure equations and captures the iteration from
    imperfect seeds); T is then released within [0.8, 1.25] x its held value."""
    M, d = X.shape
    T_lo, T_hi = 0.5 * T, 2.0 * T
    mu = 1e-6
    nun = M * d + 1
    for it in range(max_iter):
        XT, Phi = flow(field, X, T / M, dt, variational=True)
        F = np.empty(M * d + 1)
        for i in range(M):
            F[i * d:(i + 1) * d] = XT[i] - X[(i + 1) % M]
        F[-1] = f_anchor @ (X[0] - x_anchor)
        res = np.linalg.norm(F[:-1]) / (np.sqrt(M) * scale)
        # end-of-segment velocities give dF/dT = f(phi(x_i)) / M
        fT = field.f(XT)
        DF = np.zeros((M * d + 1, nun))
        for i in range(M):
            r = slice(i * d, (i + 1) * d)
            DF[r, i * d:(i + 1) * d] = Phi[i]
            j = (i + 1) % M
            DF[r, j * d:(j + 1) * d] -= np.eye(d)
            DF[r, -1] = fT[i] / M
        DF[-1, :d] = f_anchor
        if res < tol_rel:
            log(f"  iter {it:2d}: rel closure {res:.2e}  CONVERGED")
            return X, T, res, True
        fixed = it < fix_T_iters
        if it == fix_T_iters and fix_T_iters > 0:
            T_lo, T_hi = 0.8 * T, 1.25 * T
            log(f"  [release T within {T_lo:.2f}..{T_hi:.2f} s]")
        DFa = DF[:, :-1] if fixed else DF
        # Levenberg-Marquardt step with the period confined to [T_lo, T_hi]
        G = DFa.T @ DFa
        g = DFa.T @ F
        dg = np.sqrt(np.diag(G)) + 1e-30
        ok = False
        for _ in range(20):
            step = np.linalg.solve(G + mu * np.diag(dg**2), -g)
            if fixed:
                Xn = X + step.reshape(M, d)
                Tn = T
            else:
                Xn = X + step[:-1].reshape(M, d)
                Tn = T + step[-1]
            if T_lo <= Tn <= T_hi:
                XTn, _ = flow(field, Xn, Tn / M, dt, variational=False)
                Fn = np.empty(M * d + 1)
                for i in range(M):
                    Fn[i * d:(i + 1) * d] = XTn[i] - Xn[(i + 1) % M]
                Fn[-1] = f_anchor @ (Xn[0] - x_anchor)
                if np.linalg.norm(Fn) < np.linalg.norm(F):
                    ok = True
                    break
            mu *= 4.0
        if not ok:
            log(f"  iter {it:2d}: rel closure {res:.2e}  LM stalled "
                f"(mu={mu:.1e}, T={T * TPLUS_PER_S:.1f} t+)")
            return X, T, res, False
        X, T = Xn, Tn
        mu = max(mu / 3.0, 1e-14)
        log(f"  iter {it:2d}: rel closure {res:.2e}  (mu={mu:.1e}, "
            f"T={T * TPLUS_PER_S:.1f} t+)")
    return X, T, res, False


def floquet(field: RBFField, X: np.ndarray, T: float, dt: float) -> np.ndarray:
    """Multipliers from the ordered product of segment monodromies."""
    M, d = X.shape
    _, Phi = flow(field, X, T / M, dt, variational=True)
    P = np.eye(d)
    for i in range(M):
        P = Phi[i] @ P
    return np.linalg.eigvals(P)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--coords", type=Path, default=MFU / "pod/coords_do45.npz")
    p.add_argument("--model", type=Path,
                   default=MFU / "do45/alpha8.0/model_n00960.npz")
    p.add_argument("--seed-t", type=float, default=None,
                   help="seed start time (s) in the strided record")
    p.add_argument("--seed-T", type=float, default=None,
                   help="seed period (s)")
    p.add_argument("--restart-from", type=Path, default=None,
                   help="upo.npz of a previous run; densely resample its "
                        "orbit into the requested number of segments")
    p.add_argument("--segments", type=int, default=8)
    p.add_argument("--dt", type=float, default=0.025)
    p.add_argument("--max-iter", type=int, default=60)
    p.add_argument("--fix-T-iters", type=int, default=25)
    p.add_argument("--tol-rel", type=float, default=1e-8)
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--out-root", type=Path, default=MFU / "upo")
    args = p.parse_args()

    with np.load(args.coords) as z:
        A = np.asarray(z["alpha"], np.float64)[::2]
    dt_rec = 0.1
    field = RBFField(args.model)
    scale = float(np.sqrt((A**2).sum(1).mean()))

    M = args.segments
    if args.restart_from is not None:
        zr = np.load(args.restart_from, allow_pickle=True)
        T0 = float(zr["T"])
        orb = np.asarray(zr["orbit"], np.float64)
        X = orb[(np.arange(M) * orb.shape[0]) // M].copy()
        print(f"[restart] {args.restart_from} T={T0 * TPLUS_PER_S:.1f} t+ "
              f"rel_closure={float(zr['rel_closure']):.2e} -> M={M}")
    else:
        i0 = int(round(args.seed_t / dt_rec))
        nT = int(round(args.seed_T / dt_rec))
        seg_idx = i0 + (np.arange(M) * nT) // M
        X = A[seg_idx].copy()
        T0 = args.seed_T
        print(f"[seed] t={args.seed_t}s T={args.seed_T}s "
              f"({args.seed_T * TPLUS_PER_S:.1f} t+), M={M}")
    x_anchor = X[0].copy()
    f_anchor = field.f(x_anchor[None])[0]

    cov_att = field.coverage(A[np.random.default_rng(0).choice(len(A), 2000)])
    print(f"[coverage] on-attractor p5={np.percentile(cov_att, 5):.2f}")

    X, T, res, ok = shoot(field, X, T0, args.dt, f_anchor, x_anchor,
                          args.max_iter, args.tol_rel, scale,
                          fix_T_iters=args.fix_T_iters)

    # dense orbit + coverage for the gate and the diagnostics
    n_dense = 400
    Xd = np.empty((n_dense, X.shape[1]))
    x = X[0].copy()
    for j in range(n_dense):
        Xd[j] = x
        x, _ = flow(field, x[None], T / n_dense, args.dt, False)
        x = x[0]
    cov = field.coverage(Xd)
    lam = floquet(field, X, T, args.dt)
    order = np.argsort(-np.abs(lam))
    out = args.out_root / args.tag
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "upo.npz",
             X_segments=X, T=T, converged=ok, rel_closure=res,
             orbit=Xd, coverage=cov, floquet=lam,
             seed_t=args.seed_t if args.seed_t is not None else -1.0,
             seed_T=args.seed_T if args.seed_T is not None else -1.0,
             restart_from=str(args.restart_from), dt=args.dt,
             model_path=str(args.model), segments=M)
    print(f"[orbit] T={T:.3f}s = {T * TPLUS_PER_S:.1f} t+  "
          f"converged={ok}  rel_closure={res:.2e}")
    print(f"[coverage] orbit min={cov.min():.2f} / on-attractor "
          f"p5={np.percentile(cov_att, 5):.2f}")
    print("[floquet] leading |lambda|:",
          np.array2string(np.abs(lam[order][:6]), precision=3),
          " (trivial ~1 expected)")
    print(f"[done] wrote {out}/upo.npz")


if __name__ == "__main__":
    main()
