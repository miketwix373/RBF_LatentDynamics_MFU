"""Cluster-conditioned Markovianity diagnostic for the MFU_NA wall-shear ROM.

Per K-means cluster (cycle phase): is the instantaneous wall state enough, or
does the a-priori tendency residual r(t) = alpha_dot - f_RBF(a) carry structure
only history (or more wall modes) can explain? Practical Mori-Zwanzig split,
design per rom-specialist assessment (2026-07-20):

  - target alpha_dot is the STORED coords derivative (deriv_5point_full at
    native dt), identical to the fit target -> r is a pure model residual;
    3-point-FD and Savitzky-Golay variants give an estimator-uncertainty band;
  - leave-episode-out ridge CV (contiguous cluster-residence episodes), lambda
    by inner episode split; clusters with < 15 episodes flagged, not scored;
  - capacity-matched regressions (180 features): R2_state_nl on
    [a, RFF_135(a)] vs R2_memory on [a, a(t-4), a(t-8), a(t-16)] (4.5-18 t+);
    memory gain = R2_memory - R2_state_nl compares like with like;
  - future-lag placebo [a, a(t+4), a(t+8), a(t+16)]: smooth-residual
    extrapolation masquerading as memory shows up here (cf. the global
    delay-embed test in run_mfu_na_delay_embed_test.py);
  - scrambled-history null (50 perms) -> 95th-percentile gain band;
  - attribution control R2_trunc on [a, a_46..75] (instantaneous TRUNCATED wall
    POD modes from the nested do75 coords, no history): if it matches
    R2_memory, the "memory" is wall-POD truncation, not interior dynamics;
  - memory-horizon curve R2(max lag) for lags {4, 8, 16, 32, 53}; the 53-step
    (~60 t+, half SSP period) point is a phase-fitting control, not a kernel
    measurement.

Outputs under results/MFU_NA/markovianity/: summary.json, markovianity.npz.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_mfu_na_cluster_jacobians import _load_model  # noqa: E402

SHARED_ROOT = Path("/users/sbrw610/sharedscratch/RBF_ROM")
MFU = SHARED_ROOT / "results/MFU_NA"
MEM_LAGS = [4, 8, 16]
CURVE_LAGS = [4, 8, 16, 32, 53]
N_RFF = 135
N_NULL = 50
MIN_EPISODES = 15


def rbf_field_batch(A: np.ndarray, mpath: Path) -> np.ndarray:
    """f_RBF at every row of A, vectorised over the unique kernel metrics."""
    z = np.load(mpath, allow_pickle=True)
    m = _load_model(mpath)
    Sig_u = np.asarray(z["Sigma_invs_unique"], np.float64)
    pc = np.asarray(z["parent_compact"], np.int64)
    X = (A - m["mu_A"]) / m["sigma_A"]
    Phi = np.empty((A.shape[0], m["centers_std"].shape[0]))
    for q in range(Sig_u.shape[0]):
        idx = np.where(pc == q)[0]
        L = np.linalg.cholesky(Sig_u[q])
        Y, C = X @ L, m["centers_std"][idx] @ L
        d2 = ((Y**2).sum(1)[:, None] + (C**2).sum(1)[None, :] - 2.0 * Y @ C.T)
        Phi[:, idx] = np.exp(-0.5 * np.maximum(d2, 0.0))
    return (Phi / m["col_norms"]) @ m["xi"]


def episodes_of(mask: np.ndarray) -> list[np.ndarray]:
    """Contiguous runs of True in mask, as index arrays."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) > 1)
    return np.split(idx, cuts + 1)


def episode_folds(eps: list[np.ndarray], n_folds: int) -> list[np.ndarray]:
    """Group episodes into n_folds contiguous blocks of ~equal sample count."""
    sizes = np.array([e.size for e in eps])
    targets = np.linspace(0, sizes.sum(), n_folds + 1)[1:]
    folds, f, acc = [[] for _ in range(n_folds)], 0, 0
    for e, s in zip(eps, sizes):
        folds[f].append(e)
        acc += s
        if acc >= targets[f] and f < n_folds - 1:
            f += 1
    return [np.concatenate(fl) for fl in folds if fl]


def ridge_eval(Xtr, Ytr, Xte, lams, n_val):
    """Ridge with lambda from a train-tail validation split; returns preds."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = Ytr.mean(0)
    ni = Xtr.shape[0] - n_val
    G, b = Xtr[:ni].T @ Xtr[:ni], Xtr[:ni].T @ (Ytr[:ni] - ym)
    I = np.eye(G.shape[0])
    best, best_sse = lams[0], np.inf
    for lam in lams:
        W = np.linalg.solve(G + lam * I, b)
        s = float(((Ytr[ni:] - ym - Xtr[ni:] @ W) ** 2).sum())
        if s < best_sse:
            best, best_sse = lam, s
    W = np.linalg.solve(Xtr.T @ Xtr + best * I, Xtr.T @ (Ytr - ym))
    return ym + Xte @ W


def cv_r2(feats: np.ndarray, targ: np.ndarray, folds: list[np.ndarray],
          all_idx: np.ndarray, lams: np.ndarray) -> float:
    """Pooled out-of-sample R2 over episode folds (indices into feats rows)."""
    pos = {i: p for p, i in enumerate(all_idx)}
    sse, sst = 0.0, 0.0
    for f, te in enumerate(folds):
        tr = np.concatenate([folds[g] for g in range(len(folds)) if g != f])
        tp = np.array([pos[i] for i in tr])
        ep = np.array([pos[i] for i in te])
        n_val = max(10, int(0.2 * tp.size))
        pred = ridge_eval(feats[tp], targ[tp], feats[ep], lams, n_val)
        sse += float(((targ[ep] - pred) ** 2).sum())
        sst += float(((targ[ep] - targ[tp].mean(0)) ** 2).sum())
    return 1.0 - sse / sst


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--coords", type=Path, default=MFU / "pod/coords_do45.npz")
    p.add_argument("--coords-hi", type=Path,
                   default=MFU / "pod/coords_do75.npz")
    p.add_argument("--model", type=Path,
                   default=MFU / "do45/alpha8.0/model_n00960.npz")
    p.add_argument("--labels", type=Path,
                   default=MFU / "cluster_sweep/do45/K=16.npz")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=MFU / "markovianity")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    with np.load(args.coords) as z:
        a_nat = np.asarray(z["alpha"], np.float64)
        A_full = a_nat[::args.stride]
        Y_full = np.asarray(z["alpha_dot"], np.float64)[::args.stride]
        V = np.asarray(z["V"], np.float64)
        dt_nat = float(z["dt_native"])
    with np.load(args.coords_hi) as z:
        B_full = np.asarray(z["alpha"], np.float64)[::args.stride, 45:]
    dt = dt_nat * args.stride
    nx = V.shape[0] // 2
    zm = np.load(args.model, allow_pickle=True)
    n_train = int(zm["n_train"])
    assert int(zm["stride"]) == args.stride

    cent = np.asarray(np.load(args.labels)["centroids"], np.float64)
    Kc = cent.shape[0]

    # full record: the held-out 20% alone gives 2-14 residence episodes per
    # cluster, below the episode-CV floor. f_RBF is in-sample on the train
    # part; rel_err is therefore reported for both parts separately, and the
    # regression CV protects itself.
    fut_max = max(MEM_LAGS)
    idx = np.arange(max(CURVE_LAGS), A_full.shape[0] - fut_max)
    A, Y, B = A_full[idx], Y_full[idx], B_full[idx]
    lab = np.argmin(((A[:, None, :] - cent[None]) ** 2).sum(-1), axis=1)
    f_rbf = rbf_field_batch(A, args.model)
    r = Y - f_rbf

    # derivative-estimator controls: 3-point FD at strided dt, Savitzky-Golay
    # (poly 4, window 7) at native dt
    from scipy.signal import savgol_filter
    Y3 = np.full_like(A_full, np.nan)
    Y3[1:-1] = (A_full[2:] - A_full[:-2]) / (2 * dt)
    Ysg = savgol_filter(a_nat, 7, 4, deriv=1, delta=dt_nat,
                        axis=0)[::args.stride]
    r3, rsg = Y3[idx] - f_rbf, Ysg[idx] - f_rbf

    # feature blocks (rows aligned with idx)
    lag_feats = {l: A_full[idx - l] for l in CURVE_LAGS}
    fut_feats = [A_full[idx + l] for l in MEM_LAGS]
    lams = np.logspace(-4, 3, 15)

    rows = []
    curves = np.full((Kc, len(CURVE_LAGS)), np.nan)
    for k in range(Kc):
        m = lab == k
        eps = episodes_of(m)
        row = dict(k=k, n=int(m.sum()), n_episodes=len(eps))
        aY = float((Y[m] ** 2).sum())
        row["rel_err"] = float(np.sqrt((r[m] ** 2).sum() / aY))
        row["rel_err_fd3"] = float(np.sqrt((r3[m] ** 2).sum() / aY))
        row["rel_err_sg"] = float(np.sqrt((rsg[m] ** 2).sum() / aY))
        mte = m & (idx >= n_train)
        if mte.any():
            row["rel_err_test"] = float(np.sqrt(
                (r[mte] ** 2).sum() / (Y[mte] ** 2).sum()))
        rbar = r[m].mean(0)
        row["bias_frac"] = float(m.sum() * (rbar**2).sum() / aY)
        mm = m[:-1] & m[1:]
        ra, rb = r[:-1][mm] - r[:-1][mm].mean(0), r[1:][mm] - r[1:][mm].mean(0)
        row["lag1_rho"] = float((ra * rb).sum() /
                                np.sqrt((ra**2).sum() * (rb**2).sum()))
        Wr, Wt = V @ r[m].T, V @ Y[m].T
        row["tauy_frac_res"] = float((Wr[nx:] ** 2).sum() / (Wr**2).sum())
        row["tauy_frac_tend"] = float((Wt[nx:] ** 2).sum() / (Wt**2).sum())

        if len(eps) < MIN_EPISODES:
            row["assessable"] = False
            rows.append(row)
            print(f"k={k:2d} n={row['n']:5d} eps={len(eps):3d}  "
                  f"rel_err={row['rel_err']:.3f} "
                  f"[{row['rel_err_fd3']:.3f}/{row['rel_err_sg']:.3f}]  "
                  "NOT ASSESSABLE (<15 episodes)", flush=True)
            continue
        row["assessable"] = True
        folds = episode_folds(eps, args.n_folds)
        gk = np.flatnonzero(m)

        # cluster-local random Fourier features for the capacity-matched
        # nonlinear instantaneous baseline
        sub = A[m][rng.choice(m.sum(), min(400, m.sum()), replace=False)]
        d2 = ((sub[:, None] - sub[None]) ** 2).sum(-1)
        bw = np.sqrt(np.median(d2[d2 > 0]))
        Wr_ = rng.standard_normal((45, N_RFF)) / bw
        br_ = rng.uniform(0, 2 * np.pi, N_RFF)
        rff = np.cos(A @ Wr_ + br_)

        F_lin = A
        F_nl = np.concatenate([A, rff], axis=1)
        F_mem = np.concatenate([A] + [lag_feats[l] for l in MEM_LAGS], axis=1)
        F_fut = np.concatenate([A] + fut_feats, axis=1)
        F_trn = np.concatenate([A, B], axis=1)

        row["r2_state_lin"] = cv_r2(F_lin, r, folds, gk, lams)
        row["r2_state_nl"] = cv_r2(F_nl, r, folds, gk, lams)
        row["r2_memory"] = cv_r2(F_mem, r, folds, gk, lams)
        row["r2_future"] = cv_r2(F_fut, r, folds, gk, lams)
        row["r2_trunc"] = cv_r2(F_trn, r, folds, gk, lams)
        row["memory_gain"] = row["r2_memory"] - row["r2_state_nl"]

        # scrambled-history null: permute the delay block rows within cluster
        nulls = np.empty(N_NULL)
        base = np.concatenate([lag_feats[l] for l in MEM_LAGS], axis=1)
        for j in range(N_NULL):
            perm = np.arange(A.shape[0])
            perm[gk] = rng.permutation(gk)
            Fn = np.concatenate([A, base[perm]], axis=1)
            nulls[j] = cv_r2(Fn, r, folds, gk, lams)
        row["null_gain_p95"] = float(
            np.quantile(nulls - row["r2_state_nl"], 0.95))

        for ci, l in enumerate(CURVE_LAGS):
            F = np.concatenate(
                [A] + [lag_feats[q] for q in CURVE_LAGS if q <= l], axis=1)
            curves[k, ci] = cv_r2(F, r, folds, gk, lams)

        rows.append(row)
        print(f"k={k:2d} n={row['n']:5d} eps={len(eps):3d}  "
              f"rel_err={row['rel_err']:.3f} "
              f"[{row['rel_err_fd3']:.3f}/{row['rel_err_sg']:.3f}]  "
              f"bias={row['bias_frac']:.3f}  "
              f"R2 lin/nl={row['r2_state_lin']:+.2f}/{row['r2_state_nl']:+.2f} "
              f"mem={row['r2_memory']:+.2f} fut={row['r2_future']:+.2f} "
              f"trunc={row['r2_trunc']:+.2f}  gain={row['memory_gain']:+.3f} "
              f"(null95 {row['null_gain_p95']:+.3f})", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / "markovianity.npz",
             labels=lab, residual=r, adot=Y, f_rbf=f_rbf, idx=idx,
             curve_lags=np.asarray(CURVE_LAGS), curves=curves,
             mem_lags=np.asarray(MEM_LAGS), model_path=str(args.model))
    (args.out_dir / "summary.json").write_text(json.dumps(rows, indent=2))
    print(f"[done] wrote {args.out_dir}/summary.json", flush=True)


if __name__ == "__main__":
    main()
