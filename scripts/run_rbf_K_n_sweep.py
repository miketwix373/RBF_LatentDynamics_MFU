"""K-cluster anisotropic-RBF sweep over total n_rbf, per-mode error.

Generalisation of `scripts/run_rbf_K1_n_sweep.py` to K > 1. The
placement routine is the same (`rbf_centers_hier_anisotropic_pca`);
the only change is that K_shape > 1 partitions the training set into K
PCA frames first and places `n_per_cluster = total_n_rbf // K` centres
in each. So each centre carries the local cluster's anisotropy
instead of a single global one. At K=1 this script is numerically
equivalent to `run_rbf_K1_n_sweep.py` (verified by running both at
K_shape=1).

Output schema and on-disk file layout are identical to the K=1 sweep
so the existing `scripts/aggregate_rbf_K1_n_sweep.py` aggregates this
sweep too - the aggregator globs `n_*.npz` regardless of K.

The labelled `n_rbf` in each per-file npz is the GRID label (the value
the user asked for in the sweep), and `n_rbf_eff` is what was actually
placed = K * (n_rbf // K). When the grid label is not divisible by K
you lose a few centres at the rounding boundary; this matches how the
K=1 sweep treats `n_rbf_eff`.

Pipeline (one row of the sweep)
-------------------------------
1. Load coords.npz from `scripts/precompute_coords.py`.
2. Stride alpha / alpha_dot to the regression-side sample density.
3. Time-blocked train/test split.
4. `rbf_centers_hier_anisotropic_pca(K_shape=K, n_per_cluster=n_rbf//K, ...)`.
5. `rbf_features_mahal` then column-normalise by training column norms.
6. Plain lstsq.
7. Per-mode NRMSE on both partitions; save per-mode and per-(n_rbf)
   diagnostics in the same schema as the K=1 sweep.

Output
------
- Multi-n_rbf mode (default): `sweep.npz` + `rbf_K{K}_n_sweep.png`.
- Single-n_rbf mode (when --n-rbf-list has one value): writes
  `n_<value:04d>.npz` with single-n_rbf payload (scalars for cond,
  timings, n_rbf; vectors for per-mode quantities). For SLURM job
  fan-out; the aggregator combines per-n_rbf files into `sweep.npz`
  and the plot afterwards.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from chord2.sindy import (  # noqa: E402
    rbf_centers_hier_anisotropic_pca,
    rbf_features_mahal,
    ridge_svd_factor,
    ridge_svd_press,
    ridge_svd_solve,
)


def _normalise_columns(Phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    col_norms = np.linalg.norm(Phi, axis=0)
    col_norms = np.where(col_norms == 0.0, 1.0, col_norms)
    return Phi / col_norms[None, :], col_norms


def _per_mode_nrmse(Y: np.ndarray, Y_pred: np.ndarray) -> tuple[
    np.ndarray, np.ndarray, np.ndarray
]:
    rmse = np.sqrt(((Y - Y_pred) ** 2).mean(axis=0))
    std_Y = Y.std(axis=0, ddof=0)
    den = np.where(std_Y == 0.0, 1.0, std_Y)
    return rmse / den, rmse, std_Y


def run_one(
    n_rbf_total: int,
    K: int,
    A_tr: np.ndarray, Y_tr: np.ndarray,
    A_te: np.ndarray, Y_te: np.ndarray,
    *,
    alpha_tangent: float,
    energy_threshold: float,
    robust_stat: str,
    seed: int,
    n_init: int,
    standardise: bool = True,
    normal_width_floor: float = 0.0,
    ridge: str = "none",
    ridge_lambda_grid=None,
) -> dict:
    n_per_cluster = n_rbf_total // K
    if n_per_cluster < 2:
        raise SystemExit(
            f"n_rbf={n_rbf_total} with K={K} gives n_per_cluster="
            f"{n_per_cluster} < 2; pick a larger total or smaller K"
        )

    t0 = time.time()
    centers_std, Sigma_invs_std, _parent_k, rbf_meta = (
        rbf_centers_hier_anisotropic_pca(
            A_tr,
            K_shape=K,
            n_per_cluster=n_per_cluster,
            tangent_rule="energy",
            energy_threshold=energy_threshold,
            alpha_tangent=alpha_tangent,
            robust_stat=robust_stat,
            seed=seed,
            n_init=n_init,
            standardise=standardise,
            normal_width_floor=normal_width_floor,
        )
    )
    t_place = time.time() - t0

    Phi_tr_raw = rbf_features_mahal(
        A_tr, centers_std, Sigma_invs_std,
        mu_A=rbf_meta["mu_A"], sigma_A=rbf_meta["sigma_A"],
    )
    Phi_te_raw = rbf_features_mahal(
        A_te, centers_std, Sigma_invs_std,
        mu_A=rbf_meta["mu_A"], sigma_A=rbf_meta["sigma_A"],
    )
    Phi_tr, col_norms = _normalise_columns(Phi_tr_raw)
    Phi_te = Phi_te_raw / col_norms[None, :]

    t1 = time.time()
    ridge_lambda = np.nan
    if ridge == "press":
        # PRESS-selected Tikhonov ridge: tames the near-cancelling huge-||xi||
        # solution on the (deliberately) ill-conditioned wide-normal-width Phi.
        # Mirrors run_single_cluster.py. rom-specialist 2026-06-22.
        # The ridge SVD also gives cond(Phi)=S[0]/S[-1], so we skip the
        # redundant np.linalg.cond (another full SVD of the tall Phi).
        grid = (np.logspace(-10.0, -2.0, 9) if ridge_lambda_grid is None
                else np.asarray(ridge_lambda_grid, dtype=np.float64))
        U, S, Vt = ridge_svd_factor(Phi_tr)
        cond_Phi = float(S[0] / S[-1]) if S[-1] > 0.0 else float("inf")
        press_vals = np.array(
            [ridge_svd_press(U, S, Y_tr, float(lam))["press"] for lam in grid]
        )
        ridge_lambda = float(grid[int(np.argmin(press_vals))])
        xi = ridge_svd_solve(U, S, Vt, Y_tr, ridge_lambda)
    else:
        cond_Phi = float(np.linalg.cond(Phi_tr))
        xi, *_ = np.linalg.lstsq(Phi_tr, Y_tr, rcond=None)
    t_solve = time.time() - t1

    nrmse_tr, rmse_tr_abs, std_tr = _per_mode_nrmse(Y_tr, Phi_tr @ xi)
    nrmse_te, rmse_te_abs, std_te = _per_mode_nrmse(Y_te, Phi_te @ xi)

    return {
        "n_rbf_requested": int(n_rbf_total),
        "n_rbf_eff": int(Phi_tr.shape[1]),
        "n_per_cluster": int(n_per_cluster),
        "cond_Phi_train": cond_Phi,
        "nrmse_train_per_mode": nrmse_tr,
        "nrmse_test_per_mode": nrmse_te,
        "rmse_train_per_mode_abs": rmse_tr_abs,
        "rmse_test_per_mode_abs": rmse_te_abs,
        "std_alpha_dot_train_per_mode": std_tr,
        "std_alpha_dot_test_per_mode": std_te,
        "time_place_s": float(t_place),
        "time_solve_s": float(t_solve),
        "xi_norm": float(np.linalg.norm(xi)),
        "ridge_lambda": float(ridge_lambda),
        # Fitted model objects, so the sweep can persist an integrator-ready
        # model.npz per n_rbf without a refit (run_integrate_rbf_model.py).
        "centers_std": centers_std,
        "Sigma_invs_std": Sigma_invs_std,
        "parent_k": _parent_k,
        "mu_A": rbf_meta["mu_A"],
        "sigma_A": rbf_meta["sigma_A"],
        "col_norms_train": col_norms,
        "xi": xi,
    }


def save_rbf_model(out_path: Path, out: dict, args, *, dt_native: float,
                   d_o: int, M: int, n_tr: int) -> None:
    """Persist a fitted anisotropic-RBF model in the schema read by
    `run_integrate_rbf_model.py`, WITHOUT a refit.

    Sigma_invs are stored COMPACTLY: sindy broadcasts a single per-cluster
    precision matrix across that cluster's centres, so only K_valid unique
    matrices exist. We save `Sigma_invs_unique` (K_valid, r, r) plus
    `parent_compact` (n_rbf,) indexing into it; the integrator rebuilds the
    full per-centre array via `Sigma_invs_unique[parent_compact]`. This turns
    a ~550 MB n_rbf=10000, d_o=83 model into ~13 MB on disk.
    """
    parent_k = np.asarray(out["parent_k"], dtype=np.int64)
    _uniq, first_idx, parent_compact = np.unique(
        parent_k, return_index=True, return_inverse=True
    )
    Sigma_invs_unique = np.asarray(out["Sigma_invs_std"])[first_idx]
    np.savez(
        out_path,
        centers_std=out["centers_std"],
        Sigma_invs_unique=Sigma_invs_unique,
        parent_compact=parent_compact.astype(np.int64),
        parent_k=parent_k,
        mu_A=out["mu_A"],
        sigma_A=out["sigma_A"],
        col_norms_train=out["col_norms_train"],
        xi=out["xi"],
        n_rbf=np.int32(out["n_rbf_requested"]),
        n_rbf_eff=np.int32(out["n_rbf_eff"]),
        n_per_cluster=np.int32(out["n_per_cluster"]),
        K_shape=np.int32(args.K),
        alpha_tangent=np.float64(args.alpha_tangent),
        energy_threshold=np.float64(args.energy_threshold),
        robust_stat=args.robust_stat,
        normal_width_floor=np.float64(args.normal_width_floor),
        ridge_mode=args.ridge,
        ridge_lambda=np.float64(out["ridge_lambda"]),
        cond_phi_train=np.float64(out["cond_Phi_train"]),
        xi_norm=np.float64(out["xi_norm"]),
        seed=np.int32(args.seed),
        n_init=np.int32(args.n_init),
        stride=np.int32(args.stride),
        train_frac=np.float64(args.train_frac),
        dt_native=np.float64(dt_native),
        dt_strided=np.float64(dt_native * args.stride),
        d_o=np.int32(d_o),
        M=np.int32(M),
        n_train=np.int32(n_tr),
        n_test=np.int32(M - n_tr),
        standardise=np.bool_(args.standardise),
        coords_path=str(args.coords_path.resolve()),
    )


def plot_sweep(
    out_path: Path,
    n_rbf_arr: np.ndarray,
    nrmse_train: np.ndarray,
    nrmse_test: np.ndarray,
    cond_Phi: np.ndarray,
    title: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_runs, d_o = nrmse_train.shape
    cmap = plt.get_cmap("viridis")
    mode_colors = [cmap(i / max(d_o - 1, 1)) for i in range(d_o)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, mat, lab in (
        (axes[0], nrmse_train, "train"),
        (axes[1], nrmse_test, "test"),
    ):
        for k in range(d_o):
            ax.plot(n_rbf_arr, mat[:, k], color=mode_colors[k],
                    alpha=0.6, lw=0.8)
        median = np.median(mat, axis=1)
        q25 = np.quantile(mat, 0.25, axis=1)
        q75 = np.quantile(mat, 0.75, axis=1)
        ax.plot(n_rbf_arr, median, "k-", lw=2.0, label="median across modes")
        ax.fill_between(n_rbf_arr, q25, q75, color="k", alpha=0.12,
                        label="IQR (25-75%)")
        ax.axhline(1.0, color="r", ls="--", lw=0.8, alpha=0.7,
                   label="mean-predictor baseline")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$n_{\mathrm{RBF}}$ (total)")
        ax.set_ylabel(r"NRMSE $= \mathrm{RMSE}_k\, /\, \sigma(\dot a_k)$")
        ax.set_title(f"{lab}: per-mode NRMSE")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(fontsize=8, loc="best")

    axes[2].plot(n_rbf_arr, cond_Phi, "o-", color="C3")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel(r"$n_{\mathrm{RBF}}$ (total)")
    axes[2].set_ylabel(r"$\mathrm{cond}(\Phi_{\mathrm{train}})$")
    axes[2].set_title(r"conditioning of column-normalised $\Phi$")
    axes[2].grid(True, which="both", ls=":", alpha=0.4)

    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=0, vmax=d_o - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[:2], shrink=0.8, pad=0.02,
                        location="right")
    cbar.set_label("mode index")

    fig.suptitle(title)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--coords-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--K", type=int, required=True,
                        help="number of clusters (K_shape passed to "
                             "rbf_centers_hier_anisotropic_pca). Centres "
                             "are split equally across clusters: "
                             "n_per_cluster = n_rbf_total // K.")
    parser.add_argument("--n-rbf-list", type=int, nargs="+",
                        default=[10, 25, 50, 100, 200, 400, 800, 1200, 1600,
                                 2000, 2400, 2800, 3200, 3600, 4000, 4400,
                                 4800, 5200, 5600, 6000, 6400],
                        help="total n_rbf values to sweep")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--alpha-tangent", type=float, default=5.0)
    parser.add_argument("--energy-threshold", type=float, default=0.99)
    parser.add_argument("--normal-width-floor", type=float, default=0.0,
                        help="relative floor on the normal RBF width, as a "
                             "fraction of the smallest tangent width per "
                             "cluster. Default 0.0 = no-op (current behaviour); "
                             "closes the 1-D-manifold dead zone. "
                             "rom-specialist 2026-06-22.")
    parser.add_argument("--ridge", type=str, default="none",
                        choices=["none", "press"],
                        help="inner Phi->xi solve. 'none' = plain lstsq "
                             "(default). 'press' = Tikhonov ridge, lambda by "
                             "leave-one-out PRESS over --ridge-lambda-grid.")
    parser.add_argument("--ridge-lambda-grid", type=float, nargs="+",
                        default=None,
                        help="explicit lambda grid for --ridge press; "
                             "default logspace(-10, -2, 9).")
    parser.add_argument("--robust-stat", type=str, default="median",
                        choices=["median", "p75"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--no-standardise", dest="standardise",
                        action="store_false",
                        help="run K-means and per-cluster PCA on CENTRED RAW "
                             "snapshots instead of z-scored. Eigenvalues then "
                             "carry physical energy, and the PCA frame is the "
                             "standard POD frame of the cluster.")
    parser.set_defaults(standardise=True)
    parser.add_argument("--save-models", dest="save_models",
                        action="store_true", default=True,
                        help="persist an integrator-ready model_n<value>.npz "
                             "per n_rbf (compact Sigma storage). Default on so "
                             "any swept point can be integrated later without "
                             "a refit.")
    parser.add_argument("--no-save-models", dest="save_models",
                        action="store_false",
                        help="skip writing per-n_rbf model files (metrics "
                             "only).")
    parser.add_argument("--title", type=str,
                        default="K>=1 anisotropic RBF sweep")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.coords_path}", flush=True)
    with np.load(args.coords_path) as z:
        alpha = np.asarray(z["alpha"], dtype=np.float64)
        alpha_dot = np.asarray(z["alpha_dot"], dtype=np.float64)
        dt_native = float(z["dt_native"])
        d_o = int(z["d_o"])
    print(f"  alpha.shape={alpha.shape}, dt_native={dt_native}, d_o={d_o}",
          flush=True)

    A = alpha[::args.stride]
    Y = alpha_dot[::args.stride]
    M = A.shape[0]
    n_tr = int(round(args.train_frac * M))
    A_tr, A_te = A[:n_tr], A[n_tr:]
    Y_tr, Y_te = Y[:n_tr], Y[n_tr:]
    print(f"[split] K={args.K}, stride={args.stride}, M={M}, "
          f"n_train={n_tr}, n_test={M - n_tr}", flush=True)

    n_rbf_list = list(args.n_rbf_list)
    n_runs = len(n_rbf_list)
    single_mode = (n_runs == 1)
    n_rbf_arr = np.array(n_rbf_list, dtype=np.int64)
    n_rbf_eff_arr = np.zeros(n_runs, dtype=np.int64)
    n_per_cluster_arr = np.zeros(n_runs, dtype=np.int64)
    cond_Phi_arr = np.zeros(n_runs, dtype=np.float64)
    nrmse_train = np.zeros((n_runs, d_o), dtype=np.float64)
    nrmse_test = np.zeros((n_runs, d_o), dtype=np.float64)
    rmse_train_abs = np.zeros((n_runs, d_o), dtype=np.float64)
    rmse_test_abs = np.zeros((n_runs, d_o), dtype=np.float64)
    std_alpha_dot_train = np.zeros(d_o, dtype=np.float64)
    std_alpha_dot_test = np.zeros(d_o, dtype=np.float64)
    time_place = np.zeros(n_runs, dtype=np.float64)
    time_solve = np.zeros(n_runs, dtype=np.float64)
    xi_norm = np.zeros(n_runs, dtype=np.float64)

    for i, n_rbf in enumerate(n_rbf_list):
        print(f"[run {i+1}/{n_runs}] K={args.K}, n_rbf={n_rbf} "
              f"(n_per_cluster={n_rbf // args.K})", flush=True)
        out = run_one(
            n_rbf, args.K, A_tr, Y_tr, A_te, Y_te,
            alpha_tangent=args.alpha_tangent,
            energy_threshold=args.energy_threshold,
            robust_stat=args.robust_stat,
            seed=args.seed, n_init=args.n_init,
            standardise=args.standardise,
            normal_width_floor=args.normal_width_floor,
            ridge=args.ridge,
            ridge_lambda_grid=args.ridge_lambda_grid,
        )
        n_rbf_eff_arr[i] = out["n_rbf_eff"]
        n_per_cluster_arr[i] = out["n_per_cluster"]
        cond_Phi_arr[i] = out["cond_Phi_train"]
        nrmse_train[i] = out["nrmse_train_per_mode"]
        nrmse_test[i] = out["nrmse_test_per_mode"]
        rmse_train_abs[i] = out["rmse_train_per_mode_abs"]
        rmse_test_abs[i] = out["rmse_test_per_mode_abs"]
        if i == 0:
            std_alpha_dot_train = out["std_alpha_dot_train_per_mode"]
            std_alpha_dot_test = out["std_alpha_dot_test_per_mode"]
        time_place[i] = out["time_place_s"]
        time_solve[i] = out["time_solve_s"]
        xi_norm[i] = out["xi_norm"]
        if args.save_models:
            model_path = args.out_dir / f"model_n{n_rbf:05d}.npz"
            save_rbf_model(model_path, out, args, dt_native=dt_native,
                           d_o=d_o, M=M, n_tr=n_tr)
            print(f"  [model] wrote {model_path.name} "
                  f"({model_path.stat().st_size / 1e6:.1f} MB)", flush=True)
        print(f"  n_eff={out['n_rbf_eff']} "
              f"(K*{out['n_per_cluster']}), "
              f"cond(Phi)={out['cond_Phi_train']:.2e}, "
              f"med NRMSE train={np.median(nrmse_train[i]):.3e}, "
              f"med NRMSE test={np.median(nrmse_test[i]):.3e}, "
              f"t_place={out['time_place_s']:.2f}s, "
              f"t_solve={out['time_solve_s']:.2f}s",
              flush=True)

    common_kwargs = dict(
        std_alpha_dot_train_per_mode=std_alpha_dot_train,
        std_alpha_dot_test_per_mode=std_alpha_dot_test,
        stride=np.int32(args.stride),
        train_frac=np.float64(args.train_frac),
        alpha_tangent=np.float64(args.alpha_tangent),
        energy_threshold=np.float64(args.energy_threshold),
        robust_stat=args.robust_stat,
        coords_path=str(args.coords_path.resolve()),
        d_o=np.int32(d_o),
        M=np.int32(M),
        n_train=np.int32(n_tr),
        n_test=np.int32(M - n_tr),
        dt_native=np.float64(dt_native),
        K_shape=np.int32(args.K),
        standardise=np.bool_(args.standardise),
    )

    if single_mode:
        n_val = int(n_rbf_arr[0])
        out_npz = args.out_dir / f"n_{n_val:04d}.npz"
        np.savez(
            out_npz,
            n_rbf=np.int64(n_val),
            n_rbf_eff=np.int64(n_rbf_eff_arr[0]),
            n_per_cluster=np.int64(n_per_cluster_arr[0]),
            cond_Phi_train=np.float64(cond_Phi_arr[0]),
            nrmse_train_per_mode=nrmse_train[0],
            nrmse_test_per_mode=nrmse_test[0],
            rmse_train_per_mode_abs=rmse_train_abs[0],
            rmse_test_per_mode_abs=rmse_test_abs[0],
            time_place_s=np.float64(time_place[0]),
            time_solve_s=np.float64(time_solve[0]),
            xi_norm=np.float64(xi_norm[0]),
            **common_kwargs,
        )
        print(f"[done] wrote {out_npz} (single-n_rbf mode, no plot)",
              flush=True)
        return

    out_npz = args.out_dir / "sweep.npz"
    np.savez(
        out_npz,
        n_rbf=n_rbf_arr,
        n_rbf_eff=n_rbf_eff_arr,
        n_per_cluster=n_per_cluster_arr,
        cond_Phi_train=cond_Phi_arr,
        nrmse_train_per_mode=nrmse_train,
        nrmse_test_per_mode=nrmse_test,
        rmse_train_per_mode_abs=rmse_train_abs,
        rmse_test_per_mode_abs=rmse_test_abs,
        time_place_s=time_place,
        time_solve_s=time_solve,
        xi_norm=xi_norm,
        **common_kwargs,
    )
    print(f"[done] wrote {out_npz}", flush=True)

    out_png = args.out_dir / f"rbf_K{args.K}_n_sweep.png"
    title = (f"{args.title}  |  K={args.K}, alpha_tangent={args.alpha_tangent}, "
             f"stride={args.stride}, n_train={n_tr}, n_test={M - n_tr}")
    plot_sweep(out_png, n_rbf_arr, nrmse_train, nrmse_test, cond_Phi_arr,
               title)
    print(f"[done] wrote {out_png}", flush=True)


if __name__ == "__main__":
    main()
