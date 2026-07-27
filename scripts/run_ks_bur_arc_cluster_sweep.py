"""Cluster-quality K-sweep on the KS bursting d_o=8 SVD-reduced state, fed
from the arc-length-resampled coordinate file.

Pipeline
--------
1. Load ``coords_arc.npz`` (output of ``arc_resample_coords.py``). This file
   already carries the modelled reduced state and its derivative, both
   computed at native dt on the full 600k snapshots and then linearly
   interpolated to the arc-length-uniform grid (5-point stencil on the
   resampled grid would be invalid - dt is non-uniform there).
2. Set the clustering input to ``U = alpha`` and the empirical derivative
   to ``Udot = alpha_dot``. Unlike the LOR96 SVD sweep, no analytical RHS
   is invoked; KS has no closed-form Lorenz-96-style operator we can
   evaluate at the reconstructed full state, and re-differentiating
   ``alpha`` here would defeat the resampling.
3. Stride by ``--stride`` to land in the sample-count range the
   cluster-quality kNN metric can handle (LOR96 SVD used stride 4 -> ~50k
   samples; here stride 10 -> 60k).
4. Hand ``(U, Udot, Ug, Udotg)`` to ``_init_worker`` / ``_process_K`` from
   ``run_cluster_quality_sweep``; that module owns all seven metrics and
   the plotting layer.

Following the rom-specialist note of 2026-06-19, the BIC for K is *not*
corrected here for arc-length-effective sample count (consultant
recommendation #4 was deferred). All sample-count diagnostics report
raw ``m_k`` and ``m_k_eff`` from the global ``tau_int`` of ``||U||``;
read them with that caveat.
"""

from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_cluster_quality_sweep import (  # noqa: E402
    _init_worker,
    _process_K,
    integrated_autocorr_first_zero,
    plot_metric2,
    plot_metric3,
    plot_metric4,
    plot_metric5,
    plot_metric6,
    plot_metric7,
    plot_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--coords-path", type=Path,
        default=Path("/users/sbrw610/sharedscratch/RBF_ROM/results/KS_BUR/coords_arc.npz"),
    )
    parser.add_argument(
        "--svd-path", type=Path,
        default=Path("/users/sbrw610/sharedscratch/RBF_ROM/results/KS_BUR/svd_basis.npz"),
        help="optional; used only to log cumulative energy at d_o",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("/users/sbrw610/sharedscratch/RBF_ROM/results/KS_BUR/arc_cluster_sweep"),
    )
    parser.add_argument("--K-min", type=int, default=1)
    parser.add_argument("--K-max", type=int, default=12)
    parser.add_argument("--stride", type=int, default=10,
                        help="snapshot stride into coords_arc.npz "
                             "(default 10 -> M ~ 60000)")
    parser.add_argument("--energy", type=float, default=0.99,
                        help="per-cluster PCA cumulative-energy threshold")
    parser.add_argument("--q", type=int, default=20,
                        help="kNN size for the local-determinism diagnostic")
    parser.add_argument("--n-proj", type=int, default=64,
                        help="random projection count for sliced Wasserstein-1")
    parser.add_argument("--n-null", type=int, default=10,
                        help="null repetitions for i.i.d. and block-bootstrap "
                             "nulls")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--w1-seed", type=int, default=12345)
    parser.add_argument("--iid-seed", type=int, default=54321)
    parser.add_argument("--block-seed", type=int, default=67890)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--dict-size", type=int, default=50,
                        help="dictionary size p for metric 3 - the per-cluster "
                             "RBF centre count of the CHORD2 SINDy stage. "
                             "Default 50, the RBF standard used across the "
                             "cluster-quality sweeps in this repo.")
    parser.add_argument("--shell-d", type=float, default=2.0)
    parser.add_argument("--shell-tau", type=float, default=1e-3)
    parser.add_argument("--shell-ndir", type=int, default=2048)
    parser.add_argument("--shell-seed", type=int, default=98765)
    parser.add_argument("--q-boundary", type=int, default=20)
    parser.add_argument("--m7-seed", type=int, default=24680)
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    parser.add_argument("--title-prefix", type=str,
                        default="KS_BUR arc-resampled cluster-quality")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.coords_path}", flush=True)
    with np.load(args.coords_path) as z:
        alpha_full     = np.asarray(z["alpha"], dtype=np.float64)
        alpha_dot_full = np.asarray(z["alpha_dot"], dtype=np.float64)
        d_o = int(z["d_o"])
        n_state = int(z["n_state"])
        M_in = int(z["M"])
        arc_kind = str(z["arc_kind"]) if "arc_kind" in z.files else "(none)"
    print(f"  alpha.shape={alpha_full.shape}  d_o={d_o}  "
          f"arc_kind={arc_kind}", flush=True)

    if args.svd_path.exists():
        with np.load(args.svd_path) as z:
            energy_frac = np.asarray(z["energy_frac"], dtype=np.float64)
        print(f"  cumulative energy through d_o={d_o}: "
              f"{float(energy_frac[d_o - 1]):.6f}", flush=True)

    U = np.ascontiguousarray(alpha_full[::args.stride])
    Udot = np.ascontiguousarray(alpha_dot_full[::args.stride])
    M_total = int(U.shape[0])
    print(f"  after stride={args.stride}: U.shape={U.shape}, "
          f"Udot.shape={Udot.shape}", flush=True)

    norm_u = np.linalg.norm(U, axis=1)
    tau_int_state = int(integrated_autocorr_first_zero(norm_u))
    print(f"  tau_int (||U||) = {tau_int_state} samples, M = {M_total}",
          flush=True)

    # Global PCA reference for metrics 4 and 7. Mean-centred SVD on the
    # strided U matches what `run_cluster_quality_sweep` expects in the
    # `Ug`/`Udotg` channels.
    u_mean = U.mean(axis=0)
    X_global = U - u_mean
    _, S_global, Vh_global = np.linalg.svd(X_global, full_matrices=False)
    lam_global = S_global * S_global / max(M_total - 1, 1)
    cum_global = np.cumsum(lam_global) / max(lam_global.sum(), 1e-300)
    r_global = int(np.searchsorted(cum_global, args.energy) + 1)
    r_global = min(max(r_global, 1), U.shape[1])
    V_global = Vh_global[:r_global].T
    s1g = float(lam_global.sum())
    s2g = float((lam_global * lam_global).sum())
    pr_global = (s1g * s1g) / s2g if s2g > 0.0 else float("nan")
    print(f"  r_global({args.energy}) = {r_global}, "
          f"global participation ratio = {pr_global:.3f}", flush=True)

    Ug = X_global @ V_global
    Udotg = Udot @ V_global

    K_values = list(range(args.K_min, args.K_max + 1))
    print(f"[sweep] K in {K_values}, q={args.q}, energy={args.energy}, "
          f"n_proj={args.n_proj}, workers={args.workers}", flush=True)

    ctx = get_context("spawn")
    work = [(K, args.seed, args.n_init, args.energy, args.q,
             args.n_proj, args.n_null,
             args.w1_seed, args.iid_seed, args.block_seed,
             args.shell_d, args.shell_tau, args.shell_ndir, args.shell_seed,
             args.q_boundary, args.m7_seed)
            for K in K_values]

    shards: dict[int, dict] = {}
    with ctx.Pool(
        args.workers,
        initializer=_init_worker,
        initargs=(U.astype(np.float32), Udot.astype(np.float32),
                  Ug.astype(np.float32), Udotg.astype(np.float32)),
    ) as pool:
        for shard in pool.imap_unordered(_process_K, work):
            K = shard["K"]
            shards[K] = shard
            stm_arr = shard["sin_theta_max_per_pair"]
            w1_arr = shard["w1_conditional_per_pair"]
            stm = (float(np.nanmax(stm_arr))
                   if stm_arr.size and not np.all(np.isnan(stm_arr))
                   else float("nan"))
            w1 = (float(np.nanmin(w1_arr))
                  if w1_arr.size and not np.all(np.isnan(w1_arr))
                  else float("nan"))
            print(f"  K={K:2d} done: "
                  f"L={list(shard['block_length_per_cluster'])}, "
                  f"kNN={np.nanmean(shard['ratio_kNN_per_cluster']):.3g}, "
                  f"r_iid={np.nanmean(shard['ratio_iid_per_cluster']):.3g}, "
                  f"r_block={np.nanmean(shard['ratio_block_per_cluster']):.3g}, "
                  f"sin_theta_max(adj_max)={stm:.3g}, "
                  f"w1_dist(min)={w1:.3g}",
                  flush=True)

    K_sorted = sorted(K_values)

    def agg(field: str, op):
        return np.array([op(shards[K][field]) for K in K_sorted])

    def agg_adjacent(field: str, op):
        out = np.empty(len(K_sorted), dtype=np.float64)
        for i, K in enumerate(K_sorted):
            sh = shards[K]
            mask = sh["is_adjacent_per_pair"]
            arr = sh[field]
            if mask.any():
                vals = arr[mask]
                vals = vals[~np.isnan(vals)] if vals.dtype.kind == "f" else vals
                out[i] = float(op(vals)) if vals.size else float("nan")
            else:
                out[i] = float("nan")
        return out

    def agg_all_pairs(field: str, op):
        out = np.empty(len(K_sorted), dtype=np.float64)
        for i, K in enumerate(K_sorted):
            arr = shards[K][field]
            if arr.size == 0:
                out[i] = float("nan")
                continue
            vals = arr[~np.isnan(arr)] if arr.dtype.kind == "f" else arr
            out[i] = float(op(vals)) if vals.size else float("nan")
        return out

    p_dict = int(args.dict_size)
    decorr = max(2 * tau_int_state, 1)
    m_k_eff_per_K = {
        K: (shards[K]["n_k"].astype(np.float64) / decorr) for K in K_sorted
    }
    r1_overall = np.array([
        float(np.nanmean(shards[K]["r1_mean_per_cluster"])) for K in K_sorted
    ])
    fragmentation = np.array([
        K * r1_overall[i] / max(M_total, 1) for i, K in enumerate(K_sorted)
    ])

    summary = {
        "K_values": np.array(K_sorted, dtype=np.int32),
        "ratio_kNN_mean": agg("ratio_kNN_per_cluster", np.nanmean),
        "ratio_kNN_max":  agg("ratio_kNN_per_cluster", np.nanmax),
        "w1_temporal_mean": agg("w1_temporal_per_cluster", np.nanmean),
        "w1_temporal_max":  agg("w1_temporal_per_cluster", np.nanmax),
        "w1_iid_mean_mean": agg("w1_iid_mean_per_cluster", np.nanmean),
        "w1_iid_std_mean":  agg("w1_iid_std_per_cluster",  np.nanmean),
        "ratio_iid_mean":   agg("ratio_iid_per_cluster",   np.nanmean),
        "ratio_iid_max":    agg("ratio_iid_per_cluster",   np.nanmax),
        "w1_block_mean_mean": agg("w1_block_mean_per_cluster", np.nanmean),
        "w1_block_std_mean":  agg("w1_block_std_per_cluster",  np.nanmean),
        "ratio_block_mean":   agg("ratio_block_per_cluster",   np.nanmean),
        "ratio_block_max":    agg("ratio_block_per_cluster",   np.nanmax),
        "block_length_mean":  agg("block_length_per_cluster",  np.nanmean),
        "r1_mean_mean":    agg("r1_mean_per_cluster",    np.nanmean),
        "r1_mean_min":     agg("r1_mean_per_cluster",    np.nanmin),
        "r1_median_mean":  agg("r1_median_per_cluster",  np.nanmean),
        "r1_median_min":   agg("r1_median_per_cluster",  np.nanmin),
        "r1_over_tau_mean": np.array([
            float(np.nanmean(shards[K]["r1_mean_per_cluster"]))
            / max(tau_int_state, 1)
            for K in K_sorted
        ]),
        "r1_over_tau_min": np.array([
            float(np.nanmin(shards[K]["r1_mean_per_cluster"]))
            / max(tau_int_state, 1)
            for K in K_sorted
        ]),
        "fragmentation_index": fragmentation,
        "m_k_min":       np.array([int(shards[K]["n_k"].min()) for K in K_sorted]),
        "m_k_median":    np.array([float(np.median(shards[K]["n_k"])) for K in K_sorted]),
        "m_k_mean":      np.array([float(shards[K]["n_k"].mean()) for K in K_sorted]),
        "m_k_eff_min":   np.array([float(m_k_eff_per_K[K].min()) for K in K_sorted]),
        "m_k_eff_mean":  np.array([float(m_k_eff_per_K[K].mean()) for K in K_sorted]),
        "m_k_eff_over_p_min":
            np.array([float(m_k_eff_per_K[K].min()) / p_dict for K in K_sorted]),
        "m_k_eff_over_p_mean":
            np.array([float(m_k_eff_per_K[K].mean()) / p_dict for K in K_sorted]),
        "dict_size_p":   np.int32(p_dict),
        "r_k_mean":  agg("r_k", lambda a: float(np.mean(a[a > 0])) if (a > 0).any() else float("nan")),
        "r_k_max":   agg("r_k", lambda a: int(np.max(a[a > 0])) if (a > 0).any() else 0),
        "r_k_min":   agg("r_k", lambda a: int(np.min(a[a > 0])) if (a > 0).any() else 0),
        "compression_gain_mean": np.array([
            float(np.mean(shards[K]["r_k"][shards[K]["r_k"] > 0])) / max(r_global, 1)
            if (shards[K]["r_k"] > 0).any() else float("nan")
            for K in K_sorted
        ]),
        "compression_gain_max":  np.array([
            float(np.max(shards[K]["r_k"][shards[K]["r_k"] > 0])) / max(r_global, 1)
            if (shards[K]["r_k"] > 0).any() else float("nan")
            for K in K_sorted
        ]),
        "participation_ratio_mean":
            agg("participation_ratio_per_cluster", np.nanmean),
        "participation_ratio_min":
            agg("participation_ratio_per_cluster", np.nanmin),
        "r_global":         np.int32(r_global),
        "participation_ratio_global": np.float64(pr_global),
        "condition_number_mean": agg("condition_number_per_cluster", np.nanmean),
        "condition_number_max":  agg("condition_number_per_cluster", np.nanmax),
        "spectral_gap_at_elbow_mean":
            agg("spectral_gap_at_elbow_per_cluster", np.nanmean),
        "spectral_gap_at_elbow_min":
            agg("spectral_gap_at_elbow_per_cluster", np.nanmin),
        "dead_zone_fraction_mean":
            agg("dead_zone_fraction_per_cluster", np.nanmean),
        "dead_zone_fraction_max":
            agg("dead_zone_fraction_per_cluster", np.nanmax),
        "shell_d":    np.float64(args.shell_d),
        "shell_tau":  np.float64(args.shell_tau),
        "shell_ndir": np.int32(args.shell_ndir),
        "sin_theta_max_mean":    agg_adjacent("sin_theta_max_per_pair", np.mean),
        "sin_theta_max_max":     agg_adjacent("sin_theta_max_per_pair", np.max),
        "centroid_shift_mean":   agg_adjacent("centroid_shift_per_pair", np.mean),
        "centroid_shift_max":    agg_adjacent("centroid_shift_per_pair", np.max),
        "centroid_shift_ratio_mean":
            agg_adjacent("centroid_shift_ratio_per_pair", np.mean),
        "alpha_dot_jump_mean":   agg_adjacent("alpha_dot_jump_per_pair", np.mean),
        "alpha_dot_jump_max":    agg_adjacent("alpha_dot_jump_per_pair", np.max),
        "n_adjacent_pairs":      np.array([
            int(shards[K]["is_adjacent_per_pair"].sum()) for K in K_sorted
        ]),
        "switch_rate_mean": np.array([
            float(shards[K]["switch_count_per_pair"][
                shards[K]["is_adjacent_per_pair"]
            ].mean()) / max(M_total, 1)
            if shards[K]["is_adjacent_per_pair"].any() else float("nan")
            for K in K_sorted
        ]),
        "switch_rate_max": np.array([
            float(shards[K]["switch_count_per_pair"].max()) / max(M_total, 1)
            if shards[K]["switch_count_per_pair"].size else float("nan")
            for K in K_sorted
        ]),
        "switch_rate_total": np.array([
            float(shards[K]["switch_count_per_pair"].sum()) / max(M_total, 1)
            for K in K_sorted
        ]),
        "w1_conditional_min":    agg_all_pairs("w1_conditional_per_pair", np.min),
        "w1_conditional_mean":   agg_all_pairs("w1_conditional_per_pair", np.mean),
        "w1_conditional_max":    agg_all_pairs("w1_conditional_per_pair", np.max),
        "jacobian_distance_min": agg_all_pairs("jacobian_distance_per_pair", np.min),
        "jacobian_distance_mean":agg_all_pairs("jacobian_distance_per_pair", np.mean),
        "jacobian_distance_max": agg_all_pairs("jacobian_distance_per_pair", np.max),
        "q_boundary":      np.int32(args.q_boundary),
        "tau_int_state": np.int64(tau_int_state),
        "q": np.int32(args.q),
        "energy": np.float64(args.energy),
        "n_proj": np.int32(args.n_proj),
        "n_null": np.int32(args.n_null),
        "stride": np.int32(args.stride),
        "M": np.int64(M_total),
        "N": np.int64(U.shape[1]),
        "d_o": np.int64(d_o),
        "n_state": np.int64(n_state),
        "M_in_coords_arc": np.int64(M_in),
    }
    np.savez(args.out_dir / "summary.npz", **summary)

    for K in K_sorted:
        s = shards[K]
        np.savez(
            args.out_dir / f"K={K:02d}.npz",
            labels=s["labels"],
            centroids=s["centroids"],
            inertia=np.float64(s["inertia"]),
            n_iter=np.int32(s["n_iter"]),
            n_k=s["n_k"],
            r_k=s["r_k"],
            ratio_kNN_per_cluster=s["ratio_kNN_per_cluster"],
            w1_temporal_per_cluster=s["w1_temporal_per_cluster"],
            w1_iid_mean_per_cluster=s["w1_iid_mean_per_cluster"],
            w1_iid_std_per_cluster=s["w1_iid_std_per_cluster"],
            ratio_iid_per_cluster=s["ratio_iid_per_cluster"],
            w1_block_mean_per_cluster=s["w1_block_mean_per_cluster"],
            w1_block_std_per_cluster=s["w1_block_std_per_cluster"],
            ratio_block_per_cluster=s["ratio_block_per_cluster"],
            block_length_per_cluster=s["block_length_per_cluster"],
            r1_mean_per_cluster=s["r1_mean_per_cluster"],
            r1_median_per_cluster=s["r1_median_per_cluster"],
            n_runs_per_cluster=s["n_runs_per_cluster"],
            m_k_eff_per_cluster=m_k_eff_per_K[K],
            participation_ratio_per_cluster=s["participation_ratio_per_cluster"],
            condition_number_per_cluster=s["condition_number_per_cluster"],
            spectral_gap_at_elbow_per_cluster=s["spectral_gap_at_elbow_per_cluster"],
            dead_zone_fraction_per_cluster=s["dead_zone_fraction_per_cluster"],
            pair_i_m6=s["pair_i_m6"],
            pair_j_m6=s["pair_j_m6"],
            is_adjacent_per_pair=s["is_adjacent_per_pair"],
            switch_count_per_pair=s["switch_count_per_pair"],
            sin_theta_max_per_pair=s["sin_theta_max_per_pair"],
            centroid_shift_per_pair=s["centroid_shift_per_pair"],
            centroid_shift_ratio_per_pair=s["centroid_shift_ratio_per_pair"],
            alpha_dot_jump_per_pair=s["alpha_dot_jump_per_pair"],
            pair_i_m7=s["pair_i_m7"],
            pair_j_m7=s["pair_j_m7"],
            w1_conditional_per_pair=s["w1_conditional_per_pair"],
            jacobian_distance_per_pair=s["jacobian_distance_per_pair"],
        )

    tp = args.title_prefix
    plot_summary(args.out_dir, summary, q=args.q, energy=args.energy,
                 n_null=args.n_null, title_prefix=tp)
    plot_metric2(args.out_dir, summary, energy=args.energy, title_prefix=tp)
    plot_metric3(args.out_dir, summary, title_prefix=tp)
    plot_metric4(args.out_dir, summary, title_prefix=tp)
    plot_metric5(args.out_dir, summary, title_prefix=tp)
    plot_metric6(args.out_dir, summary, q=args.q_boundary, title_prefix=tp)
    plot_metric7(args.out_dir, summary, title_prefix=tp)
    print(f"[done] outputs in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
