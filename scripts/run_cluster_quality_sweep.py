"""Cluster-quality K sweep, metrics 1 (dynamical coherence), 2 (dwell time),
3 (sample sufficiency), 4 (genuine local-rank reduction), 5 (meaningful
anisotropy), 6 (soft interface to neighbours) and 7 (distinctness from
other clusters).

Implements items 1, 2, 3, 4, 5, 6 and 7 of
`docs/notes/cluster-quality-diagnostics.md`:

  1a. kNN local-determinism on `(alpha, alpha_dot)`. For each snapshot in
      cluster `k`, take its `q` nearest neighbours in `alpha`-space *within
      the cluster* and compute the trace of the covariance of their
      `alpha_dot`. Report the cluster-mean of this local variance divided
      by the within-cluster trace-covariance of `alpha_dot`. Low ratio =
      `alpha_dot` behaves as a function of `alpha` inside the cluster (one
      regime); high ratio = multi-valued at fixed `alpha`.

  1b. Split-half distribution stability on `(alpha, alpha_dot)`, with
      a random-split null. Report (i) the sliced Wasserstein-1 distance
      between the empirical joints of the two temporal halves and (ii)
      the same quantity averaged over `n_null` random half-splits of the
      cluster's snapshots. The informative quantity is the ratio
      `W1_temporal / W1_null` - on a stationary, ergodic cluster the
      ratio is ~1.0 regardless of `n_k` because both numerator and
      denominator collapse to the small-sample sliced-W1 floor at that
      population; values significantly above 1.0 mean the cluster's
      `(alpha, alpha_dot)` distribution drifts in time.

  5. Meaningful anisotropy. State-vector only. Three sub-diagnostics per
     cluster, all based on the local-PCA eigenstructure and the
     PCA-induced precision matrix `Sigma_k^{-1} = V_k diag(1/lambda_k)
     V_k^T`. Decoupled from any specific RBF dictionary so the metric
     transfers cleanly across testbeds.

       condition_number    `lambda_k^max / lambda_k^min`. The squared
                           tangent-to-normal width ratio of the
                           PCA-induced precision matrix.
       spectral_gap_at_elbow
                           `lambda_k^{(r_k)} / lambda_k^{(r_k - 1)}` -
                           the relative drop in the spectrum at the
                           energy-elbow cut. Small means a sharp elbow
                           (clean rank reduction); ~1 means no elbow
                           (the truncation is arbitrary).
       dead_zone_fraction  Sample `n_dir` uniformly distributed unit
                           vectors `v` on the sphere in R^N; for each
                           place a shell point `x = c_k + d *
                           sqrt(lambda_k^max) * v` and evaluate the
                           centroid Gaussian
                           `phi_k(x) = exp(-0.5 * (x - c_k)^T
                           Sigma_k^{-1} (x - c_k))`. Report the
                           fraction of shell directions with
                           `phi_k(x) < tau`. Direct measurement of
                           anisotropy-blindness: a high fraction means
                           the PCA-induced kernel collapses to zero
                           along low-eigenvalue directions even at
                           shell radius set by the largest eigenvalue.
                           See
                           `docs/journal/2026-06-10-anisotropic-rbf-per-cluster-design.md`.

  4. Genuine local-rank reduction. State-vector only. Per cluster the
     local PCA produces an eigenspectrum `lambda_k`; the energy-elbow
     rank at threshold `energy` (default 0.99) is `r_k`. Reported:

       r_k                per-cluster local-PCA rank at `energy`.
       compression_gain   `r_k / r_global` where `r_global` is the
                          same energy-elbow rank on the full-trajectory
                          PCA (computed once in `main`). A value
                          significantly below 1 means clustering bought
                          true local-manifold compression; ~1 means
                          clusters are just chunks of the same global
                          manifold and the per-cluster PCA overhead
                          earns nothing.
       participation_ratio  `(sum lambda_k)^2 / sum (lambda_k^2)`, a
                          single-number effective dimension of the
                          cluster's local eigenspectrum.

     The diagnostic has little dynamic range when the ambient
     dimension is small (e.g. `N=3` on LOR63), but is essential on the
     PDE testbeds (KOL, KS, RB) where `N` runs into the hundreds.

  3. Sufficient sample count for a stable local fit. State-trajectory
     only; no fit. Per cluster:

       m_k      raw snapshot count.
       m_k_eff  decorrelated count `m_k / (2 tau_int)`, with `tau_int`
                taken from the global integrated autocorrelation of
                `||u||_2` in sample units.
       m_k_eff / p
                ratio against a CLI-supplied dictionary size `p`
                (default 10, the polynomial-degree-2 SINDy library for
                3D state).

     The design-note danger zone is `m_k_eff / p < ~5`, the small-
     cluster S3 overfit pathology documented in
     `docs/journal/2026-06-09-l63-clustered-rbf-stages-A-B-E-F.md`.
     Reading the K-axis: above the K where the smallest cluster's
     `m_k_eff / p` drops below ~5, the partition cannot host a stable
     local regression at that dictionary size.

  6. Soft interface to neighbours. State-vector + PCA + analytical
     `u_dot`. For every cluster pair `(i, j)` with `i < j`:

       principal_angle    `sin(theta_max)` between `range(V_i)` and
                          `range(V_j)`, computed from the singular
                          values of `V_j^T V_i` (truncated to common
                          rank `r* = min(r_i, r_j)`). 0 = identical
                          subspaces, 1 = orthogonal.
       centroid_shift     `||V_j^T (c_i - c_j)||` - the part of the
                          centroid shift that `V_j`'s coordinate
                          system actually sees on a switch from `i`
                          to `j`. Also reported as a fraction of
                          `||c_i - c_j||`.
       alpha_dot_boundary_jump
                          `||<alpha_dot>_i - <alpha_dot>_j||` in the
                          common `V_j` frame, computed over the `q`
                          snapshots on each cluster's side that lie
                          closest to the `(i, j)` Voronoi boundary
                          (smallest signed margin). Population
                          smoothing - the kNN role of the design
                          note - with no fit. Reported only for
                          adjacent pairs (pairs with at least one
                          observed `i <-> j` transition); other
                          pairs are flagged non-adjacent and skipped.
       switch_rate        observed `i <-> j` transitions per snapshot.

     Per-K aggregates report means and worst-case (max) values over
     the *adjacent* pair set (the pair set where the boundary is
     actually crossed), together with the adjacent-pair count.

  7. Distinctness from other clusters. State vector projected to the
     global energy-elbow PCA frame `V_global` (rank `r_global`,
     computed once in `main`) so the comparison is in a common,
     testbed-independent coordinate system that does not privilege
     any one cluster. For every pair `(i, j)`:

       w1_conditional      sliced Wasserstein-1 distance between the
                           joint `(alpha_global, alpha_dot_global)`
                           empirical distributions of clusters `i`
                           and `j`, after per-feature standardisation
                           on the union. Two clusters that encode the
                           same local balance produce indistinguishable
                           joints; large W1 = genuinely distinct
                           regimes.
       jacobian_distance   Frobenius norm
                           `||J_i - J_j||_F` of the per-cluster
                           OLS Jacobian `J_k` (the slope of
                           `alpha_dot_global` regressed on
                           `alpha_global - mean_k`). A single-number
                           summary of the redundancy between the two
                           clusters' local linearised dynamics with
                           no SINDy fit.

     Per-K aggregates report the *minimum* over pairs (the closest
     pair - if even the most similar pair is far apart, the
     partition is informative) and the mean and max.

  2. Adequate dwell time. State-vector only, no PCA, no `alpha_dot`.
     Per cluster `k`, mean and median run-length `r1_k` of consecutive
     timesteps in the cluster-label sequence. The reference timescale
     is `tau_int` from the autocorrelation of `||u||_2` (computed once,
     does not depend on `K`). Reported quantities are `r1_k / tau_int`
     per cluster and the global fragmentation index
     `K * <r1> / (T_total / Delta_t)`. Equality to 1.0 in the
     fragmentation index means K-means is slicing a smooth manifold
     into K angular sectors of equal dwell.

`alpha` is the per-cluster local-PCA reduced coordinate with rank `r_k`
fixed by an energy-elbow rule (default 0.99). No SINDy fit anywhere -
`xi_k` does not enter.

On LOR63 the empirical `alpha_dot` is taken from the analytical RHS
(sigma, rho, beta read from `stats.npz`), avoiding finite-difference
noise on a known model. Generalisation to datasets without an
analytical RHS is a TODO marker, not implemented here.

Outputs under `--out-dir`:

  K=XX.npz     per-K shard: labels, centroids, per-cluster `n_k`, `r_k`,
               metric-1, -2, -3, -4, -5 arrays, plus per-pair metric-6
               and metric-7 arrays with `(pair_i, pair_j)` indexing.
  summary.npz  K-indexed cluster-aggregated statistics.
  summary.png             metric-1 plot vs K.
  summary_metric2.png     metric-2 plot vs K.
  summary_metric3.png     metric-3 plot vs K.
  summary_metric4.png     metric-4 plot vs K.
  summary_metric5.png     metric-5 plot vs K.
  summary_metric6.png     metric-6 plot vs K (adjacent-pair aggregates).
  summary_metric7.png     metric-7 plot vs K (all-pair aggregates).
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

from chord2.clustering import kmeans_fit  # noqa: E402


# ---------------------------------------------------------------------------
# LOR63-specific RHS
# ---------------------------------------------------------------------------

def lor63_rhs_batch(u: np.ndarray, sigma: float, rho: float, beta: float) -> np.ndarray:
    """Analytical L63 RHS evaluated on (M, 3)."""
    x, y, z = u[:, 0], u[:, 1], u[:, 2]
    return np.stack([
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z,
    ], axis=1)


# ---------------------------------------------------------------------------
# Per-cluster machinery
# ---------------------------------------------------------------------------

def local_pca_energy(u_k: np.ndarray, c_k: np.ndarray, energy: float
                     ) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-cluster PCA centred on the cluster centroid, energy-elbow rank.

    Returns (V, lam, r) with V of shape (N, r) and lam the full eigenvalue
    spectrum (length min(n_k, N)). r is the smallest rank with cumulative
    variance >= `energy`, capped at the ambient dimension.
    """
    X = (u_k - c_k).astype(np.float64)
    _, S, Vh = np.linalg.svd(X, full_matrices=False)
    lam = S * S / max(u_k.shape[0] - 1, 1)
    total = lam.sum()
    if total <= 0.0:
        r = 1
    else:
        cum = np.cumsum(lam) / total
        r = int(np.searchsorted(cum, energy) + 1)
    r = min(max(r, 1), X.shape[1])
    return Vh[:r].T, lam, r


def knn_local_determinism(alpha: np.ndarray, alpha_dot: np.ndarray, q: int) -> float:
    """Cluster-mean local trace-covariance of `alpha_dot` over q-NN in `alpha`,
    normalised by the within-cluster trace-covariance of `alpha_dot`.

    Returns NaN if the cluster has fewer than `q + 1` snapshots or the
    within-cluster variance is zero.
    """
    from scipy.spatial import cKDTree

    n = alpha.shape[0]
    if n < q + 1:
        return float("nan")

    tree = cKDTree(alpha)
    _, idx = tree.query(alpha, k=q)  # (n, q) includes the point itself
    nbrs = alpha_dot[idx]  # (n, q, d)
    nbr_mean = nbrs.mean(axis=1, keepdims=True)
    local_trvar = ((nbrs - nbr_mean) ** 2).sum(axis=(1, 2)) / max(q - 1, 1)

    cluster_mean = alpha_dot.mean(axis=0)
    cluster_trvar = ((alpha_dot - cluster_mean) ** 2).sum() / max(n - 1, 1)
    if cluster_trvar <= 0.0:
        return float("nan")
    return float(local_trvar.mean() / cluster_trvar)


def sliced_wasserstein1(X: np.ndarray, Y: np.ndarray, n_proj: int,
                        rng: np.random.Generator) -> float:
    """Sliced Wasserstein-1 between (nX, d) and (nY, d) empirical samples."""
    from scipy.stats import wasserstein_distance

    d = X.shape[1]
    if d == 0 or X.shape[0] < 2 or Y.shape[0] < 2:
        return float("nan")
    G = rng.standard_normal((d, n_proj))
    G /= np.linalg.norm(G, axis=0, keepdims=True) + 1e-12
    Xp = X @ G
    Yp = Y @ G
    w = np.empty(n_proj, dtype=np.float64)
    for j in range(n_proj):
        w[j] = wasserstein_distance(Xp[:, j], Yp[:, j])
    return float(w.mean())


def integrated_autocorr_first_zero(x: np.ndarray) -> int:
    """Integrated autocorrelation time of a 1D real series, in sample units.

    FFT-based unbiased autocorrelation, integrated to the first non-positive
    lag. Returns max(1, n_zero // 2) so that a series with no detectable
    correlation gives a block length of 1 (no actual blocking). Used to set
    the circular-block-bootstrap block length per cluster.
    """
    x = np.asarray(x, dtype=np.float64) - float(np.mean(x))
    n = x.shape[0]
    if n < 4:
        return 1
    n_pow = 1
    while n_pow < 2 * n:
        n_pow *= 2
    F = np.fft.rfft(x, n=n_pow)
    ac = np.fft.irfft(F * np.conj(F), n=n_pow)[:n]
    if ac[0] <= 0.0:
        return 1
    ac = ac / ac[0]
    zero = np.flatnonzero(ac <= 0.0)
    n_zero = int(zero[0]) if zero.size else n
    tau = float(ac[:n_zero].sum())
    return max(1, int(round(tau)))


def circular_block_resample(
    X: np.ndarray, m: int, L: int, rng: np.random.Generator,
) -> np.ndarray:
    """Circular block bootstrap: draw `m` samples in blocks of length `L`.

    Pick `ceil(m / L)` uniform starting indices in `[0, n)`, take blocks of
    length `L` wrapping around end-to-end, concatenate, truncate to `m`.
    Preserves within-block serial structure, destroys block-to-block order.
    """
    n = X.shape[0]
    L = max(1, min(int(L), n))
    n_blocks = (m + L - 1) // L
    starts = rng.integers(0, n, size=n_blocks)
    offsets = np.arange(L, dtype=np.int64)
    idx = (starts[:, None] + offsets[None, :]) % n
    return X[idx.reshape(-1)[:m]]


def split_half_w1_with_nulls(
    alpha: np.ndarray,
    alpha_dot: np.ndarray,
    n_proj: int,
    n_null: int,
    seed_temporal: int,
    seed_iid: int,
    seed_block: int,
) -> dict:
    """Sliced W1 of `(alpha, alpha_dot)` under temporal split, with two nulls.

    Two distinct nulls so the user can read drift apart from autocorrelation:

      - **i.i.d. random null** (existing): random partition into two halves;
        each half is approximately an i.i.d. resample. Captures sampling
        noise only. Ratio `W1_t / W1_iid` inflates with both autocorrelation
        and drift.
      - **circular block-bootstrap null** (new): each half is built from
        `m / L` random blocks of length `L` wrapping around the cluster's
        ordered subsequence, where `L` is the integrated autocorrelation
        time of the cluster's first PC (in cluster-visit units). Captures
        sampling noise *plus* within-block autocorrelation. Ratio
        `W1_t / W1_block` collapses to ~1 on a stationary cluster
        regardless of autocorrelation strength; values above 1 indicate
        genuine non-stationarity on scales longer than `L`.

    Joint is per-feature standardised before any split. The cluster's
    snapshots must be passed in trajectory-visit order so the block
    structure is meaningful.

    Returns a dict with all primary measurements plus the chosen `L`.
    """
    joint = np.concatenate([alpha, alpha_dot], axis=1)
    n = joint.shape[0]
    nan = float("nan")
    if n < 4:
        return {
            "w1_temporal": nan,
            "w1_iid_mean": nan, "w1_iid_std": nan, "ratio_iid": nan,
            "w1_block_mean": nan, "w1_block_std": nan, "ratio_block": nan,
            "block_length": 0,
        }

    mean = joint.mean(axis=0)
    std = joint.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    joint = (joint - mean) / std
    h = n // 2

    rng_t = np.random.default_rng(seed_temporal)
    w1_t = sliced_wasserstein1(joint[:h], joint[h:], n_proj, rng_t)

    # i.i.d. random null
    rng_iid = np.random.default_rng(seed_iid)
    w1_iid = np.empty(n_null, dtype=np.float64)
    for i in range(n_null):
        perm = rng_iid.permutation(n)
        w1_iid[i] = sliced_wasserstein1(
            joint[perm[:h]], joint[perm[h:]], n_proj, rng_iid
        )
    w1_iid_mean = float(w1_iid.mean())
    w1_iid_std = float(w1_iid.std())

    # Block-bootstrap null
    L = integrated_autocorr_first_zero(alpha[:, 0])
    L = max(1, min(L, n // 4))  # cap to keep ratio non-trivial
    rng_b = np.random.default_rng(seed_block)
    w1_block = np.empty(n_null, dtype=np.float64)
    for i in range(n_null):
        A = circular_block_resample(joint, h, L, rng_b)
        B = circular_block_resample(joint, n - h, L, rng_b)
        w1_block[i] = sliced_wasserstein1(A, B, n_proj, rng_b)
    w1_block_mean = float(w1_block.mean())
    w1_block_std = float(w1_block.std())

    ratio_iid = w1_t / w1_iid_mean if w1_iid_mean > 0 else nan
    ratio_block = w1_t / w1_block_mean if w1_block_mean > 0 else nan
    return {
        "w1_temporal": float(w1_t),
        "w1_iid_mean": w1_iid_mean, "w1_iid_std": w1_iid_std,
        "ratio_iid": float(ratio_iid),
        "w1_block_mean": w1_block_mean, "w1_block_std": w1_block_std,
        "ratio_block": float(ratio_block),
        "block_length": int(L),
    }


# ---------------------------------------------------------------------------
# Metric 5 helpers (anisotropy + dead-zone shell coverage)
# ---------------------------------------------------------------------------

def dead_zone_fraction(
    lam: np.ndarray, d: float, tau: float, n_dir: int,
    rng: np.random.Generator,
) -> float:
    """Fraction of shell directions on which the centroid Gaussian dies below
    `tau` at shell radius `d * sqrt(lambda_max)`.

    The PCA-induced precision matrix is `Sigma^{-1} = V diag(1/lambda) V^T`;
    in the PCA frame the kernel quadratic form is diagonal, so the dead-zone
    test is rotation-invariant and can be evaluated directly on shell
    directions sampled in R^N without constructing V. Concretely, for unit
    direction `v` and shell radius `R = d * sqrt(lambda_max)`:

        (x - c)^T Sigma^{-1} (x - c) = R^2 * sum_i v_i^2 / lambda_i,

    so `phi(x) = exp(-0.5 R^2 sum_i v_i^2 / lambda_i)`. The cluster is dead
    in direction `v` iff that exceeds `-2 log tau`.
    """
    lam = np.asarray(lam, dtype=np.float64)
    lam = lam[lam > 0.0]
    if lam.size == 0 or n_dir < 1:
        return float("nan")
    lam_max = float(lam.max())
    N = lam.size
    G = rng.standard_normal((n_dir, N))
    G /= (np.linalg.norm(G, axis=1, keepdims=True) + 1e-300)
    # Quadratic form in the PCA frame: R^2 * sum v_i^2 / lambda_i.
    qf = (d * d) * lam_max * np.einsum("ij,j->i", G * G, 1.0 / lam)
    threshold = -2.0 * np.log(max(tau, 1e-300))
    dead = qf > threshold
    return float(np.mean(dead))


# ---------------------------------------------------------------------------
# Metric 2 helpers
# ---------------------------------------------------------------------------

def cluster_run_lengths(labels: np.ndarray, K: int) -> dict:
    """Per-cluster mean / median run length and run count.

    Run lengths are taken from contiguous segments of equal label in the
    trajectory-order label sequence. NaN for clusters with no occupied
    snapshots.
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        return {
            "r1_mean": np.full(K, np.nan),
            "r1_median": np.full(K, np.nan),
            "n_runs": np.zeros(K, dtype=np.int64),
        }
    changes = np.flatnonzero(np.diff(labels) != 0) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [labels.size]))
    lengths = (ends - starts).astype(np.int64)
    run_labels = labels[starts]

    r1_mean = np.full(K, np.nan, dtype=np.float64)
    r1_median = np.full(K, np.nan, dtype=np.float64)
    n_runs = np.zeros(K, dtype=np.int64)
    for k in range(K):
        sel = run_labels == k
        if sel.any():
            L = lengths[sel]
            r1_mean[k] = float(L.mean())
            r1_median[k] = float(np.median(L))
            n_runs[k] = int(sel.sum())
    return {"r1_mean": r1_mean, "r1_median": r1_median, "n_runs": n_runs}


# ---------------------------------------------------------------------------
# Metric 6 helpers (soft interface to neighbours)
# ---------------------------------------------------------------------------

def boundary_alpha_dot_jump(
    U_i: np.ndarray, Udot_i: np.ndarray,
    U_j: np.ndarray, Udot_j: np.ndarray,
    c_i: np.ndarray, c_j: np.ndarray,
    V_j: np.ndarray, q: int,
) -> float:
    """Empirical `alpha_dot` discontinuity at the Voronoi boundary of (i, j),
    expressed in the common `V_j` frame.

    For each snapshot on side `s`, the signed Voronoi margin is
    `m = ||u - c_(not s)||^2 - ||u - c_s||^2`; small positive `m` =
    close to the boundary, large positive `m` = deep in side `s`. We
    take the `q` snapshots on each side with smallest positive margin
    (the kNN role of the design note), project their `u_dot` to `V_j`,
    and return the norm of the difference of means.
    """
    n_i, n_j = U_i.shape[0], U_j.shape[0]
    if n_i < 1 or n_j < 1:
        return float("nan")

    margin_i = np.sum((U_i - c_j) ** 2, axis=1) - np.sum((U_i - c_i) ** 2, axis=1)
    margin_j = np.sum((U_j - c_i) ** 2, axis=1) - np.sum((U_j - c_j) ** 2, axis=1)
    qi = min(q, n_i)
    qj = min(q, n_j)
    if qi < 1 or qj < 1:
        return float("nan")

    idx_i = np.argpartition(margin_i, qi - 1)[:qi]
    idx_j = np.argpartition(margin_j, qj - 1)[:qj]

    alpha_dot_i = Udot_i[idx_i] @ V_j
    alpha_dot_j = Udot_j[idx_j] @ V_j
    return float(np.linalg.norm(alpha_dot_i.mean(axis=0)
                                - alpha_dot_j.mean(axis=0)))


def pairwise_metric6(
    U: np.ndarray, Udot: np.ndarray, labels: np.ndarray, K: int,
    centroids: np.ndarray, V_list: list, q: int,
) -> dict:
    """Per-pair metric-6 quantities for all `(i, j)` with `i < j`.

    `V_list[k]` is the per-cluster PCA basis (N, r_k) or None for
    too-small clusters. Returns flat arrays of length `K*(K-1)/2`.
    """
    n_pairs = K * (K - 1) // 2
    pair_i = np.zeros(n_pairs, dtype=np.int32)
    pair_j = np.zeros(n_pairs, dtype=np.int32)
    is_adjacent = np.zeros(n_pairs, dtype=bool)
    switch_count = np.zeros(n_pairs, dtype=np.int64)
    sin_theta_max = np.full(n_pairs, np.nan, dtype=np.float64)
    centroid_shift = np.full(n_pairs, np.nan, dtype=np.float64)
    centroid_shift_ratio = np.full(n_pairs, np.nan, dtype=np.float64)
    alpha_dot_jump = np.full(n_pairs, np.nan, dtype=np.float64)

    if K < 2:
        return {
            "pair_i": pair_i, "pair_j": pair_j,
            "is_adjacent": is_adjacent,
            "switch_count": switch_count,
            "sin_theta_max": sin_theta_max,
            "centroid_shift": centroid_shift,
            "centroid_shift_ratio": centroid_shift_ratio,
            "alpha_dot_jump": alpha_dot_jump,
        }

    # Build an adjacency/switch-count table from the label sequence.
    diff = labels[1:] != labels[:-1]
    if diff.any():
        a = labels[:-1][diff]
        b = labels[1:][diff]
        # Symmetric switch count between unordered pairs.
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
    else:
        lo = np.zeros(0, dtype=labels.dtype)
        hi = np.zeros(0, dtype=labels.dtype)

    p = 0
    for i in range(K):
        Vi = V_list[i]
        ci = centroids[i]
        mask_i = labels == i
        U_i = U[mask_i]
        Udot_i = Udot[mask_i]
        for j in range(i + 1, K):
            pair_i[p] = i
            pair_j[p] = j

            sw = int(((lo == i) & (hi == j)).sum())
            switch_count[p] = sw
            is_adjacent[p] = sw > 0

            Vj = V_list[j]
            cj = centroids[j]

            # Centroid-shift magnitude in V_j frame and its ratio to total.
            shift_total = float(np.linalg.norm(ci - cj))
            if Vj is not None and Vj.size:
                shift_in_Vj = float(np.linalg.norm(Vj.T @ (ci - cj)))
                centroid_shift[p] = shift_in_Vj
                centroid_shift_ratio[p] = (
                    shift_in_Vj / shift_total if shift_total > 0 else float("nan")
                )

            # Principal angle (largest) between the two PCA subspaces.
            if Vi is not None and Vj is not None and Vi.size and Vj.size:
                r_star = min(Vi.shape[1], Vj.shape[1])
                if r_star >= 1:
                    M = Vj[:, :r_star].T @ Vi[:, :r_star]
                    sv = np.linalg.svd(M, compute_uv=False)
                    sv = np.clip(sv, 0.0, 1.0)
                    sin_theta_max[p] = float(np.sqrt(max(0.0, 1.0 - sv.min() ** 2)))

            # Empirical alpha_dot jump - only meaningful on adjacent pairs.
            if is_adjacent[p] and Vj is not None and Vj.size:
                mask_j = labels == j
                alpha_dot_jump[p] = boundary_alpha_dot_jump(
                    U_i, Udot_i,
                    U[mask_j], Udot[mask_j],
                    ci, cj, Vj, q,
                )

            p += 1

    return {
        "pair_i": pair_i, "pair_j": pair_j,
        "is_adjacent": is_adjacent,
        "switch_count": switch_count,
        "sin_theta_max": sin_theta_max,
        "centroid_shift": centroid_shift,
        "centroid_shift_ratio": centroid_shift_ratio,
        "alpha_dot_jump": alpha_dot_jump,
    }


# ---------------------------------------------------------------------------
# Metric 7 helpers (distinctness)
# ---------------------------------------------------------------------------

def pairwise_metric7(
    Ug: np.ndarray, Udotg: np.ndarray, labels: np.ndarray, K: int,
    n_proj: int, rng: np.random.Generator,
) -> dict:
    """Pairwise sliced-W1 distinctness and Jacobian-distance.

    `Ug` and `Udotg` are the snapshots and their derivatives projected
    to the global energy-elbow PCA frame `V_global` (rank `r_global`).
    The common frame makes the comparison testbed-independent and
    avoids privileging either cluster's local basis.

    Per cluster the OLS Jacobian is `J_k = argmin_J ||Udot_g - J
    (Ug - <Ug>_k)||_F^2`, restricted to the cluster's snapshots.
    """
    n_pairs = K * (K - 1) // 2
    pair_i = np.zeros(n_pairs, dtype=np.int32)
    pair_j = np.zeros(n_pairs, dtype=np.int32)
    w1_conditional = np.full(n_pairs, np.nan, dtype=np.float64)
    jac_distance = np.full(n_pairs, np.nan, dtype=np.float64)

    if K < 2:
        return {
            "pair_i": pair_i, "pair_j": pair_j,
            "w1_conditional": w1_conditional,
            "jacobian_distance": jac_distance,
        }

    joints: list = [None] * K
    jacobians: list = [None] * K
    r_global = Ug.shape[1]
    for k in range(K):
        mask = labels == k
        if int(mask.sum()) < max(4, r_global + 1):
            continue
        X = Ug[mask].astype(np.float64)
        Y = Udotg[mask].astype(np.float64)
        joints[k] = np.concatenate([X, Y], axis=1)
        Xc = X - X.mean(axis=0)
        Yc = Y - Y.mean(axis=0)
        J, *_ = np.linalg.lstsq(Xc, Yc, rcond=None)
        jacobians[k] = J  # (r_global, r_global)

    p = 0
    for i in range(K):
        for j in range(i + 1, K):
            pair_i[p] = i
            pair_j[p] = j
            Ji = joints[i]
            Jj = joints[j]
            if Ji is not None and Jj is not None:
                union = np.concatenate([Ji, Jj], axis=0)
                mean = union.mean(axis=0)
                std = union.std(axis=0)
                std = np.where(std > 0, std, 1.0)
                Ji_s = (Ji - mean) / std
                Jj_s = (Jj - mean) / std
                w1_conditional[p] = sliced_wasserstein1(Ji_s, Jj_s, n_proj, rng)
            Jaci = jacobians[i]
            Jacj = jacobians[j]
            if Jaci is not None and Jacj is not None:
                jac_distance[p] = float(np.linalg.norm(Jaci - Jacj, ord="fro"))
            p += 1

    return {
        "pair_i": pair_i, "pair_j": pair_j,
        "w1_conditional": w1_conditional,
        "jacobian_distance": jac_distance,
    }


# ---------------------------------------------------------------------------
# Worker plumbing
# ---------------------------------------------------------------------------

_U_SHARED: np.ndarray | None = None
_UDOT_SHARED: np.ndarray | None = None
_UG_SHARED: np.ndarray | None = None
_UDOTG_SHARED: np.ndarray | None = None


def _init_worker(U_arg: np.ndarray, Udot_arg: np.ndarray,
                 Ug_arg: np.ndarray, Udotg_arg: np.ndarray) -> None:
    global _U_SHARED, _UDOT_SHARED, _UG_SHARED, _UDOTG_SHARED
    _U_SHARED = U_arg
    _UDOT_SHARED = Udot_arg
    _UG_SHARED = Ug_arg
    _UDOTG_SHARED = Udotg_arg


def _process_K(payload: tuple) -> dict:
    (K, seed, n_init, energy, q, n_proj, n_null,
     w1_seed, iid_seed, block_seed,
     shell_d, shell_tau, shell_ndir, shell_seed,
     q_boundary, m7_seed) = payload
    assert _U_SHARED is not None and _UDOT_SHARED is not None
    assert _UG_SHARED is not None and _UDOTG_SHARED is not None

    U = _U_SHARED
    Udot = _UDOT_SHARED
    Ug = _UG_SHARED
    Udotg = _UDOTG_SHARED

    labels, centroids, inertia, n_iter = kmeans_fit(
        U, K, seed=seed, n_init=n_init
    )
    labels = np.asarray(labels)
    centroids = np.asarray(centroids).astype(np.float64)

    rl = cluster_run_lengths(labels, K)
    r1_mean = rl["r1_mean"]
    r1_median = rl["r1_median"]
    n_runs = rl["n_runs"]

    n_k = np.zeros(K, dtype=np.int64)
    r_k = np.zeros(K, dtype=np.int32)
    participation_ratio = np.full(K, np.nan, dtype=np.float64)
    condition_number = np.full(K, np.nan, dtype=np.float64)
    spectral_gap_at_elbow = np.full(K, np.nan, dtype=np.float64)
    dead_zone_frac = np.full(K, np.nan, dtype=np.float64)
    ratio_kNN = np.full(K, np.nan, dtype=np.float64)
    w1_temporal = np.full(K, np.nan, dtype=np.float64)
    w1_iid_mean = np.full(K, np.nan, dtype=np.float64)
    w1_iid_std = np.full(K, np.nan, dtype=np.float64)
    w1_block_mean = np.full(K, np.nan, dtype=np.float64)
    w1_block_std = np.full(K, np.nan, dtype=np.float64)
    ratio_iid = np.full(K, np.nan, dtype=np.float64)
    ratio_block = np.full(K, np.nan, dtype=np.float64)
    block_L = np.zeros(K, dtype=np.int32)

    V_list: list = [None] * K
    for k in range(K):
        mask = labels == k
        n = int(mask.sum())
        n_k[k] = n
        if n < 4:
            continue
        # Trajectory order is preserved by boolean masking; this is required
        # so the block-bootstrap null sees the cluster's true visit order.
        u_k = U[mask].astype(np.float64)
        udot_k = Udot[mask].astype(np.float64)
        c_k = centroids[k]
        V, lam_k, r = local_pca_energy(u_k, c_k, energy)
        r_k[k] = r
        V_list[k] = V
        lam_pos = lam_k[lam_k > 0.0]
        if lam_pos.size:
            s1 = float(lam_pos.sum())
            s2 = float((lam_pos * lam_pos).sum())
            participation_ratio[k] = (s1 * s1) / s2 if s2 > 0.0 else float("nan")
            lam_max_k = float(lam_pos.max())
            lam_min_k = float(lam_pos.min())
            condition_number[k] = lam_max_k / lam_min_k if lam_min_k > 0 else float("nan")
            if 1 < r <= lam_pos.size:
                spectral_gap_at_elbow[k] = (
                    float(lam_pos[r - 1]) / float(lam_pos[r - 2])
                )
            else:
                spectral_gap_at_elbow[k] = 1.0
            rng_shell = np.random.default_rng(shell_seed + k)
            dead_zone_frac[k] = dead_zone_fraction(
                lam_pos, d=shell_d, tau=shell_tau,
                n_dir=shell_ndir, rng=rng_shell,
            )
        alpha = (u_k - c_k) @ V
        alpha_dot = udot_k @ V
        ratio_kNN[k] = knn_local_determinism(alpha, alpha_dot, q)
        m = split_half_w1_with_nulls(
            alpha, alpha_dot,
            n_proj=n_proj, n_null=n_null,
            seed_temporal=w1_seed + k,
            seed_iid=iid_seed + k,
            seed_block=block_seed + k,
        )
        w1_temporal[k] = m["w1_temporal"]
        w1_iid_mean[k] = m["w1_iid_mean"]
        w1_iid_std[k] = m["w1_iid_std"]
        w1_block_mean[k] = m["w1_block_mean"]
        w1_block_std[k] = m["w1_block_std"]
        ratio_iid[k] = m["ratio_iid"]
        ratio_block[k] = m["ratio_block"]
        block_L[k] = m["block_length"]

    # Metric 6 (soft interface) and metric 7 (distinctness).
    m6 = pairwise_metric6(U, Udot, labels, K, centroids, V_list, q_boundary)
    rng_m7 = np.random.default_rng(m7_seed)
    m7 = pairwise_metric7(Ug, Udotg, labels, K, n_proj, rng_m7)

    return {
        "K": int(K),
        "labels": labels.astype(np.int32),
        "centroids": centroids.astype(np.float32),
        "inertia": float(inertia),
        "n_iter": int(n_iter),
        "n_k": n_k,
        "r_k": r_k,
        "ratio_kNN_per_cluster": ratio_kNN,
        "w1_temporal_per_cluster": w1_temporal,
        "w1_iid_mean_per_cluster": w1_iid_mean,
        "w1_iid_std_per_cluster": w1_iid_std,
        "w1_block_mean_per_cluster": w1_block_mean,
        "w1_block_std_per_cluster": w1_block_std,
        "ratio_iid_per_cluster": ratio_iid,
        "ratio_block_per_cluster": ratio_block,
        "block_length_per_cluster": block_L,
        "r1_mean_per_cluster": r1_mean,
        "r1_median_per_cluster": r1_median,
        "n_runs_per_cluster": n_runs,
        "participation_ratio_per_cluster": participation_ratio,
        "condition_number_per_cluster": condition_number,
        "spectral_gap_at_elbow_per_cluster": spectral_gap_at_elbow,
        "dead_zone_fraction_per_cluster": dead_zone_frac,
        # ---- metric 6 (per-pair, length K*(K-1)/2) ----
        "pair_i_m6": m6["pair_i"],
        "pair_j_m6": m6["pair_j"],
        "is_adjacent_per_pair": m6["is_adjacent"],
        "switch_count_per_pair": m6["switch_count"],
        "sin_theta_max_per_pair": m6["sin_theta_max"],
        "centroid_shift_per_pair": m6["centroid_shift"],
        "centroid_shift_ratio_per_pair": m6["centroid_shift_ratio"],
        "alpha_dot_jump_per_pair": m6["alpha_dot_jump"],
        # ---- metric 7 (per-pair) ----
        "pair_i_m7": m7["pair_i"],
        "pair_j_m7": m7["pair_j"],
        "w1_conditional_per_pair": m7["w1_conditional"],
        "jacobian_distance_per_pair": m7["jacobian_distance"],
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_summary(out_dir: Path, summary: dict, q: int, energy: float,
                 n_null: int, title_prefix: str = "LOR63 cluster-quality") -> None:
    """Metric 1 figure: (1a) kNN local-determinism, (1b) i.i.d. null ratio,
    (1b') circular-block-bootstrap null ratio."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = summary["K_values"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)

    ax = axes[0]
    ax.plot(K, summary["ratio_kNN_mean"], "o-", label="mean over clusters")
    ax.plot(K, summary["ratio_kNN_max"], "s--", label="max over clusters")
    ax.set_xlabel(r"$K$")
    ax.set_ylabel(r"$\langle \mathrm{Var}(\dot\alpha\,|\,\alpha) \rangle / \mathrm{Var}_k(\dot\alpha)$")
    ax.set_title(rf"(1a) kNN local-determinism, $q={q}$")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    ax = axes[1]
    ax.plot(K, summary["ratio_iid_mean"], "o-", label="mean over clusters")
    ax.plot(K, summary["ratio_iid_max"],  "s--", label="max over clusters")
    ax.axhline(1.0, color="k", lw=1, ls=":", label="i.i.d. floor")
    ax.set_xlabel(r"$K$")
    ax.set_ylabel(r"$W_1^{\rm temporal} / W_1^{\rm iid}$")
    ax.set_title(f"(1b) i.i.d. null ratio, n_null={n_null}")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[2]
    ax.plot(K, summary["ratio_block_mean"], "o-", label="mean over clusters")
    ax.plot(K, summary["ratio_block_max"],  "s--", label="max over clusters")
    ax.axhline(1.0, color="k", lw=1, ls=":", label="stationary floor")
    ax.set_xlabel(r"$K$")
    ax.set_ylabel(r"$W_1^{\rm temporal} / W_1^{\rm block}$")
    ax.set_title("(1b') Circular block-bootstrap null ratio")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.suptitle(f"{title_prefix} - metric 1, energy={energy}, "
                 f"q={q}, n_null={n_null}")
    fig.savefig(out_dir / "summary.png", dpi=150)
    plt.close(fig)


def plot_metric2(out_dir: Path, summary: dict, energy: float,
                 title_prefix: str = "LOR63 cluster-quality") -> None:
    """Metric 2 figure (dwell time): per-cluster run length, dwell-vs-tau_int
    ratio, and the global fragmentation index."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = summary["K_values"]
    tau = int(summary["tau_int_state"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)

    ax = axes[0]
    ax.plot(K, summary["r1_mean_mean"],   "o-",  label="mean of cluster-mean $r_1$")
    ax.plot(K, summary["r1_median_mean"], "v-",  label="mean of cluster-median $r_1$")
    ax.plot(K, summary["r1_mean_min"],    "s--", label="min cluster-mean $r_1$")
    ax.axhline(tau, color="k", lw=1, ls=":",
               label=fr"$\tau_{{\rm int}}={tau}$ (samples)")
    ax.set_xlabel(r"$K$")
    ax.set_ylabel(r"run length $r_1$ (samples)")
    ax.set_title("(2) Per-cluster dwell time")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    ax = axes[1]
    ax.plot(K, summary["r1_over_tau_mean"], "o-",  label="mean over clusters")
    ax.plot(K, summary["r1_over_tau_min"],  "s--", label="min over clusters")
    ax.axhline(1.0, color="k", lw=1, ls=":", label=r"$r_1 = \tau_{\rm int}$")
    ax.set_xlabel(r"$K$")
    ax.set_ylabel(r"$r_1 / \tau_{\rm int}$")
    ax.set_title("(2) Dwell-time / integral-timescale ratio")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    ax = axes[2]
    ax.plot(K, summary["fragmentation_index"], "o-")
    ax.axhline(1.0, color="k", lw=1, ls=":",
               label=r"$K\,\langle r_1 \rangle = T_{\rm total}/\Delta t$")
    ax.set_xlabel(r"$K$")
    ax.set_ylabel(r"$K\,\langle r_1 \rangle / (T_{\rm total}/\Delta t)$")
    ax.set_title("(2) Fragmentation index")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.suptitle(f"{title_prefix} - metric 2 (dwell time), "
                 fr"$\tau_{{\rm int}}={tau}$ samples, energy={energy}")
    fig.savefig(out_dir / "summary_metric2.png", dpi=150)
    plt.close(fig)


def plot_metric3(out_dir: Path, summary: dict,
                 title_prefix: str = "LOR63 cluster-quality") -> None:
    """Metric 3 figure (sample sufficiency): raw m_k, decorrelated m_k_eff,
    and m_k_eff / p ratio with the small-cluster safety zone at 5."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = summary["K_values"]
    p = int(summary["dict_size_p"])
    tau = int(summary["tau_int_state"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)

    ax = axes[0]
    ax.plot(K, summary["m_k_mean"],   "o-",  label="mean over clusters")
    ax.plot(K, summary["m_k_median"], "v-",  label="median over clusters")
    ax.plot(K, summary["m_k_min"],    "s--", label="min cluster")
    ax.set_xlabel(r"$K$")
    ax.set_ylabel(r"$m_k$ (snapshots)")
    ax.set_title("(3) Per-cluster raw count")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    ax = axes[1]
    ax.plot(K, summary["m_k_eff_mean"], "o-",  label="mean over clusters")
    ax.plot(K, summary["m_k_eff_min"],  "s--", label="min cluster")
    ax.axhline(p, color="k", lw=1, ls=":", label=fr"$p={p}$ (dictionary)")
    ax.axhline(5 * p, color="k", lw=1, ls="--",
               label=fr"$5p={5*p}$ (safety floor)")
    ax.set_xlabel(r"$K$")
    ax.set_ylabel(r"$m_k^{\rm eff} = m_k / (2\tau_{\rm int})$")
    ax.set_title(fr"(3) Decorrelated count, $\tau_{{\rm int}}={tau}$")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    ax = axes[2]
    ax.plot(K, summary["m_k_eff_over_p_mean"], "o-",  label="mean over clusters")
    ax.plot(K, summary["m_k_eff_over_p_min"],  "s--", label="min cluster")
    ax.axhline(5.0, color="r", lw=1, ls="--", label="danger-zone floor (=5)")
    ax.axhline(1.0, color="k", lw=1, ls=":",  label=r"$m_k^{\rm eff}=p$")
    ax.set_xlabel(r"$K$")
    ax.set_ylabel(r"$m_k^{\rm eff} / p$")
    ax.set_title(f"(3) Effective samples per dictionary term, p={p}")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    fig.suptitle(f"{title_prefix} - metric 3 (sample sufficiency), "
                 fr"$\tau_{{\rm int}}={tau}$ samples, p={p}")
    fig.savefig(out_dir / "summary_metric3.png", dpi=150)
    plt.close(fig)


def plot_metric4(out_dir: Path, summary: dict,
                 title_prefix: str = "LOR63 cluster-quality") -> None:
    """Metric 4 figure (genuine local-rank reduction): per-cluster r_k,
    compression gain r_k / r_global, and participation ratio."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = summary["K_values"]
    r_global = int(summary["r_global"])
    pr_global = float(summary["participation_ratio_global"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)

    ax = axes[0]
    ax.plot(K, summary["r_k_mean"], "o-",  label="mean over clusters")
    ax.plot(K, summary["r_k_max"],  "v--", label="max cluster")
    ax.plot(K, summary["r_k_min"],  "s--", label="min cluster")
    ax.axhline(r_global, color="k", lw=1, ls=":",
               label=fr"$r_{{\rm global}}={r_global}$")
    ax.set_xlabel(r"$K$")
    ax.set_ylabel(r"$r_k$ (PCA elbow rank)")
    ax.set_title("(4) Per-cluster local-PCA rank")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(K, summary["compression_gain_mean"], "o-",  label="mean over clusters")
    ax.plot(K, summary["compression_gain_max"],  "v--", label="max cluster")
    ax.axhline(1.0, color="k", lw=1, ls=":", label="no compression")
    ax.set_xlabel(r"$K$")
    ax.set_ylabel(r"$r_k / r_{\rm global}$")
    ax.set_title("(4) Compression gain")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[2]
    ax.plot(K, summary["participation_ratio_mean"], "o-",  label="mean over clusters")
    ax.plot(K, summary["participation_ratio_min"],  "s--", label="min cluster")
    ax.axhline(pr_global, color="k", lw=1, ls=":",
               label=fr"PR$_{{\rm global}}={pr_global:.2f}$")
    ax.set_xlabel(r"$K$")
    ax.set_ylabel(r"$\mathrm{PR}_k = (\sum\lambda_k)^2 / \sum\lambda_k^2$")
    ax.set_title("(4) Participation ratio (effective dim.)")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.suptitle(f"{title_prefix} - metric 4 (local-rank reduction), "
                 fr"$r_{{\rm global}}={r_global}$, "
                 fr"PR$_{{\rm global}}={pr_global:.2f}$")
    fig.savefig(out_dir / "summary_metric4.png", dpi=150)
    plt.close(fig)


def plot_metric5(out_dir: Path, summary: dict,
                 title_prefix: str = "LOR63 cluster-quality") -> None:
    """Metric 5 figure (anisotropy): condition number, spectral gap at the
    PCA elbow, and dead-zone shell-coverage fraction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = summary["K_values"]
    d = float(summary["shell_d"])
    tau = float(summary["shell_tau"])
    n_dir = int(summary["shell_ndir"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)

    ax = axes[0]
    ax.plot(K, summary["condition_number_mean"], "o-",  label="mean over clusters")
    ax.plot(K, summary["condition_number_max"],  "s--", label="max cluster")
    ax.set_xlabel(r"$K$")
    ax.set_ylabel(r"$\lambda_k^{\max} / \lambda_k^{\min}$")
    ax.set_title("(5) Per-cluster PCA condition number")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    ax = axes[1]
    ax.plot(K, summary["spectral_gap_at_elbow_mean"], "o-",  label="mean over clusters")
    ax.plot(K, summary["spectral_gap_at_elbow_min"],  "s--", label="min cluster")
    ax.axhline(1.0, color="k", lw=1, ls=":", label="no gap")
    ax.set_xlabel(r"$K$")
    ax.set_ylabel(r"$\lambda_k^{(r_k)} / \lambda_k^{(r_k - 1)}$")
    ax.set_title("(5) Spectral gap at PCA elbow")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    ax = axes[2]
    ax.plot(K, summary["dead_zone_fraction_mean"], "o-",  label="mean over clusters")
    ax.plot(K, summary["dead_zone_fraction_max"],  "s--", label="max cluster")
    ax.set_xlabel(r"$K$")
    ax.set_ylabel(r"$\Pr_v[\phi_k(c_k + R v) < \tau]$")
    ax.set_title(fr"(5) Dead-zone shell coverage, "
                 fr"$d={d}$, $\tau={tau:g}$, $n_{{\rm dir}}={n_dir}$")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()

    fig.suptitle(f"{title_prefix} - metric 5 (anisotropy)")
    fig.savefig(out_dir / "summary_metric5.png", dpi=150)
    plt.close(fig)


def plot_metric6(out_dir: Path, summary: dict, q: int,
                 title_prefix: str = "LOR63 cluster-quality") -> None:
    """Metric 6 figure (soft interface): principal angle, centroid shift,
    alpha_dot boundary jump, and observed switch rate. Aggregates are
    restricted to adjacent pairs (pairs where at least one i<->j
    transition was observed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = summary["K_values"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(K, summary["sin_theta_max_mean"], "o-",  label="mean adjacent pair")
    ax.plot(K, summary["sin_theta_max_max"],  "s--", label="max adjacent pair")
    ax.axhline(0.0, color="k", lw=1, ls=":", label="identical subspaces")
    ax.set_xlabel(r"$K$")
    ax.set_ylabel(r"$\sin\theta_{\max}(V_i, V_j)$")
    ax.set_title("(6) Largest principal angle, adjacent pairs")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(K, summary["centroid_shift_mean"], "o-",  label=r"mean $\|V_j^T(c_i-c_j)\|$")
    ax.plot(K, summary["centroid_shift_max"],  "s--", label=r"max $\|V_j^T(c_i-c_j)\|$")
    ax.plot(K, summary["centroid_shift_ratio_mean"], "v-",
            label=r"mean $\|V_j^T(c_i-c_j)\| / \|c_i-c_j\|$")
    ax.set_xlabel(r"$K$")
    ax.set_ylabel("centroid-shift magnitude / ratio")
    ax.set_title("(6) Centroid shift in $V_j$ frame")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(K, summary["alpha_dot_jump_mean"], "o-",  label="mean adjacent pair")
    ax.plot(K, summary["alpha_dot_jump_max"],  "s--", label="max adjacent pair")
    ax.set_xlabel(r"$K$")
    ax.set_ylabel(r"$\|\langle\dot\alpha\rangle_i - \langle\dot\alpha\rangle_j\|_{V_j}$")
    ax.set_title(fr"(6) Boundary $\dot\alpha$ jump, $q_{{\rm bdry}}={q}$")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(K, summary["switch_rate_mean"], "o-",
            label="mean per-pair rate (adjacent)")
    ax.plot(K, summary["switch_rate_max"],  "s--",
            label="max per-pair rate")
    ax.plot(K, summary["switch_rate_total"], "v--",
            label="total rate over all pairs")
    ax.set_xlabel(r"$K$")
    ax.set_ylabel(r"switches per snapshot")
    ax.set_title("(6) Observed switch rate")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(f"{title_prefix} - metric 6 (soft interface)")
    fig.savefig(out_dir / "summary_metric6.png", dpi=150)
    plt.close(fig)


def plot_metric7(out_dir: Path, summary: dict,
                 title_prefix: str = "LOR63 cluster-quality") -> None:
    """Metric 7 figure (distinctness): sliced-W1 between conditional
    `(alpha_g, alpha_dot_g)` distributions and Frobenius distance
    between per-cluster OLS Jacobians. The *minimum* over pairs is the
    informative quantity - it is the most similar pair, so a large
    minimum means every pair is distinct."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = summary["K_values"]
    r_global = int(summary["r_global"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    ax = axes[0]
    ax.plot(K, summary["w1_conditional_min"],  "s-",  label="min over pairs")
    ax.plot(K, summary["w1_conditional_mean"], "o--", label="mean over pairs")
    ax.plot(K, summary["w1_conditional_max"],  "v--", label="max over pairs")
    ax.set_xlabel(r"$K$")
    ax.set_ylabel(r"sliced $W_1$")
    ax.set_title(fr"(7) Pairwise $W_1[(\alpha_g,\dot\alpha_g)]$, "
                 fr"$r_{{\rm global}}={r_global}$")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(K, summary["jacobian_distance_min"],  "s-",  label="min over pairs")
    ax.plot(K, summary["jacobian_distance_mean"], "o--", label="mean over pairs")
    ax.plot(K, summary["jacobian_distance_max"],  "v--", label="max over pairs")
    ax.set_xlabel(r"$K$")
    ax.set_ylabel(r"$\|J_i - J_j\|_F$")
    ax.set_title("(7) Pairwise Jacobian distance")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.suptitle(f"{title_prefix} - metric 7 (distinctness)")
    fig.savefig(out_dir / "summary_metric7.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stats-path", type=Path,
                        default=REPO_ROOT / "data/LOR63/results/lor63/stats.npz")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "results/LOR63/cluster_quality_sweep")
    parser.add_argument("--K-min", type=int, default=1)
    parser.add_argument("--K-max", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1,
                        help="snapshot stride into stats.npz")
    parser.add_argument("--energy", type=float, default=0.99,
                        help="per-cluster PCA cumulative-energy threshold")
    parser.add_argument("--q", type=int, default=20,
                        help="kNN size for the local-determinism diagnostic")
    parser.add_argument("--n-proj", type=int, default=64,
                        help="random projection count for sliced Wasserstein-1")
    parser.add_argument("--n-null", type=int, default=10,
                        help="null repetitions (used by both i.i.d. and block-bootstrap nulls)")
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
    parser.add_argument("--shell-d", type=float, default=2.0,
                        help="metric 5 shell radius in units of "
                             "sqrt(lambda_k^max) (default 2)")
    parser.add_argument("--shell-tau", type=float, default=1e-3,
                        help="metric 5 dead-zone kernel threshold "
                             "(default 1e-3)")
    parser.add_argument("--shell-ndir", type=int, default=2048,
                        help="metric 5 shell direction count "
                             "(default 2048)")
    parser.add_argument("--shell-seed", type=int, default=98765,
                        help="metric 5 RNG seed for shell directions")
    parser.add_argument("--q-boundary", type=int, default=20,
                        help="metric 6 boundary-smoothing population size "
                             "(snapshots nearest the Voronoi boundary per "
                             "side, default 20)")
    parser.add_argument("--m7-seed", type=int, default=24680,
                        help="metric 7 RNG seed for sliced-W1 projections")
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.stats_path}", flush=True)
    with np.load(args.stats_path) as z:
        u_full = np.asarray(z["u"], dtype=np.float64)
        sigma = float(z["sigma"])
        rho = float(z["rho"])
        beta = float(z["beta"])
    U = np.ascontiguousarray(u_full[::args.stride])
    Udot = lor63_rhs_batch(U, sigma, rho, beta)
    print(f"  U.shape={U.shape}, sigma={sigma}, rho={rho}, beta={beta:.6f}",
          flush=True)

    # Metric 2 reference timescale: integral autocorrelation of ||u||_2
    # in sample units. Computed once - it does not depend on K.
    norm_u = np.linalg.norm(U, axis=1)
    tau_int_state = int(integrated_autocorr_first_zero(norm_u))
    M_total = int(U.shape[0])
    print(f"  tau_int (||u||) = {tau_int_state} samples, M = {M_total}",
          flush=True)

    # Metric 4 reference rank: energy-elbow rank of the global PCA on
    # mean-centred snapshots. Computed once - it does not depend on K.
    # The truncated right-singular basis `V_global` is also the metric-7
    # common frame, so we keep Vh.
    u_mean = U.mean(axis=0)
    X_global = U - u_mean
    _, S_global, Vh_global = np.linalg.svd(X_global, full_matrices=False)
    lam_global = S_global * S_global / max(M_total - 1, 1)
    cum_global = np.cumsum(lam_global) / max(lam_global.sum(), 1e-300)
    r_global = int(np.searchsorted(cum_global, args.energy) + 1)
    r_global = min(max(r_global, 1), U.shape[1])
    V_global = Vh_global[:r_global].T  # (N, r_global)
    s1g = float(lam_global.sum())
    s2g = float((lam_global * lam_global).sum())
    pr_global = (s1g * s1g) / s2g if s2g > 0.0 else float("nan")
    print(f"  r_global({args.energy}) = {r_global}, "
          f"global participation ratio = {pr_global:.3f}",
          flush=True)

    # Metric 7 common-frame projections of snapshots and analytical RHS.
    Ug = X_global @ V_global       # (M, r_global)
    Udotg = Udot @ V_global         # (M, r_global)

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
        """Reduce a per-pair array over the adjacent-pair subset.
        NaN when no adjacent pair exists (e.g. K=1)."""
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
        """Reduce a per-pair array over all pairs (length K*(K-1)/2)."""
        out = np.empty(len(K_sorted), dtype=np.float64)
        for i, K in enumerate(K_sorted):
            arr = shards[K][field]
            if arr.size == 0:
                out[i] = float("nan")
                continue
            vals = arr[~np.isnan(arr)] if arr.dtype.kind == "f" else arr
            out[i] = float(op(vals)) if vals.size else float("nan")
        return out

    # Metric 3 derived per-cluster series: m_k, m_k_eff = n_k / (2 tau_int),
    # m_k_eff / p. Computed here from shard `n_k` and the global tau_int.
    p_dict = int(args.dict_size)
    decorr = max(2 * tau_int_state, 1)
    m_k_eff_per_K = {
        K: (shards[K]["n_k"].astype(np.float64) / decorr) for K in K_sorted
    }

    # Per-K global mean run length for the fragmentation index.
    r1_overall = np.array([
        float(np.nanmean(shards[K]["r1_mean_per_cluster"])) for K in K_sorted
    ])
    fragmentation = np.array([
        K * r1_overall[i] / max(M_total, 1) for i, K in enumerate(K_sorted)
    ])

    summary = {
        "K_values": np.array(K_sorted, dtype=np.int32),
        # ---- metric 1a ----
        "ratio_kNN_mean": agg("ratio_kNN_per_cluster", np.nanmean),
        "ratio_kNN_max":  agg("ratio_kNN_per_cluster", np.nanmax),
        # ---- metric 1b (raw + i.i.d. + block) ----
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
        # ---- metric 2 (dwell time) ----
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
        # ---- metric 3 (sample sufficiency) ----
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
        # ---- metric 4 (genuine local-rank reduction) ----
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
        # ---- metric 5 (anisotropy + dead-zone) ----
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
        # ---- metric 6 (adjacent-pair aggregates) ----
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
        # ---- metric 7 (all-pair aggregates) ----
        "w1_conditional_min":    agg_all_pairs("w1_conditional_per_pair", np.min),
        "w1_conditional_mean":   agg_all_pairs("w1_conditional_per_pair", np.mean),
        "w1_conditional_max":    agg_all_pairs("w1_conditional_per_pair", np.max),
        "jacobian_distance_min": agg_all_pairs("jacobian_distance_per_pair", np.min),
        "jacobian_distance_mean":agg_all_pairs("jacobian_distance_per_pair", np.mean),
        "jacobian_distance_max": agg_all_pairs("jacobian_distance_per_pair", np.max),
        "q_boundary":      np.int32(args.q_boundary),
        # ---- metadata ----
        "tau_int_state": np.int64(tau_int_state),
        "q": np.int32(args.q),
        "energy": np.float64(args.energy),
        "n_proj": np.int32(args.n_proj),
        "n_null": np.int32(args.n_null),
        "stride": np.int32(args.stride),
        "M": np.int64(M_total),
        "N": np.int64(U.shape[1]),
        "sigma": np.float64(sigma),
        "rho": np.float64(rho),
        "beta": np.float64(beta),
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
            # ---- per-pair (metric 6) ----
            pair_i_m6=s["pair_i_m6"],
            pair_j_m6=s["pair_j_m6"],
            is_adjacent_per_pair=s["is_adjacent_per_pair"],
            switch_count_per_pair=s["switch_count_per_pair"],
            sin_theta_max_per_pair=s["sin_theta_max_per_pair"],
            centroid_shift_per_pair=s["centroid_shift_per_pair"],
            centroid_shift_ratio_per_pair=s["centroid_shift_ratio_per_pair"],
            alpha_dot_jump_per_pair=s["alpha_dot_jump_per_pair"],
            # ---- per-pair (metric 7) ----
            pair_i_m7=s["pair_i_m7"],
            pair_j_m7=s["pair_j_m7"],
            w1_conditional_per_pair=s["w1_conditional_per_pair"],
            jacobian_distance_per_pair=s["jacobian_distance_per_pair"],
        )

    plot_summary(args.out_dir, summary, q=args.q, energy=args.energy,
                 n_null=args.n_null)
    plot_metric2(args.out_dir, summary, energy=args.energy)
    plot_metric3(args.out_dir, summary)
    plot_metric4(args.out_dir, summary)
    plot_metric5(args.out_dir, summary)
    plot_metric6(args.out_dir, summary, q=args.q_boundary)
    plot_metric7(args.out_dir, summary)
    print(f"[done] outputs in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
