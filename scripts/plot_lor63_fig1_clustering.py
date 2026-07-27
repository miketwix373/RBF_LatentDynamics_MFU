"""Paper Figure 1 (Lorenz-63 glass box): clustering and the K-selection criterion.

A 2x3 composite.

  top row     the LOR63 attractor in (x, y, z) coloured by K-means label at
              K = 2, 3, 4, centroids overlaid; the candidate partitions the
              selection criterion judges.
  bottom row  the three selection signals of Section 2.2.3 plotted against K,
              with the selected K marked:
                1. BIC, as the marginal variation Delta BIC / Delta K
                   (Eq. bic); the elbow is its minimum.
                2. geometric (shape) distinctness over neighbouring pairs:
                   the largest principal angle sin(theta_max) (Eq. prinangle)
                   and the participation-ratio gap |P_i - P_j| (Eq. partratio).
                3. dynamical (motion) distinctness: the cluster-speed gap
                   |v_i - v_j| / (v_i + v_j) with v_k = <||u_dot||>
                   (Eq. clusterspeed).

The geometric quantities are read from the per-K shards written by
`run_cluster_quality_sweep.py`; the cluster speed is computed here from the
analytical Lorenz-63 right-hand side, avoiding finite-difference noise on a
known model. Outputs a PNG; refine interactively.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# Anchor for the paper figures; both scratch roots are the same repo.
DEFAULT_OUT = Path(
    "/users/sbrw610/sharedscratch/RBF_ROM/results/LOR63/images_paper/fig1_clustering.pdf"
)


def lor63_rhs(u: np.ndarray, sigma: float, rho: float, beta: float) -> np.ndarray:
    """Analytical Lorenz-63 right-hand side on (M, 3) snapshots."""
    x, y, z = u[:, 0], u[:, 1], u[:, 2]
    return np.stack([sigma * (y - x), x * (rho - z) - y, x * y - beta * z], axis=1)


def bic_kmeans(J_K: float, n_k: np.ndarray, M: int, r: int, K: int) -> float:
    """Bayesian information criterion for a K-means partition, Eq. (bic).

    `J_K` is the within-cluster sum of squares (the sweep's `inertia`), `n_k`
    the cluster populations, `M` the snapshot count and `r` the reduced
    dimension. Pelleg-Moore form as adapted by Colanera & Magri.
    """
    n_k = np.asarray(n_k, dtype=np.float64)
    n_k = n_k[n_k > 0]
    return (M * np.log(J_K) + K * np.log(M)
            - (2.0 / r) * float(np.sum(n_k * np.log(n_k / M))))


def neighbour_distinctness(shard: dict, speed: np.ndarray, K: int) -> dict:
    """Worst-case (max over Voronoi-adjacent pairs) distinctness at one K.

    Adjacency is the observed-transition adjacency from the sweep
    (`is_adjacent_per_pair`); a pair counts only if the trajectory actually
    crosses its boundary.
    """
    labels = np.asarray(shard["labels"])
    pr = np.asarray(shard["participation_ratio_per_cluster"], dtype=np.float64)
    sin_theta = np.asarray(shard["sin_theta_max_per_pair"], dtype=np.float64)
    adj = np.asarray(shard["is_adjacent_per_pair"], dtype=bool)
    pair_i = np.asarray(shard["pair_i_m6"], dtype=np.int64)
    pair_j = np.asarray(shard["pair_j_m6"], dtype=np.int64)

    v_k = np.array([speed[labels == k].mean() for k in range(K)])

    nan = float("nan")
    if not adj.any():
        return {"sin_theta_max": nan, "dPR_max": nan, "speed_gap_max": nan}

    ai, aj = pair_i[adj], pair_j[adj]
    s = v_k[ai] + v_k[aj]
    speed_gap = np.where(s > 0, np.abs(v_k[ai] - v_k[aj]) / s, np.nan)
    return {
        "sin_theta_max": float(np.nanmax(sin_theta[adj])),
        "dPR_max": float(np.nanmax(np.abs(pr[ai] - pr[aj]))),
        "speed_gap_max": float(np.nanmax(speed_gap)),
    }


def draw_voronoi_faces(ax, centroids: np.ndarray, traj: np.ndarray,
                       n_grid: int = 140, alpha: float = 0.22,
                       color: str = "0.4", clip_frac: float = 0.035) -> None:
    """Translucent K-means cluster boundaries clipped to the attractor.

    Each pair of centroids defines a perpendicular-bisector plane; the part
    of it that is a genuine partition boundary is where those two centroids
    are the nearest of all (the Voronoi face). We sample each plane, keep the
    face region that lies within `clip_frac` of the attractor (so the surface
    hugs the data rather than floating in empty space), drop triangles that
    would bridge gaps such as the hole of the butterfly, and render the rest
    as a translucent surface.
    """
    import matplotlib.tri as mtri
    from scipy.spatial import cKDTree

    K = centroids.shape[0]
    lo, hi = traj.min(axis=0), traj.max(axis=0)
    R = float(np.max(hi - lo))
    clip_dist = clip_frac * R
    tree = cKDTree(traj)
    ts = np.linspace(-R, R, n_grid)
    spacing = ts[1] - ts[0]
    A, B = np.meshgrid(ts, ts)

    for i in range(K):
        for j in range(i + 1, K):
            normal = centroids[i] - centroids[j]
            nn = np.linalg.norm(normal)
            if nn < 1e-9:
                continue
            nhat = normal / nn
            mid = 0.5 * (centroids[i] + centroids[j])
            seed = np.array([1.0, 0.0, 0.0]) if abs(nhat[0]) < 0.9 \
                else np.array([0.0, 1.0, 0.0])
            uvec = seed - nhat * (seed @ nhat)
            uvec /= np.linalg.norm(uvec)
            vvec = np.cross(nhat, uvec)

            P = (mid[None, None, :] + A[..., None] * uvec
                 + B[..., None] * vvec).reshape(-1, 3)
            order = np.argsort(np.linalg.norm(P[:, None, :] - centroids[None], axis=2),
                               axis=1)
            face = ((order[:, 0] == i) & (order[:, 1] == j)) | \
                   ((order[:, 0] == j) & (order[:, 1] == i))
            near = tree.query(P)[0] <= clip_dist
            keep = face & near
            if keep.sum() < 12:
                continue

            af, bf = A.ravel()[keep], B.ravel()[keep]
            tri = mtri.Triangulation(af, bf)
            # Drop triangles spanning gaps: any edge longer than a few cells.
            tv = tri.triangles
            edge = np.sqrt(np.diff(af[tv][:, [0, 1, 2, 0]], axis=1) ** 2
                           + np.diff(bf[tv][:, [0, 1, 2, 0]], axis=1) ** 2)
            tri.set_mask(edge.max(axis=1) > 2.5 * spacing)
            ax.plot_trisurf(P[keep, 0], P[keep, 1], P[keep, 2],
                            triangles=tri.get_masked_triangles(),
                            color=color, alpha=alpha, linewidth=0, shade=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stats-path", type=Path,
                        default=REPO_ROOT / "data/LOR63/results/lor63/stats.npz")
    parser.add_argument("--sweep-dir", type=Path,
                        default=REPO_ROOT / "results/LOR63/cluster_quality_sweep")
    parser.add_argument("--scatter-K", type=int, nargs="+", default=[2, 3, 4],
                        help="K values rendered as 3-D scatters (top row)")
    parser.add_argument("--metric-K", type=int, nargs="+",
                        default=list(range(1, 11)),
                        help="K values on the criteria panels (bottom row)")
    parser.add_argument("--selected-K", type=int, default=3,
                        help="K highlighted as the chosen partition")
    parser.add_argument("--stride", type=int, default=1,
                        help="snapshot stride; must match the sweep's stride")
    parser.add_argument("--max-points", type=int, default=20000,
                        help="random-thin scatter to this many points")
    parser.add_argument("--elev", type=float, default=22.0,
                        help="3-D view elevation")
    parser.add_argument("--azim", type=float, default=-60.0,
                        help="3-D view azimuth")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    import shutil
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    # MATLAB-LaTeX look: Computer Modern via a real LaTeX backend when one is
    # on PATH, otherwise matplotlib's CM mathtext as a graceful fallback.
    use_tex = shutil.which("latex") is not None and shutil.which("dvipng") is not None
    plt.rcParams.update({
        "text.usetex": use_tex,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
    })

    print(f"[load] {args.stats_path}", flush=True)
    with np.load(args.stats_path) as z:
        u = np.asarray(z["u"], dtype=np.float64)[::args.stride]
        sigma, rho, beta = float(z["sigma"]), float(z["rho"]), float(z["beta"])
    M, r = u.shape
    print(f"  u.shape={u.shape}", flush=True)

    speed = np.linalg.norm(lor63_rhs(u, sigma, rho, beta), axis=1)

    if args.max_points > 0 and M > args.max_points:
        rng = np.random.default_rng(0)
        sel = np.sort(rng.choice(M, size=args.max_points, replace=False))
    else:
        sel = slice(None)

    # --- gather the criteria over K --------------------------------------
    Kc, bic, sin_t, dpr, sgap = [], [], [], [], []
    for K in args.metric_K:
        with np.load(args.sweep_dir / f"K={K:02d}.npz") as s:
            bic.append(bic_kmeans(float(s["inertia"]), s["n_k"], M, r, K))
            d = neighbour_distinctness(dict(s), speed, K)
        Kc.append(K)
        sin_t.append(d["sin_theta_max"])
        dpr.append(d["dPR_max"])
        sgap.append(d["speed_gap_max"])
    Kc = np.array(Kc)
    bic = np.array(bic)
    dbic = np.diff(bic)               # Delta BIC / Delta K (Delta K = 1)
    K_mid = Kc[1:]
    K_star = int(K_mid[np.argmin(dbic)])
    print(f"  Delta-BIC minimum at K={K_star}", flush=True)

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#0072B2", "#E69F00", "#009E73", "#D55E00",
                           "#56B4E9", "#CC79A7"])  # Okabe-Ito (CVD-safe; no yellow)
    fig = plt.figure(figsize=(20, 9))

    # --- top row : 3-D partition scatters --------------------------------
    n_top = len(args.scatter_K)
    for i, K in enumerate(args.scatter_K):
        with np.load(args.sweep_dir / f"K={K:02d}.npz") as s:
            labels = np.asarray(s["labels"], dtype=np.int32)
            centroids = np.asarray(s["centroids"], dtype=np.float64)
        if labels.shape[0] != M:
            raise SystemExit(
                f"label count {labels.shape[0]} != snapshot count {M}; "
                "rerun with matching --stride")

        ax = fig.add_subplot(2, n_top, i + 1, projection="3d")
        # Attractor as a trajectory line coloured by cluster (no scatter),
        # decimated 1-in-2 for a light render.
        uu, ll = u[::2], labels[::2]
        pts = uu.reshape(-1, 1, 3)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = Line3DCollection(segs, colors=cmap(ll[:-1] % 6),
                              linewidths=0.5, alpha=0.7)
        ax.add_collection3d(lc)
        ax.set_xlim(u[:, 0].min(), u[:, 0].max())
        ax.set_ylim(u[:, 1].min(), u[:, 1].max())
        ax.set_zlim(u[:, 2].min(), u[:, 2].max())
        draw_voronoi_faces(ax, centroids, u[sel])
        ax.view_init(elev=args.elev, azim=args.azim)
        sel_tag = "  (selected)" if K == args.selected_K else ""
        ax.set_title(rf"$K={K}${sel_tag}", fontsize=22, y=0.86)
        ax.text2D(0.02, 0.93, f"({'abc'[i]})", transform=ax.transAxes,
                  fontsize=24, fontweight="bold")
        # Clean look: just the coloured cloud, no axes, ticks or grid.
        ax.set_axis_off()

    # --- bottom row : the three selection criteria -----------------------
    TITLE_FS, LABEL_FS, TICK_FS, LEG_FS = 23, 21, 21, 18

    # (1) BIC marginal variation.
    ax = fig.add_subplot(2, 3, 4)
    ax.plot(K_mid, dbic, "o-", color="k")
    ax.scatter([K_star], [dbic[K_mid == K_star][0]], s=110,
               facecolors="none", edgecolors="k", linewidths=1.6, zorder=5)
    ax.axvline(K_star, color="k", lw=1.0, ls="--",
               label=rf"selected, $K={K_star}$")
    ax.axhline(0.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel(r"$K$", fontsize=LABEL_FS)
    ax.set_title(r"$\Delta\mathrm{BIC}/\Delta K$", fontsize=TITLE_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=LEG_FS, loc="lower right")
    ax.text(-0.06, 1.02, "(d)", transform=ax.transAxes, fontsize=24,
            fontweight="bold", va="bottom", ha="left")

    # (2) Geometric (shape) distinctness.
    ax = fig.add_subplot(2, 3, 5)
    ax.plot(Kc, sin_t, "o-", color="k", label=r"$\sin\theta_{\max}$")
    ax.plot(Kc, dpr, "^--", color="k", label=r"$|P_i-P_j|$")
    ax.axvline(args.selected_K, color="k", lw=1.0, ls="--")
    ax.set_xlabel(r"$K$", fontsize=LABEL_FS)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("distinctness (adjacent pairs)", fontsize=TITLE_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=LEG_FS, loc="lower right")
    ax.text(-0.06, 1.02, "(e)", transform=ax.transAxes, fontsize=24,
            fontweight="bold", va="bottom", ha="left")

    # (3) Dynamical (motion) distinctness.
    ax = fig.add_subplot(2, 3, 6)
    ax.plot(Kc, sgap, "s-", color="k")
    ax.axvline(args.selected_K, color="k", lw=1.0, ls="--")
    ax.set_xlabel(r"$K$", fontsize=LABEL_FS)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(r"$|v_i-v_j|/(v_i+v_j)$", fontsize=TITLE_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(alpha=0.3)
    ax.text(-0.06, 1.02, "(f)", transform=ax.transAxes, fontsize=24,
            fontweight="bold", va="bottom", ha="left")

    fig.tight_layout(w_pad=6.0, h_pad=0.0)
    fig.subplots_adjust(hspace=-0.05)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Save PDF (paper) + PNG (quick preview); fall back to mathtext if a real
    # LaTeX backend chokes on the render.
    save_kw = dict(dpi=200, bbox_inches="tight", pad_inches=0.05)
    try:
        fig.savefig(args.out, **save_kw)
        fig.savefig(args.out.with_suffix(".png"), **save_kw)
    except RuntimeError as exc:
        print(f"[warn] LaTeX render failed ({exc}); retrying with mathtext",
              flush=True)
        plt.rcParams["text.usetex"] = False
        fig.savefig(args.out, **save_kw)
        fig.savefig(args.out.with_suffix(".png"), **save_kw)
    plt.close(fig)
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
