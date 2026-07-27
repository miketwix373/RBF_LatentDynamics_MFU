"""Fixed points of the fitted MFU_NA wall-shear RBF field.

Newton/LM on f_RBF(a) = 0 with the analytic Jacobian, seeded from (a) the
centres of the micro-orbits the UPO hunt converged onto, (b) the K=16 K-means
centroids, (c) the lowest-|f| data states. Off the kernel support the field
decays to zero, so every off-support point is a spurious root: iterations are
aborted the moment coverage drops below the on-attractor range, and every
converged root is additionally gated by coverage and by shadowing distance to
the data cloud. Credible roots get eigenspectra and wall-pattern
reconstructions V a*.

Outputs under results/MFU_NA/field_analysis/fixed_points/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from find_mfu_na_upo import RBFField, MFU, TPLUS_PER_S  # noqa: E402
from plot_mfu_na_phase_circulation import _signals, _streak_amp  # noqa: E402


def newton_root(field, x0, cov_floor, scale, max_iter=200, tol=1e-12):
    """Damped LM iteration for f(x)=0; abort if coverage leaves the support."""
    x = x0.copy()
    mu = 1e-8
    for _ in range(max_iter):
        if field.coverage(x[None])[0] < cov_floor:
            return None, "escaped"
        f, J = field.f_and_J(x[None])
        f, J = f[0], J[0]
        nf = np.linalg.norm(f)
        if nf < tol:
            return x, "converged"
        G = J.T @ J
        step = np.linalg.solve(G + mu * np.diag(np.diag(G) + 1e-30), -J.T @ f)
        ns = np.linalg.norm(step)
        if ns > 0.5 * scale:                       # trust cap
            step *= 0.5 * scale / ns
        xn = x + step
        if np.linalg.norm(field.f(xn[None])[0]) < nf:
            x = xn
            mu = max(mu / 3, 1e-14)
        else:
            mu *= 10
            if mu > 1e12:
                return None, "stalled"
    return None, "maxiter"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--coords", type=Path, default=MFU / "pod/coords_do45.npz")
    p.add_argument("--model", type=Path,
                   default=MFU / "do45/alpha8.0/model_n00960.npz")
    p.add_argument("--labels", type=Path,
                   default=MFU / "cluster_sweep/do45/K=16.npz")
    p.add_argument("--out-dir", type=Path,
                   default=MFU / "field_analysis/fixed_points")
    args = p.parse_args()

    from scipy.spatial import cKDTree

    with np.load(args.coords) as z:
        alpha = np.asarray(z["alpha"], np.float64)
        V = np.asarray(z["V"], np.float64)
    A = alpha[::2]
    rms = float(np.sqrt((A**2).sum(1).mean()))
    field = RBFField(args.model)
    rng = np.random.default_rng(0)
    cov_att = field.coverage(A[rng.choice(len(A), 2000, replace=False)])
    cov_p5 = float(np.percentile(cov_att, 5))
    cov_floor = 0.5 * cov_p5

    # SSP-plane frame from the full native record
    _, sy_d = _signals(alpha, V)
    ax_d = _streak_amp(alpha, V)
    mx, sx, my, sy_s = ax_d.mean(), ax_d.std(), sy_d.mean(), sy_d.std()

    # seeds: micro-orbit centres, centroids, lowest-|f| data states
    seeds = []
    for tag in ("refine_M16_dt0125", "T40_seed2500", "rob_a8_n4800",
                "SSP_seed2492", "SSP_seed3669"):
        fp = MFU / "upo" / tag / "upo.npz"
        if fp.exists():
            seeds.append(np.load(fp, allow_pickle=True)["orbit"].mean(0))
    cent = np.asarray(np.load(args.labels)["centroids"], np.float64)
    seeds.extend(list(cent))
    sub = A[:: max(1, len(A) // 8000)]
    nf = np.linalg.norm(field.f(sub), axis=1)
    seeds.extend(list(sub[np.argsort(nf)[:50]]))
    print(f"[seeds] {len(seeds)} (5 orbit centres + 16 centroids + 50 low-|f|)")

    roots, fates = [], {"converged": 0, "escaped": 0, "stalled": 0, "maxiter": 0}
    for s in seeds:
        x, fate = newton_root(field, np.asarray(s, np.float64), cov_floor, rms)
        fates[fate] += 1
        if x is None:
            continue
        if not any(np.linalg.norm(x - r) < 1e-6 * rms for r in roots):
            roots.append(x)
    print(f"[newton] fates: {fates};  distinct roots: {len(roots)}")

    tree = cKDTree(A)
    dsub = A[rng.choice(len(A), 3000, replace=False)]
    d_dat = tree.query(dsub, k=2)[0][:, 1]
    nn_med = float(np.median(d_dat))

    rows = []
    for i, x in enumerate(roots):
        _, J = field.f_and_J(x[None])
        ev = np.linalg.eigvals(J[0])
        lead = ev[np.argsort(-ev.real)][:6]
        cov = float(field.coverage(x[None])[0])
        d_nn = float(tree.query(x[None], k=1)[0][0])
        _, syx = _signals(x[None], V)
        axx = _streak_amp(x[None], V)
        px, py = float((axx[0] - mx) / sx), float((syx[0] - my) / sy_s)
        credible = (cov >= cov_p5) and (d_nn <= 3 * nn_med)
        rows.append(dict(
            i=i, coverage=cov, cov_p5=cov_p5, nn_dist=d_nn,
            nn_dist_over_rms=d_nn / rms, nn_dist_over_spacing=d_nn / nn_med,
            ssp=[px, py], credible=bool(credible),
            lead_eigs_per_s=[[float(e.real), float(e.imag)] for e in lead]))
        print(f"root {i}: SSP=({px:+.2f},{py:+.2f})  cov={cov:.0f}/p5={cov_p5:.0f}"
              f"  d_nn/spacing={d_nn / nn_med:.1f}  credible={credible}")
        print("   leading eigs (1/s): "
              + "  ".join(f"{e.real:+.3f}{e.imag:+.3f}i" for e in lead))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / "fixed_points.npz",
             roots=np.array(roots) if roots else np.empty((0, 45)),
             rms=rms, cov_p5=cov_p5, nn_med=nn_med,
             model_path=str(args.model))
    (args.out_dir / "summary.json").write_text(json.dumps(
        dict(fates=fates, n_roots=len(roots), rows=rows), indent=2))

    cred = [r for r in rows if r["credible"]]
    if cred:
        import shutil
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        use_tex = shutil.which("latex") and shutil.which("dvipng")
        plt.rcParams.update({"text.usetex": bool(use_tex),
                             "font.family": "serif", "mathtext.fontset": "cm"})
        n = len(cred)
        fig, axs = plt.subplots(n, 2, figsize=(10, 3.2 * n), squeeze=False)
        for r, row in enumerate(cred):
            w = V @ roots[row["i"]]
            vv = np.abs(w).max()
            for c, (part, name) in enumerate(
                    ((w[:4096], r"$\tau_x$"), (w[4096:], r"$\tau_y$"))):
                ax = axs[r, c]
                im = ax.imshow(part.reshape(64, 64), origin="lower",
                               cmap="RdBu_r", vmin=-vv, vmax=vv)
                ax.set_title(f"root {row['i']}  {name}", fontsize=12)
                ax.set_xticks([]); ax.set_yticks([])
                fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(args.out_dir / "wall_patterns.png", dpi=170)
        print(f"[fig] wall_patterns.png ({n} credible roots)")
    print(f"[done] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
