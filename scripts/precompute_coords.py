"""Pre-compute (alpha, alpha_dot) at native dt for downstream SINDy fits.

The CHORD2 SINDy entry points conflate four steps that we want to
decouple: (1) load snapshots, (2) project to reduced coordinates,
(3) estimate the derivative, (4) regress. Steps 1-3 only need to be
done once per dataset; step 4 is iterated many times across K, lambda,
RBF-kind, stride, etc. This script does steps 1-3 at native dt and
saves a stable `coords.npz` that every downstream script can consume,
so:

  - The 5-point stencil derivative is evaluated at native dt (smallest
    leading O(dt^4) error). Striding *after* differentiating no longer
    degrades the derivative.
  - `alpha_dot` has the same row count as `alpha`. The boundary rows
    use one-sided 5-point stencils (`chord2.sindy.deriv_5point_full`),
    so no snapshot is dropped.
  - The downstream SINDy fit chooses a stride freely - independence of
    rows in the regression is a regression-side knob, not a data-side
    constraint.

Two reduction paths

- `--svd-path PATH`: load `a = X @ V` from an `svd_basis.npz`
  (no centring, matching the LOR96 F=8 convention) and truncate to the
  first `--d-o` columns.
- no `--svd-path`: identity reduction, `alpha = u[:, :d_o]` (or full
  width if `--d-o` is omitted).

Both paths read native dt from the source `stats.npz` (`dtStats`).

Output schema (coords.npz)

    alpha      : (M, d_o)        reduced coordinates at native dt
    alpha_dot  : (M, d_o)        5-point derivative, full length
    t          : (M,)            sample times (copied from stats.npz)
    dt_native  : float           dt between consecutive rows
    V          : (n, d_o)        projection basis (identity if not given)
    mu         : (n,)            centring vector (zeros under the SVD
                                  convention used here)
    d_o        : int             reduced dimension
    n_state    : int             full-state dimension
    M          : int             number of snapshots
    source_stats : str           absolute path to the source stats.npz
    source_svd   : str           absolute path to svd_basis.npz, or ""

Usage
-----

LOR63 (identity reduction, full 3D coords):
    python scripts/precompute_coords.py \
        --stats-path data/LOR63/results/lor63/stats.npz \
        --out        results/LOR63/coords.npz

LOR96 F=8 (SVD reduction to d_o=35, paper convention):
    python scripts/precompute_coords.py \
        --stats-path /mnt/scratch/users/sbrw610/RBF_ROM/data/LOR96/results/vlachas_F8/stats.npz \
        --svd-path   /users/sbrw610/sharedscratch/RBF_ROM/results/LOR96/F8/svd_basis.npz \
        --d-o        35 \
        --out        /users/sbrw610/sharedscratch/RBF_ROM/results/LOR96/F8/coords.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from chord2.sindy import deriv_5point_full  # noqa: E402


def _resolve_dt(stats: dict) -> float:
    """Native snapshot dt = `dtStats` from the stats.npz contract."""
    if "dtStats" not in stats:
        raise SystemExit(
            "stats.npz is missing 'dtStats'; required to set dt_native"
        )
    return float(stats["dtStats"])


def _alpha_from_svd_basis(svd_path: Path, d_o: int) -> tuple[
    np.ndarray, np.ndarray, np.ndarray
]:
    """Load `(alpha, V[:, :d_o], mu)` from an existing svd_basis.npz.

    `mu` is the centring vector stored by the basis builder: zeros under the
    un-centred convention (KS/LOR96), or the time-mean field for centred
    bases (Kolmogorov), so reconstruction is `u ≈ mu + alpha @ V.T`.
    """
    with np.load(svd_path) as z:
        a_full = np.asarray(z["a"], dtype=np.float64)
        V_full = np.asarray(z["V"], dtype=np.float64)
        mu = (np.asarray(z["mu"], dtype=np.float64) if "mu" in z.files
              else np.zeros(V_full.shape[0], dtype=np.float64))
    if d_o < 1 or d_o > V_full.shape[1]:
        raise SystemExit(
            f"--d-o={d_o} out of range [1, {V_full.shape[1]}] for V from {svd_path}"
        )
    if a_full.shape[1] < d_o:
        raise SystemExit(
            f"svd_basis 'a' has {a_full.shape[1]} columns, < d_o={d_o}"
        )
    alpha = np.ascontiguousarray(a_full[:, :d_o])
    V_do = np.ascontiguousarray(V_full[:, :d_o])
    return alpha, V_do, mu


def _alpha_identity(u: np.ndarray, d_o: int | None) -> tuple[
    np.ndarray, np.ndarray, np.ndarray
]:
    """Identity reduction: alpha = u[:, :d_o], V = I[:, :d_o], mu = 0.

    `d_o=None` keeps the full state width.
    """
    n = u.shape[1]
    d = n if d_o is None else int(d_o)
    if d < 1 or d > n:
        raise SystemExit(f"--d-o={d} out of range [1, {n}]")
    alpha = np.ascontiguousarray(u[:, :d].astype(np.float64))
    V = np.eye(n, d, dtype=np.float64)
    mu = np.zeros(n, dtype=np.float64)
    return alpha, V, mu


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stats-path", type=Path, required=True,
        help="path to stats.npz (the FOM snapshot file)",
    )
    parser.add_argument(
        "--svd-path", type=Path, default=None,
        help="optional svd_basis.npz; if given, take alpha = a[:, :d_o] "
             "from this file and store V[:, :d_o] as the projection basis. "
             "If omitted, use identity reduction.",
    )
    parser.add_argument(
        "--d-o", type=int, default=None,
        help="reduced dimension. Required with --svd-path. Default in the "
             "identity path is the full state width.",
    )
    parser.add_argument(
        "--out", type=Path, required=True,
        help="output coords.npz path",
    )
    args = parser.parse_args()

    print(f"[load] {args.stats_path}", flush=True)
    # Lazy: read only `t` and the scalar metadata. The full `u` (32 GB for
    # KOL42) is loaded only on the identity path; the SVD path takes `alpha`
    # straight from svd_basis.npz and needs `u` for nothing.
    with np.load(args.stats_path, allow_pickle=True) as z:
        t = np.asarray(z["t"], dtype=np.float64)
        dt_native = _resolve_dt({"dtStats": z["dtStats"]} if "dtStats" in z.files
                                else {})
    M = t.shape[0]
    print(f"  t.shape={t.shape}, dtStats={dt_native}", flush=True)

    if args.svd_path is not None:
        if args.d_o is None:
            raise SystemExit("--d-o is required when --svd-path is given")
        print(f"[load] {args.svd_path}", flush=True)
        alpha, V, mu = _alpha_from_svd_basis(args.svd_path, args.d_o)
        n = V.shape[0]
        if alpha.shape[0] != M:
            raise SystemExit(
                f"svd 'a' rows ({alpha.shape[0]}) != stats rows ({M}); "
                "the svd_basis was built on a different snapshot file"
            )
        source_svd = str(args.svd_path.resolve())
    else:
        with np.load(args.stats_path) as z:
            u = np.asarray(z["u"])
        n = u.shape[1]
        alpha, V, mu = _alpha_identity(u, args.d_o)
        source_svd = ""
        print(f"[reduce] identity, d_o={alpha.shape[1]}", flush=True)

    d_o = alpha.shape[1]
    print(f"  alpha.shape={alpha.shape}, V.shape={V.shape}",
          flush=True)

    print(f"[deriv] 5-point full-length at dt={dt_native}", flush=True)
    alpha_dot = deriv_5point_full(alpha, dt_native)
    assert alpha_dot.shape == alpha.shape, (alpha_dot.shape, alpha.shape)

    # Sanity-print: derivative magnitude at boundary vs interior to
    # catch boundary-stencil regressions.
    interior_norm = float(np.linalg.norm(alpha_dot[2:-2]))
    boundary_norm = float(np.linalg.norm(
        np.concatenate([alpha_dot[:2], alpha_dot[-2:]], axis=0)
    ))
    print(f"  ||alpha_dot[interior]||={interior_norm:.3e}", flush=True)
    print(f"  ||alpha_dot[boundary]||={boundary_norm:.3e} "
          f"(4 of {M} rows)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        alpha=alpha.astype(np.float64),
        alpha_dot=alpha_dot.astype(np.float64),
        t=t,
        dt_native=np.float64(dt_native),
        V=V.astype(np.float64),
        mu=mu.astype(np.float64),
        d_o=np.int32(d_o),
        n_state=np.int32(n),
        M=np.int32(M),
        source_stats=str(args.stats_path.resolve()),
        source_svd=source_svd,
    )
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
