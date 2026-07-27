"""Global SVD reduced observable (POD).

Originally written to reproduce the dimensionality reduction of §4.1 of
Vlachas, Pathak, Hunt, Sapsis, Girvan, Ott, Koumoutsakos, *Backpropagation
algorithms and Reservoir Computing in Recurrent Neural Networks for the
forecasting of complex spatiotemporal dynamics*, Neural Networks 126 (2020)
191-217:

    "we construct observables of dimension d_o ∈ {35, 40} by performing
     Singular Value Decomposition (SVD) and keeping the most energetic d_o
     components."

Two paths, dispatched on the `stats.npz` layout:

- **scalar1d** (`u` of shape (T, N), e.g. KS / LOR96): un-centred thin SVD
  of the raw snapshot matrix, matching the Vlachas convention. The first
  mode absorbs the mean structure; `mu = 0`.

- **vector2d_uv** (`u`, `v` each (T, ny, nx), e.g. Kolmogorov flow): a single
  *joint* POD on the stacked velocity `X = [u_flat | v_flat]` of width
  `N = 2*ny*nx`. On a uniform periodic grid the Euclidean inner product on
  the stacked vector is proportional to the discrete kinetic energy, so the
  modes are KE-optimal (Holmes/Lumley/Berkooz/Rowley 2012; Sirovich 1987).
  Computed by the method-of-snapshots transpose: stream the spatial Gram
  `G = (X-mu)^T (X-mu)` in float64 over snapshot blocks, eigendecompose
  (`numpy.linalg.eigh`), `S = sqrt(eigvals)`. This keeps KOL42 (1e6 x 8192,
  32 GB on disk) lazy via a memmap into the .npz and never forms the tall
  thin-SVD `U` factor. Per the fluid-dynamics-specialist, Kolmogorov is
  CENTRED (`--center`): the forced mean shear `ū ~ sin(k_f y) e_x` is
  subtracted and stored as `mu`, so `d_o` budgets the fluctuation dynamics
  rather than a static, analytically-known base flow, and the downstream
  SINDy regression is not saddled with a large near-constant mode-0
  coordinate. Reconstruction is `u ≈ mu + a @ V.T`.

This is the *global* PCA used as a Vlachas-style baseline; it is distinct from
`chord2/local_basis.py` (per-cluster local PCA used by the quantized ROM).
One basis per dataset, saved alongside the reduced coordinates so downstream
forecasters can train on `a` directly.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from chord2.backend import asnumpy, describe_device, get_backend  # noqa: E402
from chord2.data import _open_npy_in_npz_memmap  # noqa: E402

DEFAULT_DO_REFERENCE = 35   # Vlachas et al. 2020 §4.1
DEFAULT_BLOCK = 4096        # snapshots per streaming block
DEFAULT_N_SAVE = 256        # cap modes stored in V/a (KOL42 full `a` is 32 GB)


def compute_svd_basis(u: np.ndarray) -> dict:
    """Un-centred POD-style thin SVD of the snapshot matrix `u`, shape (T, N).

    Matches the convention used by Vlachas et al. 2020 §4.1 (and the classical
    POD-Galerkin literature): SVD is applied directly to the raw snapshot
    matrix without subtracting the time-mean, so the first mode typically
    captures the mean structure. Returns the right singular vectors as columns
    of `V` (POD modes in R^N), the singular values `S`, the cumulative energy
    fraction, and the reduced coordinates `a = u @ V` such that `u ≈ a @ V.T`.
    """
    # Thin SVD: u = U_t @ diag(S) @ Vh, V is (N, N), S is (N,).
    _, S, Vh = np.linalg.svd(u, full_matrices=False)
    V = Vh.T
    a = u @ V

    energy = S ** 2
    energy_frac = np.cumsum(energy) / energy.sum()

    return {"V": V, "S": S, "energy_frac": energy_frac, "a": a}


def _fix_mode_signs(V: np.ndarray) -> None:
    """Deterministic sign: force the largest-magnitude entry of each column
    positive, in place. `eigh` sign is arbitrary; pinning it makes reruns and
    the f32/f64 input paths produce identical `a`."""
    for k in range(V.shape[1]):
        j = int(np.argmax(np.abs(V[:, k])))
        if V[j, k] < 0.0:
            V[:, k] *= -1.0


def compute_svd_basis_streaming(read_block, M: int, N: int, *,
                                center: bool, n_save: int, xp,
                                gram_stride: int = 1,
                                block: int = DEFAULT_BLOCK) -> dict:
    """Joint POD via blocked spatial-Gram eigendecomposition, numpy or cupy.

    `read_block(idx) -> (k, N) float64` host array, where `idx` is a slice or
    an integer-index ndarray into the M snapshots. Numerically equivalent to
    the thin SVD `V`/`a` up to sign and ordering.

    The mean and the spatial Gram are estimated from a `gram_stride`-strided
    subset (the basis only needs ~thousands of decorrelated snapshots), while
    the reduced coordinates `a` are projected from EVERY snapshot so the
    downstream 5-point derivative stays at native dt. The Gram is accumulated
    in float64 regardless of the snapshots' stored dtype (KOL is f32) so the
    kept modes (`S_i/S_0 >~ 1e-3`) sit well above the f64 floor despite the
    eigenvalue squaring. `S`, `energy_frac` are full length (N); `V`, `a` are
    truncated to the top `n_save` modes so the saved `a` stays bounded.

    All linear algebra runs on the backend `xp`; only `read_block` (disk I/O)
    and the final `a` buffer live on the host.
    """
    gidx = np.arange(0, M, gram_stride)
    n_gram = gidx.size

    mu_dev = xp.zeros(N, dtype=xp.float64)
    if center:                                   # pass 1: time-mean (strided)
        for c0 in range(0, n_gram, block):
            X = xp.asarray(read_block(gidx[c0:c0 + block]))
            mu_dev = mu_dev + X.sum(axis=0)
        mu_dev = mu_dev / n_gram

    G = xp.zeros((N, N), dtype=xp.float64)       # pass 2: spatial Gram (strided)
    for c0 in range(0, n_gram, block):
        X = xp.asarray(read_block(gidx[c0:c0 + block]))
        if center:
            X = X - mu_dev
        G = G + X.T @ X

    w, V = xp.linalg.eigh(G)                      # ascending eigenpairs
    w = w[::-1]
    V = V[:, ::-1]
    w = xp.clip(w, 0.0, None)
    S = asnumpy(xp.sqrt(w))
    wn = asnumpy(w)
    energy_frac = np.cumsum(wn) / wn.sum()

    n_save = min(n_save, N)
    Vt = asnumpy(V[:, :n_save]).copy()
    _fix_mode_signs(Vt)                           # deterministic sign, on host
    Vt_dev = xp.asarray(Vt)

    a = np.empty((M, n_save), dtype=np.float64)  # pass 3: project ALL snapshots
    for b0 in range(0, M, block):
        b1 = min(M, b0 + block)
        X = xp.asarray(read_block(slice(b0, b1)))
        if center:
            X = X - mu_dev
        a[b0:b1] = asnumpy(X @ Vt_dev)

    return {"V": Vt, "S": S, "energy_frac": energy_frac, "a": a,
            "mu": asnumpy(mu_dev), "n_gram": int(n_gram),
            "gram_stride": int(gram_stride)}


def _print_energy_table(S: np.ndarray, ef: np.ndarray, N: int,
                        d_o: int) -> None:
    d_o = min(d_o, N)
    print(f"Singular values:   S[0]={S[0]:.3f}  S[-1]={S[-1]:.3e}  "
          f"S[{d_o-1}]={S[d_o-1]:.3e}")
    for d in sorted({d_o, 10, 20, 30, 50, 100, N}):
        if 1 <= d <= N:
            tag = "  (full state)" if d == N else ""
            print(f"Energy fraction at d_o={d:>4}:  {ef[d-1]*100:.4f}%{tag}")


def _is_vector_uv(stats_path: Path) -> bool:
    with zipfile.ZipFile(stats_path) as zf:
        names = set(zf.namelist())
    return "v.npy" in names


def run(stats_path: Path, out_dir: Path,
        d_o: int = DEFAULT_DO_REFERENCE,
        center: bool = False,
        n_save: int = DEFAULT_N_SAVE,
        block: int = DEFAULT_BLOCK,
        device: str = "auto",
        gram_stride: int = 1) -> Path:
    if not stats_path.exists():
        raise FileNotFoundError(f"{stats_path} not found.")

    if _is_vector_uv(stats_path):
        xp, on_gpu = get_backend(device)
        u_mm = _open_npy_in_npz_memmap(stats_path, "u.npy")
        v_mm = _open_npy_in_npz_memmap(stats_path, "v.npy")
        M = int(u_mm.shape[0])
        csz = int(u_mm.shape[1] * u_mm.shape[2])
        N = 2 * csz                              # stacked state width
        n_save = min(n_save, N)

        def read_block(idx) -> np.ndarray:
            ub = np.asarray(u_mm[idx], dtype=np.float64)
            vb = np.asarray(v_mm[idx], dtype=np.float64)
            k = ub.shape[0]
            return np.concatenate([ub.reshape(k, -1), vb.reshape(k, -1)],
                                  axis=1)

        print(f"Loaded {stats_path}  vector2d_uv  M={M}  N={N} "
              f"(component_size={csz})  center={center}  n_save={n_save}\n"
              f"  backend: {describe_device(xp, on_gpu)}  "
              f"gram_stride={gram_stride} (n_gram={len(range(0, M, gram_stride))})")
        res = compute_svd_basis_streaming(
            read_block, M, N, center=center, n_save=n_save, xp=xp,
            gram_stride=gram_stride, block=block)
    else:
        d = np.load(stats_path, allow_pickle=True)
        u = d["u"].astype(np.float64)
        N = u.shape[1]
        M = u.shape[0]
        print(f"Loaded {stats_path}  scalar1d  u.shape={u.shape}  N={N}  "
              f"center={center}")
        if center:
            mu = u.mean(axis=0)
            res = compute_svd_basis(u - mu)
            res["mu"] = mu
        else:
            res = compute_svd_basis(u)
            res["mu"] = np.zeros(N, dtype=np.float64)
        n_save = N

    S, ef = res["S"], res["energy_frac"]
    _print_energy_table(S, ef, N, d_o)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "svd_basis.npz"
    np.savez(
        out_path,
        V=res["V"].astype(np.float64),
        S=res["S"].astype(np.float64),
        energy_frac=res["energy_frac"].astype(np.float64),
        a=res["a"].astype(np.float32),
        mu=res["mu"].astype(np.float64),
        N=np.int64(N),
        M=np.int64(M),
        n_save=np.int64(res["V"].shape[1]),
        centered=np.bool_(center),
        gram_stride=np.int64(res.get("gram_stride", 1)),
        n_gram=np.int64(res.get("n_gram", M)),
        d_o_report=np.int64(min(d_o, N)),
        source_stats=np.array(str(stats_path)),
    )
    print(f"Saved {out_path}  (V {res['V'].shape}, a {res['a'].shape}, "
          f"mu nonzero={np.abs(res['mu']).max():.3e})")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Global SVD reduced observable (POD)."
    )
    ap.add_argument("--stats-path", type=Path, required=True,
                    help="Path to the input stats.npz (CHORD2 schema). "
                         "scalar1d ('u' (T,N)) or vector2d_uv ('u','v' "
                         "(T,ny,nx)) layouts are auto-detected.")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Output directory; svd_basis.npz is written here.")
    ap.add_argument("--d_o", type=int, default=DEFAULT_DO_REFERENCE,
                    help="Reference reduced dimension to report energy at "
                         "(default 35).")
    ap.add_argument("--center", action="store_true",
                    help="Subtract the time-mean before POD and store it as "
                         "`mu` (reconstruction u ≈ mu + a@V.T). Required for "
                         "Kolmogorov (forced mean shear); off for KS/LOR96.")
    ap.add_argument("--n-save", type=int, default=DEFAULT_N_SAVE,
                    help="Max modes stored in V and a for the vector path "
                         "(default 256; the full `a` for KOL42 would be 32 GB).")
    ap.add_argument("--block", type=int, default=DEFAULT_BLOCK,
                    help="Snapshots per streaming block (vector path).")
    ap.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto",
                    help="Backend for the Gram/eigh/projection (vector path) "
                         "via the chord2.backend cupy shim. Default auto.")
    ap.add_argument("--gram-stride", type=int, default=1,
                    help="Estimate V/mu from every gram-stride-th snapshot "
                         "(the basis only needs ~thousands of decorrelated "
                         "samples); `a` is still projected from ALL snapshots "
                         "at native dt. Default 1.")
    args = ap.parse_args()
    run(args.stats_path, args.out_dir, d_o=args.d_o, center=args.center,
        n_save=args.n_save, block=args.block, device=args.device,
        gram_stride=args.gram_stride)
