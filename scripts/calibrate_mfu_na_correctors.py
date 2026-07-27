"""Soft-corrector calibration for the selected MFU_NA (d_o, alpha, N*) models.

The additive trust-region corrector (Appendix A) needs one model-specific knob:
the trust radius R (escape threshold) in standardised coordinates. Everything
else -- alpha, k, p -- is a fixed shape. R is the q-quantile of the
nearest-centre distance d_nn over a cloud sampled from the coefficient record,
measured in the model's own standardised metric (mirrors
run_kschao_corrector_series.py so this R equals the one a rollout would derive).

Reads the selection manifest results/MFU_NA/model_selection/best_nrbf.json,
computes R per model, writes:
  - per model, next to model_n<N>.npz:  corrector_calib.npz  (R + soft knobs)
  - aggregate: results/MFU_NA/model_selection/corrector_calibration.{json,npz}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

SHARED_ROOT = (next(_p for _p in Path(__file__).resolve().parents if (_p / "pyproject.toml").exists()))
MFU = SHARED_ROOT / "results/MFU_NA"
# Fixed soft-corrector shape used for calibration.
SOFT_ALPHA, SOFT_K, SOFT_P = 0.85, 10.0, 2.0
CLOUD_Q, N_CLOUD, CLOUD_SEED = 0.99, 10000, 0


def _coords_for(model_path: Path, d_o: int) -> Path:
    return MFU / "pod" / f"coords_do{d_o}.npz"


def calibrate_one(model_path: Path, coords_path: Path) -> dict:
    with np.load(model_path, allow_pickle=True) as z:
        centers_std = np.asarray(z["centers_std"], np.float64)
        mu_A = np.asarray(z["mu_A"], np.float64)
        sigma_A = np.asarray(z["sigma_A"], np.float64)
        stride = int(z["stride"])
        n_train = int(z["n_train"])
    with np.load(coords_path) as zc:
        A_full = np.asarray(zc["alpha"], np.float64)[::stride]

    tree = cKDTree(centers_std)
    rng = np.random.default_rng(CLOUD_SEED)
    n_cv = min(N_CLOUD, A_full.shape[0])
    sel = rng.choice(A_full.shape[0], size=n_cv, replace=False)
    A_std_cloud = (A_full[sel] - mu_A) / sigma_A
    d_nn_cloud, _ = tree.query(A_std_cloud, k=1)
    R = float(np.quantile(d_nn_cloud, CLOUD_Q))

    return {
        "escape_thresh": R,
        "soft_alpha": SOFT_ALPHA, "soft_k": SOFT_K, "soft_p": SOFT_P,
        "cloud_quantile": CLOUD_Q, "n_cloud": int(n_cv), "cloud_seed": CLOUD_SEED,
        "d_nn_cloud_median": float(np.median(d_nn_cloud)),
        "d_nn_cloud_max": float(d_nn_cloud.max()),
        "n_centers": int(centers_std.shape[0]),
        "n_train": n_train, "stride": stride,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifest", type=Path,
                   default=MFU / "model_selection/best_nrbf.json")
    p.add_argument("--out-dir", type=Path, default=MFU / "model_selection")
    args = p.parse_args()

    sel = json.loads(args.manifest.read_text())
    records = []
    print(f"{'d_o':>4} {'alpha':>7} {'N*':>6} {'R':>8} "
          f"{'d_nn_med':>9} {'d_nn_max':>9}")
    for s in sel:
        model_path = Path(s["model_path"])
        coords_path = _coords_for(model_path, s["d_o"])
        cal = calibrate_one(model_path, coords_path)

        per_model = model_path.parent / "corrector_calib.npz"
        np.savez(
            per_model,
            model_path=str(model_path), coords_path=str(coords_path),
            d_o=np.int64(s["d_o"]), alpha_tangent=np.float64(s["alpha_tangent"]),
            n_rbf=np.int64(s["best_n_rbf"]),
            escape_thresh=np.float64(cal["escape_thresh"]),
            soft_alpha=np.float64(cal["soft_alpha"]),
            soft_k=np.float64(cal["soft_k"]),
            soft_p=np.float64(cal["soft_p"]),
            cloud_quantile=np.float64(cal["cloud_quantile"]),
            n_cloud=np.int64(cal["n_cloud"]), cloud_seed=np.int64(cal["cloud_seed"]),
            d_nn_cloud_median=np.float64(cal["d_nn_cloud_median"]),
            d_nn_cloud_max=np.float64(cal["d_nn_cloud_max"]),
            n_centers=np.int64(cal["n_centers"]),
        )

        rec = {**s, **cal, "coords_path": str(coords_path),
               "calib_path": str(per_model)}
        records.append(rec)
        print(f"{s['d_o']:>4} {s['alpha_tangent']:>7.1f} {s['best_n_rbf']:>6} "
              f"{cal['escape_thresh']:>8.4f} {cal['d_nn_cloud_median']:>9.4f} "
              f"{cal['d_nn_cloud_max']:>9.4f}")

    (args.out_dir / "corrector_calibration.json").write_text(
        json.dumps(records, indent=2))
    np.savez(
        args.out_dir / "corrector_calibration.npz",
        d_o=np.array([r["d_o"] for r in records], np.int64),
        alpha_tangent=np.array([r["alpha_tangent"] for r in records], np.float64),
        n_rbf=np.array([r["best_n_rbf"] for r in records], np.int64),
        escape_thresh=np.array([r["escape_thresh"] for r in records], np.float64),
        soft_alpha=np.full(len(records), SOFT_ALPHA),
        soft_k=np.full(len(records), SOFT_K),
        soft_p=np.full(len(records), SOFT_P),
        model_path=np.array([r["model_path"] for r in records]),
        calib_path=np.array([r["calib_path"] for r in records]),
    )
    print(f"\nwrote per-model corrector_calib.npz and aggregate "
          f"{args.out_dir/'corrector_calibration.json'}")


if __name__ == "__main__":
    main()
