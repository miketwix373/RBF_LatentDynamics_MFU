# RBF-ROM for wall turbulence (RBF-FTC) — application-paper code

Reproducibility code for the RBF-FTC paper. It applies the RBF-ROM method (see the
method-paper repository) to minimal-flow-unit channel turbulence: a reduced-order
model is built from wall-shear-stress data to recover the near-wall cycle, and an
ensemble Kalman filter (EnKF) estimates the state from sparse wall observations.
Testcase: MFU_NA.

## Install

```
pip install -e .
```

Dependencies: numpy, scipy, matplotlib (see `pyproject.toml`).

## Data

All FOM data and precomputed results artefacts are hosted on Zenodo (DOI: TODO) —
see `data/README.md` and `scripts/fetch_data.sh`.

## Reproducing the figures

| Figure | Script |
|---|---|
| 1 (planes / workflow) | `figures/fig1_source/` (hand-assembled TikZ + snapshot PNGs, no script) |
| 2 (reduction grid) | `plot_mfu_na_reduction_grid.py` |
| 3 (cluster metrics) | `plot_mfu_na_cluster_metrics.py` |
| 4 (phase cycle) | `plot_mfu_na_phase_cycle_split.py` |
| 5 (alpha NRMSE) | `plot_mfu_na_alpha_nrmse_do45.py` |
| 6 (VPT vs NRMSE) | `plot_mfu_na_vpt_climate_vs_nrmse.py` |
| 7 (skin friction) | `plot_mfu_na_skin_friction_alpha_grid.py` |
| 8 (PSD modes) | `plot_mfu_na_psd_modes.py` |
| 9–12 (EnKF) | `plot_mfu_na_enkf_paper.py` |
| 14 (slow points) | `plot_ftc_fig_slowpoints.py` |
| 15 (anatomy) | `plot_ftc_fig_anatomy.py` |

(There is no Fig 13 in the paper.)

## experimental/

`experimental/mfu_opo/` holds exploratory MFU_OPO (opposition-control) code retained
for future work. It is **not** part of the published RBF-FTC paper and its data is
not included.

## Citation

```bibtex
@article{TODO,
  title   = {TODO},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO}
}
```

## License

TODO: choose a license before publishing.
