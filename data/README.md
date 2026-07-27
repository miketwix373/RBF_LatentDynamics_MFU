# Data

No data ships in this repository. Everything is hosted on Zenodo (DOI: TODO) and
fetched with `../scripts/fetch_data.sh`.

Required Zenodo inputs:

- `MFU_NA/stats.npz` (~5.2 GB) — wall-shear-stress FOM snapshots
- precomputed `results/MFU_NA/` artefacts read by the scripts:
  `pod/`, `cluster_sweep/`, `do45/`, `EnKF/`, `field_analysis/`, `markovianity/`,
  `upo/`, `paper_images/`

MFU_OPO data (for `experimental/`) is future work and is not published.
