#!/usr/bin/env bash
# Fetch the MFU_NA FOM data and precomputed results artefacts from Zenodo.
# Set ZENODO_URL to the record's file base URL, then uncomment the curl lines.
set -euo pipefail

ZENODO_URL="${ZENODO_URL:-}"
if [ -z "$ZENODO_URL" ]; then
  echo "Set ZENODO_URL to the Zenodo record base URL and uncomment the curl lines below."
  exit 1
fi

# curl -L "$ZENODO_URL/MFU_NA_stats.npz" -o data/MFU_NA/stats.npz
# curl -L "$ZENODO_URL/results_MFU_NA.tar.gz" | tar xz
