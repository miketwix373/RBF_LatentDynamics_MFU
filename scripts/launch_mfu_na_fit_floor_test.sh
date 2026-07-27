#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/RBF_ROM
#SBATCH -J mfu_floor_test
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=R-%x-%j.out
#
# Irreducible-floor + closure-gap diagnostic on the MFU_NA K=16 RBF fit.
# kNN conditional-mean floor (robust; local-CONSTANT so no parameter-count
# degeneracy at high d_o) plus a large-k local-LINEAR cross-check, and the
# truncated-mode residual R^2 (Mori-Zwanzig closure headroom). d_o=45 and 75.
# Truncated coordinates come from the 170-mode coords file.
#
# Usage: sbatch scripts/launch_mfu_na_fit_floor_test.sh

set -euo pipefail

if command -v flight >/dev/null 2>&1; then flight env activate gridware; fi
__conda_setup="$('/mnt/scratch/users/sbrw610/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then eval "$__conda_setup"; fi
unset __conda_setup
conda activate /mnt/scratch/users/sbrw610/anaconda3/envs/cfd_new

export MKL_THREADING_LAYER=GNU
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
export LD_LIBRARY_PATH="/opt/apps/flight/env/conda+jupyter/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$(pwd):$(pwd)/scripts:${PYTHONPATH:-}"
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2

OUT="results/MFU_NA/fit_floor_test"
WIDE="results/MFU_NA/pod/coords_do170.npz"

run () {  # d_o  n_rbf  sweep_dir  knn_mode  k
    python -u scripts/run_mfu_na_fit_floor_test.py --do "$1" --n-rbf "$2" \
        --sweep-dir "$3" --wide-coords "$WIDE" --out-dir "$OUT" \
        --knn-mode "$4" --k "$5" --knn-test-sub 4000 \
        > "$OUT/run_do$1_$4.log" 2>&1
}

D45="results/MFU_NA/rbf_K16_n_sweep_do45_raw_nwf0.001_e0.9999"
D75="results/MFU_NA/rbf_K16_n_sweep_do75_raw_nwf0.001_e0.9999"

mkdir -p "$OUT"
echo "Start: $(date)"
run 45 2560 "$D45" constant 200 &
run 45 2560 "$D45" linear   600 &
run 75 4800 "$D75" constant 200 &
run 75 4800 "$D75" linear   600 &
wait
echo "End: $(date)"
