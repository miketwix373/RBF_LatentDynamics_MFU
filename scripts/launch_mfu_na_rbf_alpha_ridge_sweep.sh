#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/RBF_ROM
#SBATCH -J mfu_alpha_ridge
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=R-%x-%j.out
#
# High-alpha_tangent sweep WITH PRESS ridge on the MFU_NA K=16 fit at d_o=45.
# The plain-lstsq alpha sweep showed the tangent width is the dominant lever
# (best test NRMSE 0.505 @ a=5 -> 0.344 @ a=12) but that wide kernels overfit
# past their sweet spot as cond(Phi) climbs to 1e7. The overfit is statistical
# variance from near-collinear columns, not precision (the conditioning path is
# already float64), so PRESS-selected Tikhonov ridge should suppress it and let
# the width push further. This sweeps a=12/16/20/24 with --ridge press to find
# where the floor actually bottoms.
#
# Each alpha runs the full n_rbf sweep as one background process (own sweep.npz
# + plot); the four run in parallel on one node.
# Writes: results/MFU_NA/do45/alpha<A>_ridge/sweep.npz + rbf_K16_n_sweep.png
#
# Usage: sbatch scripts/launch_mfu_na_rbf_alpha_ridge_sweep.sh

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
export MKL_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3
export OMP_NUM_THREADS=3

DO=45
K=16
COORDS="results/MFU_NA/pod/coords_do${DO}.npz"
N_RBF_LIST="160 320 640 960 1280 1920 2560 3200 4800 6400"
ALPHA_VALUES=(12.0 16.0 20.0 24.0)

echo "Start: $(date)"
echo "alpha_tangent values: ${ALPHA_VALUES[*]}  (PRESS ridge)"
pids=()
for A in "${ALPHA_VALUES[@]}"; do
    OUT="results/MFU_NA/do${DO}/alpha${A}_ridge"
    mkdir -p "$OUT"
    python -u scripts/run_rbf_K_n_sweep.py \
        --coords-path "$COORDS" --out-dir "$OUT" \
        --K "$K" --n-rbf-list $N_RBF_LIST \
        --stride 2 --train-frac 0.8 --alpha-tangent "$A" \
        --energy-threshold 0.9999 --no-standardise \
        --normal-width-floor 0.001 --ridge press --no-save-models \
        --title "MFU_NA d_o=${DO} K=${K} alpha_tangent=${A} ridge=press" \
        > "$OUT/sweep.log" 2>&1 &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then echo "[wait] PID $pid failed"; failed=$((failed+1)); fi
done
echo "End: $(date)"
[ "$failed" -eq 0 ] || { echo "ERROR: $failed proc(s) failed" >&2; exit 1; }
echo "all ${#ALPHA_VALUES[@]} alpha_tangent ridge sweeps done"
