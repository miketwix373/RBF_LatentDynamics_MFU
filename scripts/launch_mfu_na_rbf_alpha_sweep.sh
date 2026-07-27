#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/RBF_ROM
#SBATCH -J mfu_alpha_sweep
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=R-%x-%j.out
#
# alpha_tangent sweep on the MFU_NA K=16 RBF fit at d_o=45. Each alpha_tangent
# value runs the full n_rbf sweep (its own sweep.npz + plot) as one background
# process; the values run in parallel on one node. Tests whether the tangent
# RBF width moves the ~0.5 test-NRMSE floor (rom-specialist predicted it only
# shifts the bias/variance knee, not the floor; the kNN-linear floor of ~0.37
# says the fittable gap is affine structure, which width cannot supply).
#
# alpha_tangent = tangent half-width multiplier (width = alpha_tangent * sqrt(lambda)).
# Reads:  results/MFU_NA/pod/coords_do45.npz
# Writes: results/MFU_NA/rbf_K16_do45_alpha<A>/sweep.npz + rbf_K16_n_sweep.png
#
# Usage: sbatch scripts/launch_mfu_na_rbf_alpha_sweep.sh

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

DO=45
K=16
COORDS="results/MFU_NA/pod/coords_do${DO}.npz"
N_RBF_LIST="160 320 640 960 1280 1920 2560 3200 4800"
ALPHA_VALUES=(1.5 2.5 3.5 5.0 8.0 12.0)

echo "Start: $(date)"
echo "alpha_tangent values: ${ALPHA_VALUES[*]}"
pids=()
for A in "${ALPHA_VALUES[@]}"; do
    OUT="results/MFU_NA/rbf_K16_do${DO}_alpha${A}"
    mkdir -p "$OUT"
    python -u scripts/run_rbf_K_n_sweep.py \
        --coords-path "$COORDS" --out-dir "$OUT" \
        --K "$K" --n-rbf-list $N_RBF_LIST \
        --stride 2 --train-frac 0.8 --alpha-tangent "$A" \
        --energy-threshold 0.9999 --no-standardise \
        --normal-width-floor 0.001 --no-save-models \
        --title "MFU_NA d_o=${DO} K=${K} alpha_tangent=${A}" \
        > "$OUT/sweep.log" 2>&1 &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then echo "[wait] PID $pid failed"; failed=$((failed+1)); fi
done
echo "End: $(date)"
[ "$failed" -eq 0 ] || { echo "ERROR: $failed proc(s) failed" >&2; exit 1; }
echo "all ${#ALPHA_VALUES[@]} alpha_tangent sweeps done"
