#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/RBF_ROM
#SBATCH -J mfu_alpha_sweep
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=R-%x-%j.out
#
# alpha_tangent sweep (NO ridge) on the MFU_NA K=16 RBF fit, parameterised by
# d_o. Companion to the d_o=45 sweep: at d_o=45 the tangent width was the
# dominant lever (best test NRMSE 0.505 @ a=5 -> 0.325 @ a=24) and saturated
# by a~20. Higher d_o lives in a higher-dim space where product-Gaussian
# support collapses faster, so the floor is expected to need WIDER kernels;
# this brackets a up to 32 to catch it.
#
# alpha=5.0 is SKIPPED: do<DO>/alpha5.0 already holds the models-bearing base
# sweep. Each alpha runs the full n_rbf sweep as one background process (own
# sweep.npz + plot); all run in parallel on one node.
# Writes: results/MFU_NA/do<DO>/alpha<A>/sweep.npz + rbf_K16_n_sweep.png
#
# Usage: sbatch scripts/launch_mfu_na_rbf_alpha_sweep_do.sh <45|75|170>

set -euo pipefail

DO="${1:?usage: launch_mfu_na_rbf_alpha_sweep_do.sh <45|75|170>}"

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

K=16
COORDS="results/MFU_NA/pod/coords_do${DO}.npz"
N_RBF_LIST="160 320 640 960 1280 1920 2560 3200 4800 6400"
ALPHA_VALUES=(8.0 12.0 16.0 20.0 24.0 28.0 32.0)

echo "Start: $(date)  | d_o=${DO}  alpha=${ALPHA_VALUES[*]} (no ridge)"
pids=()
for A in "${ALPHA_VALUES[@]}"; do
    OUT="results/MFU_NA/do${DO}/alpha${A}"
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
echo "all ${#ALPHA_VALUES[@]} alpha sweeps done for d_o=${DO}"
