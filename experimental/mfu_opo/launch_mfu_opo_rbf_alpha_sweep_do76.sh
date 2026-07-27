#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/RBF_ROM
#SBATCH -J opo_alpha76
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=R-%x-%j.out
#
# alpha_tangent sweep (NO ridge) on the MFU_OPO K=10 RBF fit at d_o=76 (90%
# fluctuation energy). K=10 is the cluster-quality winner (distinctness knee,
# both energy cuts). d_o=76 is high-dimensional (like MFU_NA d_o=75), where
# product-Gaussian support collapses fast, so the NRMSE floor is expected to
# need WIDER kernels; the grid brackets alpha up to 32 to catch it, and keeps
# the low end (4) to locate the bias/variance knee.
#
# alpha_tangent = tangent half-width multiplier (width = alpha_tangent*sqrt(lambda)).
# Each alpha runs the full n_rbf sweep as one background process (own sweep.npz
# + plot); all run in parallel on one node. --save-models persists the fitted
# coefficients per (alpha, n_rbf) as model_n<NNNNN>.npz for downstream integration.
# Reads:  results/MFU_OPO/pod/coords_do76.npz
# Writes: results/MFU_OPO/do76/alpha<A>/{sweep.npz, model_n*.npz, rbf_K10_n_sweep.png}
#
# Usage: sbatch scripts/launch_mfu_opo_rbf_alpha_sweep_do76.sh

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

DO=76
K=10
COORDS="results/MFU_OPO/pod/coords_do${DO}.npz"
N_RBF_LIST="160 320 640 960 1280 1920 2560 3200 4800 6400"
ALPHA_VALUES=(4.0 8.0 12.0 16.0 20.0 24.0 28.0 32.0)

echo "Start: $(date)  | d_o=${DO}  K=${K}  alpha=${ALPHA_VALUES[*]} (no ridge)"
pids=()
for A in "${ALPHA_VALUES[@]}"; do
    OUT="results/MFU_OPO/do${DO}/alpha${A}"
    mkdir -p "$OUT"
    python -u scripts/run_rbf_K_n_sweep.py \
        --coords-path "$COORDS" --out-dir "$OUT" \
        --K "$K" --n-rbf-list $N_RBF_LIST \
        --stride 2 --train-frac 0.8 --alpha-tangent "$A" \
        --energy-threshold 0.9999 --no-standardise \
        --normal-width-floor 0.001 --save-models \
        --title "MFU_OPO d_o=${DO} K=${K} alpha_tangent=${A}" \
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
