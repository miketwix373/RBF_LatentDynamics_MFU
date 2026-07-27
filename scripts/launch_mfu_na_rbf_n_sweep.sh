#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/RBF_ROM
#SBATCH -J mfu_rbf_sweep
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=R-%x-%j.out
#
# MFU_NA anisotropic-RBF n_rbf sweep on the centred-POD coords, one job per
# d_o (45/75/170 = 90/95/99% fluctuation energy). K fixed at 16, the knee of
# the distinctness / velocity-contrast cluster metrics (cluster_metrics_all).
#
# Config mirrors the KOL42 high-d_o sweep (launch_kol42_rbf_n_sweep_raw_nwf.sh):
#   - --no-standardise: per-cluster PCA centred only (not z-scored), so the
#     POD mode-variance anisotropy carries physical energy.
#   - --energy-threshold 0.9999: promotes all kept directions to TANGENT
#     (5*sqrt(lambda) width) instead of leaving low-energy PCA directions as
#     ~1 sigma NORMAL directions, which on high d_o kill kernel support.
#   - --normal-width-floor 0.001: relative floor on the normal half-width so a
#     sub-percent normal drift cannot switch every kernel off.
#   - n_rbf are TOTAL centre budgets; each cluster gets n_rbf // K.
#
# Reads:  results/MFU_NA/pod/coords_do<DO>.npz   (M=79672, dt_native=0.05)
# Writes: results/MFU_NA/rbf_K<K>_n_sweep_do<DO>_raw_nwf0.001_e0.9999/
#           (n_<value>.npz + model_n<value>.npz + .log)
#
# Usage (cluster): sbatch scripts/launch_mfu_na_rbf_n_sweep.sh <45|75|170> [K]
# Usage (local):   bash   scripts/launch_mfu_na_rbf_n_sweep.sh <45|75|170> [K]
#
# After completion (per d_o):
#   python scripts/aggregate_rbf_K1_n_sweep.py \
#       --out-dir results/MFU_NA/rbf_K16_n_sweep_do<DO>_raw_nwf0.001_e0.9999 \
#       --title  "MFU_NA centred POD, K=16 raw RBF sweep (d_o=<DO>, stride 2)"

set -euo pipefail

DO="${1:?usage: launch_mfu_na_rbf_n_sweep.sh <45|75|170> [K]}"
K="${2:-16}"
NORMAL_WIDTH_FLOOR=0.001
ENERGY_THRESHOLD=0.9999
STRIDE=2
# stride 2 -> 39836 rows, ~31869 train: matches the density the cluster-quality
# metrics were computed at, and stays overdetermined for every swept n_rbf.
# n_rbf are TOTAL budgets; n_per_cluster = n_rbf // K must be >= 2, so the
# smallest total is 32 at K=16.
N_RBF_VALUES=(32 80 160 320 640 960 1280 1920 2560 3200 4800 6400)
N_PROCS=${#N_RBF_VALUES[@]}

echo "================================================="
echo "RBF K=$K n_rbf sweep | MFU_NA centred POD (d_o=$DO) | RAW (--no-standardise) | nwf=$NORMAL_WIDTH_FLOOR | e=$ENERGY_THRESHOLD | $N_PROCS parallel n_rbf on one node"
echo "================================================="
echo "Start time: $(date)"
echo "Job ID:     ${SLURM_JOB_ID:-(local)}"
echo "Node:       ${SLURM_JOB_NODELIST:-(local)}"
echo "Cores:      ${SLURM_CPUS_PER_TASK:-?}"
echo "d_o:         $DO"
echo "K (cluster): $K"
echo "n_rbf set:   ${N_RBF_VALUES[*]}"
echo ""

if command -v flight >/dev/null 2>&1; then
    flight env activate gridware
fi

__conda_setup="$('/mnt/scratch/users/sbrw610/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
fi
unset __conda_setup
conda activate /mnt/scratch/users/sbrw610/anaconda3/envs/cfd_new

export MKL_THREADING_LAYER=GNU
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
export LD_LIBRARY_PATH="/opt/apps/flight/env/conda+jupyter/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$(pwd):$(pwd)/scripts:${PYTHONPATH:-}"

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

OUT_DIR="results/MFU_NA/rbf_K${K}_n_sweep_do${DO}_raw_nwf${NORMAL_WIDTH_FLOOR}_e${ENERGY_THRESHOLD}"
COORDS="results/MFU_NA/pod/coords_do${DO}.npz"
TITLE="MFU_NA centred POD, K=${K} raw-space anisotropic RBF sweep (d_o=${DO}, stride ${STRIDE}), nwf=${NORMAL_WIDTH_FLOOR}, energy=${ENERGY_THRESHOLD}"

mkdir -p "$OUT_DIR"

echo "Coords: $COORDS"
echo "Out:    $OUT_DIR"
echo ""

pids=()
for N_RBF in "${N_RBF_VALUES[@]}"; do
    TAG=$(printf "n_%05d" "$N_RBF")
    LOG="$OUT_DIR/${TAG}.log"
    echo "[spawn] $TAG -> $LOG"
    python -u scripts/run_rbf_K_n_sweep.py \
        --coords-path "$COORDS" \
        --out-dir     "$OUT_DIR" \
        --K           "$K" \
        --n-rbf-list  "$N_RBF" \
        --stride      "$STRIDE" \
        --train-frac  0.8 \
        --alpha-tangent 5.0 \
        --energy-threshold "$ENERGY_THRESHOLD" \
        --no-standardise \
        --normal-width-floor "$NORMAL_WIDTH_FLOOR" \
        --title       "$TITLE" \
        > "$LOG" 2>&1 &
    pids+=("$!")
done

echo ""
echo "[wait] ${#pids[@]} processes; waiting on PIDs: ${pids[*]}"
failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        echo "[wait] PID $pid exited non-zero"
        failed=$((failed + 1))
    fi
done

echo ""
echo "End time: $(date)"
if [ "$failed" -gt 0 ]; then
    echo "ERROR: $failed process(es) failed; check $OUT_DIR/n_*.log" >&2
    exit 1
fi
echo "all $N_PROCS processes completed successfully"
