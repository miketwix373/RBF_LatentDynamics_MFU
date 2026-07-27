#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/RBF_ROM
#SBATCH -J mfu_enkf
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --output=R-%x-%j.out
#
# EnKF twin experiment for one MFU_NA do45 alpha_tangent model, three stages
# in one job (the pipeline that produced EnKF/enkf_ensemble_modes.pdf for
# alpha=3.5):
#   1. assimilating run: N_ens members, perturbed-obs analysis vs sparse
#      wall-shear obs every CADENCE t+, trajectories kept
#   2. free-running baseline: 1 member, zero spread, same horizon
#   3. ensemble-modes figure via plot_mfu_na_enkf_modes.py
# Writes results/MFU_NA/EnKF/alpha<ALPHA>/{ensemble_forecast.npz,freerun/,
# enkf_ensemble_modes.pdf}.
#
#   sbatch --export=ALL,ALPHA=8.0 scripts/launch_mfu_na_enkf.sh
#
# Env knobs: ALPHA (3.5), N_ENS (100), INTERVAL_S (17.72 = 200 t+),
# CADENCE (5), STRIDE (8), SEED (0). Model + trust radius come from the
# per-dir corrector_calib.npz (pinned N* selection).

set -euo pipefail

ALPHA="${ALPHA:-3.5}"

echo "Start time: $(date)"
echo "Job ID:     ${SLURM_JOB_ID:-(local)}"
echo "Node:       ${SLURM_JOB_NODELIST:-(local)}"
echo "alpha_tangent=${ALPHA}"

if command -v flight >/dev/null 2>&1; then flight env activate gridware; fi
__conda_setup="$('/mnt/scratch/users/sbrw610/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then eval "$__conda_setup"; fi
unset __conda_setup
conda activate /mnt/scratch/users/sbrw610/anaconda3/envs/cfd_new

export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
export LD_LIBRARY_PATH="/opt/apps/flight/env/conda+jupyter/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$(pwd):$(pwd)/scripts:${PYTHONPATH:-}"
export MKL_THREADING_LAYER=GNU

# Members parallelise across processes; keep BLAS single-threaded per process.
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

CALIB="results/MFU_NA/do45/alpha${ALPHA}/corrector_calib.npz"
MODEL=$(python -c "import numpy as np; \
print(str(np.load('$CALIB', allow_pickle=True)['model_path']))")
OUT="results/MFU_NA/EnKF/alpha${ALPHA}"
echo "model: $MODEL"

echo "=== stage 1: EnKF assimilating run ==="
python -u scripts/run_mfu_na_enkf.py \
    --model "$MODEL" --calib "$CALIB" \
    --assimilate --keep-traj \
    --n-ens "${N_ENS:-100}" \
    --interval-s "${INTERVAL_S:-17.72}" \
    --cadence-tplus "${CADENCE:-5}" \
    --sensor-stride "${STRIDE:-8}" --components both \
    --sigma-obs 0.1 --inflation 1.02 --additive-q 0.1 --corr-k-scale 0 \
    --init perturb --spread 0.1 --seed "${SEED:-0}" \
    --n-workers "${SLURM_CPUS_PER_TASK:-48}" \
    --out-dir "$OUT"

echo "=== stage 2: free-running baseline ==="
python -u scripts/run_mfu_na_enkf.py \
    --model "$MODEL" --calib "$CALIB" \
    --keep-traj --n-ens 1 --spread 0.0 \
    --interval-s "${INTERVAL_S:-17.72}" \
    --init perturb --seed "${SEED:-0}" \
    --n-workers 1 \
    --out-dir "$OUT/freerun"

echo "=== stage 3: ensemble-modes figure ==="
python -u scripts/plot_mfu_na_enkf_modes.py \
    --ensemble "$OUT/ensemble_forecast.npz" \
    --freerun "$OUT/freerun/ensemble_forecast.npz" \
    --out "$OUT/enkf_ensemble_modes.pdf"

echo "End time: $(date)"
