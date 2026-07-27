#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/RBF_ROM
#SBATCH -J mfu_corr_calib
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=R-%x-%j.out
#
# Corrector calibration for one MFU_NA d_o. Two stages in one job:
#   1. Refit the K=16, n_rbf=1200, alpha_tangent=8 anisotropic-RBF model with
#      model saving on (the alpha sweep ran --no-save-models), same fit config
#      as launch_mfu_na_rbf_alpha_sweep_do.sh.
#   2. Run the additive trust-region corrector series: escape radius R from the
#      training-cloud d_nn quantile (q=0.99), roll the ROM across the full test
#      horizon, record per-step d_nn / corrector strength / fire fraction.
# Writes results/MFU_NA/do<DO>/corrector_calib/{model_n01200.npz,corrector_series.npz}.
#
# One job per d_o so 45 and 75 run in parallel:
#   sbatch --export=ALL,DO=45 scripts/launch_mfu_na_corrector_calib.sh
#   sbatch --export=ALL,DO=75 scripts/launch_mfu_na_corrector_calib.sh

set -euo pipefail

DO="${DO:?set DO (45 or 75)}"

echo "Start time: $(date)"
echo "Job ID:     ${SLURM_JOB_ID:-(local)}"
echo "Node:       ${SLURM_JOB_NODELIST:-(local)}"
echo "d_o=${DO}"

if command -v flight >/dev/null 2>&1; then flight env activate gridware; fi
__conda_setup="$('/mnt/scratch/users/sbrw610/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then eval "$__conda_setup"; fi
unset __conda_setup
conda activate /mnt/scratch/users/sbrw610/anaconda3/envs/cfd_new

export MKL_THREADING_LAYER=GNU
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
export LD_LIBRARY_PATH="/opt/apps/flight/env/conda+jupyter/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$(pwd):$(pwd)/scripts:${PYTHONPATH:-}"

THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export OMP_NUM_THREADS="$THREADS"

COORDS="results/MFU_NA/pod/coords_do${DO}.npz"
OUT="results/MFU_NA/do${DO}/corrector_calib"
MODEL="${OUT}/model_n01200.npz"
mkdir -p "$OUT"

echo "=== stage 1: fit K=16 n_rbf=1200 alpha=8 (save model) ==="
python -u scripts/run_rbf_K_n_sweep.py \
    --coords-path "$COORDS" --out-dir "$OUT" \
    --K 16 --n-rbf-list 1200 \
    --stride 2 --train-frac 0.8 --alpha-tangent 8.0 \
    --energy-threshold 0.9999 --no-standardise \
    --normal-width-floor 0.001 --save-models \
    --title "MFU_NA d_o=${DO} K=16 n_rbf=1200 alpha=8 (corrector calib)"

echo "=== stage 2: additive trust-region corrector series ==="
python -u scripts/run_kschao_corrector_series.py \
    --model "$MODEL" \
    --coords-path "$COORDS" \
    --soft-alpha 0.85 --soft-k 10.0 --soft-p 2.0 \
    --cloud-escape-quantile 0.99 \
    --progress-every 2000 \
    --out "${OUT}/corrector_series.npz"

echo "End time: $(date)"
