#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/RBF_ROM
#SBATCH -J mfu_vpt
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=R-%x-%j.out
#
# Reconstruction-based VPT distribution over 96 initial conditions, one IC per
# SLURM task across 2 nodes x 48 workers. Each worker runs a free-running (no
# corrector) rollout and records the two candidate VPT criteria; a final
# aggregation step pools the partials and plots the distribution.
#
# Usage:
#   sbatch --export=ALL,DO=45 scripts/launch_mfu_na_vpt.sh

set -euo pipefail

DO="${DO:-45}"
NIC="${NIC:-96}"
T="${T:-30}"

if command -v flight >/dev/null 2>&1; then flight env activate gridware; fi
__conda_setup="$('/mnt/scratch/users/sbrw610/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then eval "$__conda_setup"; fi
unset __conda_setup
conda activate /mnt/scratch/users/sbrw610/anaconda3/envs/cfd_new

export MKL_THREADING_LAYER=GNU
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
export LD_LIBRARY_PATH="/opt/apps/flight/env/conda+jupyter/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$(pwd):$(pwd)/scripts:${PYTHONPATH:-}"
# one thread per task: 48 tasks share the node's cores
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

MODEL="results/MFU_NA/do${DO}/corrector_calib/model_n01200.npz"
COORDS="results/MFU_NA/pod/coords_do${DO}.npz"
OUT="results/MFU_NA/do${DO}/climate_validation/vpt_ics"
mkdir -p "$OUT"
rm -f "$OUT"/part_*.npz

echo "Start: $(date)  d_o=${DO}  n_ics=${NIC}  T=${T}s  (2 nodes x 48 workers)"

srun --ntasks=96 --ntasks-per-node=48 \
    python -u scripts/run_mfu_na_vpt_worker.py \
        --model "$MODEL" --coords-path "$COORDS" \
        --n-ics "$NIC" --T "$T" \
        --e-thresh 0.25 --acc-thresh 0.5 --out-dir "$OUT"

echo "=== aggregate + plot ==="
python -u scripts/plot_mfu_na_vpt_distribution.py \
    --parts-dir "$OUT" \
    --out-path "results/MFU_NA/do${DO}/climate_validation/vpt_distribution.png"

echo "End: $(date)"
