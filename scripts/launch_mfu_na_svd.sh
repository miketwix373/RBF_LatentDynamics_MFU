#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/RBF_ROM
#SBATCH -J mfu_na_svd
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=R-%x-%j.out
#
# Global CENTRED joint POD of the MFU_NA wall-shear state [tau_x | tau_y]
# (N=8192) via the streaming Gram path in chord2/baselines/svd_basis.py.
#
# Conventions:
#   - CENTRED (user decision, 2026-07-07): POD of the fluctuations about the
#     time-mean field, stored as `mu`; reconstruction u ~ mu + a@V.T. This
#     overrides the rom-specialist's uncentred recommendation (archived
#     basis at results/MFU_NA/pod_uncentred/).
#   - gram_stride=8 (~10k samples at spacing 0.4): denser never biases the
#     Gram; cost is trivial at N=8192.
#   - n_save=400: covers the 99.9% count; modes >~200 are diagnostic only.
# Build once and freeze: near-degenerate pairs rotate arbitrarily on rerun.
#
# Usage
#     sbatch scripts/launch_mfu_na_svd.sh

set -euo pipefail

echo "================================================="
echo "MFU_NA global uncentred POD"
echo "================================================="
echo "Start time: $(date)"
echo "Job ID:     ${SLURM_JOB_ID:-(local)}"
echo "Node:       ${SLURM_JOB_NODELIST:-(local)}"
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
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export OMP_NUM_THREADS="$THREADS"

python -u chord2/baselines/svd_basis.py \
    --stats-path /users/sbrw610/sharedscratch/RBF_ROM/data/MFU_NA/stats.npz \
    --out-dir    results/MFU_NA/pod \
    --d_o        45 \
    --center \
    --n-save     400 \
    --gram-stride 8 \
    --device     cpu

echo ""
echo "End time: $(date)"
