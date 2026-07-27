#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/RBF_ROM
#SBATCH -J mfu_opo_svd
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=R-%x-%j.out
#
# Global CENTRED joint POD of the MFU_OPO wall-shear state [tau_x | tau_y]
# (N=8192) via the streaming Gram path in chord2/baselines/svd_basis.py.
# One basis, frozen: near-degenerate pairs rotate arbitrarily on rerun.
#
# Conventions mirror MFU_NA (2026-07-07):
#   - CENTRED: POD of fluctuations about the time mean, stored as `mu`;
#     reconstruction u ~ mu + a@V.T.
#   - gram_stride=8, n_save=400. OPO is less compressible than NA
#     (90% at d_o=76, 95% at d_o=128), so 400 saved modes still covers
#     the 95% budget with headroom.
#
# Usage
#     sbatch scripts/launch_mfu_opo_svd.sh

set -euo pipefail

echo "MFU_OPO centred POD -- $(date) -- ${SLURM_JOB_NODELIST:-(local)}"

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

THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export OMP_NUM_THREADS="$THREADS"

python -u chord2/baselines/svd_basis.py \
    --stats-path /users/sbrw610/sharedscratch/RBF_ROM/data/MFU_OPO/stats.npz \
    --out-dir    results/MFU_OPO/pod \
    --d_o        76 \
    --center \
    --n-save     400 \
    --gram-stride 8 \
    --device     cpu

echo "End time: $(date)"
