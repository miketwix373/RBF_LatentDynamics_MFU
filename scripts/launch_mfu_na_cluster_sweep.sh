#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/RBF_ROM
#SBATCH -J mfu_cl_sweep
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=R-%x-%j.out
#
# Cluster-quality K-sweep on the MFU_NA centred-POD reduced state, one job
# per d_o (45/75/170 = 90/95/99% fluctuation energy). Seven metrics, one K
# per worker, NO RBF/SINDy fits.
#
# Reuses run_ks_bur_arc_cluster_sweep.py: it reads (alpha, alpha_dot)
# empirically from a coords file and never invokes an analytical RHS.
# Stride 2 -> ~39.8k samples (spacing 0.1, ~23 per fast advective period),
# inside the sample-count range the kNN metric handles (KS_CHAO used ~60k).
#
# Usage
#     sbatch scripts/launch_mfu_na_cluster_sweep.sh 45
#     sbatch scripts/launch_mfu_na_cluster_sweep.sh 75
#     sbatch scripts/launch_mfu_na_cluster_sweep.sh 170

set -euo pipefail

DO="${1:?usage: launch_mfu_na_cluster_sweep.sh <45|75|170>}"

echo "================================================="
echo "MFU_NA cluster-quality sweep (d_o=${DO})"
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
export PYTHONPATH="$(pwd):$(pwd)/scripts:${PYTHONPATH:-}"

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

WORKERS="${SLURM_CPUS_PER_TASK:-16}"

python -u scripts/run_ks_bur_arc_cluster_sweep.py \
    --coords-path results/MFU_NA/pod/coords_do${DO}.npz \
    --svd-path    results/MFU_NA/pod/svd_basis.npz \
    --out-dir     results/MFU_NA/cluster_sweep/do${DO} \
    --K-min 1 \
    --K-max 30 \
    --stride 2 \
    --energy 0.99 \
    --q 20 \
    --n-proj 64 \
    --dict-size 50 \
    --seed 0 \
    --w1-seed 12345 \
    --n-init 10 \
    --workers "$WORKERS" \
    --title-prefix "MFU_NA cluster-quality (d_o=${DO}, centred POD)"

echo ""
echo "=========================================="
echo "JOB COMPLETED"
echo "=========================================="
echo "End time: $(date)"
