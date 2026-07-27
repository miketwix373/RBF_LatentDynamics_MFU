#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/RBF_ROM
#SBATCH -J mfu_na_fig4regen
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=R-%x-%j.out
#
# Regenerate Fig. 4 (phase cycle, split plotter) so the axis label reads k_y
# (the images/ copy was stale at k_z), copy it into the paper images/ dir, and
# print the representative-cluster (A_x, sigma_y) coordinates to check the Q3
# quadrant placement.
#
# Usage
#     sbatch scripts/launch_regen_fig4.sh

set -euo pipefail

echo "Start: $(date)  Node: ${SLURM_JOB_NODELIST:-(local)}"

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

python -u scripts/plot_mfu_na_phase_cycle_split.py
cp -v results/MFU_NA/paper_images/phase_cycle_split_K16.pdf \
      /users/sbrw610/sharedscratch/RBF_ROM/sn-article-template/images/fig4_phase_cycle.pdf
python -u scripts/check_rep_quadrants.py

echo "End: $(date)"
