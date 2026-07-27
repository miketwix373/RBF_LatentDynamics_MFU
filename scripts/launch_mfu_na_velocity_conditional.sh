#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/RBF_ROM
#SBATCH -J mfu_velcond
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=R-%x-%j.out
#
# Conditional average of the MFU_NA x-z velocity slices on K=16 WSS cluster
# label (do45). Single chunked pass over ~40k labelled velocity frames
# (~6 GB read), writes per-cluster raw sums for downstream conditional-field
# and Reynolds-stress derivation. See run_mfu_na_velocity_conditional.py.
#
# Usage: sbatch scripts/launch_mfu_na_velocity_conditional.sh

set -euo pipefail

echo "Start: $(date)  Job: ${SLURM_JOB_ID:-(local)}  Node: ${SLURM_JOB_NODELIST:-?}"

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
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

python -u scripts/run_mfu_na_velocity_conditional.py

echo "End: $(date)"
