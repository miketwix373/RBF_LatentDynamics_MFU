#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/RBF_ROM
#SBATCH -J mfu_opo_stats
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=R-%x-%j.out
#
# Package the MFU_OPO WSS pair into the CHORD2 stats.npz schema. Single
# uniform cadence (dstep=10, dt_sim=0.005 => dt_snap=0.05); 80/20 time split
# into stats.npz (train) and stats_holdout.npz (invariant-measure holdout).
#
# Usage
#     sbatch scripts/launch_mfu_opo_stats.sh

set -euo pipefail

echo "MFU_OPO stats build -- $(date) -- ${SLURM_JOB_NODELIST:-(local)}"

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

python -u scripts/build_mfu_opo_stats.py \
    --data-dir /users/sbrw610/sharedscratch/RBF_ROM/data/MFU_OPO

echo "End time: $(date)"
