#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/RBF_ROM
#SBATCH -J mfu_opo_analyze
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=R-%x-%j.out
#
# First-look analysis of the MFU_OPO (minimal flow unit, opposition control)
# wall-shear-stress dataset: cadence, moments, PDFs, spectra, POD energy.
# Reuses scripts/analyze_mfu_na_wss.py; OPO has a single uniform cadence
# (dstep=10, dt_sim=0.005 => dt_snap=0.05), so segment_B is empty.
#
# Usage
#     sbatch scripts/launch_analyze_mfu_opo.sh

set -euo pipefail

echo "================================================="
echo "MFU_OPO first-look analysis"
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

THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export OMP_NUM_THREADS="$THREADS"

python -u scripts/analyze_mfu_na_wss.py \
    --data-dir /users/sbrw610/sharedscratch/RBF_ROM/data/MFU_OPO \
    --out-dir  results/MFU_OPO/pre_analysis \
    --dt 0.005 \
    --label "MFU\\_OPO"

echo ""
echo "End time: $(date)"
