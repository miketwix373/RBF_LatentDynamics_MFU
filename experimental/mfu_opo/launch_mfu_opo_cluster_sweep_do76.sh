#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/RBF_ROM
#SBATCH -J mfu_opo_cl76
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=R-%x-%j.out
#
# Cluster-quality K-sweep (K=1..36) on the MFU_OPO centred-POD reduced state
# at d_o=76 (90% fluctuation energy). Self-contained: precomputes
# coords_do76.npz from the frozen svd_basis.npz if absent, then runs the
# sweep. Seven metrics, one K per worker, NO RBF/SINDy fits.
#
# Reuses run_ks_bur_arc_cluster_sweep.py (reads (alpha, alpha_dot)
# empirically; never invokes an analytical RHS). Coords come from the 80%
# train split (64.4k frames); stride 2 -> 32.2k samples (spacing 0.1,
# ~322 independent samples at tau_int~100), inside the kNN metric's range.
#
# Usage
#     sbatch scripts/launch_mfu_opo_cluster_sweep_do76.sh

set -euo pipefail

DO=76
STATS=/users/sbrw610/sharedscratch/RBF_ROM/data/MFU_OPO/stats.npz
POD=results/MFU_OPO/pod
COORDS=${POD}/coords_do${DO}.npz

echo "MFU_OPO cluster sweep d_o=${DO} -- $(date) -- ${SLURM_JOB_NODELIST:-(local)}"

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

if [ ! -f "${COORDS}" ]; then
    echo "[coords] building ${COORDS}"
    python -u scripts/precompute_coords.py \
        --stats-path "${STATS}" \
        --svd-path   "${POD}/svd_basis.npz" \
        --d-o        "${DO}" \
        --out        "${COORDS}"
fi

python -u scripts/run_ks_bur_arc_cluster_sweep.py \
    --coords-path "${COORDS}" \
    --svd-path    "${POD}/svd_basis.npz" \
    --out-dir     results/MFU_OPO/cluster_sweep/do${DO} \
    --K-min 1 \
    --K-max 36 \
    --stride 2 \
    --energy 0.99 \
    --q 20 \
    --n-proj 64 \
    --dict-size 50 \
    --seed 0 \
    --w1-seed 12345 \
    --n-init 10 \
    --workers "$WORKERS" \
    --title-prefix "MFU_OPO cluster-quality (d_o=${DO}, centred POD)"

echo "End time: $(date)"
