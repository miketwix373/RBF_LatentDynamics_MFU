#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/RBF_ROM
#SBATCH -J mfu_ridge_probe
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=R-%x-%j.out
#
# Ridge-vs-lstsq probe on the MFU_NA K=16 RBF fit. The plain-lstsq sweep
# showed the ORIGINAL modes 0-44 degrading (median test NRMSE 0.51 -> 0.73)
# when the state is widened d_o=45 -> 75, at cond(Phi) ~ 1e3-3e5. rom-specialist
# (2026-07-08) flagged this as an unregularised-variance confound (R4/R5) and
# said to control the variance arm with --ridge press before reading any floor.
#
# This re-fits d_o=45 and d_o=75 at matched n_rbf with --ridge press (PRESS-
# selected Tikhonov). If modes 0-44 at d_o=75 snap back toward the d_o=45
# value, the "floor worsens with d_o" is a regularisation artefact; if not, the
# gap is intrinsic (closure) and the kNN / truncated-residual tests follow.
#
# Writes: results/MFU_NA/rbf_K16_ridge_probe_do<DO>/n_<value>.npz
#
# Usage: sbatch scripts/launch_mfu_na_rbf_ridge_probe.sh

set -euo pipefail

if command -v flight >/dev/null 2>&1; then
    flight env activate gridware
fi
__conda_setup="$('/mnt/scratch/users/sbrw610/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then eval "$__conda_setup"; fi
unset __conda_setup
conda activate /mnt/scratch/users/sbrw610/anaconda3/envs/cfd_new

export MKL_THREADING_LAYER=GNU
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
export LD_LIBRARY_PATH="/opt/apps/flight/env/conda+jupyter/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$(pwd):$(pwd)/scripts:${PYTHONPATH:-}"
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4

run_do () {
    local DO="$1"; shift
    local OUT="results/MFU_NA/rbf_K16_ridge_probe_do${DO}"
    mkdir -p "$OUT"
    for N in "$@"; do
        python -u scripts/run_rbf_K_n_sweep.py \
            --coords-path "results/MFU_NA/pod/coords_do${DO}.npz" \
            --out-dir     "$OUT" \
            --K 16 --n-rbf-list "$N" \
            --stride 2 --train-frac 0.8 --alpha-tangent 5.0 \
            --energy-threshold 0.9999 --no-standardise \
            --normal-width-floor 0.001 --ridge press \
            --no-save-models \
            --title "MFU_NA d_o=${DO} K=16 RIDGE probe" \
            > "$OUT/n_$(printf '%05d' "$N").log" 2>&1
    done
}

echo "Start: $(date)"
# matched n_rbf spanning each curve's near-optimum region
run_do 45 1920 2560 3200 &
run_do 75 1920 3200 4800 &
wait
echo "End: $(date)"
