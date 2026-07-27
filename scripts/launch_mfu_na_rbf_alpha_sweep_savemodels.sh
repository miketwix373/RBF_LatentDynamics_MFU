#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/RBF_ROM
#SBATCH -J mfu_alpha_savemodels
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=R-%x-%j.out
#
# Faithful re-run of the MFU_NA K=16 alpha_tangent sweeps WITH model saving on,
# so the fitted regressions (centres, Sigma, xi) are kept for downstream
# analysis. Same grids as the original launchers
# (launch_mfu_na_rbf_alpha_sweep.sh for do45, launch_mfu_na_rbf_alpha_sweep_do.sh
# for do75), non-ridge only. Fits are deterministic (seed=0, n_init=10) so
# sweep.npz reproduces the existing curves; --save-models adds model_n<NNNNN>.npz
# per n_rbf alongside it.
#
# One job per d_o so 45 and 75 run in parallel:
#   sbatch --export=ALL,DO=45 scripts/launch_mfu_na_rbf_alpha_sweep_savemodels.sh
#   sbatch --export=ALL,DO=75 scripts/launch_mfu_na_rbf_alpha_sweep_savemodels.sh
#
# Writes results/MFU_NA/do<DO>/alpha<A>/{sweep.npz, model_n*.npz, rbf_K16_n_sweep.png}

set -euo pipefail

DO="${DO:?set DO (45 or 75)}"

if command -v flight >/dev/null 2>&1; then flight env activate gridware; fi
__conda_setup="$('/mnt/scratch/users/sbrw610/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then eval "$__conda_setup"; fi
unset __conda_setup
conda activate /mnt/scratch/users/sbrw610/anaconda3/envs/cfd_new

export MKL_THREADING_LAYER=GNU
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
export LD_LIBRARY_PATH="/opt/apps/flight/env/conda+jupyter/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$(pwd):$(pwd)/scripts:${PYTHONPATH:-}"
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2

K=16
COORDS="results/MFU_NA/pod/coords_do${DO}.npz"

if [ "$DO" = "45" ]; then
    N_RBF_LIST="160 320 640 960 1280 1920 2560 3200 4800"
    ALPHA_VALUES=(1.5 2.5 3.5 5.0 8.0 12.0)
elif [ "$DO" = "75" ]; then
    N_RBF_LIST="160 320 640 960 1280 1920 2560 3200 4800 6400"
    ALPHA_VALUES=(5.0 8.0 12.0 16.0 20.0 24.0 28.0 32.0)
else
    echo "unsupported DO=$DO (expected 45 or 75)"; exit 1
fi

echo "Start: $(date)  | d_o=${DO}  alpha=${ALPHA_VALUES[*]}  n_rbf=${N_RBF_LIST}"
pids=()
for A in "${ALPHA_VALUES[@]}"; do
    OUT="results/MFU_NA/do${DO}/alpha${A}"
    mkdir -p "$OUT"
    python -u scripts/run_rbf_K_n_sweep.py \
        --coords-path "$COORDS" --out-dir "$OUT" \
        --K "$K" --n-rbf-list $N_RBF_LIST \
        --stride 2 --train-frac 0.8 --alpha-tangent "$A" \
        --energy-threshold 0.9999 --no-standardise \
        --normal-width-floor 0.001 --save-models \
        --title "MFU_NA d_o=${DO} K=${K} alpha_tangent=${A} (save-models)" \
        > "${OUT}/sweep.log" 2>&1 &
    pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done

echo "End: $(date)  | fail=${fail}"
exit "$fail"
