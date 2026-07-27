#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/RBF_ROM
#SBATCH -J mfu_do75_lowalpha
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=R-%x-%j.out
#
# Extend the MFU_NA d_o=75 alpha_tangent sweep to the LOW-alpha regime, to
# locate the VPT optimum below alpha=5 (the do45 curve peaks at 3.5; the do75
# curve is still climbing at its lowest fitted alpha=5). Same fit config and
# N-grid as launch_mfu_na_rbf_alpha_sweep_savemodels.sh DO=75; only the four new
# alpha values are fitted so the existing eight are untouched.
#
#   sbatch scripts/launch_mfu_na_rbf_alpha_sweep_do75_lowalpha.sh
#
# Writes results/MFU_NA/do75/alpha<A>/{sweep.npz, model_n*.npz, rbf_K16_n_sweep.png}

set -euo pipefail

if command -v flight >/dev/null 2>&1; then flight env activate gridware; fi
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

DO=75
K=16
COORDS="results/MFU_NA/pod/coords_do${DO}.npz"
N_RBF_LIST="160 320 640 960 1280 1920 2560 3200 4800 6400"
ALPHA_VALUES=(1.5 2.5 3.5 4.0)

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
        --title "MFU_NA d_o=${DO} K=${K} alpha_tangent=${A} (low-alpha extend)" \
        > "${OUT}/sweep.log" 2>&1 &
    pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done

echo "End: $(date)  | fail=${fail}"
exit "$fail"
