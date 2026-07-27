#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/RBF_ROM
#SBATCH -J mfu_lowalpha_highN
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --output=R-%x-%j.out
#
# The low-alpha_tangent NRMSE curves are still declining at the current N ceiling
# (best-N pegs the grid max), so the a-priori optimum may not be resolved. Extend
# the N-grid to 6000/8000/10000 for the low-alpha cases whose N* hit the ceiling,
# to see whether more RBFs lower the floor and move the optimal (alpha, N).
#
# Full grid re-run per case (fits are deterministic, seed=0, so existing N points
# and model files reproduce identically); sweep.npz is overwritten with the
# extended 12-point curve and model_n0{6000,8000,10000}.npz added.
#
#   sbatch scripts/launch_mfu_na_rbf_lowalpha_highN.sh
#
# Cases (do:alpha): 45:1.5 45:2.5 45:3.5  75:2.5 75:3.5 75:4.0 75:5.0

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
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export OMP_NUM_THREADS=6

K=16
N_RBF_LIST="160 320 640 960 1280 1920 2560 3200 4800 6000 8000 10000"
CASES=("45:1.5" "45:2.5" "45:3.5" "75:2.5" "75:3.5" "75:4.0" "75:5.0")

echo "Start: $(date)  | n_rbf=${N_RBF_LIST}  cases=${CASES[*]}"
pids=()
for C in "${CASES[@]}"; do
    DO="${C%%:*}"; A="${C##*:}"
    OUT="results/MFU_NA/do${DO}/alpha${A}"
    COORDS="results/MFU_NA/pod/coords_do${DO}.npz"
    mkdir -p "$OUT"
    python -u scripts/run_rbf_K_n_sweep.py \
        --coords-path "$COORDS" --out-dir "$OUT" \
        --K "$K" --n-rbf-list $N_RBF_LIST \
        --stride 2 --train-frac 0.8 --alpha-tangent "$A" \
        --energy-threshold 0.9999 --no-standardise \
        --normal-width-floor 0.001 --save-models \
        --title "MFU_NA d_o=${DO} K=${K} alpha_tangent=${A} (highN extend)" \
        > "${OUT}/sweep_highN.log" 2>&1 &
    pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done

echo "End: $(date)  | fail=${fail}"
exit "$fail"
