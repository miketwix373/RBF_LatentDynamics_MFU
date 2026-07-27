#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/RBF_ROM
#SBATCH -J mfu_upo_hunt
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=90G
#SBATCH --time=04:00:00
#SBATCH --output=R-%x-%j.out
#
# UPO hunt on the MFU_NA wall-shear RBF field (find_mfu_na_upo.py). One sbatch
# job per GROUP, one background worker per run (the EnKF-sweep pattern):
#   refine -- refine/robustness battery around the near-converged T=107.8 t+
#             orbit (upo/SSP_seed2380): dt- and M-refinement + re-convergence
#             on other model fits (alpha_tangent and n_rbf perturbations)
#   seeds  -- remaining recurrence seeds: SSP-window (~95-125 t+), the ~40 t+
#             tier, and a double-period candidate
#
#   for g in refine seeds; do
#       sbatch --export=ALL,GROUP=$g scripts/launch_mfu_na_upo_hunt.sh
#   done

set -euo pipefail
GROUP="${GROUP:?set GROUP=refine|seeds|trav1|trav2}"

echo "Start time: $(date)"
echo "Job ID:     ${SLURM_JOB_ID:-(local)}"
echo "group=$GROUP"

if command -v flight >/dev/null 2>&1; then flight env activate gridware; fi
__conda_setup="$('/mnt/scratch/users/sbrw610/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then eval "$__conda_setup"; fi
unset __conda_setup
conda activate /mnt/scratch/users/sbrw610/anaconda3/envs/cfd_new

export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
export LD_LIBRARY_PATH="/opt/apps/flight/env/conda+jupyter/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$(pwd):$(pwd)/scripts:${PYTHONPATH:-}"
export MKL_THREADING_LAYER=GNU

MFU=/users/sbrw610/sharedscratch/RBF_ROM/results/MFU_NA
BEST="$MFU/upo/SSP_seed2380/upo.npz"
A8=$MFU/do45/alpha8.0
A35=$MFU/do45/alpha3.5
LOGDIR="$MFU/upo/logs"
mkdir -p "$LOGDIR"

RUNS=()   # entries: "tag|args"
case "$GROUP" in
refine)
    RUNS+=("refine_M8_dt0125|--restart-from $BEST --segments 8 --dt 0.0125 --fix-T-iters 5")
    RUNS+=("refine_M16_dt025|--restart-from $BEST --segments 16 --dt 0.025 --fix-T-iters 5")
    RUNS+=("refine_M16_dt0125|--restart-from $BEST --segments 16 --dt 0.0125 --fix-T-iters 5")
    RUNS+=("refine_M4_dt025|--restart-from $BEST --segments 4 --dt 0.025 --fix-T-iters 5")
    RUNS+=("rob_a8_n1920|--restart-from $BEST --model $A8/model_n01920.npz --segments 8 --fix-T-iters 10")
    RUNS+=("rob_a8_n4800|--restart-from $BEST --model $A8/model_n04800.npz --segments 8 --fix-T-iters 10")
    RUNS+=("rob_a35_n4800|--restart-from $BEST --model $A35/model_n04800.npz --segments 8 --fix-T-iters 10")
    RUNS+=("rob_a35_n1920|--restart-from $BEST --model $A35/model_n01920.npz --segments 8 --fix-T-iters 10")
    ;;
trav1)
    # traversal-filtered seeds 1-8: near-recurrent AND sweeping both the
    # high-sigma_y (burst) and low-sigma_y (quiescent) cluster groups
    RUNS+=("trav_t1207|--seed-t 1207.4 --seed-T 12.0 --segments 24 --fix-T-iters 50")
    RUNS+=("trav_t2707|--seed-t 2707.2 --seed-T 9.5 --segments 24 --fix-T-iters 50")
    RUNS+=("trav_t1763|--seed-t 1763.8 --seed-T 10.4 --segments 24 --fix-T-iters 50")
    RUNS+=("trav_t1026|--seed-t 1026.5 --seed-T 11.8 --segments 24 --fix-T-iters 50")
    RUNS+=("trav_t2638|--seed-t 2638.3 --seed-T 11.4 --segments 24 --fix-T-iters 50")
    RUNS+=("trav_t2715|--seed-t 2715.9 --seed-T 8.4 --segments 24 --fix-T-iters 50")
    RUNS+=("trav_t0953|--seed-t 953.4 --seed-T 9.3 --segments 24 --fix-T-iters 50")
    RUNS+=("trav_t2834|--seed-t 2834.5 --seed-T 9.8 --segments 24 --fix-T-iters 50")
    ;;
trav2)
    # traversal-filtered seeds 9-16
    RUNS+=("trav_t2946|--seed-t 2946.9 --seed-T 11.5 --segments 24 --fix-T-iters 50")
    RUNS+=("trav_t1313|--seed-t 1313.2 --seed-T 11.4 --segments 24 --fix-T-iters 50")
    RUNS+=("trav_t3084|--seed-t 3084.9 --seed-T 11.6 --segments 24 --fix-T-iters 50")
    RUNS+=("trav_t0842|--seed-t 842.1 --seed-T 12.4 --segments 24 --fix-T-iters 50")
    RUNS+=("trav_t2826|--seed-t 2826.7 --seed-T 10.0 --segments 24 --fix-T-iters 50")
    RUNS+=("trav_t0109|--seed-t 109.2 --seed-T 12.4 --segments 24 --fix-T-iters 50")
    RUNS+=("trav_t0316|--seed-t 316.7 --seed-T 12.4 --segments 24 --fix-T-iters 50")
    RUNS+=("trav_t1201|--seed-t 1201.3 --seed-T 8.4 --segments 24 --fix-T-iters 50")
    ;;
seeds)
    RUNS+=("SSP_seed3669|--seed-t 3669.0 --seed-T 10.5 --segments 8")
    RUNS+=("SSP_seed2492|--seed-t 2492.6 --seed-T 10.9 --segments 8")
    RUNS+=("SSP_seed2318|--seed-t 2318.7 --seed-T 10.5 --segments 8")
    RUNS+=("SSP_seed2314|--seed-t 2314.5 --seed-T 9.8 --segments 8")
    RUNS+=("SSP_seed2376|--seed-t 2376.3 --seed-T 8.4 --segments 8")
    RUNS+=("T40_seed2324|--seed-t 2324.9 --seed-T 3.6 --segments 6")
    RUNS+=("T40_seed2500|--seed-t 2500.0 --seed-T 3.4 --segments 6")
    RUNS+=("T40_seed2158|--seed-t 2158.3 --seed-T 3.9 --segments 6")
    RUNS+=("SSP2_seed2380|--seed-t 2380.3 --seed-T 16.8 --segments 16")
    ;;
*)  echo "unknown GROUP=$GROUP"; exit 1 ;;
esac

NT=$(( ${SLURM_CPUS_PER_TASK:-48} / ${#RUNS[@]} ))
[ "$NT" -lt 1 ] && NT=1
export MKL_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT OMP_NUM_THREADS=$NT
echo "Fanning out ${#RUNS[@]} workers, $NT BLAS threads each"

for entry in "${RUNS[@]}"; do
    tag="${entry%%|*}"
    extra="${entry#*|}"
    echo "  worker $tag"
    # shellcheck disable=SC2086
    python -u scripts/find_mfu_na_upo.py \
        $extra --max-iter 120 --tag "$tag" \
        > "$LOGDIR/$tag.log" 2>&1 &
done

wait
echo "All $GROUP workers finished."
echo "End time: $(date)"
