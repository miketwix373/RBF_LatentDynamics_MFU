#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/RBF_ROM
#SBATCH -J mfu_enkf_sweep_a8
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=R-%x-%j.out
#
# EnKF parametric sweep for the alpha_tangent=8.0 MFU_NA model. One sbatch job
# per GROUP, one background worker per run within the node (the qsweep
# pattern). Groups:
#   qstar        -- additive-q grid for the alpha8 q* (chi2/N_obs = 1 crossing)
#   cadence      -- q*(cadence) autonomy-horizon curve, mini q-grid per cadence
#   cadence_twin -- perfect-model twin at each cadence (--twin): the DA q* floor
#   sigma        -- q* robustness vs obs-noise level
#   nens         -- q* robustness vs ensemble size
#   stride_q     -- q* robustness vs sensor density
#   obs          -- observability map: sensor stride x components at the
#                   production DA point (infl 1.02, q 0.1)
#   resolved     -- one long 2000 t+ run for mode- and cluster-resolved q*
# Robustness groups keep the alpha3.5 qsweep conventions (inflation 1.0,
# corr-k-scale 0, seed 0) so crossings are comparable. Cadence groups scale
# --interval-s to hold ~25 analysis cycles per run. All runs store the full
# ensemble trajectory.
#
#   for g in qstar cadence cadence_twin sigma nens stride_q obs resolved; do
#       sbatch --export=ALL,GROUP=$g scripts/launch_mfu_na_enkf_sweep_a8.sh
#   done

set -euo pipefail

GROUP="${GROUP:?set GROUP=qstar|cadence|cadence_twin|sigma|nens|stride_q|obs|resolved|resolved_full}"
ALPHA="8.0"

echo "Start time: $(date)"
echo "Job ID:     ${SLURM_JOB_ID:-(local)}"
echo "Node:       ${SLURM_JOB_NODELIST:-(local)}"
echo "group=$GROUP alpha_tangent=$ALPHA"

if command -v flight >/dev/null 2>&1; then flight env activate gridware; fi
__conda_setup="$('/mnt/scratch/users/sbrw610/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then eval "$__conda_setup"; fi
unset __conda_setup
conda activate /mnt/scratch/users/sbrw610/anaconda3/envs/cfd_new

export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
export LD_LIBRARY_PATH="/opt/apps/flight/env/conda+jupyter/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$(pwd):$(pwd)/scripts:${PYTHONPATH:-}"
export MKL_THREADING_LAYER=GNU
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

CALIB="results/MFU_NA/do45/alpha${ALPHA}/corrector_calib.npz"
MODEL=$(python -c "import numpy as np; \
print(str(np.load('$CALIB', allow_pickle=True)['model_path']))")
echo "model: $MODEL"

OUT_ROOT="results/MFU_NA/EnKF/sweep_a8/$GROUP"
LOGDIR="$OUT_ROOT/logs"
mkdir -p "$LOGDIR"

# Baseline operating point (the alpha3.5 qsweep conventions, alpha8 model);
# per-group extra flags override via argparse last-one-wins.
BASE="--assimilate --keep-traj --n-ens 100 --interval-s 10.0 \
--cadence-tplus 5 --sensor-stride 8 --components both --sigma-obs 0.1 \
--inflation 1.0 --additive-q 0.1 --corr-k-scale 0 --seed 0"

# interval holding ~25 analysis cycles at cadence $1 (t+), floor 10 s
civ () { python -c "print(max(10.0, round(25*$1/11.289, 2)))"; }

RUNS=()   # entries: "tag|extra flags"
QGRID=(0.0 0.1 0.2 0.4)
case "$GROUP" in
qstar)
    for q in 0.0 0.05 0.1 0.15 0.2 0.3 0.4; do
        RUNS+=("q$q|--additive-q $q")
    done ;;
cadence)
    for cad in 2.5 7.5 10 15 20 30; do          # 5 t+ covered by qstar
        iv=$(civ "$cad")
        for q in "${QGRID[@]}"; do
            RUNS+=("cad${cad}_q$q|--cadence-tplus $cad --interval-s $iv --additive-q $q")
        done
    done ;;
cadence_twin)
    for cad in 2.5 5 7.5 10 15 20 30; do
        iv=$(civ "$cad")
        for q in 0.0 0.1 0.2; do
            RUNS+=("cad${cad}_q$q|--twin --cadence-tplus $cad --interval-s $iv --additive-q $q")
        done
    done ;;
sigma)
    for sg in 0.05 0.2 0.4; do
        for q in "${QGRID[@]}"; do
            RUNS+=("sig${sg}_q$q|--sigma-obs $sg --additive-q $q")
        done
    done ;;
nens)
    for ne in 25 50 200; do
        for q in "${QGRID[@]}"; do
            RUNS+=("ne${ne}_q$q|--n-ens $ne --additive-q $q")
        done
    done ;;
stride_q)
    for st in 4 16 32; do
        for q in "${QGRID[@]}"; do
            RUNS+=("st${st}_q$q|--sensor-stride $st --additive-q $q")
        done
    done ;;
obs)
    for st in 2 4 8 16 32; do
        for comp in x y both; do
            RUNS+=("st${st}_${comp}|--sensor-stride $st --components $comp --inflation 1.02")
        done
    done ;;
resolved)
    RUNS+=("long2000|--interval-s 177.2")       # ~2000 t+, ~400 cycles
    ;;
resolved_full)
    # full test window (7960 of 7966 strided steps): ~8990 t+, ~2000 cycles;
    # the truth visits all 16 K-means clusters (all-16 coverage needs ~5400)
    RUNS+=("long9000|--interval-s 796.0")
    ;;
*)  echo "unknown GROUP=$GROUP"; exit 1 ;;
esac

NW=$(( ${SLURM_CPUS_PER_TASK:-48} / ${#RUNS[@]} ))
[ "$NW" -lt 2 ] && NW=2
echo "Fanning out ${#RUNS[@]} workers, $NW cores each"

for entry in "${RUNS[@]}"; do
    tag="${entry%%|*}"
    extra="${entry#*|}"
    echo "  worker $tag  ->  $OUT_ROOT/$tag"
    # shellcheck disable=SC2086
    python -u scripts/run_mfu_na_enkf.py \
        --model "$MODEL" --calib "$CALIB" \
        $BASE $extra \
        --n-workers "$NW" \
        --out-dir "$OUT_ROOT/$tag" \
        > "$LOGDIR/$tag.log" 2>&1 &
done

wait
echo "All $GROUP workers finished."
echo "End time: $(date)"
