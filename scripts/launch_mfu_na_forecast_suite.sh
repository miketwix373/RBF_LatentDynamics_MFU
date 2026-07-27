#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/RBF_ROM
#SBATCH -J mfu_forecast
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=16:00:00
#SBATCH --output=R-%x-%j.out
#
# Forecast suite for the calibrated MFU_NA models: ONE node, one background
# worker per model (<=48 workers/node), each running 100 short forecasts +
# one long test with its own calibrated corrector. Reads the calibration
# manifest and fans out run_mfu_na_forecast_suite.py per row.
#
#   sbatch scripts/launch_mfu_na_forecast_suite.sh
#
# Env knobs: N_ICS (default 100), HORIZON_S (default 30). Per-worker threads
# scale with N via threads_for_n().

set -euo pipefail

echo "Start time: $(date)"
echo "Job ID:     ${SLURM_JOB_ID:-(local)}"
echo "Node:       ${SLURM_JOB_NODELIST:-(local)}"

if command -v flight >/dev/null 2>&1; then flight env activate gridware; fi
__conda_setup="$('/mnt/scratch/users/sbrw610/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then eval "$__conda_setup"; fi
unset __conda_setup
conda activate /mnt/scratch/users/sbrw610/anaconda3/envs/cfd_new

export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
export LD_LIBRARY_PATH="/opt/apps/flight/env/conda+jupyter/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$(pwd):$(pwd)/scripts:${PYTHONPATH:-}"
export MKL_THREADING_LAYER=GNU

# Per-worker thread budget scales with N (RBF centres): the big hosts dominate
# wall time (rate ~ 1/N), so they get more threads. Cost is a brief early
# oversubscription (sum > 48) that resolves as the cheap small-N workers finish.
threads_for_n() {
    local n="$1"
    if   [ "$n" -ge 6000 ]; then echo 16
    elif [ "$n" -ge 2500 ]; then echo 8
    elif [ "$n" -ge 1000 ]; then echo 4
    else echo 2; fi
}

MANIFEST="results/MFU_NA/model_selection/corrector_calibration.json"
N=$(python -c "import json;print(len(json.load(open('$MANIFEST'))))")

# IDX_LIST (space/comma-separated manifest row indices) restricts the fan-out to
# a subset -- e.g. to forecast only newly-added models without recomputing the
# rest. Unset => all rows.
if [ -n "${IDX_LIST:-}" ]; then
    IFS=', ' read -r -a INDICES <<< "$IDX_LIST"
else
    INDICES=($(seq 0 $((N - 1))))
fi
echo "Fanning out ${#INDICES[@]} workers (of $N manifest rows), threads scaled by N"

LOGDIR="results/MFU_NA/model_selection/forecast_logs"
mkdir -p "$LOGDIR"

for i in "${INDICES[@]}"; do
    read MODEL COORDS RESC SA SK SP NRBF DO ALP < <(python - "$MANIFEST" "$i" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))[int(sys.argv[2])]
print(r["model_path"], r["coords_path"], r["escape_thresh"],
      r["soft_alpha"], r["soft_k"], r["soft_p"],
      r["best_n_rbf"], r["d_o"], r["alpha_tangent"])
PY
)
    tag="do${DO}_a${ALP}"
    THIS_THREADS=$(threads_for_n "$NRBF")
    echo "  worker $i: $tag  N=$NRBF  threads=$THIS_THREADS  R=$RESC"
    MKL_NUM_THREADS="$THIS_THREADS" OPENBLAS_NUM_THREADS="$THIS_THREADS" \
    OMP_NUM_THREADS="$THIS_THREADS" \
    python -u scripts/run_mfu_na_forecast_suite.py \
        --model "$MODEL" --coords-path "$COORDS" \
        --escape-thresh "$RESC" \
        --soft-alpha "$SA" --soft-k "$SK" --soft-p "$SP" \
        --n-ics "${N_ICS:-100}" --horizon-s "${HORIZON_S:-30}" \
        > "$LOGDIR/worker_${i}_${tag}.log" 2>&1 &
done

wait
echo "All workers finished."
echo "End time: $(date)"
