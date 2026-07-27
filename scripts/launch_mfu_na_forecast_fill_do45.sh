#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/RBF_ROM
#SBATCH -J mfu_forecast_fill
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=16:00:00
#SBATCH --output=R-%x-%j.out
#
# Fill-in forecast suite for the do45 alphas fitted after the original suite
# ran (4.0, 6.0, 7.0, 10.0, 16.0, 20.0): they have sweep.npz + saved models
# but no calibration row, so no forecast_ics.npz / long_test.npz and no VPT
# record. ONE node, four stages in one job:
#   1. rebuild the N* selection manifest (select_mfu_na_best_nrbf.py; its
#      PANELS already list the new alphas)
#   2. rebuild the corrector calibration manifest
#      (calibrate_mfu_na_correctors.py; seeded, existing rows recompute
#      identically)
#   3. fan out run_mfu_na_forecast_suite.py workers ONLY for manifest rows
#      whose model dir is missing long_test.npz or forecast_ics.npz, computed
#      dynamically from the fresh manifest -- no hardcoded indices
#   4. regenerate vpt_vs_alpha.{png,json} and the VPT/climate-vs-NRMSE figure
#
#   sbatch scripts/launch_mfu_na_forecast_fill_do45.sh
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

echo "=== stage 1: rebuild N* selection manifest ==="
python -u scripts/select_mfu_na_best_nrbf.py

echo "=== stage 2: rebuild corrector calibration manifest ==="
python -u scripts/calibrate_mfu_na_correctors.py

MANIFEST="results/MFU_NA/model_selection/corrector_calibration.json"

# Rows still missing forecast products -- the fill-in set.
INDICES=($(python - "$MANIFEST" <<'PY'
import json, sys
from pathlib import Path
for i, r in enumerate(json.load(open(sys.argv[1]))):
    d = Path(r["model_path"]).parent
    if not ((d / "long_test.npz").exists() and (d / "forecast_ics.npz").exists()):
        print(i)
PY
))
echo "Fanning out ${#INDICES[@]} fill-in workers (rows: ${INDICES[*]:-none}), threads scaled by N"

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

echo "=== stage 4: regenerate VPT json + figures ==="
python -u scripts/plot_mfu_na_vpt_vs_alpha.py \
    --manifest "$MANIFEST" \
    --out-path results/MFU_NA/paper_images/vpt_vs_alpha.png
python -u scripts/plot_mfu_na_vpt_climate_vs_nrmse.py

echo "End time: $(date)"
