#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/RBF_ROM
#SBATCH -J mfu_corr_val
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=R-%x-%A_%a.out
#SBATCH --array=0-13
#
# Validate the per-model soft-corrector calibration for the 14 selected MFU_NA
# (d_o, alpha, N*) models. One array task per manifest row: roll the calibrated
# ROM across the full test horizon at the STORED trust radius R (no
# recomputation drift) and record per-step corrector behaviour. Writes
# corrector_series.npz next to each selected model.
#
#   sbatch scripts/launch_mfu_na_corrector_validate.sh
#
# Reads results/MFU_NA/model_selection/corrector_calibration.json; the array
# range 0-13 must match its length (14 entries).

set -euo pipefail

echo "Start time: $(date)"
echo "Job ID:     ${SLURM_JOB_ID:-(local)}  Array task: ${SLURM_ARRAY_TASK_ID:-0}"
echo "Node:       ${SLURM_JOB_NODELIST:-(local)}"

if command -v flight >/dev/null 2>&1; then flight env activate gridware; fi
__conda_setup="$('/mnt/scratch/users/sbrw610/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then eval "$__conda_setup"; fi
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

MANIFEST="results/MFU_NA/model_selection/corrector_calibration.json"
IDX="${SLURM_ARRAY_TASK_ID:-0}"

read MODEL COORDS RESC SA SK SP OUT < <(python - "$MANIFEST" "$IDX" <<'PY'
import json, sys
recs = json.load(open(sys.argv[1]))
r = recs[int(sys.argv[2])]
from pathlib import Path
out = str(Path(r["model_path"]).parent / "corrector_series.npz")
print(r["model_path"], r["coords_path"], r["escape_thresh"],
      r["soft_alpha"], r["soft_k"], r["soft_p"], out)
PY
)

echo "task $IDX: MODEL=$MODEL"
echo "  COORDS=$COORDS  R=$RESC  soft=($SA,$SK,$SP)"
echo "  OUT=$OUT"

python -u scripts/run_kschao_corrector_series.py \
    --model "$MODEL" \
    --coords-path "$COORDS" \
    --escape-thresh "$RESC" \
    --soft-alpha "$SA" --soft-k "$SK" --soft-p "$SP" \
    --progress-every 2000 \
    --out "$OUT"

echo "End time: $(date)"
