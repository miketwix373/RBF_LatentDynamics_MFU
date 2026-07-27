#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/RBF_ROM
#SBATCH -J mfu_enkf_qsweep
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=R-%x-%j.out
#
# Additive-inflation (q) sweep for q* calibration: ONE node, one background
# worker per q value (each an independent EnKF run), the members within each
# run split over that worker's core slice. q* is the q where chi2/N_obs = 1.
#
#   sbatch scripts/launch_mfu_na_enkf_qsweep.sh
#
# Env knobs: N_ENS (100), INTERVAL_S (10), CADENCE (5), STRIDE (8).

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
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

QS=(0.0 0.05 0.1 0.15 0.2 0.3)
NW=$(( ${SLURM_CPUS_PER_TASK:-48} / ${#QS[@]} ))   # cores per worker
[ "$NW" -lt 1 ] && NW=1
echo "Fanning out ${#QS[@]} q-workers, $NW cores each"

LOGDIR="results/MFU_NA/EnKF/qsweep/logs"
mkdir -p "$LOGDIR"

for q in "${QS[@]}"; do
    echo "  worker q=$q  ->  qsweep/q$q  (n-workers=$NW)"
    python -u scripts/run_mfu_na_enkf.py \
        --assimilate --n-ens "${N_ENS:-100}" \
        --interval-s "${INTERVAL_S:-10.0}" --cadence-tplus "${CADENCE:-5}" \
        --sensor-stride "${STRIDE:-8}" --components both \
        --inflation 1.0 --additive-q "$q" --corr-k-scale 0 \
        --n-workers "$NW" \
        --out-dir "results/MFU_NA/EnKF/qsweep/q$q" \
        > "$LOGDIR/q${q}.log" 2>&1 &
done

wait
echo "All q-workers finished."
echo "End time: $(date)"
