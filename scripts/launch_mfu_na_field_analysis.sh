#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/RBF_ROM
#SBATCH -J mfu_field_analysis
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --output=R-%x-%j.out
#
# Heavy legs of the MFU_NA field-analysis menu (items 3 and 5C), one GROUP per
# job:
#   ftg  -- finite-time transient-growth map, N=3000 states x tau+ = 5, 10
#   lyap -- full Lyapunov spectrum of the alpha8/n960 field (Benettin QR)
#
#   for g in ftg lyap; do
#       sbatch --export=ALL,GROUP=$g scripts/launch_mfu_na_field_analysis.sh
#   done

set -euo pipefail
GROUP="${GROUP:?set GROUP=ftg|lyap|siblings}"

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
export MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16

MFU=/users/sbrw610/sharedscratch/RBF_ROM/results/MFU_NA
MODEL=$MFU/do45/alpha8.0/model_n00960.npz

case "$GROUP" in
ftg)
    python -u scripts/analyze_mfu_na_finite_time_growth.py \
        --model "$MODEL" --n-states 3000
    ;;
lyap)
    python -u scripts/run_mfu_na_lyapunov.py \
        --model "$MODEL" \
        --coords-path "$MFU/pod/coords_do45.npz" \
        --out-dir "$MFU/field_analysis/interpretables/lyapunov" \
        --base data --t-settle 50 --t-trans 100 --t-total 600
    ;;
siblings)
    # consultant-mandated persistence battery (2026-07-20 audit): rerun the
    # field diagnostics on sibling fits so fitted-field properties can be
    # separated from regression furniture; plus the data-cloud intrinsic dim.
    SIBS="a8_n1920:do45/alpha8.0/model_n01920.npz \
a8_n4800:do45/alpha8.0/model_n04800.npz \
a35_n4800:do45/alpha3.5/model_n04800.npz"
    ROOT="$MFU/field_analysis/siblings"
    LOGDIR="$ROOT/logs"
    mkdir -p "$LOGDIR"
    export MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 OMP_NUM_THREADS=3
    for sib in $SIBS; do
        tag="${sib%%:*}"
        mdl="$MFU/${sib#*:}"
        python -u scripts/find_mfu_na_fixed_points.py \
            --model "$mdl" --out-dir "$ROOT/$tag/fixed_points" \
            > "$LOGDIR/${tag}_fixed_points.log" 2>&1 &
        python -u scripts/analyze_mfu_na_trace_contraction.py \
            --model "$mdl" --out-dir "$ROOT/$tag/trace" \
            > "$LOGDIR/${tag}_trace.log" 2>&1 &
        python -u scripts/analyze_mfu_na_liftup_matrix.py \
            --model "$mdl" --out "$ROOT/$tag/liftup" \
            > "$LOGDIR/${tag}_liftup.log" 2>&1 &
        python -u scripts/run_mfu_na_lyapunov.py \
            --model "$mdl" --coords-path "$MFU/pod/coords_do45.npz" \
            --out-dir "$ROOT/$tag/lyapunov" \
            --base data --t-settle 50 --t-trans 100 --t-total 600 \
            > "$LOGDIR/${tag}_lyapunov.log" 2>&1 &
    done
    python -u scripts/estimate_intrinsic_dim.py \
        --coords-path "$MFU/pod/coords_do45.npz" \
        > "$LOGDIR/intrinsic_dim.log" 2>&1 &
    wait
    echo "All sibling workers finished."
    ;;
*)  echo "unknown GROUP=$GROUP"; exit 1 ;;
esac

echo "End time: $(date)"
