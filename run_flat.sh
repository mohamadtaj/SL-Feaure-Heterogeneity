#!/usr/bin/env bash

set -euo pipefail

DATASETS=(

    cdc_3class_balanced
    cdc_binary5050_stratified
    s_500_num_not_corr
    s_500_cat_not_corr
    s_500_mix_not_corr   
    glioma
    gallstone
    thyroid  
    diabetes  

)

OVERLAPS=(20 30 40 50 60 70 80 90)
NODES=(2 3 4 6)

METHODS=(

    model_imputation
    surrogate_split_agreement_pr_01
    informed_marginal_cat_simple
    probability_informed_cat_simple_fallback_fixed
    intersection  
    baseline_intersection
    treewise_filtering
    probability_weighted  

)
# ----------------------------------------------------------------------


MAX_JOBS=${MAX_JOBS:-32}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

supports_waitn=0
if (( BASH_VERSINFO[0] > 4 )) || { (( BASH_VERSINFO[0] == 4 )) && (( BASH_VERSINFO[1] >= 3 )); }; then
  supports_waitn=1
fi

printf '[run_flat] START %(%F %T)T\n' -1
echo   "[run_flat] MAX_JOBS=${MAX_JOBS}"
echo   "[run_flat] OMP_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS MKL_NUM_THREADS=$MKL_NUM_THREADS NUMEXPR_NUM_THREADS=$NUMEXPR_NUM_THREADS"
echo   "[run_flat] wait -n available: $supports_waitn"
echo   "[run_flat] DATASETS:"; for x in "${DATASETS[@]}"; do echo "  - $x"; done
echo   "[run_flat] OVERLAPS: ${OVERLAPS[*]}"
echo   "[run_flat] NODES:    ${NODES[*]}"
echo   "[run_flat] METHODS:";  for x in "${METHODS[@]}";  do echo "  - $x"; done

active=0

pids=()

launch_one () {
  local ds="$1" ov="$2" nn="$3" m="$4"
  local logdir="logs/${ds}/p${nn}_${ov}"
  mkdir -p "$logdir"
  local out="${logdir}/${m}.out"

  stdbuf -oL -eL python -u run.py \
    --dataset "$ds" \
    --overlap_percentage "$ov" \
    --method "$m" \
    --nodes "$nn" \
    >"$out" 2>&1
}

for ds in "${DATASETS[@]}"; do
  for ov in "${OVERLAPS[@]}"; do
    for nn in "${NODES[@]}"; do
      for m in "${METHODS[@]}"; do

        launch_one "$ds" "$ov" "$nn" "$m" &

        if (( supports_waitn )); then
          active=$((active+1))
          if (( active >= MAX_JOBS )); then

            if ! wait -n; then
              echo "[run_flat] WARN: a job exited non-zero" >&2
            fi
            active=$((active-1))
          fi
        else

          pids+=($!)
          if (( ${#pids[@]} >= MAX_JOBS )); then
            if ! wait "${pids[0]}"; then
              echo "[run_flat] WARN: a job exited non-zero" >&2
            fi
            pids=("${pids[@]:1}")
          fi
        fi

      done
    done
  done
done


if (( supports_waitn )); then
  wait   
else
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      echo "[run_flat] WARN: PID $pid failed" >&2
    fi
  done
fi

printf '[run_flat] DONE  %(%F %T)T\n' -1