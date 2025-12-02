#!/bin/bash

if [[ "$#" -ne 3 && "$#" -ne 4 ]]; then
    echo "Usage: $0 container_path config_or_experiment_dir_path estimator_alias [run_ope]"
    exit 1
fi

container_path="$1"

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    config_path="$2"
else
    config_path="${2}/trial_$(printf "%03d" "$SLURM_ARRAY_TASK_ID")/config.yml"
fi

estimator_alias="$3"

run_ope=$( [ "$4" == "run_ope" ] && echo "--run_ope" )

cd ~
rsync -r ppdev "$TMPDIR" --exclude="*_env"
cd "${TMPDIR}/ppdev"

apptainer exec --bind "${TMPDIR}:/mnt" --nv "$container_path" python scripts/fit_estimator.py \
    --config_path "$config_path" \
    --estimator_alias "$estimator_alias" \
    $run_ope

if [ "$SLURM_ARRAY_JOB_ID" ]; then
    log_path="${2}/logs/"
    log_path+="${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
    results_path=$(awk '/^results:/ {flag=1}
        /^  root_dir:/ && flag {print $2; flag=0}' "$config_path")
    cp "$log_path" "$results_path"
fi
