#!/bin/bash

if [[ "$#" -ne 2 && "$#" -ne 3 ]]; then
    echo "Usage: $0 container_path config_or_experiment_dir_path [extract_cohort]"
    exit 1
fi

container_path="$1"

if [ -z "$SLURM_ARRAY_TASK_ID" ]
then
    config_path="$2"
else
    config_path="${2}/trial_$(printf "%03d" "$SLURM_ARRAY_TASK_ID")/config.yml"
fi

extract_cohort=$( [ "$3" == "extract_cohort" ] && echo "--extract_cohort" )

cd ~
rsync -r ppdev "$TMPDIR" --exclude="*_env"
cd "${TMPDIR}/ppdev"

apptainer exec --bind "${TMPDIR}:/mnt" --nv "$container_path" python -m cudf.pandas scripts/make_data.py \
    --config_path "$config_path" \
    $extract_cohort
