#!/bin/bash

if [[ "$#" -ne 4 && "$#" -ne 5 ]]; then
    echo "Usage: $0 container_path experiment_dir_path target_policy mu_estimator_alias [pi_estimator_alias]"
    exit 1
fi

container_path="$1"
experiment_dir_path="$2"
target_policy="$3"
mu_estimator_alias="$4"
pi_estimator_alias=$( [ -n "$5" ] && echo "--pi_estimator_alias $5" )

cd ~
rsync -r ppdev "$TMPDIR" --exclude="*_env"
cd "${TMPDIR}/ppdev"

apptainer exec --bind "${TMPDIR}:/mnt" --nv "$container_path" python scripts/run_ope.py \
    --experiment_dir_path "$experiment_dir_path" \
    --target_policy "$target_policy" \
    --mu_estimator_alias "$mu_estimator_alias" \
    $pi_estimator_alias
