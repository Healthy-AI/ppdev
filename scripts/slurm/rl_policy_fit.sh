#!/bin/bash
#SBATCH -A NAISS2024-5-480 -p alvis
#SBATCH -N 1 --constraint=NOGPU
#SBATCH -t 0-01:00:00

container="/mimer/NOBACKUP/groups/inpole/ppdev/ppdev_env.sif"

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 experiment_dir"
    exit 1
fi

experiment_dir="$1"

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "This script must be run as a Slurm array job. For example:"
    echo "  sbatch --output=\"\${experiment_dir}/logs/%x_%A_%a.out\" --job-name=\"fit_rl_policy\" --array=1-100 scripts/slurm/fit_rl_policy.sh \"\$experiment_dir\""
    exit 1
fi

cd ~
rsync -r ppdev "$TMPDIR" --exclude="*_env"
cd "${TMPDIR}/ppdev"

apptainer exec --bind "${TMPDIR}:/mnt" --nv "$container" python scripts/fit_rl_policy.py \
    --trial_dir_path "${experiment_dir}/trial_$(printf "%03d" "$SLURM_ARRAY_TASK_ID")"
