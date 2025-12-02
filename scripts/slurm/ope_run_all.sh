#!/bin/bash

# Set Slurm parameters.
account="NAISS2024-5-480"
partition="alvis"
gpu="none"
container="/mimer/NOBACKUP/groups/inpole/ppdev/ppdev_env.sif"

# Check if the correct number of arguments were passed in.
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 experiment_dir estimator1 [estimator2 ...]"
    exit 1
fi

# Get the experiment directory.
experiment_dir="$1"

# Shift input arguments to get the list of estimators.
shift

# Get the log files directory.
logs_dir="${experiment_dir}/logs"

# Run OPE.

if [ "$gpu" == "none" ]; then
        gpu_resources="--constraint=NOGPU"
    else
        gpu_resources="--gpus-per-node=${gpu}:1"
    fi

for estimator in "$@"; do

    # RL policy.
    sbatch \
        --account="$account" \
        --partition="$partition" \
        --nodes=1 \
        $gpu_resources \
        --output="${logs_dir}/%x_%A_%a.out" \
        --time="1-0:0" \
        --job-name="run_ope_rl_${estimator}" \
        "scripts/slurm/ope_run.sh" "$container" "$experiment_dir" "rl" "$estimator"

    # RL (DQN) policy.
    sbatch \
        --account="$account" \
        --partition="$partition" \
        --nodes=1 \
        $gpu_resources \
        --output="${logs_dir}/%x_%A_%a.out" \
        --time="1-0:0" \
        --job-name="run_ope_dqn_${estimator}" \
        "scripts/slurm/ope_run.sh" "$container" "$experiment_dir" "rl_dqn" "$estimator"

    # RL (BCQ) policy.
    sbatch \
        --account="$account" \
        --partition="$partition" \
        --nodes=1 \
        $gpu_resources \
        --output="${logs_dir}/%x_%A_%a.out" \
        --time="1-0:0" \
        --job-name="run_ope_bcq_${estimator}" \
        "scripts/slurm/ope_run.sh" "$container" "$experiment_dir" "rl_bcq" "$estimator"

    # RL (CQL) policy.
    sbatch \
        --account="$account" \
        --partition="$partition" \
        --nodes=1 \
        $gpu_resources \
        --output="${logs_dir}/%x_%A_%a.out" \
        --time="1-0:0" \
        --job-name="run_ope_cql_${estimator}" \
        "scripts/slurm/ope_run.sh" "$container" "$experiment_dir" "rl_cql" "$estimator"

    # Random policy.
    sbatch \
        --account="$account" \
        --partition="$partition" \
        --nodes=1 \
        $gpu_resources \
        --output="${logs_dir}/%x_%A_%a.out" \
        --time="1-0:0" \
        --job-name="run_ope_random_${estimator}" \
        "scripts/slurm/ope_run.sh" "$container" "$experiment_dir" "random" "$estimator"

    # Most likely policy.
    sbatch \
        --account="$account" \
        --partition="$partition" \
        --nodes=1 \
        $gpu_resources \
        --output="${logs_dir}/%x_%A_%a.out" \
        --time="1-0:0" \
        --job-name="run_ope_most_likely_${estimator}" \
        "scripts/slurm/ope_run.sh" "$container" "$experiment_dir" "most_likely" "$estimator" "$estimator"

    # Outcome policy.
    sbatch \
        --account="$account" \
        --partition="$partition" \
        --nodes=1 \
        $gpu_resources \
        --output="${logs_dir}/%x_%A_%a.out" \
        --time="1-0:0" \
        --job-name="run_ope_outcome_${estimator}" \
        "scripts/slurm/ope_run.sh" "$container" "$experiment_dir" "outcome" "$estimator" "$estimator"

done
