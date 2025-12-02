#!/bin/bash

make_data="false"
make_cohorts="false"

# Set Slurm parameters.
account="NAISS2024-5-480"
partition="alvis"
gpu="T4"
container="/mimer/NOBACKUP/groups/inpole/ppdev/ppdev_env.sif"

# Check if the correct number of arguments were passed in.
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 parameters_file default_config_file estimator1 [estimator2 ...]"
    exit 1
fi

# Get the input files.
parameters_file="$1"
default_config_file="$2"

# Shift input arguments to get the list of estimators.
shift 2

# Extract the root directory for the results from the default config file.
results_root_dir=$(awk '/^results:/ {flag=1} 
    /^  root_dir:/ && flag {print $2; flag=0}' "$default_config_file")

# Check that it was found.
if [ -z "$results_root_dir" ]; then
    echo "Could not extract a root directory for the results from $default_config_file."
    exit 1
fi

# Create a new experiment directory in the root directory for the results.
timestamp=$(date +"%y%m%d_%H%M")
experiment_dir="${results_root_dir}/${timestamp}"
mkdir -p "$experiment_dir"

# Save the input files to the experiment directory.
cp "$default_config_file" "${experiment_dir}/default_config.yml"
cp "$parameters_file" "${experiment_dir}/parameters.csv"

# Create a directory for the log files.
logs_dir="${experiment_dir}/logs"
mkdir -p "$logs_dir"

# Define a function to update a parameter value in the config file.
update_config_file() {
    local config_file="$1"
    local parameter_name="$2"
    local new_parameter_value="$3"
    sed -ri "s/^(\s*)(${parameter_name}\s*:\s*)\S+.*/\1${parameter_name}: $new_parameter_value/" "$config_file"
}

dependencies=""

i=0
while IFS="," read -r -a parameter_values; do

    if [ $i -eq 0 ]; then
        # Extract the parameter names.
        parameter_names=("${parameter_values[@]}")

        if [ "$make_data" == "true" ]; then
            make_data_job_id=$(
                sbatch \
                    --account="$account" \
                    --partition="$partition" \
                    --nodes=1 \
                    --gpus-per-node="$gpu:1" \
                    --output="$logs_dir/%x_%A.out" \
                    --time="1-0:0" \
                    --job-name="make_data" \
                    "scripts/slurm/data_make.sh" "$container" "$default_config_file" \
                | awk '{print $4}'
            )
            dependencies="afterok:${make_data_job_id}"
        fi

        i=$((i + 1))

        continue
    fi

    # Create a results directory for the current parameters.
    results_dir="${experiment_dir}/trial_$(printf "%03d" "$i")"
    mkdir -p "$results_dir"
    
    # Copy the default config file to the results directory.
    config_file="${results_dir}/config.yml"
    cp "$default_config_file" "$config_file"

    # Update the config file with the current parameters.
    for index in "${!parameter_names[@]}"; do
        parameter_name="${parameter_names[$index]}"
        parameter_value="${parameter_values[$index]}"
        update_config_file "$config_file" "$parameter_name" "$parameter_value"
    done

    # Update the root directory for the results.
    #
    # Note: Only the first "root_dir" is replaced.
    awk -v repl="$results_dir" '{
        if (!found && $0 ~ /root_dir/) {
            sub(/root_dir:.*/, "root_dir: " repl);
            found = 1
        }
        print
    }' "$config_file" > temp.yml && mv temp.yml "$config_file"

    # Save the current parameters to a file.
    echo "$(IFS=,; echo "${parameter_names[*]}")" > "${results_dir}/parameters.csv"
    echo "$(IFS=,; echo "${parameter_values[*]}")" >> "${results_dir}/parameters.csv"

    i=$((i + 1))

done < <(tr -d '\r' < "$parameters_file")

# Make cohorts.
if [ "$make_cohorts" == "true" ]; then
    make_cohort_job_id=$(
        sbatch \
            --account="$account" \
            --partition="$partition" \
            --nodes=1 \
            --gpus-per-node="$gpu:1" \
            --output="${logs_dir}/%x_%A_%a.out" \
            --time="0-1:0" \
            --array=1-$((i - 1)) \
            --dependency="$dependencies" \
            --job-name="make_cohort" \
            "scripts/slurm/data_make.sh" "$container" "$experiment_dir" "extract_cohort" \
        | awk '{print $4}'
    )
    if [ -z "$dependencies" ]; then
        dependencies="afterok:${make_cohort_job_id}"
    else
        dependencies="${dependencies}:${make_cohort_job_id}"
    fi
fi

# Fit estimators.
for estimator in "$@"; do
    if [ "$estimator" == "rnn" ]; then
        gpu_resources="--gpus-per-node=${gpu}:4"
    else
        gpu_resources="--constraint=NOGPU"
    fi
    sbatch \
        --account="$account" \
        --partition="$partition" \
        --nodes=1 \
        $gpu_resources \
        --output="${logs_dir}/%x_%A_%a.out" \
        --time="1-0:0" \
        --array=1-$((i - 1)) \
        --job-name="fit_${estimator}" \
        --dependency="$dependencies" \
        "scripts/slurm/estimator_fit.sh" "$container" "$experiment_dir" "$estimator"
done
