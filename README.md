# Pragmatic Policy Development via Interpretable Behavior Cloning

This repository contains the code used for the experiments in our paper [_Pragmatic Policy Development via Interpretable Behavior Cloning_](https://arxiv.org/pdf/2507.17056).

## Installation

First clone the following repositories from GitHub:
```bash
git clone https://github.com/antmats/ppdev.git
git clone https://github.com/antmats/ReassessDTR.git
```

We use `pixi` as our package manager. To install the project and its dependencies, make sure to install `pixi` according to [these instructions](https://pixi.sh/latest/#installation).

Then run the following command:
```bash
pixi install
```

## Data

To preprocess the data, run
```bash
pixi run scripts/make_data.py --config_path configs/config.yml
```

## Experiments

To reproduce the results for the RA experiment, run the following commands:
```bash
cd ~/ppdev
./scripts/slurm/estimator_fit_all.sh seeds.csv configs/ra.yml dt dt_switch dt_stage_switch rnn
./scripts/slurm/rl_policy_fit.sh <experiment_dir>
apptainer exec --bind ~/ppdev:/mnt/ppdev <image> python ./scripts/save_split_indices_to_file.py \
    --config_path conifgs/ra.yml \
    --output_dir_path ~/ReassessDTR/DTRGym/RAEnv \
    --num_seeds 50
cd ~/ReassessDTR && ./run_ra_experiment.sh <experiment_dir> && cd ~/ppdev
./scripts/slurm/ope_run_all.sh <experiment_dir> dt_stage_switch
```

Similarly, run the following commands to reproduce the results for the sepsis experiment:
```bash
cd ~/ppdev
./scripts/slurm/estimator_fit_all.sh seeds.csv configs/sepsis.yml dt dt_switch dt_stage_switch rnn
./scripts/slurm/rl_policy_fit.sh <experiment_dir>
apptainer exec --bind ~/ppdev:/mnt/ppdev <image> python ./scripts/save_split_indices_to_file.py \
    --config_path conifgs/sepsis.yml \
    --output_dir_path ~/ReassessDTR/DTRGym/MIMIC3SepsisEnv \
    --num_seeds 50
cd ~/ReassessDTR && ./run_mimic_experiment.sh <experiment_dir> && cd ~/ppdev
./scripts/slurm/ope_run_all.sh <experiment_dir> dt_stage_switch
```
