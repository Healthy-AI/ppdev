# Pragmatic Policy Development via Interpretable Behavior Cloning

This repository contains the code used for the experiments in our ML4H 2025 paper [_Pragmatic Policy Development via Interpretable Behavior Cloning_](https://arxiv.org/pdf/2507.17056).

## Installation

First clone the following repositories from GitHub:
```bash
git clone https://github.com/antmats/ppdev.git
git clone https://github.com/antmats/ReassessDTR.git
```

The `ReassessDTR` repository is used for training the offline reinforcement learning policies included in our experiments.

We use `pixi` as our package manager. Before installing dependencies, make sure `pixi` is installed by following the instructions [here](https://pixi.sh/latest/#installation).

Once `pixi` is installed, run the following command in **each** of the cloned repositories:
```bash
pixi install
```

## Configuration files

For each experiment (rheumatoid arthritis (RA) and sepsis), a corresponding configuration file is provided in [configs](configs). These configuration files specify details such as the path to the dataset, the evaluation metrics to be used, and the directory where the results will be saved.

## Data

### RA

To preprocess the RA data and extract the relevant patient cohort, run:
```bash
pixi run scripts/make_data.py --config_path configs/ra.yml --extract_cohort
```

Note that the RA data are available from [CorEvitas, LLC](https://www.ppd.com/our-solutions/clinical/real-world-data/registries/rheumatology/) through a commercial subscription agreement and are not publicly available.

### Sepsis

The Sepsis data were preprocessed as decribed in [Komorowski et al. (2018)](https://www.nature.com/articles/s41591-018-0213-5). To obtain the preprocessed dataset, follow the instructions provided [here](https://github.com/antmats/case_based_ope?tab=readme-ov-file#collect-sepsis-dataset).


## Experiments

The results and figures presented in the paper are available in [this notebook](notebooks/paper_results.ipynb).

We conducted our experiments on the [Alvis cluster](https://www.c3se.chalmers.se/about/Alvis/) using [Slurm](https://slurm.schedmd.com/documentation.html) for workload management. If you have access to a cluster that uses Slurm, you can reproduce our experiments by following the steps below. Note that you will need to update some variables at the top of the bash scripts used for launching jobs (e.g., Slurm accounting information).

### Containers

We use [Apptainer containers](https://apptainer.org/) to run the code. To create a container, copy the file `container.def` to a storage directory with sufficient space. Then, assuming the `ppdev` repository is cloned to your home directory and you are located in the storage directory, run the following command:
```bash
apptainer build --bind $HOME:/mnt ppdev_env.sif container.def
```

To verify the container, run the following command:
```bash
apptainer exec ppdev_env.sif python --version
```

A separate container for the `ReassessDTR` repository is needed and can be created in the same way.

### Reproducing the RA experiment

Run the following commands to reproduce the results for the RA experiment.

```bash
cd ~/ppdev
./scripts/slurm/estimator_fit_all.sh seeds.csv configs/ra.yml dt dt_switch dt_stage_switch rnn
```

The command above creates an experiment directory. Assume its path is stored in the variable `experiment_dir`.

Next, train the reinforcement learning policies.

```bash
sbatch --output="${experiment_dir}/logs/%x_%A_%a.out" --job-name="fit_rl_policy" --array=1-50 scripts/slurm/rl_policy_fit.sh "$experiment_dir"
cd ~/ReassessDTR
sbatch --output="${experiment_dir}/logs/%x_%A_%a.out" --job-name="rl_ra" --array=1-50 run_ra_experiment.sh "${experiment_dir}"
```

After all jobs have completed, run off-policy evaluation.
```bash
cd ~/ppdev
./scripts/slurm/ope_run_all.sh "$experiment_dir" dt_stage_switch
```

### Reproducing the Sepsis experiment

Run the following commands to reproduce the results for the Sepsis experiment.

```bash
cd ~/ppdev
./scripts/slurm/estimator_fit_all.sh seeds.csv configs/sepsis.yml dt dt_switch dt_stage_switch rnn
```

Assume the experiment directory path is stored in the variable `experiment_dir`. Train the reinforcement learning policies using the following commands.
```bash
sbatch --output="${experiment_dir}/logs/%x_%A_%a.out" --job-name="fit_rl_policy" --array=1-50 scripts/slurm/rl_policy_fit.sh "$experiment_dir"
cd ~/ReassessDTR
sbatch --output="${experiment_dir}/logs/%x_%A_%a.out" --job-name="rl_sepsis" --array=1-50 run_mimic_experiment.sh "${experiment_dir}"
```

After all jobs have completed, run off-policy evaluation.
```bash
cd ~/ppdev
./scripts/slurm/ope_run_all.sh "$experiment_dir" dt_stage_switch
```

### Notes
- Training reinforcement learning policies using the `ReassessDTR` repository requires a [Weights & Biases account](https://wandb.ai/site/). An API key must be set in the scripts `run_ra_experiment.sh` and `run_mimic_experiment.sh`.
- For the Sepsis experiment, the data must be saved as a zip file named `mimictable.zip` in [this directory](https://github.com/antmats/ReassessDTR/tree/master/DTRGym/MIMIC3SepsisEnv).
