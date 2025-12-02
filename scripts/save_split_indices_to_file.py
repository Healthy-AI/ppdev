import os
import copy
import argparse
import json
from ppdev.utils import load_config
from ppdev.data import get_data_handler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save split indices to file.")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--output_dir_path", type=str, required=True)
    parser.add_argument("--num_seeds", type=int, default=10)
    args = parser.parse_args()

    default_config = load_config(args.config_path)
    group = default_config["experiment"]["identifier"]

    for seed in range(args.num_seeds):
        config = copy.deepcopy(default_config)
        config["experiment"]["seed"] = seed

        data_handler = get_data_handler(config)
        X_train, X_valid, X_test = data_handler.split_data()

        split_indices = {
             "train": X_train[group].unique().tolist(),
             "valid": X_valid[group].unique().tolist(),
             "test": X_test[group].unique().tolist(),
        }
        file_path = os.path.join(args.output_dir_path, f"split_{seed}.json")
        with open(file_path, "w") as f:
             json.dump(split_indices, f)
