import argparse
from os.path import join

import joblib

from ppdev.policy_evaluation import RLPolicyRA, RLPolicySepsis
from ppdev.utils import load_config
from ppdev.data import get_data_handler, get_dataset
from ppdev.data import utils as utils

DEFAULT_N_CLUSTERS_LIST_RA = range(5, 55, 5)
DEFAULT_N_CLUSTERS_LIST_SEPSIS = range(500, 1050, 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial_dir_path", type=str, required=True)
    args = parser.parse_args()

    config = load_config(join(args.trial_dir_path, "config.yml"))
    # We do not need any validation data here.
    config["experiment"]["valid_size"] = None

    data_handler = get_data_handler(config)

    data_train, _data_valid, _data_test = data_handler.split_data()

    if config["experiment"]["alias"] == "ra":
        data_train = utils.filter_cohort(data_train, c_stage="stage")

        dataset_train = get_dataset(data_train, config)

        # Get a preprocessor for imputing missing values. Numerical features are
        # scaled to have zero mean and unit variance.
        preprocessor = data_handler.get_preprocessor("scale", "none")

        policy = RLPolicyRA(
            n_clusters_list=DEFAULT_N_CLUSTERS_LIST_RA,
            preprocessor=preprocessor,
        )
        policy.fit(dataset_train)

    else:  # sepsis
        dataset_train = get_dataset(data_train, config)

        preprocessor = data_handler.get_preprocessor("scale", "shift")

        policy = RLPolicySepsis(
            n_clusters_list=DEFAULT_N_CLUSTERS_LIST_SEPSIS,
            preprocessor=preprocessor,
        )
        policy.fit(dataset_train)

    joblib.dump(policy, join(args.trial_dir_path, "rl_policy.pkl"))
