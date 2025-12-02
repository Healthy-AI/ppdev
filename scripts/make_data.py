import os
import argparse
from pathlib import Path

import pandas as pd

from ppdev.utils import load_config
from ppdev.utils import print_log
from ppdev.data import preprocess_data
from ppdev.data import get_data_handler
from ppdev.data import select_cohort


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--extract_cohort", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config_path)

    if not os.path.exists(config["data"]["preprocessed_data_path"]):
        if not os.path.exists(config["data"]["raw_data_path"]):
            raise FileNotFoundError(
                f"The raw data file was not found: {config['data']['raw_data_path']}."
            )
        
        print_log("Preprocessing the raw data...")
        raw_data = pd.read_pickle(config["data"]["raw_data_path"])
        preprocessed_data = preprocess_data(raw_data)

        p = Path(os.path.dirname(config["data"]["preprocessed_data_path"]))
        p.mkdir(parents=True, exist_ok=True)
        preprocessed_data.to_pickle(config["data"]["preprocessed_data_path"])
    
    if args.extract_cohort:
        if not "preprocessed_data" in locals():
            preprocessed_data = pd.read_pickle(config["data"]["preprocessed_data_path"])
        cohort_info_path = get_data_handler(config).get_cohort_info_path()
        if not os.path.exists(cohort_info_path):
            print_log("Collecting cohort information...")
            cohort_info = select_cohort(
                preprocessed_data,
                **config["experiment"]["cohort_params"],
            )
            cohort_info.to_pickle(cohort_info_path)
