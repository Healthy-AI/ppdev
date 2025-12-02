import os
import argparse
from functools import partial

import joblib
import torch
import pandas as pd
from sklearn import set_config

from ppdev.utils import load_config, create_results_dir, save_yaml
from ppdev.fit_evaluate import fit_estimator, evaluate_estimator
from ppdev.estimators import CalibratedClassifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--estimator_alias", type=str, required=True)
    parser.add_argument("--new_output_dir", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config_path)

    if args.new_output_dir:
        results_dir, config = create_results_dir(config, update_config=True)
        save_yaml(config, results_dir, "config")
    else:
        results_dir = config["results"]["root_dir"]
    
    get_path = partial(os.path.join, results_dir)

    set_config(enable_metadata_routing=True)

    estimator_config = config["estimators"][args.estimator_alias]

    if (
        torch.cuda.device_count() > 1
        and estimator_config["hparams"] is not None
    ):
        from dask_cuda import LocalCUDACluster
        from dask.distributed import Client
        cluster = LocalCUDACluster()
        client = Client(cluster)
        estimator, cv_results = fit_estimator(config, args.estimator_alias)
        for worker in cluster.workers.values():
            process = worker.process.process
            if process.is_alive():
                process.terminate()
        client.shutdown()
    else:
        estimator, cv_results = fit_estimator(config, args.estimator_alias)

    joblib.dump(estimator, get_path(f"{args.estimator_alias}_estimator.pkl"))

    if cv_results is not None:
        cv_results.to_csv(
            get_path(f"{args.estimator_alias}_search.csv"),
            index=False,
        )

    scoring_args = (config, estimator, args.estimator_alias)
    scores1 = evaluate_estimator(*scoring_args)
    scores2 = evaluate_estimator(*scoring_args, subset="switches")
    scores3 = evaluate_estimator(*scoring_args, subset="stays")

    scores = pd.concat([scores1, scores2, scores3], ignore_index=True)

    if isinstance(estimator, CalibratedClassifier):
        scoring_args = (config, estimator.estimator, args.estimator_alias)
        scores4 = evaluate_estimator(*scoring_args)
        scores5 = evaluate_estimator(*scoring_args, subset="switches")
        scores6 = evaluate_estimator(*scoring_args, subset="stays")
        extra_scores = pd.concat(
            [scores4, scores5, scores6], ignore_index=True,
        )
        extra_scores.metric = extra_scores.metric + "_uncalibrated"
        scores = pd.concat([scores, extra_scores], ignore_index=True)

    scores.to_csv(get_path(f"{args.estimator_alias}_scores.csv"), index=False)

    set_config(enable_metadata_routing=False)
