import types

import torch
import pandas as pd
import numpy as np
from skorch.callbacks import Checkpoint
from sklearn.model_selection import PredefinedSplit

from .pipeline import get_pipeline
from .estimators import CalibratedClassifier, BaseTherapyClassifier, expects_groups
from .scoring import get_scoring
from .search import GridSearchCV, RandomizedSearchCV
from .data import get_data_handler, get_dataset, get_params


def _requires_switch_mask(scoring_metrics):
    if isinstance(scoring_metrics, list):
        return any([metric.endswith("_switches") for metric in scoring_metrics])
    else:
        return scoring_metrics.endswith("_switches")


def _get_ccp_alphas(path, n=10):
    ccp_alphas = path.ccp_alphas[:-1]
    min_ccp_alpha = np.maximum(ccp_alphas.min(), 0.0)
    max_ccp_alpha = ccp_alphas.max()
    return np.linspace(min_ccp_alpha, max_ccp_alpha, n)


def _prepare_search(
    pipeline, hparams, dataset_train, config, estimator_alias, data_train=None
):
    estimator_config = config["estimators"][estimator_alias]
    search_config = config["model_selection"]

    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["estimator"]

    if estimator_config.get("include_ccp_alphas", False):
        assert hasattr(estimator, "cost_complexity_pruning_path")
        assert data_train is not None
        # TODO: This is not quite compatible with data handler 4, since the
        # training dataset contains both training and validation data in this
        # case.
        Xt_train = preprocessor.fit_transform(dataset_train.inputs)
        y_train = dataset_train.targets
        params_train = get_params(data_train, config, estimator_alias)
        ccp_path = estimator.cost_complexity_pruning_path(
            Xt_train, y_train, **params_train
        )
        if isinstance(estimator, BaseTherapyClassifier):
            for e, p in ccp_path.items():
                hparams[f"estimator__{e}__ccp_alpha"] = _get_ccp_alphas(p)
        else:
            hparams["estimator__ccp_alpha"] = _get_ccp_alphas(ccp_path)

    if isinstance(search_config["scoring"], list):
        scoring = {
            scoring_metric: get_scoring(estimator, scoring_metric)
            for scoring_metric in search_config["scoring"]
        }
    else:
        scoring = get_scoring(estimator, search_config["scoring"])

    n_jobs = None if estimator_config.get("is_net_estimator", False) else -1

    splits = dataset_train.get_splits(
        n_splits=search_config["n_splits"],
        test_size=search_config["test_size"],
        seed=search_config["seed"],
    )

    parameters_to_index = None
    if estimator_config.get("is_net_estimator", False):
        checkpoint = next(
            (c for c in estimator.callbacks if isinstance(c, Checkpoint)),
            None,
        )
        if checkpoint is not None:
            parameters_to_index = {
                "estimator__callbacks__Checkpoint__fn_prefix": checkpoint.fn_prefix
            }

    search_strategy = estimator_config.get("search", search_config["search"])

    if search_strategy == "exhaustive":
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=hparams,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=search_config["refit"],
            cv=splits,
            verbose=2,
            error_score="raise",
            parameters_to_index=parameters_to_index,
        )
    elif search_strategy == "random":
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=hparams,
            n_iter=search_config["n_iter"],
            scoring=scoring,
            n_jobs=n_jobs,
            refit=search_config["refit"],
            cv=splits,
            verbose=2,
            random_state=search_config["seed"],
            error_score="raise",
            parameters_to_index=parameters_to_index,
        )
    else:
        raise ValueError(
            f"Invalid search method: {search_strategy}."
        )

    return search


def fit_estimator(config, estimator_alias):
    estimator_config = config["estimators"][estimator_alias]

    data_handler = get_data_handler(config)
    data_train, data_valid, _data_test = data_handler.split_data()

    if (
        estimator_config.get("is_net_estimator", False)
        or estimator_config["calibrate"]
        or config["experiment"]["data_handler"] == "4"
    ):
        assert len(data_valid) > 0, "Validation data is required for this estimator and/or configuration."

    if config["experiment"]["data_handler"] == "4":
        data_train_orig = data_train  # Keep a copy of the original data
        n_train, n_valid = len(data_train), len(data_valid)
        data_train = pd.concat([data_train, data_valid])
        dataset_train = get_dataset(data_train, config)
        def get_splits(self, n_splits, test_size, seed):
            splitter = PredefinedSplit([-1] * n_train + [0] * n_valid)
            return splitter.split()
        dataset_train.get_splits = types.MethodType(get_splits, dataset_train)
        Xy_train = (dataset_train.inputs, dataset_train.targets)
    else:
        dataset_train = get_dataset(data_train, config)
        Xy_train = (dataset_train.inputs, dataset_train.targets)

    pipeline = get_pipeline(config, estimator_alias)

    params_train = get_params(data_train, config, estimator_alias, training=True)

    if estimator_config.get("is_net_estimator", False):
        preprocessor = pipeline.named_steps["preprocessor"]
        if config["experiment"]["data_handler"] == "4":
            preprocessor.fit(dataset_train.inputs.iloc[:n_train])
        else:
            preprocessor.fit(dataset_train.inputs)
        dataset_valid = get_dataset(data_valid, config)
        params_train["X_valid"] = preprocessor.transform(dataset_valid.inputs)
        params_train["y_valid"] = dataset_valid.targets
        if expects_groups(pipeline.named_steps["estimator"]):
            params_train["groups_valid"] = dataset_valid.groups

    hparams = estimator_config.get("hparams", None)
    fixed_params = estimator_config.get("fixed_params", None)

    if fixed_params is not None:
        if isinstance(hparams, list):
            for hparam_dict in hparams:
                for fixed_param_name in fixed_params:
                    hparam_dict.pop(fixed_param_name, None)
        elif isinstance(hparams, dict):
            for fixed_param_name in fixed_params:
                hparams.pop(fixed_param_name, None)
        pipeline.set_params(**fixed_params)

    if hparams is not None:
        search = _prepare_search(
            pipeline,
            hparams,
            dataset_train,
            config,
            estimator_alias,
            data_train,
        )

        if _requires_switch_mask(config["model_selection"]["scoring"]):
            params_train["mask"] = (
                data_train["therapy"] != data_train["prev_therapy"]
            )

        if config["experiment"]["data_handler"] == "4":
            # We set `refit` to False to avoid refitting the best estimator on
            # the combined training and validation data. Only the training data
            # is used for training the best estimator. The validation data is
            # used for calibration.
            search.refit = False

        if (
            estimator_config.get("is_net_estimator", False)
            and torch.cuda.device_count() > 1
        ):
            from joblib import parallel_backend
            with parallel_backend("dask"):
                search.fit(*Xy_train, **params_train)
        else:
            search.fit(*Xy_train, **params_train)

        cv_results = pd.DataFrame(search.cv_results_)

        if config["experiment"]["data_handler"] == "4":
            refit = config["model_selection"]["refit"]
            best_params = cv_results.loc[
                cv_results[f"rank_test_{refit}"].idxmin(), "params"
            ]
            best_pipeline = pipeline.set_params(**best_params)
            dataset_train_orig = get_dataset(data_train_orig, config)
            Xy_train_orig = (dataset_train_orig.inputs, dataset_train_orig.targets)
            params_train_orig = get_params(data_train_orig, config, estimator_alias, training=True)
            if estimator_config.get("is_net_estimator", False):
                params_train_orig["X_valid"] = params_train["X_valid"]
                params_train_orig["y_valid"] = params_train["y_valid"]
                if expects_groups(pipeline.named_steps["estimator"]):
                    params_train_orig["groups_valid"] = params_train["groups_valid"]
            best_pipeline.fit(*Xy_train_orig, **params_train_orig)
        else:
            best_pipeline = search.best_estimator_

    else:
        best_pipeline = pipeline.fit(*Xy_train, **params_train)
        cv_results = None

    if estimator_config["calibrate"]:
        dataset_valid = get_dataset(data_valid, config)
        params_valid = get_params(data_valid, config, estimator_alias)
        calibrated_best_pipeline = CalibratedClassifier(
            estimator=best_pipeline,
            method="sigmoid",
        ).fit(dataset_valid.inputs, dataset_valid.targets, **params_valid)
        return calibrated_best_pipeline, cv_results

    else:
        return best_pipeline, cv_results


def evaluate_estimator(config, estimator, estimator_alias, subset="all"):
    if subset not in ["all", "stays", "switches"]:
        raise ValueError(f"Invalid subset: {subset}.")

    data_handler = get_data_handler(config)
    _data_train, _data_valid, data_test = data_handler.split_data()

    t = "therapy" if config["experiment"]["alias"] == "ra" else "treatment"
    if subset == "stays":
        stay = data_test[t] == data_test[f"prev_{t}"]
        data_test = data_test[stay]
    elif subset == "switches":
        switch = data_test[t] != data_test[f"prev_{t}"]
        data_test = data_test[switch]

    estimator_config = config["estimators"][estimator_alias]
    dataset_test = get_dataset(data_test, config)
    params_test = get_params(data_test, config, estimator_alias)

    scores = []
    for metric in config["experiment"]["evaluation_metrics"]:
        scores.append(
            (
                metric if subset == "all" else f"{metric}_{subset}",
                estimator.score(
                    dataset_test.inputs,
                    dataset_test.targets,
                    metric=metric,
                    **params_test,
                ),
            ),
        )
    
    return pd.DataFrame(scores, columns=["metric", "score"])
