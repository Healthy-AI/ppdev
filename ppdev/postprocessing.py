import os
import warnings
import glob
from os.path import join, basename

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from seaborn._statistics import EstimateAggregator

from .policy_evaluation import OPEResult


def _collect_results(results_dir, suffix, **parameters):
    suffix = f"_{suffix}.csv"
    all_results = []
    for results_path in glob.glob(join(results_dir, f"*{suffix}")):
        results = pd.read_csv(results_path)
        estimator_alias = basename(results_path).removesuffix(suffix)
        results["estimator_alias"] = estimator_alias
        results = results.assign(**parameters)
        all_results.append(results)
    return all_results


def collect_results(experiment_dir):
    parameters_file_path = join(experiment_dir, "parameters.csv")
    parameters_file = pd.read_csv(parameters_file_path)
    parameters_file.index += 1  # Start indexing from 1

    all_scores = []
    all_ope_estimates = []
    
    for i, parameters in parameters_file.iterrows():
        results_dir = join(experiment_dir, f"trial_{i:03d}")

        scores = _collect_results(results_dir, "scores", **parameters)
        all_scores.extend(scores)

        ope_estimates = _collect_results(results_dir, "ope_estimates", **parameters)
        all_ope_estimates.extend(ope_estimates)

    all_scores = pd.concat(all_scores)
    default_columns = ["estimator_alias", "metric", "score"]
    experiment_columns = [c for c in all_scores.columns if not c in default_columns]
    all_scores = all_scores.pivot_table(
        index=["estimator_alias"] + experiment_columns,
        columns="metric",
        values="score",
    ).reset_index()

    if len(all_ope_estimates) > 0:
        all_ope_estimates = pd.concat(all_ope_estimates)

    return all_scores, all_ope_estimates


def collect_cv_results(experiment_dir, estimator_alias, exclude_params=None):
    results_dirs = [x for x in os.listdir(experiment_dir) if x.startswith("trial")]
    results_dirs.sort()

    if exclude_params is None:
        exclude_params = {}

    all_cv_results = []
    for results_dir in results_dirs:
        params = pd.read_csv(join(experiment_dir, results_dir, "parameters.csv"))
        skip_dir = False
        for p, v in exclude_params.items():
            if p in params and params[p].item() == v:
                skip_dir = True
        if skip_dir:
            continue
        for cv_results_path in glob.glob(join(experiment_dir, results_dir, "*_search.csv")):
            if basename(cv_results_path).removesuffix("_search.csv") == estimator_alias:
                cv_results = pd.read_csv(cv_results_path)
                params = pd.concat([params] * len(cv_results), ignore_index=True)
                cv_results = pd.concat([cv_results, params], axis=1)
                all_cv_results.append(cv_results)

    return pd.concat([df for df in all_cv_results])


def get_scoring_table(
    scores,
    metric,
    groupby="estimator_alias",
    include_cis=False,
    multirow=False,
):
    agg = EstimateAggregator(np.mean, "ci", n_boot=1000, seed=0)
    
    table = scores.groupby(groupby).apply(agg, var=metric)
    table = table * 100  # Convert to percentage

    if include_cis:
        f = lambda row: (
            r"\begin{tabular}[c]{@{}c@{}}"
            + rf"{row[metric]:.1f}\\({row[f'{metric}min']:.1f}, {row[f'{metric}max']:.1f})" 
            + r"\end{tabular}"
        ) if multirow else (
            f"{row[metric]:.1f} ({row[f'{metric}min']:.1f}, {row[f'{metric}max']:.1f})"
        )
    else:
        f = lambda row: f"{row[metric]:.1f}"
    
    table[metric] = table[[metric, f"{metric}min", f"{metric}max"]].apply(f, axis=1)
    table = table.drop(columns=[f"{metric}min", f"{metric}max"])

    return table


def inspect_hparams(
    experiment_dir,
    estimator_alias,
    metric="score",
    use_log_scale=None,
    exclude_params=None,
):
    results_dirs = [x for x in os.listdir(experiment_dir) if x.startswith("trial")]
    results_dirs.sort()

    if exclude_params is None:
        exclude_params = {}

    cv_results = []
    for results_dir in results_dirs:
        try:
            params = pd.read_csv(join(experiment_dir, results_dir, "parameters.csv"))
        except FileNotFoundError:
            params = {}
        skip_dir = False
        for p, v in exclude_params.items():
            if p in params and params[p].item() == v:
                skip_dir = True
        if skip_dir:
            continue
        for cv_results_path in glob.glob(join(experiment_dir, results_dir, "*_search.csv")):
            if basename(cv_results_path).removesuffix("_search.csv") == estimator_alias:
                cv_results.append(pd.read_csv(cv_results_path))

    cv_results = pd.concat([pd.DataFrame(d) for d in cv_results])

    if use_log_scale is None:
        use_log_scale = []

    for param in [c for c in cv_results.columns if c.startswith("param_")]:
        _fig, ax = plt.subplots()
        try:
            ax.scatter(
                cv_results[f"mean_test_{metric}"],
                cv_results[param],
            )
        except:
            print(f"Could not produce plot for parameter {param}.")
            continue
        if param in use_log_scale:
            ax.set_yscale("log")
        ax.set_title(param)


def combine_scoring_tables(scores, metrics, **kwargs):
    tables = []
    for metric in metrics:
        table = get_scoring_table(scores, metric, **kwargs)
        tables.append(table)
    return pd.concat(tables, axis=1)


def collect_ope_estimates(
    experiment_dir,
    keep_weights=False,
    estimates_to_recompute=None,
    threshold=100,
    normalize_stages=False,
):
    all_ope_estimates = []
    for ope_estimates in [x for x in os.listdir(experiment_dir) if x.startswith("pi_")]:
        ope_estimates = pd.read_csv(join(experiment_dir, ope_estimates))
        ope_estimates = ope_estimates.fillna(dict(s=-1, k=-1, p=-1))
        groupby = ["seed", "s", "k", "p"]
        if estimates_to_recompute is not None:
            assert isinstance(estimates_to_recompute, list)
            ope_estimates["threshold"] = threshold
            ope_estimates_ = ope_estimates.copy()
            if not normalize_stages:
                ope_estimates_["n_stages"] = 1
            for estimate in estimates_to_recompute:
                assert f"{estimate}_" in ope_estimates.columns
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    new_estimates = ope_estimates_.groupby(groupby).apply(
                        lambda x: getattr(OPEResult.from_dataframe(x), estimate)()
                    ).reset_index(name=f"{estimate}_")
                ope_estimates.drop(columns=f"{estimate}_", inplace=True)
                ope_estimates = pd.merge(ope_estimates, new_estimates, on=groupby, how="left")
        if not keep_weights:
            ope_estimates = ope_estimates.groupby(groupby).first().reset_index()
        all_ope_estimates.append(ope_estimates)
    return pd.concat(all_ope_estimates).reset_index(drop=True)
