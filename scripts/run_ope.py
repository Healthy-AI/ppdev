import os
import json
import argparse
from os.path import join, isfile

import joblib
import numpy as np
import pandas as pd
from sklearn import set_config

from ppdev.utils import print_log, compile_dataframe, load_config
from ppdev.policy_evaluation import OPEEstimator
from ppdev.policy_evaluation import RandomPolicy, PropensityPolicy
from ppdev.policy_evaluation import yield_ope_inputs
from ppdev.policy_evaluation import keep_top_proba_and_renormalize

TARGET_POLICIES = [
    "rl",
    "rl_dqn",
    "rl_bcq",
    "rl_cql",
    "random",
    "most_likely",
    "outcome",
]

K_VALUES_RA = [1, 2, 3, 4, 5, 6, 7, 8]
K_VALUES_SEPSIS = [1, 2, 3, 4, 5, 10, 15, 20, 25]
P_VALUES = [0, 0.01]
S_VALUES = [0, 0.1, 0.2, 0.3, 0.4, 0.5]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir_path", type=str, required=True)
    parser.add_argument("--target_policy", type=str, required=True, choices=TARGET_POLICIES)
    parser.add_argument("--mu_estimator_alias", type=str, required=True)
    parser.add_argument("--pi_estimator_alias", type=str, default=None)
    args = parser.parse_args()

    config = load_config(join(args.experiment_dir_path, "default_config.yml"))
    k_values = K_VALUES_RA if config["experiment"]["alias"] == "ra" else K_VALUES_SEPSIS

    if not args.target_policy in TARGET_POLICIES:
        raise ValueError(f"Invalid target policy: {args.target_policy}.")

    if (
        args.target_policy in ["most_likely", "outcome"]
        and args.pi_estimator_alias is None
    ):
        raise ValueError(
            "The 'pi_estimator_alias' argument must be provided together with "
            "the 'most_likely' and 'outcome' target policies."
        )

    set_config(enable_metadata_routing=True)

    ope_estimates = []

    for (
        dataset,
        parameters,
        (mu_estimator, mu_estimator_params),
        (pi_estimator, pi_estimator_params),
    ) in yield_ope_inputs(
        args.experiment_dir_path,
        args.mu_estimator_alias,
        args.pi_estimator_alias,
    ):
        print_log(
            f"Evaluating target policy '{args.target_policy}' with the "
            f"following parameters:\n{parameters.to_string()}\n"
        )

        if mu_estimator is None:
            print_log(
                "Skipping trial because the behavior policy model "
                f"{args.mu_estimator_alias} was not found."
            )
            continue

        if args.pi_estimator_alias is not None and pi_estimator is None:
            print_log(
                "Skipping trial because the target policy requires a behavior "
                f"policy model {args.pi_estimator_alias} which was not found."
            )
            continue

        def run_ope(target_policy, s=None, k=None, p=None):
            ope_estimator = OPEEstimator(
                target_policy=target_policy,
                behavior_policy_estimator=mu_estimator,
            )
            ope_estimate = ope_estimator.estimate(
                dataset,
                params_mu=mu_estimator_params,
                params_pi=pi_estimator_params,
            )
            return compile_dataframe(
                ope_estimate,
                parameters,
                {
                    "target_policy": args.target_policy,
                    "estimator_alias_mu": args.mu_estimator_alias,
                    "estimator_alias_pi": args.pi_estimator_alias,
                    "s": s,
                    "k": k,
                    "p": p,
                },
            )

        if args.target_policy == "rl":
            trial = f"trial_{parameters.name:03d}"
            trial_dir_path = join(args.experiment_dir_path, trial)
            rl_policy_path = join(trial_dir_path, "rl_policy.pkl")
            if not isfile(rl_policy_path):
                print_log(f"Skipping trial because no RL policy was found.")
                continue
            target_policy = joblib.load(rl_policy_path)
            for p in P_VALUES:
                target_policy.p = p
                ope_estimates.append(run_ope(target_policy, p=p))

        elif (
            args.target_policy == "rl_dqn"
            or args.target_policy == "rl_bcq"
            or args.target_policy == "rl_cql"
        ):
            trial = f"trial_{parameters.name:03d}"
            trial_dir_path = join(args.experiment_dir_path, trial)

            env_dir = next(
                (x for x in os.listdir(trial_dir_path) if "Env" in x), None
            )
            if env_dir is None:
                print_log("Skipping trial because no environment directory was found.")
                continue
            env_dir_path = join(trial_dir_path, env_dir)

            rl_algo = args.target_policy.split("_")[-1]
            rl_dir = next(
                (x for x in os.listdir(env_dir_path) if rl_algo in x), None
            )
            if rl_dir is None:
                print_log(f"Skipping trial because no direcotry for {rl_algo} was found.")
                continue
            rl_dir_path = join(env_dir_path, rl_dir)

            best_candidate = None
            best_score = float("-inf")

            for candidate in os.listdir(rl_dir_path):
                summary_path = join(rl_dir_path, candidate, "run_summary.json")
                if not os.path.exists(summary_path):
                    continue
                try:
                    with open(summary_path, "r") as f:
                        summary = json.load(f)
                except Exception as e:
                    print_log(f"Error loading run_summary.json for candidate {candidate}: {e}.")
                    continue
                else:
                    score = summary.get("best_epoch-all_val-WIS_truncated", float("-inf"))
                    if score > best_score:
                        best_score = score
                        best_candidate = candidate

            if best_candidate is None:
                print_log(f"No valid policy candidates found in {rl_dir_path}.")
                continue

            proba_path = join(rl_dir_path, best_candidate, "proba-all_test.json")
            if not os.path.exists(proba_path):
                print_log(f"No probabilities found for {best_candidate}.")
                continue
            try:
                with open(proba_path, "r") as f:
                    proba = json.load(f)
            except Exception as e:
                print_log(f"Error loading probabilities for {best_candidate}: {e}.")
                continue

            class PolicyWrapper:
                def __init__(self, proba, p=0):
                    self.proba = np.array(proba)
                    self.p = p

                def predict_proba(self, inputs):
                    return keep_top_proba_and_renormalize(
                        self.proba, k=1, p=self.p
                    )

            for p in P_VALUES:
                target_policy = PolicyWrapper(proba, p=p)
                ope_estimates.append(run_ope(target_policy, p=p))

        elif args.target_policy == "random":
            for p in P_VALUES:
                target_policy = RandomPolicy(mu_estimator.classes_.size, p=p)
                ope_estimates.append(run_ope(target_policy, p=p))

        elif args.target_policy == "most_likely":
            for s in S_VALUES:
                for k in k_values:
                    target_policy = PropensityPolicy(pi_estimator, s, k)
                    ope_estimates.append(run_ope(target_policy, s=s, k=k))

        else:  # target_policy == "outcome"
            for s in S_VALUES:
                for k in k_values:
                    target_policy = PropensityPolicy(pi_estimator, s, k, use_outcomes=True)
                    ope_estimates.append(run_ope(target_policy, s=s, k=k))

    ope_estimates = pd.concat(ope_estimates).reset_index(drop=True)

    output_name = (
        f"pi_{args.target_policy}"
        f"__estimator_mu_{args.mu_estimator_alias}"
        f"__estimator_pi_{args.pi_estimator_alias}"
    )

    output_path = join(args.experiment_dir_path, f"{output_name}.csv")

    ope_estimates.to_csv(output_path, index=False)

    set_config(enable_metadata_routing=False)
