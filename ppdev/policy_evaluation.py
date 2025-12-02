import copy
import types
from os.path import join, isfile
from functools import partial

import joblib
import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.cluster import KMeans
from mdptoolbox.mdp import QLearning

from .utils import load_config
from .estimators import CalibratedClassifier
from .data import OPEDataset, get_data_handler, get_dataset, get_params
from .pipeline import CustomPipeline


def get_ope_inputs(
    config, subset="test", mu_estimator_alias=None, pi_estimator_alias=None
):
    data_handler = get_data_handler(config)
    i_subset = -1 if subset == "test" else 0
    data = data_handler.split_data()[i_subset]
    dataset = get_dataset(data, config)

    mu_estimator_params = None
    if mu_estimator_alias is not None:
        mu_estimator_params = get_params(data, config, mu_estimator_alias)
        mu_estimator_params.pop("sample_weight", None)

    pi_estimator_params = None
    if pi_estimator_alias is not None:
        pi_estimator_params = get_params(data, config, pi_estimator_alias)
        pi_estimator_params.pop("sample_weight", None)

    return dataset, mu_estimator_params, pi_estimator_params


def yield_ope_inputs(
    experiment_dir_path,
    mu_estimator_alias=None,
    pi_estimator_alias=None,
    subset="test",
):
    assert subset in ["train", "test"]

    parameters_file_path = join(experiment_dir_path, "parameters.csv")
    parameters_file = pd.read_csv(parameters_file_path)
    parameters_file.index += 1  # Start indexing from 1

    for i, parameters in parameters_file.iterrows():
        results_dir_path = join(experiment_dir_path, f"trial_{i:03d}")
        get_path = partial(join, results_dir_path)

        mu_estimator = None
        if mu_estimator_alias is not None:
            mu_estimator_path = get_path(f"{mu_estimator_alias}_estimator.pkl")
            if isfile(mu_estimator_path):
                mu_estimator = joblib.load(mu_estimator_path)
            else:
                mu_estimator = None

        pi_estimator = None
        if pi_estimator_alias is not None:
            pi_estimator_path = get_path(f"{pi_estimator_alias}_estimator.pkl")
            if isfile(pi_estimator_path):
                pi_estimator = joblib.load(pi_estimator_path)
            else:
                pi_estimator = None

        config_path = get_path("config.yml")
        config = load_config(config_path)
        dataset, mu_estimator_params, pi_estimator_params = get_ope_inputs(
            config, subset, mu_estimator_alias, pi_estimator_alias
        )

        yield (
            dataset,
            parameters,
            (mu_estimator, mu_estimator_params),
            (pi_estimator, pi_estimator_params),
        )


def extract_preprocessor_and_estimator(estimator):
    if isinstance(estimator, CalibratedClassifier):
        estimator = estimator.estimator
    if isinstance(estimator, CustomPipeline):
        return estimator[:-1], estimator[-1]
    else:
        return None, estimator


def keep_top_proba_and_renormalize(proba, k, p=0):
    # TODO: To align with Komorowski et al., the probability of selecting any
    # action other than the top-k actions should be 1.

    m, n = proba.shape

    mask = np.zeros((m, n), dtype=bool)
    sorted_indices = np.argsort(proba, axis=1)
    mask[np.arange(m)[:, None], sorted_indices[:, -k:]] = True

    proba[~mask] = 0

    remaining_mass = 1 - p*n

    proba = proba * remaining_mass / np.sum(proba, axis=1, keepdims=True)

    return (proba + p)


class PropensityPolicy():
    def __init__(self, estimator, s=0, k=1, p=0, use_outcomes=False):
        self.estimator = estimator
        self.s = s
        self.k = k
        self.p = p
        self.use_outcomes = use_outcomes

    def _modify_predict_proba(self, estimator):
        _, estimator = extract_preprocessor_and_estimator(estimator)
        if not hasattr(estimator, "estimator_s_"):
            raise ValueError(
                "The estimator must have an attribute `estimator_s_`."
            )
        original_predict_proba = estimator.estimator_s_.predict_proba
        s = self.s
        def new_predict_proba(self, X):
            yp = original_predict_proba(X)
            yp[:, 1] = np.minimum(1, yp[:, 1] + s)
            yp[:, 0] = 1 - yp[:, 1]
            return yp
        estimator.estimator_s_.predict_proba = types.MethodType(
            new_predict_proba, estimator.estimator_s_
        )

    def _outcome_selection(self, proba, outcome_per_class):
        proba_out = np.zeros_like(proba)
        for i, (p, o) in enumerate(zip(proba, outcome_per_class)):
            mask = p > 0
            try:
                best_idx = np.nanargmax(o[mask])  # [0, sum(mask) - 1]
            except ValueError:
                proba_out[i] = p
                continue
            best_class = np.where(mask)[0][best_idx]  # [0, n_classes - 1]
            proba_out[i, best_class] = 1.0
        return proba_out

    def predict_proba(self, inputs, **predict_params):
        estimator = copy.deepcopy(self.estimator)

        if self.s > 0:
            # Increase the likelihood of switching treatment.
            self._modify_predict_proba(estimator)

        # TODO: If the estimator includes a calibration step, the effect of
        # increasing the probaility of switching is likely reduced. Should we
        # consider raw predictions instead of calibrated ones?
        proba = estimator.predict_proba(inputs, **predict_params)
        proba = keep_top_proba_and_renormalize(proba, self.k)

        if self.use_outcomes:
            preprocessor, clf = extract_preprocessor_and_estimator(estimator)
            if not hasattr(clf, "outcome_per_class"):
                raise ValueError(
                    "The estimator must have an attribute `outcome_per_class`."
                )

            if preprocessor is not None:
                inputs = preprocessor.transform(inputs)
            outcome_per_class = clf.outcome_per_class(inputs, **predict_params)

            proba = self._outcome_selection(proba, outcome_per_class)

        proba = keep_top_proba_and_renormalize(proba, proba.shape[1], self.p)

        assert np.allclose(proba.sum(axis=1), 1)

        return proba


class RandomPolicy():
    def __init__(self, n_actions, p=0.01):
        self.n_actions = n_actions
        self.p = p

    def predict_proba(self, inputs):
        n_inputs = len(inputs)
        proba = np.full((n_inputs, self.n_actions), self.p / (self.n_actions - 1))
        chosen_actions = np.random.randint(low=0, high=self.n_actions, size=n_inputs)
        proba[np.arange(n_inputs), chosen_actions] = 1 - self.p
        return proba


class RLPolicySepsis():
    def __init__(
        self,
        preprocessor,
        n_clusters_list,
        transition_threshold=None,
        random_state=None,
        discount=0.99,
        p=0,
    ):
        self.preprocessor = preprocessor
        self.n_clusters_list = n_clusters_list
        self.transition_threshold = transition_threshold
        self.random_state = random_state
        self.discount = discount
        self.p = p

    def fit(self, dataset):
        self.preprocessor_ = self.preprocessor.fit(dataset.inputs)

        inputs = self.preprocessor_.transform(dataset.inputs)
        actions = dataset.targets
        rewards = dataset.outcomes
        groups = dataset.groups

        self.clusterings_ = []
        for n_clusters in self.n_clusters_list:
            kmeans = KMeans(
                n_clusters=n_clusters,
                init="k-means++",
                n_init=1,
                verbose=1,
                random_state=self.random_state,
            )
            kmeans.fit(inputs)
            self.clusterings_.append(kmeans)

        if not self.clusterings_:
            raise ValueError("No valid clusterings were found.")

        idx_best_clustering = np.argmin(
            [clustering.inertia_ for clustering in self.clusterings_]
        )
        self.best_clustering_ = self.clusterings_[idx_best_clustering]

        self.n_states_ = self.best_clustering_.n_clusters + 2
        self.n_actions_ = len(set(actions))

        states = self.best_clustering_.predict(inputs)
        next_states = pd.Series(states).groupby(groups).shift(-1, fill_value=-1).values

        T = np.zeros((self.n_states_, self.n_actions_, self.n_states_))
        C = np.zeros((self.n_states_, self.n_actions_))

        for s, a, s_next, r in zip(states, actions, next_states, rewards):
            if s_next == -1:
                s_next = self.n_states_ - 2 if r < 0 else self.n_states_ - 1
            T[s, a, s_next] += 1
            C[s, a] += 1

        # Exclude infrequent transitions.
        if self.transition_threshold is not None:
            mask = C <= self.transition_threshold
            T[mask] = C[mask] = 0

        # Normalize the transition matrix.
        for s in range(self.n_states_):
            for a in range(self.n_actions_):
                if C[s, a] > 0:
                    T[s, a] /= C[s, a]
                else:
                    T[s, a] = np.full(self.n_states_, 1 / self.n_states_)

        # Define the reward matrix.
        R = np.zeros((self.n_states_, self.n_actions_, self.n_states_))
        R[:, :, self.n_states_ - 2] = -100
        R[:, :, self.n_states_ - 1] = 100
        R = np.sum(T * R, axis=2)

        # Permute the transition matrix to match the MDP toolbox format.
        T = np.transpose(T, axes=(1, 0, 2))

        # Run Q-learning.
        self.qlearning_ = QLearning(
            transitions=T,
            reward=R,
            discount=self.discount,
            n_iter=10_000,
        )
        self.qlearning_.run()

        return self

    def predict_proba(self, inputs):
        assert hasattr(self, "qlearning_"), "There is no policy learned!"
        optimal_actions = np.array(self.qlearning_.policy)
        inputs = self.preprocessor_.transform(inputs)
        states = self.best_clustering_.predict(inputs)
        proba = (
            optimal_actions[states][:, None] == np.arange(self.n_actions_)
        ).astype(int)
        return keep_top_proba_and_renormalize(proba, 1, self.p)


class RLPolicyRA():
    def __init__(
        self,
        preprocessor,
        n_clusters_list,
        transition_threshold=None,
        random_state=None,
        discount=0.99,
        p=0,
    ):
        self.preprocessor = preprocessor
        self.n_clusters_list = n_clusters_list
        self.transition_threshold = transition_threshold
        self.random_state = random_state
        self.discount = discount
        self.p = p

    def _summarize_comorbidities(self, inputs):
        """Summarize comorbidities and past comorbidities."""
        c_comor = [c for c in inputs.columns if "__comor" in c]
        c_hxcomor = [c for c in inputs.columns if "__hxcomor" in c]
        inputs["comor_count"] = inputs[c_comor].sum(axis=1)
        inputs["hxcomor_count"] = inputs[c_hxcomor].sum(axis=1)
        return inputs.drop(c_comor + c_hxcomor, axis=1)

    def fit(self, dataset):
        self.preprocessor_ = self.preprocessor.fit(dataset.inputs)

        inputs = pd.DataFrame(
            self.preprocessor_.transform(dataset.inputs),
            columns=self.preprocessor_.get_feature_names_out(),
        )
        inputs = self._summarize_comorbidities(inputs)
        actions = dataset.targets
        rewards = dataset.outcomes
        groups = dataset.groups

        self.categorical_ = [
            i for i, c in enumerate(inputs.columns) if not (
                c.startswith("numerical")
                or c.startswith("comor_count")
                or c.startswith("hxcomor_count")
            )
        ]

        self.clusterings_ = []
        for n_clusters in self.n_clusters_list:
            kprototypes = KPrototypes(
                n_clusters=n_clusters,
                init="Cao",
                n_init=1,
                verbose=1,
                random_state=self.random_state,
                n_jobs=-1,
            )
            try:
                kprototypes.fit(inputs.values, categorical=self.categorical_)
                self.clusterings_.append(kprototypes)
            except ValueError:
                pass

        if not self.clusterings_:
            raise ValueError("No valid clusterings were found.")

        idx_best_clustering = np.argmin(
            [clustering.cost_ for clustering in self.clusterings_]
        )
        self.best_clustering_ = self.clusterings_[idx_best_clustering]

        self.n_states_ = self.best_clustering_.n_clusters
        self.n_actions_ = len(set(actions))

        states = self.best_clustering_.predict(inputs.values, categorical=self.categorical_)
        next_states = pd.Series(states).groupby(groups).shift(-1, fill_value=-1).values

        T = np.zeros((self.n_states_, self.n_actions_, self.n_states_))
        R = np.zeros((self.n_states_, self.n_actions_))
        C = np.zeros_like(R)  # Frequency of state-action pairs

        for s, a, s_next, r in zip(states, actions, next_states, rewards):
            if s_next == -1:
                continue
            T[s, a, s_next] += 1
            R[s, a] += r
            C[s, a] += 1

        # Exclude infrequent transitions.
        if self.transition_threshold is not None:
            mask = C <= self.transition_threshold
            T[mask] = C[mask] = 0

        # Normalize the transition matrix.
        for s in range(self.n_states_):
            for a in range(self.n_actions_):
                if C[s, a] > 0:
                    T[s, a] /= C[s, a]
                else:
                    T[s, a] = np.full(self.n_states_, 1 / self.n_states_)

        # Compute the average reward for each state-action pair.
        R = np.divide(R, C, where=C > 0)

        # Permute the transition matrix to match the MDP toolbox format.
        T = np.transpose(T, axes=(1, 0, 2))

        # Run Q-learning.
        self.qlearning_ = QLearning(
            transitions=T,
            reward=R,
            discount=self.discount,
            n_iter=10_000,
        )
        self.qlearning_.run()

        return self

    def predict_proba(self, inputs):
        assert hasattr(self, "qlearning_"), "There is no policy learned!"
        optimal_actions = np.array(self.qlearning_.policy)
        inputs = pd.DataFrame(
            self.preprocessor_.transform(inputs),
            columns=self.preprocessor_.get_feature_names_out(),
        )
        inputs = self._summarize_comorbidities(inputs)
        states = self.best_clustering_.predict(inputs.values, categorical=self.categorical_)
        proba = (
            optimal_actions[states][:, None] == np.arange(self.n_actions_)
        ).astype(int)
        return keep_top_proba_and_renormalize(proba, 1, self.p)


class OPEResult():
    def __init__(
        self,
        weights,
        returns,
        weighted_returns,
        n_stages,
        threshold=100,
    ):
        self.weights = weights
        self.returns = returns
        self.weighted_returns = weighted_returns
        self.n_stages = n_stages
        self.threshold = threshold

        # Vanilla IS.
        self.is_estimate_ = self.is_estimate()

        # Weighted IS.
        self.wis_estimate_ = self.wis_estimate()

        # Truncated weighted IS.
        self.twis_estimate_ = self.twis_estimate()

        # Per-decision IS.
        self.pdis_estimate_ = self.pdis_estimate()

        # Weighted per-decision IS.
        self.wpdis_estimate_ = None

        # Effective sample size.
        self.ess_ = np.sum(weights) ** 2 / np.sum(weights ** 2)

    def is_estimate(self):
        return (
            np.sum(np.multiply(self.weights, self.returns) / self.n_stages)
            / len(self.weights)
        )

    def wis_estimate(self):
        return (
            np.sum(np.multiply(self.weights, self.returns) / self.n_stages)
            / np.sum(self.weights)
        )

    def twis_estimate(self):
        weights_ = np.minimum(self.weights, self.threshold)
        return (
            np.sum(np.multiply(weights_, self.returns) / self.n_stages)
            / np.sum(weights_)
        )

    def pdis_estimate(self):
        return (
            np.sum(self.weighted_returns / self.n_stages)
            / len(self.weights)
        )

    def wpdis_estimate(self, per_step_weights):
        weights = np.array(per_step_weights)

        # Avoid division by zero.
        weight_sum = np.sum(weights)
        if weight_sum == 0:
            return 0.0

        return (
            np.sum(self.weighted_returns / self.n_stages)
            / weight_sum
        )

    @classmethod
    def from_dataframe(cls, df):
        weights = df["weights"].to_numpy()
        returns = df["returns"].to_numpy()
        weighted_returns = df["weighted_returns"].to_numpy()
        n_stages = df["n_stages"].to_numpy()
        assert df["threshold"].nunique() == 1
        threshold = df["threshold"].unique()[0]
        return cls(weights, returns, weighted_returns, n_stages, threshold)


class OPEEstimator:
    def __init__(self, target_policy, behavior_policy_estimator):
        self.pi = target_policy
        self.mu = behavior_policy_estimator

    def _check_dataset_and_params(self, dataset, params_mu, params_pi):
        if not isinstance(dataset, OPEDataset):
            raise ValueError("Data must be an instance of `OPEDataset`.")
        input_size = len(dataset.inputs)
        message = "Parameters {} must have the same length as the inputs."
        if params_mu is not None:
            if any(len(p) != input_size for p in params_mu.values()):
                raise ValueError(message.format("`params_mu`"))
        if params_pi is not None:
            if any(len(p) != input_size for p in params_pi.values()):
                raise ValueError(message.format("`params_pi`"))

    def _precompute_proba(self, dataset, params_mu=None, params_pi=None):
        self._check_dataset_and_params(dataset, params_mu, params_pi)
        params_mu_ = params_mu if params_mu is not None else {}
        params_pi_ = params_pi if params_pi is not None else {}
        proba_mu = self.mu.predict_proba(dataset.inputs, **params_mu_)
        proba_pi = self.pi.predict_proba(dataset.inputs, **params_pi_)
        self.proba_mu_ = pd.DataFrame(proba_mu, index=dataset.groups)
        self.proba_pi_ = pd.DataFrame(proba_pi, index=dataset.groups)

    def _get_proba(self, dataset, original_groups=None):
        original_groups = dataset.groups.unique() if original_groups is None else original_groups
        rows, columns = dataset.inputs.index, dataset.targets
        proba_pi = self.proba_pi_.loc[original_groups].values[rows, columns]
        proba_mu = self.proba_mu_.loc[original_groups].values[rows, columns]
        return proba_pi, proba_mu

    def _collect_weights_and_returns(self, dataset, original_groups=None):
        outcomes, groups = dataset.outcomes, dataset.groups
        proba_pi, proba_mu = self._get_proba(dataset, original_groups)
        ratios = np.divide(proba_pi, proba_mu)
        cum_weights = pd.Series(ratios).groupby(groups).cumprod()
        weights = cum_weights.groupby(groups).last()
        returns = pd.Series(outcomes).groupby(groups).sum()
        weighted_returns = (cum_weights * outcomes).groupby(groups).sum()
        n_stages = groups.value_counts().reindex(groups.unique())
        return (
            weights.to_numpy(),
            returns.to_numpy(),
            weighted_returns.to_numpy(),
            n_stages,
            cum_weights,  # per-step weights
        )

    def estimate(self, dataset, params_mu=None, params_pi=None, original_groups=None):
        if original_groups is None:
            self._precompute_proba(dataset, params_mu, params_pi)
            weights, returns, weighted_returns, n_stages, per_step_weights = (
                self._collect_weights_and_returns(dataset)
            )
        else:
            assert hasattr(self, "proba_mu_") and hasattr(self, "proba_pi_")
            weights, returns, weighted_returns, n_stages, per_step_weights = (
                self._collect_weights_and_returns(dataset, original_groups)
            )
        ope_result = OPEResult(weights, returns, weighted_returns, n_stages)
        ope_result.wpdis_estimate_ = ope_result.wpdis_estimate(per_step_weights)
        return ope_result
