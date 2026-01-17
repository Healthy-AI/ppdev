import warnings
from functools import partial
from abc import ABCMeta, abstractmethod

import torch
import numpy as np

import skorch

import sklearn.calibration as calibration
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.calibration import IsotonicRegression, _SigmoidCalibration
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.utils import indexable, check_random_state, column_or_1d
from sklearn.utils.multiclass import type_of_target, check_classification_targets
from sklearn.utils.validation import check_is_fitted, _check_sample_weight
from sklearn.utils.metadata_routing import (
    process_routing,
    MetadataRouter,
    MethodMapping,
    _MetadataRequester,
)

from .metrics import sce
from .data import OPEDataset
from .utils import pad_pack_sequences, seed_torch, Dataset
from .tree import ExtendedTree

MAX_INT = np.iinfo(np.int32).max


def expects_groups(estimator):
    return hasattr(estimator, "dataset") and estimator.dataset == OPEDataset


def get_estimator(
    estimator_alias,
    output_dir=None,
    output_dim=None,
    random_state=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if estimator_alias == "dummy_most_frequent":
        return (
            CustomDummyClassifier(strategy="most_frequent")
            .set_fit_request(sample_weight=True)
            .set_score_request(sample_weight=True, metric=True)
        )

    elif estimator_alias == "dummy_prior":
        return (
            CustomDummyClassifier(strategy="prior")
            .set_fit_request(sample_weight=True)
            .set_score_request(sample_weight=True, metric=True)
        )

    elif estimator_alias == "dt":
        return (
            CustomDecisionTreeClassifier(random_state=random_state)
            .set_fit_request(o=True, sample_weight=True)
            .set_score_request(sample_weight=True, metric=True, mask=True)
        )

    elif estimator_alias == "dt_stage":
        return (
            StageTherapyClassifier(
                estimator_bl=CustomDecisionTreeClassifier().set_fit_request(o=True, sample_weight=True),
                estimator_fu=CustomDecisionTreeClassifier().set_fit_request(o=True, sample_weight=True),
                random_state=random_state,
            )
            .set_fit_request(stages=True, o=True, sample_weight=True)
            .set_predict_request(stages=True)
            .set_predict_proba_request(stages=True)
            .set_score_request(stages=True, sample_weight=True, metric=True, mask=True)
        )
    
    elif estimator_alias == "dt_switch":
        return (
            SwitchTherapyClassifier(
                estimator_s=CustomDecisionTreeClassifier().set_fit_request(o=True, sample_weight=True),
                estimator_t=CustomDecisionTreeClassifier().set_fit_request(o=True, sample_weight=True),
                random_state=random_state,
            )
            .set_fit_request(y_prev=True, o=True, sample_weight=True)
            .set_predict_request(y_prev=True)
            .set_predict_proba_request(y_prev=True)
            .set_score_request(y_prev=True, sample_weight=True, metric=True, mask=True)
        )

    elif estimator_alias == "dt_stage_switch":
        return (
            StageSwitchTherapyClassifier(
                estimator_bl=CustomDecisionTreeClassifier().set_fit_request(o=True, sample_weight=True),
                estimator_s=CustomDecisionTreeClassifier().set_fit_request(o=True, sample_weight=True),
                estimator_t=CustomDecisionTreeClassifier().set_fit_request(o=True, sample_weight=True),
                random_state=random_state,
            )
            .set_fit_request(stages=True, y_prev=True, o=True, sample_weight=True)
            .set_predict_request(stages=True, y_prev=True)
            .set_predict_proba_request(stages=True, y_prev=True)
            .set_score_request(stages=True, y_prev=True, sample_weight=True, metric=True, mask=True)
        )

    elif estimator_alias == "mlp":
        return (
            NeuralNetClassifier(
                seed=random_state,
                module=MLPClassifier,
                module__output_dim=output_dim,
                criterion=torch.nn.CrossEntropyLoss,
                optimizer=torch.optim.Adam,
                optimizer__lr=0.001,
                max_epochs=100,
                batch_size=32,
                iterator_train__shuffle=True,
                dataset=Dataset,
                train_split=None,
                callbacks=[
                    skorch.callbacks.EarlyStopping(patience=10),
                    skorch.callbacks.Checkpoint(
                        f_optimizer=None,
                        f_criterion=None,
                        fn_prefix="mlp_",
                        dirname=output_dir,
                        load_best=True,
                    ),
                ],
                verbose=1,
                device=device,
            )
            .set_fit_request(X_valid=True, y_valid=True)
            .set_score_request(sample_weight=True, metric=True, mask=True)
        )

    elif estimator_alias == "rnn":
        return (
            NeuralNetClassifier(
                seed=random_state,
                module=RNNClassifier,
                module__output_dim=output_dim,
                criterion=torch.nn.CrossEntropyLoss,
                optimizer=torch.optim.Adam,
                optimizer__lr=0.001,
                max_epochs=100,
                batch_size=32,
                iterator_train__shuffle=True,
                iterator_train__collate_fn=pad_pack_sequences,
                iterator_valid__collate_fn=pad_pack_sequences,
                dataset=OPEDataset,
                train_split=None,
                callbacks=[
                    skorch.callbacks.EarlyStopping(patience=10),
                    skorch.callbacks.Checkpoint(
                        f_optimizer=None,
                        f_criterion=None,
                        fn_prefix="rnn_",
                        dirname=output_dir,
                        load_best=True,
                    ),
                ],
                verbose=1,
                device=device,
            )
            .set_fit_request(groups=True, X_valid=True, y_valid=True, groups_valid=True)
            .set_predict_request(groups=True)
            .set_predict_proba_request(groups=True)
            .set_score_request(groups=True, sample_weight=True, metric=True, mask=True)
        )

    else:
        raise ValueError(f"Unknown estimator alias: {estimator_alias}.")


# =============================================================================
# == Mixin class for all classifiers. =========================================
# =============================================================================

class ClassifierMixin:
    _estimator_type = "classifier"

    accepted_metrics = {
        "accuracy": {
            "function": metrics.accuracy_score,
            "requires_proba": False,
            "kwargs": {},
            "lower_is_better": False,
        },
        "roc_auc_ovr": {
            "function": metrics.roc_auc_score,
            "requires_proba": True,
            "kwargs": {"multi_class": "ovr", "average": "macro"},
            "lower_is_better": False,
        },
        "roc_auc_ovr_weighted": {
            "function": metrics.roc_auc_score,
            "requires_proba": True,
            "kwargs": {"multi_class": "ovr", "average": "weighted"},
            "lower_is_better": False,
        },
        "neg_log_loss": {
            "function": metrics.log_loss,
            "requires_proba": True,
            "kwargs": {"normalize": True},
            "lower_is_better": True,
        },
        "sce": {
            "function": sce,
            "requires_proba": True,
            "kwargs": {},
            "lower_is_better": True,
            "one_hot_y": True,
        },
    }

    def score(self, X, y, metric="accuracy", sample_weight=None, mask=None, **predict_params):
        if not metric in self.accepted_metrics:
            raise ValueError(
                f"Got invalid metric {metric}. "
                f"Valid metrics are {self.accepted_metrics.keys()}."
            )

        score_params = self.accepted_metrics[metric]["kwargs"]
        if sample_weight is not None:
            score_params["sample_weight"] = sample_weight if mask is None \
                else sample_weight[mask]

        if self.accepted_metrics[metric].get("one_hot_y", False):
            n_classes = len(self.classes_)
            y = np.eye(n_classes)[y]

        if self.accepted_metrics[metric]["requires_proba"]:
            yp = self.predict_proba(X, **predict_params)
        else:
            yp = self.predict(X, **predict_params)
        
        if mask is not None:
            y = y[mask]
            yp = yp[mask]

        try:
            return self.accepted_metrics[metric]["function"](y, yp, **score_params)
        except ValueError:
            return np.nan

    def _more_tags(self):
        return {"requires_y": True}


# =============================================================================
# == scikit-learn classifiers. ================================================
# =============================================================================

class CustomDecisionTreeClassifier(ClassifierMixin, DecisionTreeClassifier):
    """Decision tree classifier."""

    def fit(self, X, y, o=None, sample_weight=None, check_input=True):
        super().fit(X, y, sample_weight, check_input)
        if o is not None:
            self.tree_ = ExtendedTree(
                self.n_features_in_, self.n_classes_, tree=self.tree_,
            )
            X = self._validate_X_predict(X, check_input)
            self.tree_.store_outcomes(X, y, o)
        return self


class CustomDummyClassifier(ClassifierMixin, DummyClassifier):
    """Dummy classifier."""


# =============================================================================
# == Custom classifiers for treatment prediction. =============================
# =============================================================================

def _set_random_states(estimator, random_state=None):
    """Set all `random_state` parameters of an estimator.
    See: github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_base.py
    """
    random_state = check_random_state(random_state)
    to_set = {}
    for key in sorted(estimator.get_params(deep=True)):
        if key == "random_state" or key.endswith("__random_state"):
            to_set[key] = random_state.randint(np.iinfo(np.int32).max)
    if to_set:
        estimator.set_params(**to_set)


class BaseTherapyClassifier(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, random_state=None):
        self.random_state = random_state

    @property
    @abstractmethod
    def _estimator_names(self):
        pass

    @property
    @abstractmethod
    def outcome_per_class(self):
        pass

    def _check_estimators(self):
        for estimator_name in self._estimator_names:
            if getattr(self, estimator_name) is None:
                raise ValueError(
                    f"Estimator `{estimator_name}` cannot be `None`."
                )
    
    def _make_estimator(self, estimator_name, random_state=None):
        estimator = clone(getattr(self, estimator_name))
        if random_state is not None:
            _set_random_states(estimator, random_state)
        return estimator

    def _validate_y(self, y):
        y = column_or_1d(y, warn=True)
        y_type = type_of_target(y, input_name="y")
        if y_type not in ["binary", "multiclass"]:
            raise ValueError(
                f"Unknown target type: {y_type}."
                "Valid target types are: 'binary' and 'multiclass'."
            )
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        return y

    def _fit(self, X, y, input_transformer, **fit_params):
        self._check_estimators()
        random_state = check_random_state(self.random_state)
        y = self._validate_y(y)

        routed_params = process_routing(self, "fit", **fit_params)

        n_estimators = len(self._estimator_names)
        seeds = random_state.randint(MAX_INT, size=(n_estimators))
        for i, estimator_name in enumerate(self._estimator_names):
            estimator = self._make_estimator(estimator_name, random_state=seeds[i])
            fit_params_ = routed_params.get(estimator_name).fit.copy()
            o = fit_params_.pop("o", None)
            sw = fit_params_.pop("sample_weight", None)
            Xi, yi, oi, swi = input_transformer(
                estimator_name, X, y, o=o, sample_weight=sw
            )
            # TODO: Check if estimator takes `o` as input.
            fit_params_.update(dict(o=oi, sample_weight=swi))
            estimator.fit(Xi, yi, **fit_params_)
            setattr(self, f"{estimator_name}_", estimator)

        return self

    def get_metadata_routing(self):
        router = MetadataRouter(owner=self.__class__.__name__)
        for estimator_name in self._estimator_names:
            router.add(
                **{estimator_name: getattr(self, estimator_name)},
                method_mapping=MethodMapping()
                .add(caller="fit", callee="fit"),
            )
        router.add_self_request(self)
        return router

    def _get_ccp_paths(self, X, y, input_transformer, sample_weight=None):
        self._check_estimators()
        random_state = check_random_state(self.random_state)
        y = self._validate_y(y)

        ccp_paths = {}

        n_estimators = len(self._estimator_names)
        seeds = random_state.randint(MAX_INT, size=(n_estimators))
        for i, estimator_name in enumerate(self._estimator_names):
            estimator = self._make_estimator(estimator_name, random_state=seeds[i])
            if hasattr(estimator, "cost_complexity_pruning_path"):
                Xi, yi, _oi, swi = input_transformer(
                    estimator_name, X, y, sample_weight=sample_weight
                )
                path = estimator.cost_complexity_pruning_path(Xi, yi, swi)
                ccp_paths[estimator_name] = path

        return ccp_paths


class StageTherapyClassifier(ClassifierMixin, BaseTherapyClassifier):
    _estimator_names = ["estimator_bl", "estimator_fu"]

    def __init__(
        self,
        estimator_bl=CustomDecisionTreeClassifier(),  # Baseline estimator
        estimator_fu=CustomDecisionTreeClassifier(),  # Post-baseline estimator
        random_state=None,
    ):
        super().__init__(random_state=random_state)
        self.estimator_bl = estimator_bl
        self.estimator_fu = estimator_fu

    def outcome_per_class(self, X, stages):
        check_is_fitted(self)
        is_baseline = stages == 1
        leaves_bl = self.estimator_bl_.apply(X[is_baseline])
        leaves_fu = self.estimator_fu_.apply(X[~is_baseline])
        out = np.full((X.shape[0], self.n_classes_), np.nan)
        out[np.ix_(is_baseline, self.estimator_bl_.classes_)] = \
            self.estimator_bl_.tree_.outcome_per_class[leaves_bl]
        out[~is_baseline] = self.estimator_fu_.tree_.outcome_per_class[leaves_fu]
        return

    def _prepare_fit_inputs(self, stages, estimator_name, X, y, o=None, sample_weight=None):
        include = (
            stages == 1 if estimator_name == "estimator_bl" else stages != 1
        )
        X = X[include]
        y = y[include]
        if o is not None:
            o = o[include]
        if sample_weight is not None:
            sample_weight = sample_weight[include]
        return X, y, o, sample_weight

    def fit(self, X, y, *, stages, o=None, sample_weight=None):
        input_transformer = partial(self._prepare_fit_inputs, stages)
        return super()._fit(
            X, y, input_transformer, o=o, sample_weight=sample_weight
        )

    def cost_complexity_pruning_path(self, X, y, *, stages, sample_weight=None):
        input_transformer = partial(self._prepare_fit_inputs, stages)
        return super()._get_ccp_paths(
            X, y, input_transformer, sample_weight=sample_weight
        )

    def predict(self, X, *, stages):
        check_is_fitted(self)
        proba = self.predict_proba(X, stages=stages)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X, *, stages):
        check_is_fitted(self)
        proba = np.zeros((X.shape[0], self.n_classes_))
        is_baseline = stages == 1
        proba[np.ix_(is_baseline, self.estimator_bl_.classes_)] = \
            self.estimator_bl_.predict_proba(X[is_baseline])
        proba[~is_baseline] = self.estimator_fu_.predict_proba(X[~is_baseline])
        assert np.allclose(proba.sum(axis=1), np.ones(X.shape[0]))
        return proba

    def score(self, X, y, *, stages, metric, sample_weight=None, mask=None):
        return super().score(
            X,
            y,
            stages=stages,
            metric=metric,
            sample_weight=sample_weight,
            mask=mask,
        )


class SwitchTherapyClassifier(ClassifierMixin, BaseTherapyClassifier):
    _estimator_names = ["estimator_s", "estimator_t"]

    def __init__(
        self,
        estimator_s=CustomDecisionTreeClassifier(),  # Switch estimator
        estimator_t=CustomDecisionTreeClassifier(),  # Therapy estimator
        random_state=None,
    ):
        super().__init__(random_state=random_state)
        self.estimator_s = estimator_s
        self.estimator_t = estimator_t

    def outcome_per_class(self, X, y_prev):
        check_is_fitted(self)
        leaves_s = self.estimator_s_.apply(X)
        leaves_t = self.estimator_t_.apply(X)
        out = np.full((X.shape[0], self.n_classes_), np.nan)
        for i, (ls, lt, yp) in enumerate(zip(leaves_s, leaves_t, y_prev)):
            # TODO: How can we integrate the outcome of patients who switch?
            out[i] = self.estimator_t_.tree_.outcome_per_class[lt]
            out[i, yp] = self.estimator_s_.tree_.outcome_per_class[ls, 0]  # First class is stay
        return out

    def _prepare_fit_inputs(self, y_prev, estimator_name, X, y, o=None, sample_weight=None):
        y_switch = 1 * (y_prev != y)
        if estimator_name == "estimator_s":
            return X, y_switch, o, sample_weight
        else:
            switch = y_switch > 0
            if o is not None:
                o = o[switch]
            if sample_weight is not None:
                sample_weight = sample_weight[switch]
            return X[switch], y[switch], o, sample_weight

    def fit(self, X, y, *, y_prev, o=None, sample_weight=None):
        input_transformer = partial(self._prepare_fit_inputs, y_prev)
        return super()._fit(
            X, y, input_transformer, o=o, sample_weight=sample_weight
        )

    def cost_complexity_pruning_path(self, X, y, *, y_prev, sample_weight=None):
        input_transformer = partial(self._prepare_fit_inputs, y_prev)
        return super()._get_ccp_paths(
            X, y, input_transformer, sample_weight=sample_weight
        )

    def predict_proba(self, X, y_prev):
        check_is_fitted(self)

        # One-hot encode the previous treatment.
        y_prev = np.eye(self.n_classes_)[y_prev]

        # Make predictions.
        y_sp = self.estimator_s_.predict_proba(X)[:, 1, np.newaxis]
        y_tp = self.estimator_t_.predict_proba(X)

        # Remove probability of the previous treatment and renormalize.
        #
        # TODO: Use a prior instead of equal probabilities below.
        y_tp = (1-y_prev) * y_tp
        mask = y_tp.sum(axis=1) == 0
        # Assign equal probability to all treatments except the previous one.
        y_tp[mask] = 1 - y_prev[mask]
        y_tp = y_tp / y_tp.sum(axis=1, keepdims=True)

        # Mix in the probability of staying.
        y_p = (1-y_sp)*y_prev + y_sp*y_tp

        # Check that the probabilities sum to one.
        y_p_sum = y_p.sum(axis=1)
        assert np.allclose(y_p_sum, np.ones_like(y_p_sum))

        return y_p
    
    def predict(self, X, y_prev):
        check_is_fitted(self)
        yp = self.predict_proba(X, y_prev)
        return np.argmax(yp, axis=1)
    
    def score(self, X, y, *, y_prev, metric, sample_weight=None, mask=None):
        return super().score(
            X,
            y,
            y_prev=y_prev,
            metric=metric,
            sample_weight=sample_weight,
            mask=mask,
        )


class StageSwitchTherapyClassifier(ClassifierMixin, BaseTherapyClassifier):
    _estimator_names = ["estimator_bl", "estimator_s", "estimator_t"]

    def __init__(
        self,
        estimator_bl=CustomDecisionTreeClassifier(),  # Baseline estimator
        estimator_s=CustomDecisionTreeClassifier(),  # Switch estimator
        estimator_t=CustomDecisionTreeClassifier(),  # Therapy estimator
        random_state=None,
    ):
        super().__init__(random_state=random_state)
        self.estimator_bl = estimator_bl
        self.estimator_s = estimator_s
        self.estimator_t = estimator_t

    def outcome_per_class(self, X, stages, y_prev):
        check_is_fitted(self)
        leaves_bl = self.estimator_bl_.apply(X)
        leaves_s = self.estimator_s_.apply(X)
        leaves_t = self.estimator_t_.apply(X)
        out = np.full((X.shape[0], self.n_classes_), np.nan)
        iterables = zip(leaves_bl, leaves_s, leaves_t, stages, y_prev)
        for i, (lbl, ls, lt, s, yp) in enumerate(iterables):
            if s == 1:
                out[i, self.estimator_bl_.classes_] = (
                    self.estimator_bl_.tree_.outcome_per_class[lbl]
                )
            else:
                out[i] = self.estimator_t_.tree_.outcome_per_class[lt]
                out[i, yp] = self.estimator_s_.tree_.outcome_per_class[ls, 0]
        return out

    def _prepare_fit_inputs(self, stages, y_prev, estimator_name, X, y, o=None, sample_weight=None):
        include = (
            stages == 1 if estimator_name == "estimator_bl" else stages != 1
        )

        y_prev = y_prev[include]
        X = X[include]
        y = y[include]
        if o is not None:
            o = o[include]
        if sample_weight is not None:
            sample_weight = sample_weight[include]

        if estimator_name == "estimator_bl":
            return X, y, o, sample_weight

        y_switch = 1 * (y_prev != y)
        if estimator_name == "estimator_s":
            return X, y_switch, o, sample_weight
        else:
            switch = y_switch > 0
            if o is not None:
                o = o[switch]
            if sample_weight is not None:
                sample_weight = sample_weight[switch]
            return X[switch], y[switch], o, sample_weight

    def fit(self, X, y, *, stages, y_prev, o=None, sample_weight=None):
        input_transformer = partial(self._prepare_fit_inputs, stages, y_prev)
        return super()._fit(
            X, y, input_transformer, o=o, sample_weight=sample_weight
        )

    def cost_complexity_pruning_path(self, X, y, *, stages, y_prev, sample_weight=None):
        input_transformer = partial(self._prepare_fit_inputs, stages, y_prev)
        return super()._get_ccp_paths(
            X, y, input_transformer, sample_weight=sample_weight
        )

    def predict(self, X, *, stages, y_prev):
        check_is_fitted(self)
        proba = self.predict_proba(X, stages=stages, y_prev=y_prev)
        return np.argmax(proba, axis=1)

    def _predict_proba_fu(self, X, y_prev):
        # One-hot encode the previous treatment.
        y_prev = np.eye(self.n_classes_)[y_prev]

        # Make predictions.
        y_sp = self.estimator_s_.predict_proba(X)[:, 1, np.newaxis]
        y_tp = self.estimator_t_.predict_proba(X)

        # Remove probability of the previous treatment and renormalize.
        y_tp = (1-y_prev) * y_tp
        mask = y_tp.sum(axis=1) == 0
        y_tp[mask] = 1 - y_prev[mask]  # Assign equal probability to all treatments except the previous one
        y_tp = y_tp / y_tp.sum(axis=1, keepdims=True)

        # Mix in the probability of staying.
        y_p = (1-y_sp)*y_prev + y_sp*y_tp

        # Check that the probabilities sum to one.
        y_p_sum = y_p.sum(axis=1)
        assert np.allclose(y_p_sum, np.ones_like(y_p_sum))

        return y_p

    def predict_proba(self, X, *, stages, y_prev):
        check_is_fitted(self)
        proba = np.zeros((X.shape[0], self.n_classes_))
        is_baseline = stages == 1
        # TODO: Handle `is_baseline.sum() == 0`` and `~is_baseline.sum() == 0`.
        proba[np.ix_(is_baseline, self.estimator_bl_.classes_)] = (
            self.estimator_bl_.predict_proba(X[is_baseline])
        )
        proba[~is_baseline] = self._predict_proba_fu(
            X[~is_baseline], y_prev[~is_baseline]
        )
        assert np.allclose(proba.sum(axis=1), np.ones(X.shape[0]))
        return proba

    def score(self, X, y, *, stages, y_prev, metric, sample_weight=None, mask=None):
        return super().score(
            X,
            y,
            stages=stages,
            y_prev=y_prev,
            metric=metric,
            sample_weight=sample_weight,
            mask=mask,
        )


# =============================================================================
# == Neural network classifiers. ==============================================
# =============================================================================

class LazyRNNEncoder(torch.nn.Module):
    def __init__(
        self,
        output_dim=64,
        num_layers=1,
        nonlinearity="tanh",
    ):
        super(LazyRNNEncoder, self).__init__()

        self.output_dim = output_dim
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.rnn = None

    def forward(self, inputs):
        assert isinstance(inputs, torch.nn.utils.rnn.PackedSequence)
        if self.rnn is None:
            input_dim = inputs.data.size(-1)
            self.rnn = torch.nn.RNN(
                input_dim,
                self.output_dim,
                num_layers=self.num_layers,
                nonlinearity=self.nonlinearity,
                batch_first=True,
                device=inputs.data.device,
            )
        encodings, _ = self.rnn(inputs)
        encodings = torch.nn.utils.rnn.unpack_sequence(encodings)
        return torch.cat(encodings)


class RNNClassifier(torch.nn.Module):
    def __init__(self, output_dim, **kwargs):
        super(RNNClassifier, self).__init__()

        encoder_params = skorch.utils.params_for("encoder", kwargs)
        self.encoder = LazyRNNEncoder(**encoder_params)

        self.output_layer = torch.nn.Linear(self.encoder.output_dim, output_dim)

    def forward(self, x):
        encodings = self.encoder(x)
        return self.output_layer(encodings)


class LazyMLPEncoder(torch.nn.Module):
    def __init__(
        self,
        output_dim=64,
        hidden_dims=(),
        nonlinearity=torch.nn.ReLU(),
    ):
        super(LazyMLPEncoder, self).__init__()

        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.nonlinearity = nonlinearity
        self.layers = None

    def forward(self, inputs):
        if self.layers is None:
            input_dim = inputs.size(-1)
            sizes = [input_dim, *self.hidden_dims, self.output_dim]
            self.layers = torch.nn.ModuleList([])
            for i in range(len(sizes) - 1):
                in_features = sizes[i]
                out_features = sizes[i+1]
                self.layers += [torch.nn.Linear(in_features, out_features)]
        for layer in self.layers:
            inputs = self.nonlinearity(layer(inputs))
        return inputs


class MLPClassifier(torch.nn.Module):
    def __init__(self, output_dim, **kwargs):
        super(MLPClassifier, self).__init__()

        encoder_params = skorch.utils.params_for("encoder", kwargs)
        self.encoder = LazyMLPEncoder(**encoder_params)

        self.output_layer = torch.nn.Linear(self.encoder.output_dim, output_dim)

    def forward(self, x):
        encodings = self.encoder(x)
        return self.output_layer(encodings)


class NeuralNetClassifier(
    ClassifierMixin, skorch.NeuralNetClassifier, _MetadataRequester
):
    def __init__(self, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
        if seed is not None:
            seed_torch(seed)

    @property
    def _default_callbacks(self):
        return [
            ("epoch_timer", skorch.callbacks.EpochTimer()),
            ("train_loss", skorch.callbacks.PassthroughScoring(name="train_loss", on_train=True)),
            ("valid_loss", skorch.callbacks.PassthroughScoring(name="valid_loss")),
            ("print_log", skorch.callbacks.PrintLog())
        ]

    def get_params(self, deep=True, **kwargs):
        params = BaseEstimator.get_params(self, deep=deep, **kwargs)

        # Callback parameters are not returned by .get_params and need a
        # special treatment.
        params_cb = self._get_params_callbacks(deep=deep)
        params.update(params_cb)

        to_exclude = {"_modules", "_criteria", "_optimizers", "_metadata_request"}
        return {key: val for key, val in params.items() if key not in to_exclude}

    def fit(self, X, y, groups=None, X_valid=None, y_valid=None, groups_valid=None):
        return super().fit(
            X,
            y,
            groups=groups,
            X_valid=X_valid,
            y_valid=y_valid,
            groups_valid=groups_valid,
        )

    def get_dataset(self, X, y, groups=None):
        if issubclass(self.dataset, skorch.dataset.Dataset):
            return self.dataset(X, y)
        return self.dataset(inputs=X, targets=y, groups=groups)

    def get_split_datasets(self, X, y, groups=None, X_valid=None, y_valid=None, groups_valid=None):
        if X_valid is not None and y_valid is not None:
            if self.train_split is not None:
                warnings.warn(
                    "Validation data was explicitly passed while the parameter "
                    "`train_split` is set. Consider setting `train_split=None`."
                )
            dataset_train = self.get_dataset(X, y, groups)
            dataset_valid = self.get_dataset(X_valid, y_valid, groups_valid)
            return dataset_train, dataset_valid
        dataset = self.get_dataset(X, y, groups)
        if not self.train_split:
            return dataset, None
        return self.train_split(dataset)

    def run_single_epoch(self, iterator, training, prefix, step_fn, **fit_params):
        # Catch parameters passed to the `self.get_split_datasets` call
        # since they should not be passed to the step function.
        fit_params.pop("groups", None)
        fit_params.pop("X_valid", None)
        fit_params.pop("y_valid", None)
        fit_params.pop("groups_valid", None)

        if iterator is None:
            return

        batch_count = 0
        for batch in iterator:
            self.notify("on_batch_begin", batch=batch, training=training)
            step = step_fn(batch, **fit_params)
            self.history.record_batch(prefix + "_loss", step["loss"].item())
            if isinstance(batch, (tuple, list)):
                if isinstance(batch[0], torch.nn.utils.rnn.PackedSequence):
                    batch_size = int(batch[0].batch_sizes[0])
                else:
                    batch_size = skorch.dataset.get_len(batch[0])
            else:
                batch_size = skorch.dataset.get_len(batch)
            self.history.record_batch(prefix + "_batch_size", batch_size)
            self.notify("on_batch_end", batch=batch, training=training, **step)
            batch_count += 1

        self.history.record(prefix + "_batch_count", batch_count)

    def forward(self, X, groups=None, training=False, device="cpu"):
        y_infer = list(self.forward_iter(X, groups, training=training, device=device))
        is_multioutput = len(y_infer) > 0 and isinstance(y_infer[0], tuple)
        if is_multioutput:
            return tuple(map(torch.cat, zip(*y_infer)))
        return torch.cat(y_infer)

    def forward_iter(self, X, groups=None, training=False, device="cpu"):
        dataset = self.get_dataset(X, y=None, groups=groups)
        iterator = self.get_iterator(dataset, training=training)
        for batch in iterator:
            yp = self.evaluation_step(batch, training=training)
            yield skorch.utils.to_device(yp, device=device)

    def predict(self, X, groups=None):
        return self.predict_proba(X, groups).argmax(axis=1)

    def predict_proba(self, X, groups=None):
        nonlin = self._get_predict_nonlinearity()
        y_probas = []
        for yp in self.forward_iter(X, groups, training=False):
            yp = yp[0] if isinstance(yp, tuple) else yp
            yp = nonlin(yp)
            y_probas.append(skorch.utils.to_numpy(yp))
        y_proba = np.concatenate(y_probas, 0)
        return y_proba

    def score(self, X, y, *, metric, groups=None, sample_weight=None, mask=None):
        return super().score(
            X,
            y,
            groups=groups,
            metric=metric,
            sample_weight=sample_weight,
            mask=mask,
        )


# =============================================================================
# == Classififer calibration. =================================================
# =============================================================================

class _CalibratedClassifier(calibration._CalibratedClassifier):
    def predict_proba(self, X, **params):
        predictions = self.estimator.predict_proba(X, **params)

        original_estimator = self.estimator

        class PredictionWrapper:
            _estimator_type = "classifier"

            def __init__(self, predictions):
                self.predictions = predictions

            def predict_proba(self, X):
                return self.predictions

        self.estimator = PredictionWrapper(predictions)
        self.estimator.classes_ = original_estimator.classes_

        try:
            return super().predict_proba(X)
        finally:
            self.estimator = original_estimator


def _fit_calibrator(clf, predictions, y, classes, method, sample_weight=None):
    """Fit calibrator(s) and return a `_CalibratedClassifier` instance.
    Source: sklearn/calibration.py
    """
    Y = label_binarize(y, classes=classes)
    label_encoder = LabelEncoder().fit(classes)
    pos_class_indices = label_encoder.transform(clf.classes_)
    calibrators = []
    for class_idx, this_pred in zip(pos_class_indices, predictions.T):
        if method == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds="clip")
        else:  # "sigmoid"
            calibrator = _SigmoidCalibration()
        calibrator.fit(this_pred, Y[:, class_idx], sample_weight)
        calibrators.append(calibrator)
    pipeline = _CalibratedClassifier(clf, calibrators, method=method, classes=classes)
    return pipeline


class CalibratedClassifier(ClassifierMixin, MetaEstimatorMixin, BaseEstimator):
    """This class allows for passing parameters to the `predict_proba` method."""

    def __init__(
        self,
        estimator,
        method="sigmoid",
    ):
        self.estimator = estimator
        self.method = method

    def fit(self, X, y, sample_weight=None, **params):
        if self.estimator is None:
            raise ValueError("Estimator cannot be `None`.")

        check_classification_targets(y)
        X, y = indexable(X, y)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        check_is_fitted(self.estimator, attributes=["classes_"])
        self.classes_ = self.estimator.classes_

        routed_params = process_routing(self, "predict_proba", **params)
        predictions = self.estimator.predict_proba(
            X, **routed_params.estimator.predict_proba
        )

        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        calibrated_classifier = _fit_calibrator(
            self.estimator,
            predictions,
            y,
            self.classes_,
            self.method,
            sample_weight,
        )

        self.calibrated_classifier_ = calibrated_classifier

        if hasattr(calibrated_classifier, "n_features_in_"):
            self.n_features_in_ = calibrated_classifier.n_features_in_

        if hasattr(calibrated_classifier, "feature_names_in_"):
            self.feature_names_in_ = calibrated_classifier.feature_names_in_

        return self

    def predict_proba(self, X, **params):
        check_is_fitted(self)
        routed_params = process_routing(self, "predict_proba", **params)
        return self.calibrated_classifier_.predict_proba(
            X, **routed_params.estimator.predict_proba
        )

    def predict(self, X, **params):
        check_is_fitted(self)
        return self.classes_[
            np.argmax(self.predict_proba(X, **params), axis=1)
        ]

    def get_metadata_routing(self):
        router = (
            MetadataRouter(owner=self.__class__.__name__)
            .add_self_request(self)
            .add(
                estimator=self.estimator,
                method_mapping=MethodMapping().add(caller="fit", callee="predict_proba"),
            )
            .add(
                estimator=self.estimator,
                method_mapping=MethodMapping().add(caller="predict_proba", callee="predict_proba"),
            )
        )
        return router
