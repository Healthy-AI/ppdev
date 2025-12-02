from sklearn.utils.metadata_routing import (
    _MetadataRequester,
    MetadataRequest,
    _routing_enabled,
)

from .estimators import (
    StageTherapyClassifier,
    SwitchTherapyClassifier,
    StageSwitchTherapyClassifier,
    NeuralNetClassifier,
    expects_groups,
)


def get_scoring(estimator, metric):
    switches_only = metric.endswith("_switches")
    metric = metric.removesuffix("_switches")
    if isinstance(estimator, SwitchTherapyClassifier):
        return (
            SwitchTherapyScorer(estimator=estimator, metric=metric)
            .set_score_request(
                y_prev=True, sample_weight=True, mask=switches_only
            )
        )
    elif isinstance(estimator, StageTherapyClassifier):
        return (
            StageTherapyScorer(estimator=estimator, metric=metric)
            .set_score_request(
                stages=True, sample_weight=True, mask=switches_only
            )
        )
    elif isinstance(estimator, StageSwitchTherapyClassifier):
        return (
            StageSwitchTherapyScorer(estimator=estimator, metric=metric)
            .set_score_request(
                stages=True, y_prev=True, sample_weight=True, mask=switches_only
            )
        )
    elif isinstance(estimator, NeuralNetClassifier) and expects_groups(estimator):
        return (
            NeuralNetScorer(estimator=estimator, metric=metric)
            .set_score_request(
                groups=True, sample_weight=True, mask=switches_only
            )
        )
    else:
        return (
            BaseScorer(estimator=estimator, metric=metric)
            .set_score_request(sample_weight=True, mask=switches_only)
        )


class BaseScorer(_MetadataRequester):
    def __init__(self, estimator, metric):
        self.estimator = estimator
        self.metric = metric

    def __call__(self, estimator, X, y, *, sample_weight=None, mask=None):
        return estimator.score(
            X,
            y,
            metric=self.metric,
            sample_weight=sample_weight,
            mask=mask,
        )

    def set_score_request(self, **kwargs):
        if not _routing_enabled():
            raise RuntimeError(
                "This method is only available when metadata routing is enabled."
            )
        self._metadata_request = MetadataRequest(owner=self.__class__.__name__)
        for param, alias in kwargs.items():
            self._metadata_request.score.add_request(param=param, alias=alias)
        return self


class SwitchTherapyScorer(BaseScorer):
    def __call__(self, estimator, X, y, *, y_prev, sample_weight=None, mask=None):
        return estimator.score(
            X,
            y,
            y_prev=y_prev,
            metric=self.metric,
            sample_weight=sample_weight,
            mask=mask,
        )


class StageTherapyScorer(BaseScorer):
    def __call__(self, estimator, X, y, *, stages, sample_weight=None, mask=None):
        return estimator.score(
            X,
            y,
            stages=stages,
            metric=self.metric,
            sample_weight=sample_weight,
            mask=mask,
        )


class StageSwitchTherapyScorer(BaseScorer):
    def __call__(self, estimator, X, y, *, stages, y_prev, sample_weight=None, mask=None):
        return estimator.score(
            X,
            y,
            stages=stages,
            y_prev=y_prev,
            metric=self.metric,
            sample_weight=sample_weight,
            mask=mask,
        )


class NeuralNetScorer(BaseScorer):
    def __call__(self, estimator, X, y, *, groups, sample_weight=None, mask=None):
        return estimator.score(
            X,
            y,
            groups=groups,
            metric=self.metric,
            sample_weight=sample_weight,
            mask=mask,
        )
