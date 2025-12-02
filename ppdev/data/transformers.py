import pandas as pd

from sklearn.utils import check_scalar, check_random_state
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import _transform_one
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ["DataSelector", "CustomColumnTransformer"]


class DataSelector(BaseEstimator, TransformerMixin):
    def __init__(self, frac_pre_bl=1.0, frac_post_bl=1.0, random_state=None):
        self.frac_pre_bl = frac_pre_bl
        self.frac_post_bl = frac_post_bl
        self.random_state = random_state

    def fit_transform(self, X, y=None, ids=None, exclude_reasons=None):
        return self.fit(X).transform(X, ids=ids, exclude_reasons=exclude_reasons)

    def fit(self, X, y=None):
        return self

    def _sample(self, data, groups, size, random_state):
        if isinstance(size, float):
            size = int(size * groups.nunique())
        sampled_groups = random_state.choice(
            groups.unique(), size=size, replace=False
        )
        return data.loc[groups.isin(sampled_groups)]

    def _check_transform_inputs(self, X, ids, exclude_reasons):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")
        if not isinstance(ids, pd.Series):
            ids = pd.Series(ids)
        if not isinstance(exclude_reasons, pd.Series):
            exclude_reasons = pd.Series(exclude_reasons)
        X = X.reset_index(drop=True)
        ids = ids.reset_index(drop=True)
        exclude_reasons = exclude_reasons.reset_index(drop=True)
        return X, ids, exclude_reasons

    def transform(self, X, y=None, ids=None, exclude_reasons=None):
        if ids is None and exclude_reasons is None:
            return X

        if ids is None or exclude_reasons is None:
            provided = "ids" if ids else "exclude_reasons"
            raise ValueError(
                "Sampling requires both 'ids' and 'exclude_reasons'. Only "
                f"'{provided}' was provided."
            )

        X, ids, exclude_reasons = self._check_transform_inputs(
            X, ids, exclude_reasons,
        )

        kwargs = dict(min_val=0.0, max_val=1.0)
        check_scalar(self.frac_pre_bl, "frac_pre_bl", float, **kwargs)
        check_scalar(self.frac_post_bl, "frac_post_bl", float, **kwargs)

        random_state = check_random_state(self.random_state)

        mask_fn = exclude_reasons.str.contains

        mask_pre_bl = mask_fn("No b/tsDMARD initiated")
        X_pre_bl = self._sample(
            X[mask_pre_bl], ids[mask_pre_bl], self.frac_pre_bl, random_state
        )

        mask_post_bl = mask_fn("History of b/tsDMARDs at registry enrollment")
        X_post_bl = self._sample(
            X[mask_post_bl], ids[mask_post_bl], self.frac_post_bl, random_state
        )

        X_other = X[~mask_pre_bl & ~mask_post_bl]

        X = pd.concat([X_other, X_pre_bl, X_post_bl]).sort_index()

        return X


def _custom_transform_one(transformer, X, y, weight, params=None):
    """
    Custom implementation of `_transform_one` (see sklearn/pipeline.py) that
    selects parameters for the transformer's `transform` method via the key
    "transform", instead of the transform attribute, as the latter is
    incompatible with the Dask joblib backend.
    """
    res = transformer.transform(X, **params["transform"])
    if weight is None:
        return res
    return res * weight


class CustomColumnTransformer(ColumnTransformer):
    def _call_func_on_transformers(self, X, y, func, column_as_labels, routed_params):
        if func is _transform_one:
            func = _custom_transform_one
        return super()._call_func_on_transformers(
            X, y, func, column_as_labels, routed_params
        )
