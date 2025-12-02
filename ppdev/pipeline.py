import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.base import _fit_context
from sklearn.utils._user_interface import _print_elapsed_time

from .data import get_data_handler
from .data import get_dataset
from .data import DataSelector
from .estimators import get_estimator


class CustomPipeline(Pipeline):
    def _fit(self, X, y=None, routed_params=None):
        if isinstance(self.steps[0][1], DataSelector):
            # Make a shallow copy of the steps.
            steps = list(self.steps)

            if not isinstance(X, pd.DataFrame):
                raise ValueError(
                    "The input data must be a pandas DataFrame."
                )

            if not isinstance(X.index, pd.RangeIndex):
                X = X.reset_index(drop=True)
            
            # Select the data to be used for model fitting. The final step is 
            # included since super()._fit iterates over all steps except the
            # last one.
            self.steps = [self.steps[0], self.steps[-1]]
            Xt = super()._fit(X, y, routed_params)

            # Update any parameters passed to the remaining steps, including
            # the final estimator.
            selected_indices = Xt.index.to_numpy()
            for step_name, step_params in routed_params.items():
                if step_name == self.steps[0][0]:
                    continue
                for method_name, method_params in step_params.items():
                    for param_name, param_value in method_params.items():
                        if (
                            isinstance(param_value, (list, np.ndarray, pd.Series))
                            and len(param_value) == len(X)
                        ):
                            routed_params[step_name][method_name][param_name] = \
                                np.asarray(param_value)[selected_indices]

            # Fit the remaining transformers.
            self.steps = steps[1:]
            Xt = super()._fit(Xt, y, routed_params)
            
            self.steps = steps

            return Xt, selected_indices

        selected_indices = np.arange(X.shape[0])
        return super()._fit(X, y, routed_params), selected_indices

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y=None, **params):
        routed_params = self._check_method_params(method="fit", props=params)
        Xt, selected_indices = self._fit(X, y, routed_params)
        if y is not None:
            y = y[selected_indices]
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                last_step_fit_params = routed_params[self.steps[-1][0]]["fit"]
                self._final_estimator.fit(Xt, y, **last_step_fit_params)
        return self


def get_pipeline(config, estimator_alias):
    estimator_config = config["estimators"][estimator_alias]

    data_handler = get_data_handler(config)

    if config["experiment"]["alias"] == "ra":
        data_selector = (
            DataSelector()
            .set_transform_request(ids=False, exclude_reasons=True)
            .set_transform_request(ids=True, exclude_reasons=True)
        )
    else:
        data_selector = None

    preprocessor = data_handler.get_preprocessor(
        estimator_config.get("numerical_transformation", "none"),
        estimator_config.get("categorical_transformation", "none"),
    )

    estimator_kwargs = {"random_state": config["experiment"]["seed"]}

    if estimator_config.get("is_net_estimator", False):
        estimator_kwargs["output_dir"] = config["results"]["root_dir"]
        data_train = data_handler.split_data()[0]
        dataset_train = get_dataset(data_train, config)
        estimator_kwargs["output_dim"] = len(set(dataset_train.targets))

    estimator = get_estimator(estimator_alias, **estimator_kwargs)

    pipeline = CustomPipeline(
        [
            ("data_selector", data_selector),
            ("preprocessor", preprocessor),
            ("estimator", estimator),
        ],
    )
    pipeline.set_score_request(sample_weight=False)
    return pipeline
