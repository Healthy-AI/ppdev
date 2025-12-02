import os
import pickle
from contextlib import contextmanager
from abc import ABCMeta, abstractmethod

import torch
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from sklearn.model_selection import GroupShuffleSplit as CVSplitter
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    KBinsDiscretizer,
    LabelEncoder,
    FunctionTransformer,
    MinMaxScaler,
)
from sklearn.compose import make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from ppdev.data import utils as utils
from .transformers import CustomColumnTransformer

__all__ = [
    "get_data_handler",
    "get_dataset",
    "get_params",
    "OPEDataset",
]


def get_data_handler(config):
    experiment_config = config["experiment"]
    data_config = config["data"]
    data_handler_str = config["experiment"]["data_handler"]
    if data_handler_str == "1":
        return RADataHandler1(experiment_config, data_config)
    elif data_handler_str == "2":
        return RADataHandler2(experiment_config, data_config)
    elif data_handler_str == "3":
        return RADataHandler3(experiment_config, data_config)
    elif data_handler_str == "4":
        return RADataHandler4(experiment_config, data_config)
    elif data_handler_str == "5":
        return RADataHandler5(experiment_config, data_config)
    elif data_handler_str == "6":
        return RADataHandler6(experiment_config, data_config)
    elif data_handler_str == "sepsis":
        return SepsisDataHandler(experiment_config, data_config)
    else:
        raise ValueError(f"Invalid data handler: {data_handler_str}.")


def get_dataset(data, config, encode_targets=True):
    experiment_config = config["experiment"]
    inputs = data[experiment_config["input"]]
    groups = data[experiment_config["identifier"]]
    targets = data[experiment_config["target"]]
    if encode_targets:
        targets = LabelEncoder().fit_transform(targets)
    outcomes = data[experiment_config["outcome"]]
    return OPEDataset(inputs, groups, targets, outcomes)


def get_params(data, config, estimator_alias, training=False):
    estimator_config = config["estimators"][estimator_alias]

    params_mapper = estimator_config.get("params", None)
    if params_mapper is not None:
        params = {k: data[v].to_numpy() for k, v in params_mapper.items()}
    else:
        params = {}

    if estimator_config.get("use_sample_weights", False):
        c_target = config["experiment"]["target"]
        targets = LabelEncoder().fit_transform(data[c_target])
        sample_weights = compute_sample_weight("balanced", targets)
    else:
        sample_weights = None
    params["sample_weight"] = sample_weights

    if training and config["experiment"]["alias"] == "ra":
        # The first step of the pipeline is an instance of `DataSelector`. When
        # fitting the pipeline, the data selector samples data based on patient
        # IDs and exclude reasons.
        params["ids"] = data[config["experiment"]["identifier"]].to_numpy()
        params["exclude_reasons"] = data.exclude_reason.to_numpy()

    if not training:
        # TODO: Consider defining params_train and params_test in the config.
        params.pop("o", None)

    return params


class OPEDataset(torch.utils.data.Dataset):
    return_mode_ = "inputs_targets"

    def __init__(self, inputs, groups, targets=None, outcomes=None):
        self.inputs = self._check_inputs_and_groups(inputs)
        self.groups = self._check_inputs_and_groups(groups)
        self.targets = self._check_targets_and_outcomes(targets)
        self.outcomes = self._check_targets_and_outcomes(outcomes)

    def _check_inputs_and_groups(self, input):
        if input is None:
            raise ValueError("Inputs or groups cannot be None.")
        if input.ndim == 1 and not isinstance(input, pd.Series):
            return pd.Series(input)
        if input.ndim > 1 and not isinstance(input, pd.DataFrame):
            return pd.DataFrame(input)
        return input.reset_index(drop=True)

    def _check_targets_and_outcomes(self, input):
        if input is not None and not isinstance(input, np.ndarray):
            return np.array(input)
        return input

    def get_splits(self, n_splits=5, test_size=0.2, seed=None):
        splitter = CVSplitter(n_splits, test_size=test_size, random_state=seed)
        return splitter.split(X=self.inputs, groups=self.groups)

    @property
    def unique_groups(self):
        if not hasattr(self, "_unique_groups"):
            self._unique_groups = self.groups.unique()
        return self._unique_groups

    @property
    def group_indices(self):
        if not hasattr(self, "_group_indices"):
            self._group_indices = self.inputs.groupby(self.groups).indices
        return self._group_indices

    def __len__(self):
        return len(self.unique_groups)

    def __getitem__(self, i):
        group = self.unique_groups[i]
        indices = self.group_indices[group]
        inputs = self.inputs.iloc[indices]
        targets = self.targets[indices] if self.targets is not None else None
        outcomes = self.outcomes[indices] if self.outcomes is not None else None
        if self.return_mode_ == "inputs_targets":
            return inputs, targets
        elif self.return_mode_ == "inputs_targets_outcomes":
            return inputs, targets, outcomes
        else:
            raise ValueError(f"Invalid return_mode: {self.return_mode_}.")

    def __iter__(self):
        for i in range(len(self)):
            yield self.__getitem__(i)

    def save_data(self, filepath):
        data = {
            "inputs": self.inputs,
            "groups": self.groups,
            "targets": self.targets,
            "outcomes": self.outcomes,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    @contextmanager
    def set_return_mode(self, return_mode):
        old_return_mode = self.return_mode_
        if return_mode not in ["inputs_targets", "inputs_targets_outcomes"]:
            raise ValueError(f"Invalid return_mode: {return_mode}")
        self.return_mode_ = return_mode
        try:
            yield
        finally:
            self.return_mode_ = old_return_mode


class BaseDataHandler(metaclass=ABCMeta):
    def __init__(self, experiment_config, data_config):
        self.experiment_config = experiment_config
        self.data_config = data_config

    @abstractmethod
    def load_data(self):
        """Load all available data."""

    def _split_once(self, data, test_size=None, seed=None):
        splitter = CVSplitter(1, test_size=test_size, random_state=seed)
        groups = data[self.experiment_config["identifier"]]
        train, valid = next(splitter.split(data, groups=groups))
        return data.iloc[train], data.iloc[valid]

    @abstractmethod
    def split_data(self):
        """Split data into a training set, a validation, and a testing set."""


def get_NEWS2_score(respiratory_rate, SpO2, on_vent, blood_pressure, heart_rate, is_CVPU, temperature):
    """Calculate the National Early Warning 2 (NEWS2) score.

    Reference: https://www.rcplondon.ac.uk/projects/outputs/national-early-warning-score-news-2.
    """

    score_table = {
        "respiratory_rate": {
            "range": [0, 9, 12, 21, 25, float("inf")],
            "score": [3, 1, 0, 2, 3],
        },
        "SpO2_scale1": {
            "range": [0, 92, 94, 96, float("inf")],
            "score": [3, 2, 1, 0],
        },
        "SpO2_scale2": {
            "on_vent": {
                "range": [93, 95, 97, 100.01],
                "score": [4, 3, 2],
            },
            "on_air": {
                "range": [0, 84, 86, 88, 93],
                "score": [4, 3, 2, 0],
            }
        },
        "on_vent": {
            "range": [0, 1, float("inf")],
            "score": [2, 0],
        },
        "blood_pressure": {
            "range": [0, 90, 100, 110, 220, float("inf")],
            "score": [3, 2, 1, 0, 3],
        },
        "heart_rate": {
            "range": [0, 40, 50, 90, 110, 130, float("inf")],
            "score": [3, 1, 0, 1, 2, 3],
        },
        "is_CVPU": {
            "range": [0, 1, float("inf")],
            "score": [0, 3],
        },
        "temperature": {
            "range": [0, 35, 36, 38, 39, float("inf")],
            "score": [3, 1, 0, 1, 2],
        }
    }

    scoring_dict = {
        "respiratory_rate": respiratory_rate,
        "SpO2_scale1": SpO2,  # Assume no HRF
        "on_vent": on_vent,
        "blood_pressure": blood_pressure,
        "heart_rate": heart_rate,
        "is_CVPU": is_CVPU,
        "temperature": temperature,
    }

    def get_single_score(selector, value):
        score_dict = score_table[selector]
        assert len(score_dict["range"]) == len(score_dict["score"]) + 1
        for idx, upper_bound in enumerate(score_dict["range"]):
            if value < upper_bound:
                return score_dict["score"][idx - 1]
        raise ValueError("{} not in range of {}.".format(selector, score_dict["range"]))

    total_score = 0
    for selector, value in scoring_dict.items():
        total_score += get_single_score(selector, value)
    return total_score


def calculate_news2(row):
    is_CVPU = 1 if row["GCS"] < 15 else 0
    return get_NEWS2_score(
        respiratory_rate=row["RR"],
        SpO2=row["SpO2"],
        on_vent=row["mechvent"],
        blood_pressure=row["SysBP"],
        heart_rate=row["HR"],
        is_CVPU=is_CVPU,
        temperature=row["Temp_C"],
    )


class SepsisDataHandler(BaseDataHandler):
    SHIFT = ["gender", "mechvent", "re_admission"]
    LOG_SCALE = [
        "SpO2", "BUN", "Creatinine", "SGOT", "SGPT", "Total_bili", "INR",
        "input_total", "input_4hourly", "output_total", "output_4hourly",
        "max_dose_vaso", "prev_input_4hourly", "prev_max_dose_vaso",
    ]
    SCALE = [
        "age", "Weight_kg", "GCS", "HR", "SysBP", "MeanBP", "DiaBP", "RR",
        "Temp_C", "FiO2_1", "Potassium", "Sodium", "Chloride", "Glucose",
        "Magnesium", "Calcium", "Hb", "WBC_count", "Platelets_count", "PTT",
        "PT", "Arterial_pH", "paO2", "paCO2", "Arterial_BE", "HCO3",
        "Arterial_lactate", "SOFA", "SIRS", "Shock_Index", "PaO2_FiO2",
        "cumulated_balance", "elixhauser",
    ]

    def _discretize_doses(self, doses, num_levels=5):
        discrete_doses = np.zeros_like(doses)  # 0 is default (zero dose)
        is_nonzero = doses > 0
        ranked_nonzero_doses = rankdata(doses[is_nonzero]) / np.sum(is_nonzero)
        discrete_nonzero_doses = np.digitize(
            ranked_nonzero_doses,
            bins=np.linspace(0, 1, num=num_levels),
            right=True,
        )
        discrete_doses[is_nonzero] = discrete_nonzero_doses
        return discrete_doses

    def _get_action(self, data):
        """Discretize the doses of IV fluids and vasopressors into 4 levels.

        Source: https://github.com/GilesLuo/ReassessDTR/blob/master/DTRGym/MIMIC3SepsisEnv/preprocess_util.py#L95.
        """
        iv_fluid_action = np.digitize(
            data["input_4hourly"],
            bins=[1e-10, 50, 180, 530],
            right=True,
        )
        vasopressor_action = np.digitize(
            data["max_dose_vaso"],
            bins=[1e-10, 0.08, 0.22, 0.45],
            right=True,
        )

        combined_actions = np.stack(
            (iv_fluid_action, vasopressor_action), axis=-1
        )
        _, final_action = np.unique(combined_actions, axis=0, return_inverse=True)

        return final_action

    def load_data(self):
        data = pd.read_csv(self.data_config["preprocessed_data_path"])

        data["step"] = data.bloc.astype(int) - 1

        # Filter out patients with discontinuous data.
        max_steps = data.groupby("icustayid").step.max()
        actual_counts = data.groupby("icustayid").size()
        expected_counts = max_steps + 1
        discontinuous_patients = expected_counts[expected_counts != actual_counts].index
        data = data[~data.icustayid.isin(discontinuous_patients)]
        data.sort_values(["icustayid", "step"], ascending=True, inplace=True)

        # Filter out patients with less than 20--24 hours of data.
        #
        # 0--4: 0 | 4--8: 1 | 8--12: 2 | 12--16: 3 | 16--20: 4 | 20--24: 5
        p_gt24 = data.groupby("icustayid").step.max() >= 5
        data = data[
            data.icustayid.isin([p for p, true in p_gt24.items() if true])
        ]

        # Take the first `max_len` steps of each patient.
        data = data.groupby("icustayid").head(self.experiment_config["max_len"])

        data.reset_index(drop=True, inplace=True)

        # Add NEWS2 score.
        data["NEWS2"] = data.apply(calculate_news2, axis=1)
        data[f"r_NEWS2"] = -1 * (data["NEWS2"] - 0) / (18 - 0)

        data["treatment"] = self._get_action(data)

        # Add a column `prev_treatment` to indicate the previous treatment.
        data["prev_treatment"] = (
            data.groupby("icustayid").treatment.shift(fill_value=0)
        )

        treatments = ["input_4hourly", "max_dose_vaso"]
        previous_doses = data.groupby("icustayid")[treatments].shift(fill_value=0)
        mapper = {c: f"prev_{c}" for c in treatments}
        previous_doses = previous_doses.rename(columns=mapper)
        data = pd.concat([data, previous_doses], axis=1)

        # Add a column `outcome` to indicate the outcome of the current treatment.
        data["outcome"] = 0.0
        last_indices = data.groupby("icustayid").tail(1).index
        died = data.loc[last_indices, "died_within_48h_of_out_time"].eq(1)
        data.loc[last_indices, "outcome"] = np.where(died, -100, 100)

        data["survival_90d"] = data["mortality_90d"].eq(0).astype("int")
        data["survival_48h"] = data["died_within_48h_of_out_time"].eq(0).astype("int")

        return data

    def split_data(self):
        data = self.load_data()
        data_train, data_test = self._split_once(
            data,
            self.experiment_config["test_size"],
            self.experiment_config["seed"],
        )
        if self.experiment_config["valid_size"] is None:
            data_valid = pd.DataFrame(columns=data.columns)
        else:
            data_train, data_valid = self._split_once(
                data_train,
                test_size=self.experiment_config["valid_size"],
                seed=self.experiment_config["seed"],
            )
        return data_train, data_valid, data_test

    def get_shift_transformer(self):
        return MinMaxScaler((-0.5, 0.5))

    def get_scale_transformer(self):
        return StandardScaler()

    def _add_log(self, x):
        return np.log(0.1 + x)

    def get_log_scale_transformer(self):
        return make_pipeline(
            FunctionTransformer(self._add_log, feature_names_out="one-to-one"),
            StandardScaler(),
        )

    def get_shifted_columns(self, X):
        return sorted(list(set(X).intersection(self.SHIFT)))

    def get_scaled_columns(self, X):
        return sorted(list(set(X).intersection(self.SCALE)))

    def get_log_scaled_columns(self, X):
        return sorted(list(set(X).intersection(self.LOG_SCALE)))

    def get_preprocessor(
        self,
        numerical_transformation,
        categorical_transformation,
    ):
        encode_numerical = numerical_transformation != "none"
        encode_categorical = categorical_transformation != "none"

        t1 = self.get_shift_transformer() if encode_categorical else "passthrough"
        t1 = ("shift_transformer", t1, self.get_shifted_columns)

        t2 = self.get_scale_transformer() if encode_numerical else "passthrough"
        t2 = ("scale_transformer", t2, self.get_scaled_columns)

        t3 = self.get_log_scale_transformer() if encode_numerical else "passthrough"
        t3 = ("log_scale_transformer", t3, self.get_log_scaled_columns)

        return CustomColumnTransformer(
            transformers=[t1, t2, t3], remainder="passthrough",
        )


class BaseRADataHandler(BaseDataHandler, metaclass=ABCMeta):
    def get_cohort_info_path(self):
        cohort_params = self.experiment_config["cohort_params"]
        cohort_params_str = []
        for pname, pvalue in cohort_params.items():
            cohort_params_str.append(f"{pname}={pvalue}")
        cohort_filename = "cohort_" + "_".join(cohort_params_str) + ".pkl"
        return os.path.join(self.data_config["root_dir"], cohort_filename)

    def load_data(self):
        # Load data and add cohort information.
        data = pd.read_pickle(self.data_config["preprocessed_data_path"])
        cohort_info_path = self.get_cohort_info_path()
        cohort_info = pd.read_pickle(cohort_info_path)
        assert all(data.index == cohort_info.index)
        data = pd.concat([data, cohort_info], axis=1)

        # Add a column `id2` to indicate the original patient ID.
        data["id2"] = data.id.str.extract("^([0-9-]+)", expand=False)

        # Decode categorical variables.
        for v, i in utils.get_all_variables().items():
            if v in data.columns and "categories" in i:
                data.loc[:, v] = data[v].map(i["categories"])
        
        # Convert `hx2X` columns to boolean type.
        # TODO: Run the pre-processing script again and remove these lines.
        c_hx2 = [c for c in data.columns if c.startswith("hx2")]
        data[c_hx2] = data[c_hx2].astype("boolean")

        # Add a column `prev_therapy` to indicate the previous therapy.
        data["prev_therapy"] = data.groupby("id2").therapy.shift()
        data.loc[data.prev_therapy.eq("Other therapy"), "prev_therapy"] = np.nan
        mask = data.prev_therapy.isna() & data.stage2.eq(1)
        data.loc[mask, "prev_therapy"] = "csDMARD therapy"
        mask = data.prev_therapy.isna()
        data.loc[mask, "prev_therapy"] = data.loc[mask, "therapy"]

        # Exclude non-registry visits.
        data = data.loc[data.stage.ne(-2)]

        # Add a column `y_prev` to indicate the previous therapy as an integer.
        data["y_prev"] = LabelEncoder().fit_transform(data.prev_therapy)

        # Add a column `outcome` to indicate the outcome of the current therapy.
        data["outcome"] = 10 - data.groupby("id").cdai.shift(-1)

        # Fill missing values by propagting the last valid observation.
        c_ffill = self.experiment_config["ffill"]
        data[c_ffill] = data.groupby(by="id2")[c_ffill].ffill()

        return data

    def _get_numerical_transformer(self, numerical_transformation):
        imputer = SimpleImputer(strategy="mean")
        if numerical_transformation == "discretize":
            encoder = KBinsDiscretizer(subsample=None)
        elif numerical_transformation == "scale":
            encoder = StandardScaler()
        elif numerical_transformation == "minmax":
            encoder = MinMaxScaler()
        elif numerical_transformation == "none":
            encoder = "passthrough"
        else:
            raise ValueError(
                "Invalid transformation for numerical columns: "
                f"{numerical_transformation}."
            )
        return make_pipeline(imputer, encoder)

    def _get_categorical_transformer(self, categorical_transformation):
        imputer = SimpleImputer(strategy="most_frequent")
        if categorical_transformation == "onehot":
            encoder = OneHotEncoder(
                drop="if_binary",
                handle_unknown="ignore",
                sparse_output=False,
            )
        elif categorical_transformation == "none":
            encoder = "passthrough"
        else:
            raise ValueError(
                "Invalid transformation for categorical columns: "
                f"{categorical_transformation}."
            )
        return make_pipeline(imputer, encoder)

    def _get_boolean_transformer(self):
        return SimpleImputer(strategy="most_frequent")

    def get_preprocessor(
        self,
        numerical_transformation,
        categorical_transformation,
    ):
        numerical_transformer = (
            "numerical_transformer",
            self._get_numerical_transformer(numerical_transformation),
            make_column_selector(dtype_include="float64"),
        )

        categorical_transformer = (
            "categorical_transformer",
            self._get_categorical_transformer(categorical_transformation),
            make_column_selector(dtype_include="category"),
        )

        boolean_transformer = (
            "boolean_transformer",
            self._get_boolean_transformer(),
            make_column_selector(dtype_include="boolean"),
        )

        return CustomColumnTransformer(
            transformers=[
                numerical_transformer,
                categorical_transformer,
                boolean_transformer,
            ],
            remainder="passthrough",
        )


class RADataHandler1(BaseRADataHandler):
    """RA data handler 1.

    Split the data into a training set and a testing set. The testing set
    contains all patinets in the cohort. The training set contains all other
    patients. If the `valid_size` parameter is not None, the training set is
    further split into a training set and a validation set.
    """

    def split_data(self):
        data = self.load_data()
        data_test = utils.filter_cohort(data, c_stage="stage")
        data_train = data[~data.id2.isin(data_test.id2)]
        if self.experiment_config["valid_size"] is None:
            data_valid = pd.DataFrame(columns=data.columns)
        else:
            data_train, data_valid = self._split_once(
                data_train,
                test_size=self.experiment_config["valid_size"],
                seed=self.experiment_config["seed"],
            )
        return data_train, data_valid, data_test


class RADataHandler2(BaseRADataHandler):
    """RA data handler 2.

    Split the data into a training set and a testing set of equal size using a
    random split. Only cohort patients are then selected for the final testing
    set. If the `valid_size` parameter is not None, the training set is
    further split into a training set and a validation set.
    """

    def split_data(self):
        data = self.load_data()
        data_train, data_test = self._split_once(
            data, 0.5, self.experiment_config["seed"]
        )
        data_test = utils.filter_cohort(data_test, c_stage="stage")
        if self.experiment_config["valid_size"] is None:
            data_valid = pd.DataFrame(columns=data.columns)
        else:
            data_train, data_valid = self._split_once(
                data_train,
                test_size=self.experiment_config["valid_size"],
                seed=self.experiment_config["seed"],
            )
        return data_train, data_valid, data_test


class RADataHandler3(BaseRADataHandler):
    """RA data handler 3.

    Extract cohort patients from the original dataset and split the remaining
    patients into a training set and a testing set. The size of the testing set
    is determined by the `test_size` parameter in the configuration file. If
    the `valid_size` parameter is not None, the training set is further split
    into a training set and a validation set.
    """

    def split_data(self):
        data = self.load_data()
        data = utils.filter_cohort(data, c_stage="stage")
        data_train, data_test = self._split_once(
            data,
            self.experiment_config["test_size"],
            self.experiment_config["seed"],
        )
        if self.experiment_config["valid_size"] is None:
            data_valid = pd.DataFrame(columns=data.columns)
        else:
            data_train, data_valid = self._split_once(
                data_train,
                test_size=self.experiment_config["valid_size"],
                seed=self.experiment_config["seed"],
            )
        return data_train, data_valid, data_test


class RADataHandler4(BaseRADataHandler):
    """RA data handler 4.

    Split the data into a training set and a testing set of equal size using a
    random split. Only cohort patients are then selected for the final testing
    set. If the `valid_size` parameter is not None, the training set is
    further split into a training set and a validation set. Only cohort
    patients are selected for the validation set (this is the only difference
    compared with RA data handler 2).
    """

    def split_data(self):
        data = self.load_data()
        data_train, data_test = self._split_once(
            data, 0.5, self.experiment_config["seed"]
        )
        data_test = utils.filter_cohort(data_test, c_stage="stage")
        if self.experiment_config["valid_size"] is None:
            data_valid = pd.DataFrame(columns=data.columns)
        else:
            data_train, data_valid = self._split_once(
                data_train,
                test_size=self.experiment_config["valid_size"],
                seed=self.experiment_config["seed"],
            )
            data_valid = utils.filter_cohort(data_valid, c_stage="stage")
        return data_train, data_valid, data_test


class RADataHandler5(BaseRADataHandler):
    """RA data handler 5.

    Extract baseline patients from the original dataset and split the remaining
    patients into a training set and a testing set. The size of the testing set
    is determined by the `test_size` parameter in the configuration file. If
    the `valid_size` parameter is not None, the training set is further split
    into a training set and a validation set.
    """

    def split_data(self):
        data = self.load_data()
        data = data[data.stage.eq(1)]
        data_train, data_test = self._split_once(
            data,
            self.experiment_config["test_size"],
            self.experiment_config["seed"],
        )
        if self.experiment_config["valid_size"] is None:
            data_valid = pd.DataFrame(columns=data.columns)
        else:
            data_train, data_valid = self._split_once(
                data_train,
                test_size=self.experiment_config["valid_size"],
                seed=self.experiment_config["seed"],
            )
        return data_train, data_valid, data_test


class RADataHandler6(BaseRADataHandler):
    """RA data handler 6.

    Create a training set by extracting cohort patients from the original
    dataset. Since this data handler should be used to fit trees for
    visualization, no testing set is created. If the `valid_size` parameter is
    not None, the training set is further split into a training set and a
    validation set.
    """

    def split_data(self):
        data = self.load_data()
        data_train = utils.filter_cohort(data, c_stage="stage")
        data_test = pd.DataFrame(columns=data.columns)
        if self.experiment_config["valid_size"] is None:
            data_valid = pd.DataFrame(columns=data.columns)
        else:
            data_train, data_valid = self._split_once(
                data_train,
                test_size=self.experiment_config["valid_size"],
                seed=self.experiment_config["seed"],
            )
        return data_train, data_valid, data_test
