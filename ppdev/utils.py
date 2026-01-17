import os
import sys
import copy
import argparse
import datetime
from pathlib import Path

import skorch.dataset
import yaml
import torch
import skorch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


def _get_single_dataframe(value_estimate, parameters, additional_info=None):
    d = vars(value_estimate) | parameters.to_dict() | additional_info
    try:
        return pd.DataFrame(d)
    except ValueError:
        return pd.DataFrame(d, index=[0])


def compile_dataframe(value_estimates, parameters, additional_info=None):
    if additional_info is None:
        additional_info = {}
    if isinstance(value_estimates, list):
        for i_bs, value_estimate in enumerate(value_estimates, start=1):
            value_estimate.bootstrap_idx = i_bs
    else:
        value_estimates = [value_estimates]
    return pd.concat(
        [
            _get_single_dataframe(value_estimate, parameters, additional_info)
            for value_estimate in value_estimates
        ]
    )


def print_log(output):
    print(output)
    sys.stdout.flush()


def dict_to_namespace(d):
    """Recursively convert a dictionary to an argparse namespace."""
    if isinstance(d, dict):
        for k, v in d.items():
            d[k] = dict_to_namespace(v)
        return argparse.Namespace(**d)
    else:
        return d


def namespace_to_dict(namespace):
    """Recursively convert an argparse namespace to a dictionary."""
    if isinstance(namespace, argparse.Namespace):
        return {k: namespace_to_dict(v) for k, v in vars(namespace).items()}
    else:
        return namespace


def _change_to_local_paths(d, cluster_project_path, local_project_path):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            recursive = {
                k: _change_to_local_paths(
                    v,
                    cluster_project_path,
                    local_project_path,
                )
            }
            out.update(recursive)
        elif (
            isinstance(k, str) 
            and ("path" in k or "dir" in k) 
            and isinstance(v, str)
        ):
            out[k] = v.replace(
                cluster_project_path,
                local_project_path,
            )
        else:
            out[k] = v
    return out


def load_config(config_path, as_object=False):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    local_home_path = os.environ.get("LOCAL_HOME_PATH")
    cluster_project_path = os.environ.get("CLUSTER_PROJECT_PATH")
    local_project_path = os.environ.get("LOCAL_PROJECT_PATH")
    
    if (
        local_home_path == str(Path.home())
        and cluster_project_path is not None
        and local_project_path is not None
    ):
        config = _change_to_local_paths(
            config,
            cluster_project_path,
            local_project_path,
        )
    
    return dict_to_namespace(config) if as_object else config


def create_results_dir(
    config,
    suffix=None,
    update_config=False,
):
    """Create a new results directory.

    Parameters
    ----------
    config : argparse.Namespace or dict
        Configuration object, either a `argparse.Namespace` or dictionary. Must 
        contain a key or attribute "results.root_dir" which specifies the base 
        directory for saving results.
    suffix : str, optional (default=None)
        Suffix to append to the results directory name.
    update_config : bool, optional (default=False)
        Whether to update the configuration dictionary with the new results path.
    """
    if isinstance(config, argparse.Namespace):
        results_root_dir = config.results.root_dir
    else:
        results_root_dir = config["results"]["root_dir"]
    
    time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    if suffix is not None:
        time_stamp += "_" + suffix
    
    results_dir = os.path.join(results_root_dir, time_stamp)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    if update_config:
        config = copy.deepcopy(config)
        if isinstance(config, argparse.Namespace):
            config.results.root_dir = results_dir
        else:
            config["results"]["root_dir"] = results_dir
        return results_dir, config
    else:
        return results_dir


def save_yaml(data, path, filename, **kwargs):
    if isinstance(data, argparse.Namespace):
        data = namespace_to_dict(data)
    Path(path).mkdir(parents=True, exist_ok=True)
    path = os.path.join(path, filename + ".yaml")
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, **kwargs)


def pad_pack_sequences(batch):
    sequences, targets = zip(*batch)

    # Pad and pack sequences.
    sequences = [torch.Tensor(sequence.values) for sequence in sequences]
    lengths = [sequence.size(0) for sequence in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True)
    packed_padded_sequences = pack_padded_sequence(
        padded_sequences,
        batch_first=True,
        lengths=lengths,
        enforce_sorted=False,
    )

    # Concatenate targets.
    tensor_targets = []
    for i, target in enumerate(targets):
        if target is None:
            target = [-1] * lengths[i]
        tensor_targets.append(torch.LongTensor(target))
    tensor_targets = torch.cat(tensor_targets)

    return packed_padded_sequences, tensor_targets


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)


class Dataset(skorch.dataset.Dataset):
    def transform(self, X, y):
        X, y = super().transform(X, y)
        X = torch.from_numpy(X).type(torch.float32)
        return X, y


class suppress_print:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
