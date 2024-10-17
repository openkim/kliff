import os
import pickle
import random
import subprocess
import tarfile
from collections.abc import Sequence
from pathlib import Path
from typing import Union

import numpy as np
import requests
import yaml


def length_equal(a, b):
    if isinstance(a, Sequence) and isinstance(b, Sequence):
        if len(a) == len(b):
            return True
        else:
            return False
    else:
        return True


def torch_available():
    try:
        import torch

        return True
    except ModuleNotFoundError:
        return False


def torch_geometric_available():
    try:
        import torch_geometric

        return True
    except ModuleNotFoundError:
        return False


def split_string(string: str, length=80, starter: str = None):
    r"""
    Insert `\n` into long string such that each line has size no more than `length`.

    Args:
        string: The string to split.
        length: Targeted length of the each line.
        starter: String to insert at the beginning of each line.
    """

    if starter is not None:
        target_end = length - len(starter) - 1
    else:
        target_end = length

    sub_string = []
    while string:
        end = target_end
        if len(string) > end:
            while end >= 0 and string[end] != " ":
                end -= 1
            end += 1
        sub = string[:end].strip()
        if starter is not None:
            sub = starter + " " + sub
        sub_string.append(sub)
        string = string[end:]

    return "\n".join(sub_string) + "\n"


def seed_all(seed=35, cudnn_benchmark=False, cudnn_deterministic=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if torch_available():
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic


def to_path(path: Union[str, Path]) -> Path:
    """
    Convert str (or filename) to pathlib.Path.
    """
    return Path(path).expanduser().resolve()


def download_dataset(dataset_name: str) -> Path:
    """
    Download dataset and untar it.

    Args:
        dataset_name: name of the dataset

    Returns:
        Path to the dataset
    """
    path = to_path(dataset_name)

    if not path.exists():
        tarball = path.with_suffix(".tar.gz")

        # download
        url = (
            f"https://raw.githubusercontent.com/openkim/kliff/master/examples/"
            f"{dataset_name}.tar.gz"
        )

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(tarball, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        # untar
        tf = tarfile.open(tarball, "r:gz")
        tf.extractall(path.parent)

        # # remove tarball
        # tarball.unlink()

    return path


def create_directory(path: Union[str, Path], is_directory: bool = False):
    p = to_path(path)
    if is_directory:
        dirname = p
    else:
        dirname = p.parent
    if not dirname.exists():
        os.makedirs(dirname)


def yaml_dump(data, filename: Union[Path, str]):
    """
    Dump data to a yaml file.
    """
    create_directory(filename)
    with open(to_path(filename), "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def yaml_load(filename: Union[Path, str]):
    """
    Load data from a yaml file.
    """
    with open(to_path(filename), "r") as f:
        data = yaml.safe_load(f)

    return data


def pickle_dump(data, filename: Union[Path, str]):
    """
    Dump data to a pickle file.
    """

    create_directory(filename)
    with open(to_path(filename), "wb") as f:
        pickle.dump(data, f)


def pickle_load(filename: Union[Path, str]):
    """
    Load data from a pikel file.
    """
    with open(to_path(filename), "rb") as f:
        data = pickle.load(f)

    return data


def stress_to_voigt(input_stress: np.ndarray) -> list:
    """
    Convert stress from 3x3 tensor notation to 6x1 Voigt notation.
    :math:`\sigma_{ij} = [\sigma_{11}, \sigma_{22}, \sigma_{33}, \sigma_{23}, \sigma_{13}, \sigma_{12}]`

    Args:
        input_stress: Stress tensor in Voigt notation or tensor notation.

    Returns:
        stress: Stress tensor Voigt notation.
    """
    stress = [0.0] * 6
    if input_stress.ndim == 2:
        # tensor -> Voigt
        stress[0] = input_stress[0, 0]
        stress[1] = input_stress[1, 1]
        stress[2] = input_stress[2, 2]
        stress[3] = input_stress[1, 2]
        stress[4] = input_stress[0, 2]
        stress[5] = input_stress[0, 1]
    else:
        raise ValueError("input_stress must be a 2D array")

    return stress


def stress_to_tensor(input_stress: list) -> np.ndarray:
    """
    Convert stress from 6x1 Voigt notation to 3x3 tensor notation.

    Args:
        input_stress: Stress tensor in Voigt notation.

    Returns:
        stress: Stress tensor notation.
    """
    stress = np.zeros((3, 3))
    stress[0, 0] = input_stress[0]
    stress[1, 1] = input_stress[1]
    stress[2, 2] = input_stress[2]
    stress[1, 2] = stress[2, 1] = input_stress[3]
    stress[0, 2] = stress[2, 0] = input_stress[4]
    stress[0, 1] = stress[1, 0] = input_stress[5]

    return stress


def is_kim_model_installed(model_name: str) -> bool:
    """
    Check if the KIM model is installed in any collection.

       Args:
           model_name: name of the model.
    """
    model_list = subprocess.run(
        ["kim-api-collections-management", "list"], capture_output=True, text=True
    )
    if model_name in model_list.stdout:
        return True
    else:
        return False


def install_kim_model(model_name: str, collection: str = "user") -> bool:
    """
    Install the KIM model

    Args:
        model_name: name of the model.
        collection: name of the collection.

    Returns:
        True if the model is now installed, False otherwise.
    """
    if not is_kim_model_installed(model_name):
        output = subprocess.run(
            ["kim-api-collections-management", "install", collection, model_name],
            check=True,
        )
        return output.returncode == 0
    else:
        return True

def get_n_configs_in_xyz(file_path: str) -> int:
    """
    Get the number of configurations in a xyz file. It uses the grep command to count the number of lines
    that contain only numbers.
    Args:
        file_path: Path to the xyz file.

    Returns:

    """
    pattern = "^[0-9]+$"
    # Run the grep command and capture the output
    result = subprocess.run(
        ["grep", "-Ec", pattern, file_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Check if there is any error
    if result.returncode != 0:
        raise Exception(result.stderr)
    else:
        num_atoms = int(result.stdout.strip())

    return num_atoms
