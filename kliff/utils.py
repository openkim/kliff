import os
import random
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
    except ImportError:
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
            f"https://raw.githubusercontent.com/mjwen/kliff/master/examples/"
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


def create_directory(path: Union[str, Path], is_directory=False):
    p = to_path(path)
    if is_directory:
        dirname = p
    else:
        dirname = p.parent
    if not dirname.exists():
        os.makedirs(dirname)


def yaml_dump(obj, filename):
    create_directory(filename)
    with open(to_path(filename), "w") as f:
        yaml.dump(obj, f, default_flow_style=False)


def yaml_load(filename):
    with open(to_path(filename), "r") as f:
        obj = yaml.safe_load(f)
    return obj
