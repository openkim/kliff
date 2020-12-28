from collections.abc import Sequence
from pathlib import Path
from typing import Union


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


def split_string(string, length=80, starter=None):
    r"""Insert `\n` into long string such that each line has size no more than `length`.

    Parameters
    ----------
    length: int
        Targeted length of the each line.

    starter: string
        String to insert at the beginning of each line.

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


def to_path(path: Union[str, Path]) -> Path:
    """
    Convert str (or filename) to pathlib.Path.
    """
    return Path(path).expanduser().resolve()
