import logging
import os
import pickle
import sys
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from monty.json import MSONable

logger = logging.getLogger(__name__)


class Parameter(MSONable):
    """
    Potential parameter.

    To follow the KIM model paradigm, a parameter is a list of floats that can be
    adjusted to fit the potential to some parameters.

    This class is mainly by physically-motivated potentials.

    Args:
        value: parameter value, which can have multiple components.
        fixed: whether parameter component should be fixed (i.e. not used for fitting).
            Should have the same length of `value`. Default to not fixed for all
            components.
        lower_bound: lower bound of the allowed value for the parameter. Default to
            `None`, i.e. no lower bound is applied.
        upper_bound: upper bound of the allowed value for the parameter. Default to
            `None`, i.e. no lower bound is applied.
        name: name of the parameter.
    """

    def __init__(
        self,
        value: Union[Sequence[float]],
        fixed: Optional[Sequence[bool]] = None,
        lower_bound: Optional[Sequence[float]] = None,
        upper_bound: Optional[Sequence[float]] = None,
        name: Optional[str] = None,
    ):

        self._value = _check_shape(value, "parameter value")
        self._fixed = (
            [False] * len(self._value)
            if fixed is None
            else _check_shape(fixed, "fixed")
        )
        self._lower_bound = (
            [None] * len(self._value)
            if lower_bound is None
            else _check_shape(lower_bound, "lower_bound")
        )
        self._upper_bound = (
            [None] * len(self._value)
            if upper_bound is None
            else _check_shape(upper_bound, "upper_bound")
        )
        self._name = name

    @property
    def value(self) -> List[float]:
        """
        Parameter value.
        """
        return self._value

    def set_value(self, index: int, v: float):
        """
        Set the value of a component.
        """
        self._value[index] = v

    @property
    def fixed(self) -> List[bool]:
        """
        Whether each parameter component is fixed or not (i.e. allow to fitting or not).
        """
        return self._fixed

    def set_fixed(self, index: int, v: bool):
        """
        Set the fixed status of a component of the parameter.

        Args:
            index: index of the component
            v: fix status
        """
        self._fixed[index] = v

    @property
    def lower_bound(self) -> List[float]:
        """
        Lower bound of parameter.
        """
        return self._lower_bound

    def set_lower_bound(self, index: int, v: bool):
        """
        Set the lower bound of a component of the parameter.

        Args:
            index: index of the component
            v: fix status
        """
        self._lower_bound[index] = v

    @property
    def upper_bound(self) -> List[float]:
        """
        Upper bound of parameter.
        """
        return self._upper_bound

    def set_upper_bound(self, index: int, v: bool):
        """
        Set the upper bound of a component of the parameter.

        Args:
            index: index of the component
            v: fix status
        """
        self._upper_bound[index] = v

    @property
    def name(self) -> str:
        """
        Name of the parameter.
        """
        return self._name

    def __len__(self):
        return len(self.value)

    def __getitem__(self, i: int):
        return self._value[i]

    def __setitem__(self, i: int, v: float):
        self._value[i] = v

    def as_dict(self):
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "value": self._value,
            "fixed": self._fixed,
            "lower_bound": self._lower_bound,
            "upper_bound": self._upper_bound,
            "name": self._name,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            value=d["value"],
            fixed=d["fixed"],
            lower_bound=d["lower_bound"],
            upper_bound=d["upper_bound"],
            name=d["name"],
        )


class FittingParameter:
    """
    Parameters of a model that will be optimized.

    This can be all the model parameters or a subset of the model parameters.
    It interacts with optimizer to provide initial guesses of parameter values and
    receive updated parameters from the optimizer (and will be used by calculator to
    update model parameters).

    Args:
        model_params: {name, parameter} all the parameters of a model.
    """

    def __init__(self, model_params: Dict[str, Parameter]):

        self.model_params = model_params

        # key: parameter name
        # values: {'size', 'value', 'use_default', 'fix', 'lower_bound', 'upper_bound'}
        self.params = OrderedDict()

        # components of params that are optimized
        self._index = []

    def read(self, filename: Path):
        """
        Read the parameters that will be optimized from a file.

        Each parameter is a 1D array, and each component of the parameter array should be
        listed in a new line. Each line can contains 1, 2, or 3 elements, described in
        details below:

        1st element: float or `DEFAULT`
            Initial guess of the parameter component. If `DEFAULT` (case insensitive), the
            value from the calculator is used as the initial guess.

        The 2nd and 3rd elements are optional.

        If 2 elements are provided:

        2nd element: `FIX` (case insensitive)
            If `FIX`, the corresponding component will not be optimized.

        If 3 elements are provided:

        2nd element: float or `INF` (case insensitive)
            Lower bound of the parameter. `INF` indicates that the lower bound is
            negative infinite, i.e. no lower bound is applied.

        3rd element: float or `INF` (case insensitive)
            Upper bound of the parameter. `INF` indicates that the upper bound is
            positive infinite, i.e. no upper bound is applied.

        Instead of reading fitting parameters from a file, you can also setting them
        using a dictionary by calling the `set()` method.

        Args:
            filename: path to file that includes the fitting parameters.

        Example:

            # put the below in a file, say `params.txt` and you can read the fitting
            # parameters by this_class.read(filename="params.txt")

            A
            DEFAULT
            1.1

            B
            DEFAULT FIX
            1.1     FIX

            C
            DEFAULT  0.1  INF
            1.0      INF  2.1
            2.0      FIX
        """

        with open(filename, "r") as fin:
            lines = fin.readlines()
            lines = _remove_comments(lines)

        num_line = 0
        while num_line < len(lines):
            line = lines[num_line].strip()
            num_line += 1

            if line in self.params:
                warnings.warn(
                    f"file `{filename}`, line {num_line}. Parameter `{line}` already "
                    f"set. Reset it.",
                    category=Warning,
                )

            if line not in self.model_params:
                raise ParameterError(
                    f"file `{filename}`, line {num_line}. Parameter `{line}` not "
                    f"supported."
                )

            name = line
            size = self.model_params[name].size

            settings = []
            for j in range(size):
                settings.append(lines[num_line].split())
                num_line += 1

            self.set_one(name, settings)

    def set(self, **kwargs):
        """
        Set the parameters that will be optimized.

        One or more parameters can be set. Each argument is for one parameter, where the
        argument name is the parameter name, the value of the argument is the
        settings(including initial value, fix flag, lower bound, and upper bound).

        The value of the argument should be a list of list, where each inner list is for
        one component of the parameter, which can contain 1, 2, or 3 elements.  See
        `~kliff.model.parameter.FittingParameter.read()` for the options of the elements.

        Example:
            instance.set(A=[['DEFAULT'], [2.0, 1.0, 3.0]], B=[[1.0, 'FIX'], [2.0, 'INF', 3.0]])
        """
        for name, settings in kwargs.items():
            if name in self.params:
                msg = 'Parameter "{}" already set.'.format(name)
                warnings.warn(msg, category=Warning)
            if name not in self.model_params:
                raise ParameterError('Parameter "{}" not supported.'.format(name))
            self.set_one(name, settings)

    def set_one(self, name: str, settings: List[List[Any]]):
        """
        Set one parameter that will be optimized.

        The name of the parameter should be given as the first entry of a list (or tuple),
        and then each data line should be given in in a list.

        Args:
            name: name of a fitting parameter
            settings: initial value, flag to fix a parameter, lower and upper bounds of a
                parameter.

        Example:
            name = 'param_A'
            settings = [['default', 0, 20], [2.0, 'fix'], [2.2, 'inf', 3.3]]
            instance.set_one(name, settings)
        """
        # TODO update Example in docs, see test_parameter.py for example

        size = self.model_params[name].size
        if len(settings) != size:
            raise ParameterError(
                f"Expect {size} components for parameter `{name}`, got {len(settings)}."
            )

        # TODO a new class (subclassing Parameter) that having the below keys,
        #  and replacing the below (using Monty to serilization).
        tmp_dict = {
            "size": size,
            "value": [None for _ in range(size)],
            "use_default": [False for _ in range(size)],
            "fix": [False for _ in range(size)],
            "lower_bound": [None for _ in range(size)],
            "upper_bound": [None for _ in range(size)],
        }
        self.params[name] = tmp_dict

        for j, line in enumerate(settings):
            num_items = len(line)
            if num_items == 1:
                self._read_1_item(name, j, line)
            elif num_items == 2:
                self._read_2_item(name, j, line)
            elif num_items == 3:
                self._read_3_item(name, j, line)
            else:
                raise ParameterError(
                    f"More than 3 elements listed at data line {j+1} for "
                    f"parameter `{name}`."
                )
            self._check_bounds(name)

        self._set_index(name)

    def echo_params(self, filename: Optional[Path] = None, print_size: bool = True):
        """
        Print the optimizing parameters to file or stdout.

        Args:
            filename: Path to the file to output the optimizing parameters. If `None`,
                print to stdout.
            print_size: Whether to print the size of parameters. (Each parameter
                may have one or more components).
        """

        if filename is not None:
            fout = open(filename, "w")
        else:
            fout = sys.stdout

        print("#" + "=" * 80, file=fout)
        print("# Model parameters that are optimized.", file=fout)
        print("#" + "=" * 80, file=fout)
        print(file=fout)

        for name, attr in self.params.items():
            if print_size:
                print(name, attr["size"], file=fout)
            else:
                print(name, file=fout)

            for i in range(attr["size"]):
                print("{:24.16e}".format(attr["value"][i]), end=" ", file=fout)

                if attr["fix"][i]:
                    print("fix", end=" ", file=fout)

                lb = attr["lower_bound"][i]
                ub = attr["upper_bound"][i]
                has_lb = lb is not None
                has_ub = ub is not None
                has_bounds = has_lb or has_ub
                if has_bounds:
                    if has_lb:
                        print("{:24.16e}".format(lb), end=" ", file=fout)
                    else:
                        print("None", end=" ", file=fout)
                    if has_ub:
                        print("{:24.16e}".format(ub), end=" ", file=fout)
                    else:
                        print("None", end=" ", file=fout)
                print(file=fout)

            print(file=fout)

        if filename is not None:
            fout.close()

    # TODO change save and load to use yaml file instead of pickle
    def save(self, filename: Path):
        """Save the fitting parameters to file."""
        filename = os.path.abspath(filename)
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(filename, "wb") as f:
            pickle.dump(self.params, f)

    def load(self, filename: Path):
        """Load the fitting parameters from file."""
        # restore parameters
        with open(filename, "rb") as f:
            self.params = pickle.load(f)
        # restore index
        self._index = []
        for name in self.params.keys():
            self._set_index(name)

    def get_names(self) -> List[str]:
        """
        Return a list of parameter names.
        """
        return list(self.params.keys())

    def get_size(self, name: str):
        """
        Get the parameter size.

        Args:
            name: parameter name
        """
        return self.params[name]["size"]

    def get_value(self, name: str):
        """
        Get the value of the parameter.

        Args:
            name: parameter name
        """

        return self.params[name]["value"].copy()

    def set_value(self, name: str, value: List[float]):
        """
        Set the parameter value.

        Typically, you will not call this, but call set_one().

        Args:
            name: name of the parameter
            value: parameter values
        """
        self.params[name]["value"] = np.asarray(value)

    def get_lower_bound(self, name: str):
        """
        Lower bonds of a parameter.

        Args:
            name: parameter name
        """
        return self.params[name]["lower_bound"].copy()

    def get_upper_bound(self, name: str):
        """
        Upper bonds of parameter.

        Args:
            name: parameter name
        """
        return self.params[name]["upper_bound"].copy()

    def get_fix(self, name: str):
        """
        Whether parameter is fixed.

        Args:
            name: parameter name
        """
        return self.params[name]["fix"].copy()

    def get_number_of_opt_params(self) -> int:
        """
        Number of optimizing parameters.
        """
        return len(self._index)

    def get_opt_params(self) -> np.ndarray:
        """
        Nest all parameter values (except the fix ones) to a list.

        This is the opposite operation of update_params(). This can be fed to the
        optimizer as the starting parameters.

        Returns:
            opt_params: A 1D array of nested optimizing parameter values.
        """
        opt_x0 = []
        for idx in self._index:
            name = idx.name
            c_idx = idx.c_idx
            opt_x0.append(self.params[name]["value"][c_idx])
        if len(opt_x0) == 0:
            raise ParameterError("No parameters specified to optimize.")
        return np.asarray(opt_x0)

    def update_params(self, opt_x: List[float]):
        """
        Update parameter values from optimizer.

        This is the opposite operation of get_opt_params().

        Args:
            opt_x: updated parameter values from the optimizer.
        """
        for k, val in enumerate(opt_x):
            name = self._index[k].name
            c_idx = self._index[k].c_idx
            self.params[name]["value"][c_idx] = val

    def get_opt_param_name_value_and_indices(self, k):
        """
        Get the `name`, `value`, `parameter_index`, and `component_index` of an
        optimizing parameter given the slot `k`.
        """
        name = self._index[k].name
        p_idx = self._index[k].p_idx
        c_idx = self._index[k].c_idx
        value = self.params[name]["value"][c_idx]
        return name, value, p_idx, c_idx

    def get_opt_params_bounds(self):
        """Get the lower and upper parameter bounds."""
        bounds = []
        for idx in self._index:
            name = idx.name
            c_idx = idx.c_idx
            lower = self.params[name]["lower_bound"][c_idx]
            upper = self.params[name]["upper_bound"][c_idx]
            bounds.append([lower, upper])
        return bounds

    def has_opt_params_bounds(self):
        """Whether bounds are set for some parameters."""
        bounds = self.get_opt_params_bounds()
        for lb, up in bounds:
            if lb is not None or up is not None:
                return True
        return False

    def _read_1_item(self, name, j, line):
        self._read_1st_item(name, j, line[0])

    def _read_2_item(self, name, j, line):
        self._read_1st_item(name, j, line[0])
        if line[1].lower() == "fix":
            self.params[name]["fix"][j] = True
        else:
            raise ParameterError("Data at line {} of {} corrupted.".format(j + 1, name))

    def _read_3_item(self, name, j, line):
        self._read_1st_item(name, j, line[0])

        if (line[1] is not None) and (
            not (isinstance(line[1], str) and line[1].lower() == "none")
        ):
            try:
                self.params[name]["lower_bound"][j] = float(line[1])
            except ValueError as e:
                raise ParameterError(
                    "{}.\nData at line {} of {} corrupted.".format(e, j + 1, name)
                )

        if (line[2] is not None) and (
            not (isinstance(line[2], str) and line[2].lower() == "none")
        ):
            try:
                self.params[name]["upper_bound"][j] = float(line[2])
            except ValueError as e:
                raise ParameterError(
                    "{}.\nData at line {} of {} corrupted.".format(e, j + 1, name)
                )

    def _read_1st_item(self, name, j, first):
        if isinstance(first, str) and first.lower() == "default":
            self.params[name]["use_default"][j] = True
            model_value = self.model_params[name].value
            self.params[name]["value"][j] = model_value[j]
        else:
            try:
                self.params[name]["value"][j] = float(first)
            except ValueError as e:
                raise ParameterError(
                    '{}.\nData at line {} of parameter "{}" corrupted.'.format(
                        e, j + 1, name
                    )
                )

    def _check_bounds(self, name: str):
        """
        Check whether the initial guess of a parameter is within its lower and
        upper bounds.
        """
        attr = self.params[name]
        for i in range(attr["size"]):
            lower_bound = attr["lower_bound"][i]
            upper_bound = attr["upper_bound"][i]
            value = attr["value"][i]
            if lower_bound is not None:
                if value < lower_bound:
                    raise ParameterError(
                        'Initial guess at line {} of parameter "{}" '
                        "out of bounds.".format(i + 1, name)
                    )
            if upper_bound is not None:
                if value > upper_bound:
                    raise ParameterError(
                        'Initial guess at line {} of parameter "{}" '
                        "out of bounds.".format(i + 1, name)
                    )

    def _set_index(self, name: str):
        """
        Check whether a parameter component will be optimized or not (by checking its
        'fix' attribute). If yes, include it in the index list.

        Given a parameter and its values such as:

        PARAM_FREE_B
        1.1
        2.2  fix
        4.4  3.3  5.5

        the first slot (1.1) and the third slot (4.4) will be included in self._index,
        and later be optimized.
        """

        # TODO check if there is alternative so as not to use OrderedDict
        p_idx = list(self.model_params.keys()).index(name)
        size = self.params[name]["size"]
        fix = self.params[name]["fix"]

        for c_idx in range(size):
            if not fix[c_idx]:
                idx = _Index(name, p_idx, c_idx)
                # check whether already in self._index
                already_in = False
                for k, i in enumerate(self._index):
                    if idx == i:
                        already_in = k
                        break
                if already_in is not False:
                    self._index[already_in] = idx
                else:
                    self._index.append(idx)


class _Index(object):
    def __init__(self, name, parameter_index=None, component_index=None):
        self.name = name
        self.parameter_index = self.p_idx = parameter_index
        self.component_index = self.c_idx = component_index

    def set_parameter_index(self, index):
        self.parameter_index = self.p_idx = index

    def set_component_index(self, index):
        self.component_index = self.c_idx = index

    def __expr__(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ != other.__dict__
        else:
            return True


def _remove_comments(lines: List[str]):
    """
    Remove lines in a string list that start with # and content after #.
    """
    processed_lines = []
    for line in lines:
        line = line.strip()
        if not line or line[0] == "#":
            continue
        if "#" in line:
            line = line[0 : line.index("#")]
        processed_lines.append(line)
    return processed_lines


def _check_shape(x: Any, key="parameter"):
    """Check x to be a 1D array or list-like sequence."""
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, Sequence):
        if any(isinstance(i, Sequence) for i in x):
            raise ParameterError(f"{key} should be a 1D array (or list).")
    else:
        raise ParameterError(f"{key} should be a 1D array (or list).")

    return x


class ParameterError(Exception):
    def __init__(self, msg):
        super(ParameterError, self).__init__(msg)
        self.msg = msg
