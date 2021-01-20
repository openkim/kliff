import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple, Union

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
        index: integer index of the parameter.
    """

    def __init__(
        self,
        value: Union[Sequence[float]],
        fixed: Optional[Sequence[bool]] = None,
        lower_bound: Optional[Sequence[float]] = None,
        upper_bound: Optional[Sequence[float]] = None,
        name: Optional[str] = None,
        index: Optional[int] = None,
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
        self._index = index

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
        self._value[index] = float(v)

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

    def set_lower_bound(self, index: int, v: float):
        """
        Set the lower bound of a component of the parameter.

        Args:
            index: index of the component
            v: lower bound value
        """
        self._lower_bound[index] = v

    @property
    def upper_bound(self) -> List[float]:
        """
        Upper bound of parameter.
        """
        return self._upper_bound

    def set_upper_bound(self, index: int, v: float):
        """
        Set the upper bound of a component of the parameter.

        Args:
            index: index of the component
            v: upper bound value
        """
        self._upper_bound[index] = v

    @property
    def name(self) -> str:
        """
        Name of the parameter.
        """
        return self._name

    @property
    def index(self) -> int:
        """
        Integer index of the parameter.
        """
        return self._index

    def __len__(self):
        return len(self.value)

    def __getitem__(self, i: int):
        return self._value[i]

    def __setitem__(self, i: int, v: float):
        self._value[i] = float(v)

    def as_dict(self):
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "value": self._value,
            "fixed": self._fixed,
            "lower_bound": self._lower_bound,
            "upper_bound": self._upper_bound,
            "name": self._name,
            "index": self._index,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            value=d["value"],
            fixed=d["fixed"],
            lower_bound=d["lower_bound"],
            upper_bound=d["upper_bound"],
            name=d["name"],
            index=d["index"],
        )


class OptimizingParameters(MSONable):
    """
    A collection of paramters that will be optimized.

    This can be all the parameters of a model or a subset of the parameters of a model.
    The behavior of individual component of a parameter can also be controlled. For
    example, keep the 2nd component of a parameters fixed, while optimizing the other
    components.

    It interacts with optimizer to provide initial guesses of parameter values; it also
    receives updated parameters from the optimizer and update model parameters.

    Args:
        model_params: {name, parameter} all the parameters of a model. The attributes
            of these parameters will be modified to reflect whether it is optimized.
    """

    def __init__(self, model_params: Dict[str, Parameter]):
        self.model_params = model_params

        # list of optimizing param names
        self._params = []

        # individual components of parameters that are optimized
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

            # put the below in a file, say `model_params.txt` and you can read the fitting
            # parameters by this_class.read(filename="model_params.txt")

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
            name = lines[num_line].strip()
            num_line += 1

            if name in self._params:
                warnings.warn(
                    f"file `{filename}`, line {num_line}. Parameter `{name}` already "
                    f"set. Reset it.",
                    category=Warning,
                )

            if name not in self.model_params:
                raise ParameterError(
                    f"file `{filename}`, line {num_line}. Parameter `{name}` not "
                    f"supported."
                )

            num_components = len(self.model_params[name])

            settings = []
            for j in range(num_components):
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
        `~kliff.model.parameter.OptimizingParameters.read()` for the options of the
        elements.

        Example:
            instance.set(A=[['DEFAULT'], [2.0, 1.0, 3.0]], B=[[1.0, 'FIX'], [2.0, 'INF', 3.0]])
        """
        for name, settings in kwargs.items():
            if name in self._params:
                msg = f"Parameter `{name}` already set. Reset it."
                warnings.warn(msg, category=Warning)

            if name not in self.model_params:
                raise ParameterError(f"Parameter `{name}` not supported.")

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
        size = len(self.model_params[name])
        if len(settings) != size:
            raise ParameterError(
                f"Expect {size} components for parameter `{name}`, got {len(settings)}."
            )

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

            # probably setting cutoff?
            if "cut" in name:
                # and not fix it?
                if not (num_items == 2 and line[0].lower() == "fix"):
                    warnings.warn(
                        f"Parameter `{name}` seems to be a cutoff distance. KLIFF does "
                        f"not support optimizing cutoff. Remove it from your "
                        f"optimizing parameter or make it a `fix` parameter if you "
                        f"want to use a different cutoff distance from the default "
                        f"value in the model. Ignore this if `{name}` is not a cutoff "
                        f"distance."
                    )

        self._set_index(name)

        if name not in self._params:
            self._params.append(name)

    def echo_opt_params(
        self, filename: Union[Path, TextIO, None] = sys.stdout, echo_size: bool = True
    ) -> str:
        """
        Get the optimizing parameters as a string and/or print to file (stdout).

        Args:
            filename: Path to the file to output the optimizing parameters. If `None`,
                print to stdout.
            echo_size: Whether to print the size of parameters. (Each parameter
                may have one or more components).

        Returns:
            Optimizing parameters as a string.
        """

        s = "#" + "=" * 80 + "\n"
        s += "# Model parameters that are optimized.\n"
        s += "#" + "=" * 80 + "\n\n"

        for name in self._params:
            p = self.model_params[name]

            if echo_size:
                s += f"{name} {len(p)}\n"
            else:
                s += f"{name}\n"

            for i in range(len(p)):
                s += f"{p[i]:24.16e} "

                if p.fixed[i]:
                    s += "fix "

                lb = p.lower_bound[i]
                ub = p.upper_bound[i]
                has_lb = lb is not None
                has_ub = ub is not None
                has_bounds = has_lb or has_ub
                if has_bounds:
                    if has_lb:
                        s += f"{lb:24.16e} "
                    else:
                        s += "None "
                    if has_ub:
                        s += f"{ub:24.16e} "
                    else:
                        s += "None "
                s += "\n"

            s += "\n"

        if filename is not None:
            if isinstance(filename, (str, Path)):
                with open(filename, "w") as f:
                    f.write(s)
            else:
                filename.write(s)

        return s

    def get_num_opt_params(self) -> int:
        """
        Number of optimizing parameters.

        This is the total number of model parameter components. For example,
        if the model has two parameters set to be optimized and each have two components,
        this will be four.
        """
        return len(self._index)

    def get_opt_params(self) -> np.ndarray:
        """
        Nest all optimizing parameter values (except the fixed ones) to a 1D array.

        The obtained values can be provided to the optimizer as the starting parameters.

        This is the opposite operation of update_model_params().

        Returns:
            opt_params: A 1D array of nested optimizing parameter values.
        """
        params = []
        for idx in self._index:
            params.append(self.model_params[idx.name][idx.c_idx])
        if len(params) == 0:
            raise ParameterError("No parameters specified to optimize.")

        return np.asarray(params)

    def update_opt_params(self, params: Sequence[float]):
        """
        Update the optimizing parameter values from a sequence of float.

        This is the opposite operation of get_opt_params().

        Args:
            params: updated parameter values from the optimizer.
        """
        for k, val in enumerate(params):
            name = self._index[k].name
            c_idx = self._index[k].c_idx
            self.model_params[name][c_idx] = val

    def get_opt_param_name_value_and_indices(
        self, index: int
    ) -> Tuple[str, float, int, int]:
        """
        Get the `name`, `value`, `parameter_index`, and `component_index` of optimizing
        parameter in slot `index`.

        Args:
            index: slot index of the optimizing parameter
        """
        name = self._index[index].name
        p_idx = self._index[index].p_idx
        c_idx = self._index[index].c_idx
        value = self.model_params[name][c_idx]

        return name, value, p_idx, c_idx

    def get_opt_params_bounds(self) -> List[Tuple[int, int]]:
        """
        Get the lower and upper bounds of optimizing parameters.
        """
        bounds = []
        for idx in self._index:
            name = idx.name
            c_idx = idx.c_idx
            lower = self.model_params[name].lower_bound[c_idx]
            upper = self.model_params[name].upper_bound[c_idx]
            bounds.append([lower, upper])

        return bounds

    def has_opt_params_bounds(self) -> bool:
        """
        Whether bounds are set for some parameters.
        """
        bounds = self.get_opt_params_bounds()
        for low, up in bounds:
            if low is not None or up is not None:
                return True
        return False

    def _read_1_item(self, name, j, line):
        self._read_1st_item(name, j, line[0])

    def _read_2_item(self, name, j, line):
        self._read_1st_item(name, j, line[0])

        if line[1].lower() == "fix":
            self.model_params[name].set_fixed(j, True)
        else:
            raise ParameterError(f"Data at line {j+1} of {name} corrupted.")

    def _read_3_item(self, name, j, line):
        self._read_1st_item(name, j, line[0])

        value = self.model_params[name][j]

        # lower bound
        if (line[1] is not None) and (
            not (isinstance(line[1], str) and line[1].lower() == "none")
        ):
            try:
                self.model_params[name].set_lower_bound(j, float(line[1]))
            except ValueError as e:
                raise ParameterError(f"{e}.\nData at line {j+1} of {name} corrupted.")

            if float(line[1]) > value:
                raise ParameterError(
                    f"Lower bound ({line[1]}) larger than value({value}) at line {j+1} "
                    f"of parameter `{name}`"
                )

        # upper bound
        if (line[2] is not None) and (
            not (isinstance(line[2], str) and line[2].lower() == "none")
        ):
            try:
                self.model_params[name].set_upper_bound(j, float(line[2]))
            except ValueError as e:
                raise ParameterError(f"{e}.\nData at line {j+1} of {name} corrupted.")

            if float(line[2]) < value:
                raise ParameterError(
                    f"Upper bound ({line[2]}) smaller than value({value}) at line {j+1} "
                    f"of parameter `{name}`"
                )

    def _read_1st_item(self, name, j, first):
        if isinstance(first, str) and first.lower() == "default":
            pass
        else:
            try:
                self.model_params[name][j] = float(first)
            except ValueError as e:
                raise ParameterError(
                    f"{e}.\nData at line {j+1} of parameter `{name}` corrupted."
                )

    def _set_index(self, name: str):
        """
        Include parameter component that will be optimized (i.e. `fixed` is False) in
        the optimizing parameter index list.

        Given a parameter and its values such as:

        PARAM_FREE_B
        1.1
        2.2  fix
        4.4  3.3  5.5

        the first slot (1.1) and the third slot (4.4) will be included in self._index,
        and later be optimized.

        Args:
            name: name of the parameter
        """

        size = len(self.model_params[name])
        fix = self.model_params[name].fixed
        p_idx = self.model_params[name].index

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
                    warnings.warn(
                        f"Parameter `{name}` component `{c_idx}` already set. Reset it.",
                        category=Warning,
                    )
                    self._index[already_in] = idx
                else:
                    self._index.append(idx)

    def as_dict(self):
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "model_params": {k: v.as_dict() for k, v in self.model_params.items()},
            "params": self._params,
        }

    @classmethod
    def from_dict(cls, d):
        c = cls(
            model_params={
                k: Parameter.from_dict(v) for k, v in d["model_params"].items()
            }
        )
        c._params = d["params"]
        for name in d["params"]:
            c._set_index(name)

        return c


class _Index:
    """
    Mapping of a component of the optimizing parameter list to the model parameters dict.
    """

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

    return x.copy()


class ParameterError(Exception):
    def __init__(self, msg):
        super(ParameterError, self).__init__(msg)
        self.msg = msg
