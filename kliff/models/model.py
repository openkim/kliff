import copy
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple, Union

import numpy as np

from kliff.dataset.dataset import Configuration
from kliff.models.parameter import OptimizingParameters, Parameter
from kliff.models.parameter_transform import ParameterTransform
from kliff.utils import yaml_dump, yaml_load


class ComputeArguments:
    """
    Compute property (e.g. energy, forces, and stress) for a configuration.

    This is the base class for other compute arguments. Typically, a user will not
    directly use this.

    Args:
        conf: atomic configurations
        supported_species: species supported by the potential model, with chemical
            symbol as key and integer code as value.
        influence_distance: influence distance (aka cutoff distance) to calculate neighbors
        compute_energy: whether to compute energy
        compute_forces: whether to compute forces
        compute_stress: whether to compute stress
    """

    implemented_property = []

    def __init__(
        self,
        conf: Configuration,
        supported_species: Dict[str, int],
        influence_distance: float,
        compute_energy: bool = True,
        compute_forces: bool = True,
        compute_stress: bool = False,
    ):

        self.conf = conf
        self.supported_species = supported_species
        self.influence_distance = influence_distance
        self.compute_energy = compute_energy
        self.compute_forces = compute_forces
        self.compute_stress = compute_stress

        self.compute_property = self._check_compute_property()
        self.results = {p: None for p in self.implemented_property}

    def compute(self, params: Dict[str, Parameter]):
        """
        Compute the properties required by the compute flags, and store them in
        self.results.

        Args:
            params: the parameters of the model.

        Example:
            energy = a_func_to_compute_energy()
            forces = a_func_to_compute_forces()
            stress = a_func_to_compute_stress()
            self.results['energy'] = energy
            self.results['forces'] = forces
            self.results['stress'] = stress
        """
        raise NotImplementedError('"compute" method not implemented.')

    def get_compute_flag(self, name: str) -> bool:
        """
        Check whether the model is asked to compute property.

        Args:
            name: name of the property, e.g. energy, forces, and stresses
        """
        if name in self.compute_property:
            return True
        else:
            return False

    def get_property(self, name: str) -> Any:
        """
        Get a property by name.

        Args:
            name: name of the property, e.g. energy, forces, and stresses
        """
        if name not in self.compute_property:
            ModelError(f"Model not initialized to comptue `{name}`.")
        return self.results[name]

    def get_energy(self) -> float:
        """
        Potential energy.
        """
        return self.get_property("energy")

    def get_forces(self) -> np.ndarray:
        """
        2D array of shape (N,3) of the forces on atoms, where N is the number of atoms
        in the configuration.
        """
        return self.get_property("forces")

    def get_stress(self) -> np.ndarray:
        """
        1D array of the virial stress, in Voigt notation.
        """
        return self.get_property("stress")

    def get_prediction(self) -> np.ndarray:
        """
        1D array of prediction from the model for the configuration.
        """
        if self.compute_energy:
            energy = self.results["energy"]
            pred = np.asarray([energy])
        else:
            pred = np.asarray([])

        if self.compute_forces:
            forces = self.results["forces"]
            pred = np.concatenate((pred, forces.ravel()))

        if self.compute_stress:
            stress = self.results["stress"]
            pred = np.concatenate((pred, stress))

        return pred

    def get_reference(self) -> np.ndarray:
        """
        1D array of reference values for the configuration.
        """
        if self.compute_energy:
            energy = self.conf.energy
            ref = np.asarray([energy])
        else:
            ref = np.asarray([])

        if self.compute_forces:
            forces = self.conf.forces
            ref = np.concatenate((ref, forces.ravel()))

        if self.compute_stress:
            stress = self.conf.stress
            ref = np.concatenate((ref, stress))

        return ref

    def _check_compute_property(self):
        compute_property = []
        if self.compute_energy:
            if "energy" not in self.implemented_property:
                raise NotImplementedError("`Energy` not implemented in model.")
            else:
                compute_property.append("energy")
        if self.compute_forces:
            if "forces" not in self.implemented_property:
                raise NotImplementedError("`Forces` not implemented in model.")
            else:
                compute_property.append("forces")
        if self.compute_stress:
            if "stress" not in self.implemented_property:
                raise NotImplementedError("`Stress` not implemented in model.")
            else:
                compute_property.append("stress")

        return compute_property


class Model:
    """
    Base class for all physics-motivated models.

    Typically, a user will not directly use this.

    Args:
        model_name: name of the model.
        params_transform: optional transformation of parameters. Let's call the
            parameters initialized in `init_model_params()` as original parameter
            space. Sometimes, it's easier to work in another parameter (e.g.
            optimizers can perform better in the log space). Then we can use this
            parameter transformation class to transform between the original
            parameter space and the new easy-to-work space. Typically, a model only
            knows how to work in its original space to compute, e.g. energy and
            forces, so we need to inverse transform parameters back to original space
            (after an optimizer update its value in the log space).
            A `params_transform` instance should implement both a `transform` and an
            `inverse_transform` method to accomplish the above tasks.
            Note, all the parameters of this (the `Model`) class
            (e.g. `self.model_params`, and `self.opt_params`) are in the transformed
            easy-to-work space.
    """

    def __init__(
        self,
        model_name: str = None,
        params_transform: Optional[ParameterTransform] = None,
    ):
        self.model_name = model_name
        self.params_transform = params_transform

        self.model_params = self.init_model_params()

        if self.params_transform is not None:
            # make a copy since params_transform can make changes in place
            transformed = copy.deepcopy(self.model_params)
            self.model_params_transformed = self.params_transform(transformed)
        else:
            self.model_params_transformed = self.model_params

        self.opt_params = OptimizingParameters(self.model_params_transformed)
        self.influence_distance = self.init_influence_distance()
        self.supported_species = self.init_supported_species()

    def init_model_params(self, *args, **kwargs) -> Dict[str, Parameter]:
        """
        Initialize the parameters of the model.

        Should return a dictionary of parameters.

        Example:
            model_params = {"sigma": Parameter([0.5]
                      "epsilon": Parameter([0.4])

            return model_params
        """
        raise NotImplementedError("`init_model_params` not implemented.")

    def init_influence_distance(self, *args, **kwargs) -> float:
        """
        Initialize the influence distance (aka cutoff distance) of the model.

        This is used to compute the neighbor list of each atom.

        Example:
            return 5.0
        """
        raise NotImplementedError("`init_influence_distance` not implemented.")

    def init_supported_species(self, *args, **kwargs) -> Dict[str, int]:
        """
        Initialize the supported species of the model.

        Should return a dict with chemical symbol as key and integer code as value.

        Example:
            return {"C:0, "O":1}
        """
        raise NotImplementedError("`init_supported_species` not implemented.")

    def get_compute_argument_class(self):
        # NOTE, change to compute argument class as needed
        # return ComputeArguments
        raise NotImplementedError("`get_compute_argument_class` not implemented.")

    def write_kim_model(self, path: Path = None):
        raise NotImplementedError("`write_kim_model` not implemented yet.")

    def get_influence_distance(self) -> float:
        """
        Return influence distance (aka cutoff distance) of the model.
        """
        return self.influence_distance

    def get_supported_species(self) -> Dict[str, int]:
        """
        Return supported species of the model, a dict with chemical symbol as key and
        integer code as value.
        """
        return self.supported_species

    def get_model_params(self) -> Dict[str, Parameter]:
        """
        Return all parameters of the model.
        """
        return self.model_params

    def echo_model_params(
        self,
        filename: Union[Path, TextIO, None] = sys.stdout,
        params_space: str = "original",
    ) -> str:
        """
        Echo the model parameters.

        Args:
            filename: Path to write the model parameter info (e.g. sys.stdout).
                If `None`, do not write.
            params_space: In which space to show the parameters, options are `original`
                and `transformed`.

        Returns:
            model parameter info in a string
        """

        if params_space == "original":
            params = self.model_params
        elif params_space == "transformed":
            if self.params_transform is None:
                raise RuntimeError(
                    "Cannot obtain parameters in transformed space when "
                    "`params_transform` of model is not set."
                )
            else:
                params = self.model_params_transformed
        else:
            supported = ["original", "transformed"]
            raise ValueError(
                f"Expect `params_space` to be one of {supported}; got {params_space}"
            )

        s = "#" + "=" * 80 + "\n"
        s += "# Available parameters to optimize.\n"
        s += f"# Parameters in `{params_space}` space.\n"
        name = self.__class__.__name__ if self.model_name is None else self.model_name
        s += f"# Model: {name}\n"

        s += "#" + "=" * 80 + "\n\n"

        for name, p in params.items():
            s += f"name: {name}\n"
            s += f"value: {p.value}\n"
            s += f"size: {len(p)}\n\n"

        if filename is not None:
            if isinstance(filename, (str, Path)):
                with open(filename, "w") as f:
                    f.write(s)
            else:
                print(s, file=filename)

        return s

    def read_opt_params(self, filename: Path):
        """
        Read optimizing parameters from a file.

        Each parameter is a 1D array, and each component of the parameter array should
        be listed in a new line. Each line can contain 1, 2, or 3 elements, described
        in details below:

        1st element: float or `DEFAULT`
            Initial guess of the parameter component. If `DEFAULT` (case insensitive),
            the value from the calculator is used as the initial guess.

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
        using a dictionary by calling the `set_opt_params()` or `set_one_opt_params()`
        method.

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
        self.opt_params.read(filename)
        self.model_params = self._inverse_transform_params(self.opt_params.model_params)

        # reset influence distance in case it depends on parameters and changes
        self.init_influence_distance()

    def set_opt_params(self, **kwargs):
        """
        Set the parameters that will be optimized.

        One or more parameters can be set. Each argument is for one parameter, where the
        argument name is the parameter name, the value of the argument is the
        settings(including initial value, fix flag, lower bound, and upper bound).

        The value of the argument should be a list of list, where each inner list is for
        one component of the parameter, which can contain 1, 2, or 3 elements.
         See `~kliff.model.model.Model.read_opt_params()` for the options of the elements.

        Example:
           instance.set(A=[['DEFAULT'], [2.0, 1.0, 3.0]], B=[[1.0, 'FIX'], [2.0, 'INF', 3.0]])
        """
        self.opt_params.set(**kwargs)
        self.model_params = self._inverse_transform_params(self.opt_params.model_params)

        # reset influence distance in case it depends on parameters and changes
        self.init_influence_distance()

    def set_one_opt_param(self, name: str, settings: List[List[Any]]):
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
        self.opt_params.set_one(name, settings)
        self.model_params = self._inverse_transform_params(self.opt_params.model_params)

        # reset influence distance in case it depends on parameters and changes
        self.init_influence_distance()

    def echo_opt_params(self, filename: [Path, TextIO, None] = sys.stdout):
        """
        Echo the optimizing parameter to a file.
        """
        return self.opt_params.echo_opt_params(filename)

    def get_num_opt_params(self) -> int:
        """
        Number of optimizing parameters.

        This is the total number of model parameter components. For example,
        if the model has two parameters set to be optimized and each have two components,
        this will be four.
        """
        return self.opt_params.get_num_opt_params()

    def get_opt_params(self) -> np.ndarray:
        """
         Nest all optimizing parameter values (except the fixed ones) to a 1D array.

        The obtained values can be provided to the optimizer as the starting parameters.

        This is the opposite operation of update_model_params().

        Returns:
            opt_params: A 1D array of nested optimizing parameter values.
        """
        return self.opt_params.get_opt_params()

    def update_model_params(self, params: Sequence[float]):
        """
        Update the optimizing parameter values from a sequence of float.

        This is the opposite operation of get_opt_params().

        Note, self.model_params will be updated as well, since self.opt_params is
        initialized from self.model_params without copying the `Parameter` instance.

        Args:
            params: updated parameter values from the optimizer.
        """
        self.model_params_transformed = self.opt_params.update_opt_params(params)
        self.model_params = self._inverse_transform_params(
            self.model_params_transformed
        )

    def get_opt_param_name_value_and_indices(
        self, index: int
    ) -> Tuple[str, float, int, int]:
        """
        Get the `name`, `value`, `parameter_index`, and `component_index` of optimizing
        parameter in slot `index`.

        Args:
            index: slot index of the optimizing parameter
        """
        return self.opt_params.get_opt_param_name_value_and_indices(index)

    def get_opt_params_bounds(self) -> List[Tuple[int, int]]:
        """
        Get the lower and upper bounds of optimizing parameters.
        """
        return self.opt_params.get_opt_params_bounds()

    def has_opt_params_bounds(self) -> bool:
        """
        Whether bounds are set for some parameters.
        """
        return self.opt_params.has_opt_params_bounds()

    def save(self, filename: Path = "trained_model.yaml"):
        """
        Save a model to disk.

        Args:
            filename: Path where to store the model.
        """
        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "opt_params": self.opt_params.as_dict(),
        }
        yaml_dump(d, filename)

    def load(self, filename: Path = "trained_model.yaml"):
        """
        Load a model on disk into memory.

        Args:
            filename: Path where the model is stored.
        """
        d = yaml_load(filename)
        self.opt_params = OptimizingParameters.from_dict(d["opt_params"])

        # Set model_params to opt_params.model_params since they should share the
        # underlying `Parameter` objects
        self.model_params = self.opt_params.model_params

    def _inverse_transform_params(self, model_params_transformed: Dict[str, Parameter]):
        """
        Inverse transform the parameters.
        """
        if self.params_transform is not None:
            # make a copy since params_transform can make changes in place
            transformed = copy.deepcopy(model_params_transformed)
            return self.params_transform.inverse_transform(transformed)
        else:
            return model_params_transformed


class ModelError(Exception):
    def __init__(self, msg):
        super(ModelError, self).__init__(msg)
        self.msg = msg
