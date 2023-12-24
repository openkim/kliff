import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple, Union

import numpy as np

from kliff.dataset.dataset import Configuration
from kliff.models.parameter import Parameter
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
            ModelError(f"Model not initialized to compute `{name}`.")
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
    def __init__(
        self,
        model_name: str = None,
    ):
        self.model_name = model_name
        self.model_params = self.init_model_params()
        self.influence_distance = self.init_influence_distance()
        self.supported_species = self.init_supported_species()
        self.mutable_param_list = []

    def init_model_params(self, *args, **kwargs) -> Dict[str, Parameter]:
        raise NotImplementedError("`init_model_params` not implemented.")

    def init_influence_distance(self, *args, **kwargs) -> float:
        raise NotImplementedError("`init_influence_distance` not implemented.")

    def init_supported_species(self, *args, **kwargs) -> Dict[str, int]:
        raise NotImplementedError("`init_supported_species` not implemented.")

    def get_compute_argument_class(self):
        raise NotImplementedError("`get_compute_argument_class` not implemented.")

    def write_kim_model(self, path: Path = None):
        raise NotImplementedError("`write_kim_model` not implemented yet.")

    def get_influence_distance(self) -> float:
        return self.influence_distance

    def get_supported_species(self) -> Dict[str, int]:
        return self.supported_species

    def get_model_params(self) -> Dict[str, Parameter]:
        return self.model_params

    def echo_model_params(
        self,
        filename: Union[Path, TextIO, None] = sys.stdout,
    ) -> str:
        params = self.model_params
        s = "#" + "=" * 80 + "\n"
        s += "# Available parameters to optimize.\n"
        name = self.__class__.__name__ if self.model_name is None else self.model_name
        s += f"# Model: {name}\n"

        s += "#" + "=" * 80 + "\n\n"

        for name, p in params.items():
            s += f"name: {name}\n"
            s += f"value: {p.get_numpy_array_model_space()}\n"  # `.numpy()` converts any transform to original space
            s += f"size: {len(p)}\n\n"

        if filename is not None:
            if isinstance(filename, (str, Path)):
                with open(filename, "w") as f:
                    f.write(s)
            else:
                print(s, file=filename)

        return s

    def read_opt_params(self, filename: Path):
        pass

    def set_params_mutable(self, opt_params: List[str]):
        pass

    def set_opt_params(self, **kwargs):
        keys = list(kwargs.keys())
        optimizable_keys = []
        for key in keys:
            if len(kwargs[key][0]) == 2:
                try:
                    if kwargs[key][0][1] == "fix":
                        continue
                except IndexError:
                    optimizable_keys.append(key)
            else:
                optimizable_keys.append(key)

        self.set_params_mutable(optimizable_keys)
        for name, setting in kwargs.items():
            self.set_one_opt_param(name, setting)

    def set_one_opt_param(self, name: str, settings: List[List[Any]]):
        param = self.model_params[name]
        # check the val kind
        param_old = param.get_numpy_array_param_space()
        for i in range(param_old.shape[0]):
            supplied_value = settings[i][0]
            if supplied_value == "default":
                param_old[i] = param.get_numpy_array_param_space()[i]
            elif isinstance(supplied_value, (int, float)):
                param_old[i] = supplied_value
            elif isinstance(supplied_value, np.ndarray) or isinstance(
                supplied_value, Parameter
            ):
                param_old[i] = supplied_value[0]
            else:
                raise ValueError("Settings array is not properly formatted")
        # When model is operating with transformed parameters
        # input is expected in transformed space
        param.copy_from_param_space(param_old)
        self.influence_distance = self.init_influence_distance()

    def echo_opt_params(self, filename: [Path, TextIO, None] = sys.stdout):
        """
        Echo the optimizing parameter to a file.
        """
        for param_key in self.model_params:
            if self.model_params[param_key].is_mutable:
                print(
                    f"Parameter:{param_key} : {self.model_params[param_key].get_numpy_array_model_space()}"
                )

        # return self.opt_params.echo_opt_params(filename)

    def get_num_opt_params(self) -> int:
        """
        Count and return number of optimizable parameters.
        Utilizes `Parameter` class.
        """
        i = 0
        for param_key in self.model_params:
            if self.model_params[param_key].is_mutable:
                i += 1
        return i

    def get_opt_params(self) -> np.ndarray:
        """
        Get optimizable parameters, concatenated as a single numpy array. Obtained numpy array is the state for
        the optimizer to optimize.
        Utilizes `Parameter` class.
        """
        opt_param = np.array([])
        for param_key in self.mutable_param_list:
            if self.model_params[param_key].is_mutable:  # additional check
                opt_param = np.append(
                    opt_param,
                    self.model_params[param_key].get_opt_numpy_array_param_space(),
                )
            else:
                # This should not happen
                raise AttributeError(
                    f"Parameter {param_key}, is not optimizable. Please report this error"
                )
        return opt_param

    def update_model_params(self, params: np.ndarray):
        """
        Copy and update the parameter from incoming params array. This method utilizes the
        parameters internal function to copy the parameter in a consistent manner.

        Args:
            params: numpy array with the shape of optimized parameter concatenated array.
        """
        i = 0
        for param_key in self.mutable_param_list:
            if self.model_params[param_key].is_mutable:
                param_size = (
                    self.model_params[param_key]
                    .get_opt_numpy_array_param_space()
                    .shape[0]
                )
                self.model_params[param_key].copy_from_param_space(
                    params[i : i + param_size]
                )
                i += param_size
            else:
                raise AttributeError(
                    f"Parameter {param_key}, is not optimizable. Please report this error"
                )

    def get_opt_param_name_value_and_indices(
        self, index: int
    ) -> Tuple[str, Union[float, np.ndarray], int]:
        # ) -> Tuple[str, float, int, int]:
        for param_key in self.mutable_param_list:
            if self.model_params[param_key].is_mutable:
                if index == self.model_params[param_key].index:
                    return self.model_params[
                        param_key
                    ].get_opt_param_name_value_and_indices()

    def get_formatted_param_bounds(self) -> Tuple[Tuple[int, int]]:
        """
        Get the lower and upper bounds of optimizing parameters, to be supplied directly
        to the scipy optimizer.

        Returns:
            tuple with bounds values. Unbound variables are provided with value (None, None)
        """
        bounds = []
        for param_key in self.mutable_param_list:
            if self.model_params[param_key].is_mutable:
                bounds.extend(self.model_params[param_key].get_bounds_param_space())
        return tuple(bounds)

    def opt_params_has_bounds(self) -> bool:
        """
        Whether bounds are set for any of the parameters.

        Returns:
            boolean true if any of the parameters are marked mutable.
        """
        has_bounds = False
        for param in self.model_params:
            if self.model_params[param].bounds is not None:
                has_bounds = True
                break
        return has_bounds

    def save(self, filename: Path = "trained_model.yaml"):
        """
        Save a model to disk.

        Args:
            filename: Path where to store the model.
        """
        opt_params = {}
        for param_key in self.model_params:
            if self.model_params[param_key].is_mutable:
                opt_params[param_key] = self.model_params[param_key].as_dict()
        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "opt_params": opt_params,
        }
        yaml_dump(d, filename)

    def load(self, filename: Path = "trained_model.yaml"):
        """
        Load a model on disk into memory.

        Args:
            filename: Path where the model is stored.
        """
        d = yaml_load(filename)
        for param in d["opt_params"]:
            self.model_params[param].from_dict(d["opt_params"][param])

    def named_parameters(self):
        """
        Get a dict of parameters that are marked as mutable, and hence can be optimized.
        The parameter values are subjected to change as per the transformations applied.

        Returns:
            Dictionary of parameters (~kliff.models.parameters.Parameter)
        """
        param_opt_dict = {}
        for param_key in self.model_params:
            if self.model_params[param_key].is_mutable:
                param_opt_dict[param_key] = self.model_params[param_key]
        return param_opt_dict

    def parameters(self):
        """
        Get a list of parameters that are marked as mutable, and hence can be optimized.

        Returns:
            List of parameters (~kliff.models.parameters.Parameter)
        """
        param_opt_list = []
        for param_key in self.model_params:
            if self.model_params[param_key].is_mutable:
                param_opt_list.append(self.model_params[param_key])
        return param_opt_list

    # def parameters(self):
    #     """
    #     Get an iterator of parameters that are marked as mutable, and hence can be optimized.
    #
    #     Returns:
    #         Iterator of parameters (~kliff.models.parameters.Parameter)
    #     """
    #     param_opt_list = []
    #     for param_key in self.model_params:
    #         if self.model_params[param_key].is_mutable:
    #             yield self.model_params[param_key]
    #


class ModelError(Exception):
    def __init__(self, msg):
        super(ModelError, self).__init__(msg)
        self.msg = msg
