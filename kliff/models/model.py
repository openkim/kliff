import copy
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple, Union

import numpy as np

from kliff.dataset.dataset import Configuration
from kliff.models.parameter import Parameter
# from kliff.models.parameter_transform import LogTransform, ParameterTransform
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
        # if params_space == "original":
        params = self.model_params
        s = "#" + "=" * 80 + "\n"
        s += "# Available parameters to optimize.\n"
        # s += f"# Parameters in `{params_space}` space.\n"
        name = self.__class__.__name__ if self.model_name is None else self.model_name
        s += f"# Model: {name}\n"

        s += "#" + "=" * 80 + "\n\n"

        for name, p in params.items():
            s += f"name: {name}\n"
            s += f"value: {p.numpy()}\n" # `.numpy()` converts any transform to original space
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

    # def set_opt_params(self, **kwargs):
    def set_opt_params(self, list_of_params):
        pass

    def set_one_opt_param(self, name: str, settings: List[List[Any]]):
        pass

    def echo_opt_params(self, filename: [Path, TextIO, None] = sys.stdout):
        """
        Echo the optimizing parameter to a file.
        """
        for param_key in self.model_params:
            # print(param_key, param_val)
            if self.model_params[param_key].is_trainable:
                print(f"Parameter:{param_key} : {self.model_params[param_key]}")

        # return self.opt_params.echo_opt_params(filename)

    def get_num_opt_params(self) -> int:
        return self.opt_params.get_num_opt_params()

    def get_opt_params(self) -> np.ndarray:
        opt_param = np.array([])
        for param_key in self.model_params:
            if self.model_params[param_key].is_trainable:
                opt_param = np.append(
                    opt_param, self.model_params[param_key]
                )
        return opt_param

    def update_model_params(self, params: Sequence[float]):
        i = 0
        for param_key in self.model_params:
            if self.model_params[param_key].is_trainable:
                self.model_params[param_key].copy_(params[i])
                i += 1

    def get_opt_param_name_value_and_indices(
        self, index: int
    ) -> Tuple[str, float, int, int]:
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
        return False#self.opt_params.has_opt_params_bounds()

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
        self.opt_params = Parameter(d["opt_params"]["value"])
        self.model_params = self.opt_params.model_params

    def parameters(self):
        """
        Get a dict of parameters that will be optimized.
        """
        param_opt_dict = {}
        for param_key in self.model_params:
            if self.model_params[param_key].is_trainable:
                param_opt_dict[param_key] = self.model_params[param_key]
        return param_opt_dict


class ModelError(Exception):
    def __init__(self, msg):
        super(ModelError, self).__init__(msg)
        self.msg = msg
