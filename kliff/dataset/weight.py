from typing import Any, Dict, Optional
import copy

import numpy as np


class Weight:
    """
    Base class for weight.

    This class mimics the behavior  provided by the ``residual_data`` argument from the
    previous version.
    Subclass can implement
        - compute_weight

    Args:
        config_weight: the weight for the entire configuration
        energy_weight: the weight for the energy
        forces_weight: the weight for the forces
        stress_weight: the weight for the stress
    """

    def __init__(
        self,
        config_weight: float = 1.0,
        energy_weight: float = 1.0,
        forces_weight: float = 1.0,
        stress_weight: float = 1.0,
    ):
        self._config_weight = config_weight  # weight for the entire configuration
        self._energy_weight = energy_weight
        self._forces_weight = forces_weight
        self._stress_weight = stress_weight

    @staticmethod
    def compute_weight(config):
        pass

    @property
    def config_weight(self):
        return self._config_weight

    @property
    def energy_weight(self):
        return self._energy_weight

    @property
    def forces_weight(self):
        return self._forces_weight

    @property
    def stress_weight(self):
        return self._stress_weight


default_weight_params = {
    "energy_weight_params": [1.0, 0.0],
    "forces_weight_params": [1.0, 0.0],
    "stress_weight_params": [1.0, 0.0],
}


class NonuniformWeight(Weight):
    r"""
    Non-uniform weight that is computed from the data. The calculation follows Lenosky et
    al. (1997), with some modification in notation,

    ..math:
       \frac{1}{w_m^2} = c_1^2 + c_2^2 \Vert f_m \Vert ^2.

    Args:
        config_weight: the weight for the entire configuration
        weight_params: a dictionary containing parameters c1 and c2 for calculating the
            weight of each property. The supported key value pairs are:
            - energy_weight_params: float or array-like (default: [1.0, 0.0])
            - forces_weight_params: float or array-like (default: [1.0, 0.0])
            - stress_weight_params: float or array-like (default: [1.0, 0.0])
            If a float is given, this number will be used to set c1, while c2 is set to
            zero. If an array-like with 2 elements is given, it should contain c1 as the
            first element and c2 as the second element.
    """

    def __init__(
        self, config_weight: float = 1.0, weight_params: Optional[Dict[str, Any]] = None
    ):
        self._config_weight = config_weight
        # Initiate the weight values. They will be changed latter.
        self._energy_weight = 0.0
        self._forces_weight = 0.0
        self._stress_weight = 0.0

        self._weight_params = self._check_weight_params(
            weight_params, default_weight_params
        )

    def compute_weight(self, config):
        """
        Compute the weights of the energy, forces, and stress data.
        """
        energy = config._energy
        forces = config._forces
        stress = config._stress

        # Energy
        if energy is not None:
            # Use the absolute value of the energy
            energy_norm = np.abs(energy)
            self._energy_weight = self._compute_weight_one_property(
                energy_norm, self._weight_params["energy_weight_params"]
            )
        # Forces
        if forces is not None:
            # Use the magnitude of the force vector
            forces_norm = np.linalg.norm(forces, axis=1)
            self._forces_weight = np.repeat(
                self._compute_weight_one_property(
                    forces_norm, self._weight_params["forces_weight_params"]
                ),
                3,
            )
        # Stress
        if stress is not None:
            # Use the Frobenius norm of the stress tensor
            normal_stress_norm = np.linalg.norm(stress[:3])
            shear_stress_norm = np.linalg.norm(stress[3:])
            stress_norm = np.sqrt(normal_stress_norm ** 2 + 2 * shear_stress_norm ** 2)
            self._stress_weight = self._compute_weight_one_property(
                stress_norm, self._weight_params["stress_weight_params"]
            )

    @staticmethod
    def _compute_weight_one_property(data_norm, property_weight_params):
        """
        Compute the weight based for one property.
        """
        c1, c2 = property_weight_params
        sigma2 = c1 ** 2 + (c2 * data_norm) ** 2
        weight = 1 / np.sqrt(sigma2)
        return weight

    @staticmethod
    def _check_weight_params(weight_params: Dict[str, Any], default: Dict[str, Any]):
        """
        Check the weight parameters and set it to the needed format, i.e., list with 2
        elements for each property.
        """
        if weight_params is not None:
            for key, value in weight_params.items():
                if key not in default:
                    raise WeightError(
                        f"Expect the keys of `weight_params` to be one or combinations "
                        f"of {', '.join(default.keys())}; got {key}. "
                    )
                else:
                    if (
                        np.ndim(value) == 0
                    ):  # If there is only a number given, use it to set c1
                        default[key][0] = value
                    else:  # To set c1 and c2, a list with 2 elements need to be passed in
                        default[key] = value
        return default


class WeightError(Exception):
    def __init__(self, msg):
        super(WeightError, self).__init__(msg)
        self.msg = msg
