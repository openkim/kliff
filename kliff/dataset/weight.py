import copy
from typing import Any, Dict, Optional, Union

import numpy as np
from loguru import logger


class Weight:
    """Base class for weight.

    This class mimics the behavior  provided by the ``residual_data`` argument from the
    previous version.
    Subclass can implement
        - compute_weight

    Parameters
    ----------
    config_weight: float (optional)
        The weight for the entire configuration
    energy_weight: float (optional)
        The weight for the energy
    forces_weight: float (optional)
        The weight for the forces
    stress_weight: float (optional)
        The weight for the stress
    """

    def __init__(
        self,
        config_weight: Union[float, None] = 1.0,
        energy_weight: Union[float, None] = 1.0,
        forces_weight: Union[float, None] = 1.0,
        stress_weight: Union[float, None] = 1.0,
    ):
        self._config_weight = config_weight
        self._energy_weight = energy_weight
        self._forces_weight = forces_weight
        self._stress_weight = stress_weight

    def compute_weight(self, config):
        self._check_compute_flag(config)

    @property
    def config_weight(self):
        return self._config_weight

    @config_weight.setter
    def config_weight(self, value):
        self._config_weight = value

    @property
    def energy_weight(self):
        return self._energy_weight

    @energy_weight.setter
    def energy_weight(self, value):
        self._energy_weight = value

    @property
    def forces_weight(self):
        return self._forces_weight

    @forces_weight.setter
    def forces_weight(self, value):
        self._forces_weight = value

    @property
    def stress_weight(self):
        return self._stress_weight

    @stress_weight.setter
    def stress_weight(self, value):
        self._stress_weight = value

    def __repr__(self):
        return f"Weights: config={self.config_weight}, energy={self.energy_weight}, forces={self.forces_weight}, stress={self.stress_weight}"

    def _check_compute_flag(self, config):
        """
        Check whether compute flag correctly set when the corresponding weight in
        residual data is 0.
        """
        ew = self.energy_weight
        fw = self.forces_weight
        sw = self.stress_weight
        msg = (
            '"{0}_weight" are near zero. Seems you do not want to use {0} in the '
            'fitting. You can set "use_{0}" in "calculator.create()" to "False" to speed'
            "up the fitting."
        )

        # If the weight are really small, but not zero, then warn the user. Zero weight
        # usually means that the property is used.
        if config._energy is not None and ew is not None and np.all(ew < 1e-12):
            logger.warning(msg.format("energy", ew))
        if config._forces is not None and fw is not None and np.all(fw < 1e-12):
            logger.warning(msg.format("forces", fw))
        if config._stress is not None and sw is not None and np.all(sw < 1e-12):
            logger.warning(msg.format("stress", sw))

    def to_dict(self):
        return {
            "config": self.config_weight,
            "energy": self.energy_weight,
            "forces": self.forces_weight,
            "stress": self.stress_weight,
        }


class MagnitudeInverseWeight(Weight):
    r"""Non-uniform weight that is computed from the data. The calculation follows
    Lenosky et al. (1997), with some modification in notation,

    ..math:
       \frac{1}{w_m^2} = c_1^2 + c_2^2 \Vert f_m \Vert ^2.

    Parameters
    ----------
    config_weight: float (optional)
        The weight for the entire configuration
    weight_params: dict (optional)
        A dictionary containing parameters c1 and c2 for calculating the
        weight of each property. The supported key value pairs are:
        - energy_weight_params: float or array-like (default: [1.0, 0.0])
        - forces_weight_params: float or array-like (default: [1.0, 0.0])
        - stress_weight_params: float or array-like (default: [1.0, 0.0])
        If a float is given, this number will be used to set c1, while c2 is set to
        zero. If an array-like with 2 elements is given, it should contain c1 as the
        first element and c2 as the second element.

    References
    ----------
    .. [Lenosky1997] T. J. Lenosky et al., “Highly optimized tight-binding model of
       silicon,” Phys. Rev. B, vol. 55, no. 3, pp. 15281544, Jan. 1997, doi:
       10.1103/PhysRevB.55.1528.

    """

    # Default parameters
    default_weight_params = {
        "energy_weight_params": [1.0, 0.0],
        "forces_weight_params": [1.0, 0.0],
        "stress_weight_params": [1.0, 0.0],
    }

    def __init__(
        self,
        config_weight: float = 1.0,
        weight_params: Optional[Dict[str, Any]] = None,
    ):
        self._config_weight = config_weight
        # Initiate the weight values. They will be changed latter.
        self._energy_weight = 0.0
        self._forces_weight = 0.0
        self._stress_weight = 0.0

        self._weight_params = self._check_weight_params(
            weight_params, self.default_weight_params
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
                [energy_norm], self._weight_params["energy_weight_params"], "energy"
            )[0]
        # Forces
        if forces is not None:
            # Use the magnitude of the force vector
            forces_norm = np.linalg.norm(forces, axis=1)
            self._forces_weight = np.repeat(
                self._compute_weight_one_property(
                    forces_norm, self._weight_params["forces_weight_params"], "forces"
                ),
                3,
            )
        # Stress
        if stress is not None:
            # Use the Frobenius norm of the stress tensor
            normal_stress_norm = np.linalg.norm(stress[:3])
            shear_stress_norm = np.linalg.norm(stress[3:])
            stress_norm = np.sqrt(normal_stress_norm**2 + 2 * shear_stress_norm**2)
            self._stress_weight = self._compute_weight_one_property(
                [stress_norm], self._weight_params["stress_weight_params"], "stress"
            )[0]

        self._check_compute_flag(config)

    @staticmethod
    def _compute_weight_one_property(data_norm, property_weight_params, property_type):
        """
        Compute the weight based for one property.
        """
        c1, c2 = property_weight_params
        sigma = np.array([np.linalg.norm([c1, c2 * dn]) for dn in data_norm])
        weight = 1 / sigma
        if np.any(sigma < 1e-12):
            logger.warning(
                f"Found near zero inverse {property_type} weight. Be aware that some "
                f"{property_type} data might be overweight."
            )
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
                    if np.ndim(value) == 0:
                        # If there is only a number given, use it to set c1
                        default[key][0] = value
                    elif np.ndim(value) == 1 and len(value) == 2:
                        # To set c1 and c2, a list with 2 elements need to be passed in
                        default[key] = value
                    else:
                        raise WeightError(
                            "Expect a float or a list of floats with format [c1, c2]"
                        )
        return default


class WeightError(Exception):
    def __init__(self, msg):
        super(WeightError, self).__init__(msg)
        self.msg = msg
