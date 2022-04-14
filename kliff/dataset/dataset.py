import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from kliff.dataset.extxyz import read_extxyz, write_extxyz
from kliff.utils import to_path
from loguru import logger

# map from file_format to file extension
SUPPORTED_FORMAT = {"xyz": ".xyz"}

# This is the default parameters to set the weight
default_weight_params = {
    "energy_weight_params": [1.0, 0.0],
    "forces_weight_params": [1.0, 0.0],
    "stress_weight_params": [1.0, 0.0],
}
# This is the default values of the weight, which correspond to ordinary least squares
default_weight = {
    "energy_weight": 1.0,
    "forces_weight": 1.0,
    "stress_weight": 1.0,
}


class Configuration:
    r"""
    Class of atomic configuration.

    This is used to store the information of an atomic configuration, e.g. supercell,
    species, coords, energy, and forces.

    Args:
        cell: A 3x3 matrix of the lattice vectors. The first, second, and third rows are
            :math:`a_1`, :math:`a_2`, and :math:`a_3`, respectively.
        species: A list of N strings giving the species of the atoms, where N is the
            number of atoms.
        coords: A Nx3 matrix of the coordinates of the atoms, where N is the number of
            atoms.
        PBC: A list with 3 components indicating whether periodic boundary condition
            is used along the directions of the first, second, and third lattice vectors.
        energy: energy of the configuration.
        forces: A Nx3 matrix of the forces on atoms, where N is the number of atoms.
        stress: A list with 6 components in Voigt notation, i.e. it returns
            :math:`\sigma=[\sigma_{xx},\sigma_{yy},\sigma_{zz},\sigma_{yz},\sigma_{xz},
            \sigma_{xy}]`. See: https://en.wikipedia.org/wiki/Voigt_notation
        weight: weight of the configuration and data point in the loss function.
        identifier: a (unique) identifier of the configuration

    """

    def __init__(
        self,
        cell: np.ndarray,
        species: List[str],
        coords: np.ndarray,
        PBC: List[bool],
        energy: float = None,
        forces: Optional[np.ndarray] = None,
        stress: Optional[List[float]] = None,
        weight: Optional[Dict[str, Any]] = None,
        weight_params: Optional[Dict[str, Any]] = None,
        identifier: Optional[Union[str, Path]] = None,
    ):
        self._cell = cell
        self._species = species
        self._coords = coords
        self._PBC = PBC
        self._energy = energy
        self._forces = forces
        self._stress = stress

        if weight is None:
            if weight_params is None:
                weight = {
                    "energy_weight": 1.0,
                    "forces_weight": 1.0,
                    "stress_weight": 1.0,
                }
            else:
                weight_params = _check_weight_params(
                    weight_params, default_weight_params
                )
                weight = _compute_energy_forces_stress_weights(
                    energy, forces, stress, weight_params
                )
        self._weight = weight

        self._identifier = identifier
        self._path = None

    # TODO enable config weight read in from file
    @classmethod
    def from_file(
        cls,
        filename: Path,
        weight_params: dict = None,
        file_format: str = "xyz",
    ):
        """
        Read configuration from file.

        Args:
            filename: Path to the file that stores the configuration.
            weight_params: Parameters that are used to compute the weight from the data.
            file_format: Format of the file that stores the configuration (e.g. `xyz`).
        """

        if file_format == "xyz":
            cell, species, coords, PBC, energy, forces, stress = read_extxyz(filename)
        else:
            raise ConfigurationError(
                f"Expect data file_format to be one of {list(SUPPORTED_FORMAT.keys())}, "
                f"got: {file_format}."
            )

        cell = np.asarray(cell)
        species = [str(i) for i in species]
        coords = np.asarray(coords)
        PBC = [bool(i) for i in PBC]
        energy = float(energy) if energy is not None else None
        forces = np.asarray(forces) if forces is not None else None
        stress = [float(i) for i in stress] if stress is not None else None

        # Set the weights
        weight_params = _check_weight_params(weight_params, default_weight_params)
        weight = _compute_energy_forces_stress_weights(
            energy, forces, stress, weight_params
        )

        self = cls(
            cell,
            species,
            coords,
            PBC,
            energy,
            forces,
            stress,
            weight,
            identifier=str(filename),
        )
        self._path = to_path(filename)

        return self

    def to_file(self, filename: Path, file_format: str = "xyz"):
        """
        Write the configuration to file.

        Args:
            filename: Path to the file that stores the configuration.
            file_format: Format of the file that stores the configuration (e.g. `xyz`).
        """
        filename = to_path(filename)
        if file_format == "xyz":
            write_extxyz(
                filename,
                self.cell,
                self.species,
                self.coords,
                self.PBC,
                self._energy,
                self._forces,
                self._stress,
            )
        else:
            raise ConfigurationError(
                f"Expect data file_format to be one of {list(SUPPORTED_FORMAT.keys())}, "
                f"got: {file_format}."
            )

    @property
    def cell(self) -> np.ndarray:
        """
        3x3 matrix of the lattice vectors of the configurations.
        """
        return self._cell

    @property
    def PBC(self) -> List[bool]:
        """
        A list with 3 components indicating whether periodic boundary condition
        is used along the directions of the first, second, and third lattice vectors.
        """
        return self._PBC

    @property
    def species(self) -> List[str]:
        """
        Species string of all atoms.
        """
        return self._species

    @property
    def coords(self) -> np.ndarray:
        """
        A Nx3 matrix of the Cartesian coordinates of all atoms.
        """
        return self._coords

    @property
    def energy(self) -> Union[float, None]:
        """
        Potential energy of the configuration.
        """
        if self._energy is None:
            raise ConfigurationError("Configuration does not contain energy.")
        return self._energy

    @property
    def forces(self) -> np.ndarray:
        """
        Return a `Nx3` matrix of the forces on each atoms.
        """
        if self._forces is None:
            raise ConfigurationError("Configuration does not contain forces.")
        return self._forces

    @property
    def stress(self) -> List[float]:
        r"""
        Stress of the configuration.

        The stress is given in Voigt notation i.e
        :math:`\sigma=[\sigma_{xx},\sigma_{yy},\sigma_{zz},\sigma_{yz},\sigma_{xz},
        \sigma_{xy}]`.

        """
        if self._stress is None:
            raise ConfigurationError("Configuration does not contain stress.")
        return self._stress

    @property
    def weight(self):
        """
        Get the weight of the configuration if the loss function.
        """
        return self._weight

    @weight.setter
    def weight(self, weight: float):
        """
        Set the weight of the configuration if the loss function.
        """
        self._weight = weight

    @property
    def identifier(self) -> str:
        """
        Return identifier of the configuration.
        """
        return self._identifier

    @identifier.setter
    def identifier(self, identifier: str):
        """
        Set the identifier of the configuration.
        """
        self._identifier = identifier

    @property
    def path(self) -> Union[Path, None]:
        """
        Return the path of the file containing the configuration. If the configuration
        is not read from a file, return None.
        """
        return self._path

    def get_num_atoms(self) -> int:
        """
        Return the total number of atoms in the configuration.
        """
        return len(self.species)

    def get_num_atoms_by_species(self) -> Dict[str, int]:
        """
        Return a dictionary of the number of atoms with each species.
        """
        return self.count_atoms_by_species()

    def get_volume(self) -> float:
        """
        Return volume of the configuration.
        """
        return abs(np.dot(np.cross(self.cell[0], self.cell[1]), self.cell[2]))

    def count_atoms_by_species(
        self, symbols: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """
        Count the number of atoms by species.

        Args:
            symbols: species to count the occurrence. If `None`, all species present
                in the configuration are used.

        Returns:
            {specie, count}: with `key` the species string, and `value` the number of
                atoms with each species.
        """

        unique, counts = np.unique(self.species, return_counts=True)
        symbols = unique if symbols is None else symbols

        natoms_by_species = dict()
        for s in symbols:
            if s in unique:
                natoms_by_species[s] = counts[list(unique).index(s)]
            else:
                natoms_by_species[s] = 0

        return natoms_by_species

    def order_by_species(self):
        """
        Order the atoms according to the species such that atoms with the same species
        have contiguous indices.
        """
        if self.forces is not None:
            species, coords, forces = zip(
                *sorted(
                    zip(self.species, self.coords, self.forces),
                    key=lambda pair: pair[0],
                )
            )
            self._species = np.asarray(species).tolist()
            self._coords = np.asarray(coords)
            self._forces = np.asarray(forces)
        else:
            species, coords = zip(
                *sorted(zip(self.species, self.coords), key=lambda pair: pair[0])
            )
            self._species = np.asarray(species)
            self._coords = np.asarray(coords)


def _compute_energy_forces_stress_weights(energy, forces, stress, weight_params):
    """
    Compute the weights of the energy, forces, and stress data.
    """
    # Energy
    if energy is not None:
        # Use the absolute value of the energy
        energy_norm = np.abs(energy)
        energy_weight = _compute_weight(
            energy_norm, weight_params["energy_weight_params"]
        )
    else:
        energy_weight = 0.0
    # Forces
    if forces is not None:
        # Use the magnitude of the force vector
        forces_norm = np.linalg.norm(forces, axis=1)
        forces_weight = np.repeat(
            _compute_weight(forces_norm, weight_params["forces_weight_params"]),
            3,
        )
    else:
        forces_weight = 0.0
    # Stress
    if stress is not None:
        # Use the Frobenius norm of the stress tensor
        normal_stress_norm = np.linalg.norm(stress[:3])
        shear_stress_norm = np.linalg.norm(stress[3:])
        stress_norm = np.sqrt(normal_stress_norm ** 2 + 2 * shear_stress_norm ** 2)
        stress_weight = _compute_weight(
            stress_norm, weight_params["stress_weight_params"]
        )
    else:
        stress_weight = 0.0

    weight = {
        "energy_weight": energy_weight,
        "forces_weight": forces_weight,
        "stress_weight": stress_weight,
    }
    return weight


def _compute_weight(data_norm, weight_params):
    """
    Compute the weight based on proposal by Lenosky et al. (1997), with some
    modification in notation.
    """
    c1, c2 = weight_params
    sigma2 = c1 ** 2 + (c2 * data_norm) ** 2
    weight = 1 / np.sqrt(sigma2)
    return weight


class Dataset:
    """
    A dataset of multiple configurations (:class:`~kliff.dataset.Configuration`).

    Args:
        path: Path of a file storing a configuration or filename to a directory containing
            multiple files. If given a directory, all the files in this directory and its
            subdirectories with the extension corresponding to the specified file_format
            will be read.
        weight_params: A dictionary containing parameters used to compute the weight from
            the data. The weights are calculated by

            ..math::
               w_m = \frac{1}{\sqrt{c_1^2 + (c_2 * \Vert f_m \Vert)^2}}

            For forces data, :math:`\Vert f_m \Vert` is taken to be the magnitude of the
            force for atom m, such that each force components acting on atom m is weighted
            equally.
            The supported keys are
            - energy_weight_params: set the values of c_1 and c_2 for the energy data
            - forces_weight_params: set the values of c_1 and c_2 for the forces data
            - stress_weight_params: set the values of c_1 and c_2 for the stress data
            The default values are
            {
                "energy_weight_params": [1.0, 0.0],
                "forces_weight_params": [1.0, 0.0],
                "stress_weight_params": [1.0, 0.0],
            }
            For the last 3 keys, if only 1 number is given as a float, and not a list,
            then this number will be used as c_1 and c_2 will be zero.
        file_format: Format of the file that stores the configuration, e.g. `xyz`.
    """

    def __init__(
        self,
        path: Optional[Path] = None,
        weight_params: Optional[Dict[str, Any]] = None,
        file_format="xyz",
    ):
        self.file_format = file_format
        self.weight_params = _check_weight_params(weight_params, default_weight_params)

        if path is not None:
            self.configs = self._read(path, self.weight_params, file_format)

        else:
            self.configs = []

    def add_configs(
        self,
        path: Path,
        weight_params: Optional[Dict[str, Any]] = default_weight_params,
    ):
        """
        Read configurations from filename and added them to the existing set of
        configurations.

        This is a convenience function to read configurations from different directory
        on disk.

        Args:
            path: Path the directory (or filename) storing the configurations.
            weight_params: Parameters used to compute the weights from the data.
        """

        configs = self._read(path, weight_params, self.file_format)
        self.configs.extend(configs)

    def get_configs(self) -> List[Configuration]:
        """
        Get the configurations.
        """
        return self.configs

    def get_num_configs(self) -> int:
        """
        Return the number of configurations in the dataset.
        """
        return len(self.configs)

    @staticmethod
    def _read(
        path: Path,
        weight_params: Optional[Dict[str, Any]] = default_weight_params,
        file_format: str = "xyz",
    ):
        """
        Read atomic configurations from path.
        """
        try:
            extension = SUPPORTED_FORMAT[file_format]
        except KeyError:
            raise DatasetError(
                f"Expect data file_format to be one of {list(SUPPORTED_FORMAT.keys())}, "
                f"got: {file_format}."
            )

        path = to_path(path)

        if path.is_dir():
            parent = path
            all_files = []
            for root, dirs, files in os.walk(parent):
                for f in files:
                    if f.endswith(extension):
                        all_files.append(to_path(root).joinpath(f))
            all_files = sorted(all_files)
        else:
            parent = path.parent
            all_files = [path]

        configs = [
            Configuration.from_file(f, weight_params, file_format) for f in all_files
        ]

        if len(configs) <= 0:
            raise DatasetError(
                f"No dataset file with file format `{file_format}` found at {parent}."
            )

        logger.info(f"{len(configs)} configurations read from {path}")

        return configs


def _check_weight_params(
    weight_params: Dict[str, Any],
    default: Dict[str, Any] = default_weight_params,
):
    """
    Check the weight parameters and set it to the needed format, i.e., list with 2
    elements for each property.
    """
    if weight_params is not None:
        for key, value in weight_params.items():
            if np.ndim(value) == 0:  # If there is only a number given, use it to set c1
                default[key][0] = value
            else:  # To set c1 and c2, a list with 2 elements need to be passed in
                default[key] = value
    return default


class ConfigurationError(Exception):
    def __init__(self, msg):
        super(ConfigurationError, self).__init__(msg)
        self.msg = msg


class DatasetError(Exception):
    def __init__(self, msg):
        super(DatasetError, self).__init__(msg)
        self.msg = msg
