import copy
import os
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
from loguru import logger
from monty.dev import requires

from kliff.dataset.extxyz import read_extxyz, write_extxyz
from kliff.dataset.weight import Weight
from kliff.utils import stress_to_tensor, stress_to_voigt, to_path

# For type checking
if TYPE_CHECKING:
    from colabfit.tools.configuration import Configuration as ColabfitConfiguration
    from colabfit.tools.database import MongoDatabase

# check if colabfit-tools is installed
try:
    from colabfit.tools.database import MongoDatabase
except ImportError:
    MongoDatabase = None

import ase.io
from ase.data import chemical_symbols

# map from file_format to file extension
SUPPORTED_FORMAT = {"xyz": ".xyz"}
SUPPORTED_PARSERS = ["ase"]


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
        weight: an instance that computes the weight of the configuration in the loss
            function.
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
        weight: Optional[Weight] = None,
        identifier: Optional[Union[str, Path]] = None,
    ):
        self._cell = cell
        self._species = species
        self._coords = coords
        self._PBC = PBC
        self._energy = energy
        self._forces = forces
        self._stress = stress
        self._fingerprint = None

        self._identifier = identifier
        self._path = None

        self._metadata: dict = {}

        self._weight = Weight() if weight is None else weight
        self._weight.compute_weight(self)  # Compute the weight
        # TODO: Dynamic loading from colabfit-tools dataset. Is it needed?

    # TODO enable config weight read in from file
    @classmethod
    def from_file(
        cls,
        filename: Path,
        weight: Optional[Weight] = None,
        file_format: str = "xyz",
    ):
        """
        Read configuration from file.

        Args:
            filename: Path to the file that stores the configuration.
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

    @classmethod
    def from_colabfit(
        cls,
        database_client: "MongoDatabase",
        data_object: dict,
        weight: Optional[Weight] = None,
    ):
        """
        Read configuration from colabfit database .

        Args:
            database_client: Instance of connected MongoDatabase client, which can be used to
                fetch database from colabfit-tools dataset.
            data_object: colabfit data object dictionary to be associated with current
                configuration and property.
            weight: an instance that computes the weight of the configuration in the loss
                function.
        """
        try:
            configuration_id = data_object["relationships"][0]["configuration"]
            fetched_configuration = database_client.configurations.find_one(
                {"colabfit-id": data_object["relationships"][0]["configuration"]}
            )
            fetched_properties = list(
                database_client.property_instances.find(
                    {
                        "colabfit-id": {
                            "$in": data_object["relationships"][0]["property_instance"]
                        }
                    }
                )
            )
        except:
            raise ConfigurationError(
                "Looks like Mongo database did not return appropriate response. "
                f"Please run db.configurations.find('_id':{data_object}) to verify response. "
            )
        cell = np.asarray(fetched_configuration["cell"])
        # TODO: consistent Z -> symbol mapping -> Z mapping across all kliff
        species = [
            chemical_symbols[int(i)] for i in fetched_configuration["atomic_numbers"]
        ]
        coords = np.asarray(fetched_configuration["positions"])
        PBC = [bool(i) for i in fetched_configuration["pbc"]]

        energy = None
        forces = None
        stress = None
        for property in fetched_properties:
            if property["type"] == "potential-energy":
                energy = float(property["potential-energy"]["energy"]["source-value"])
            elif property["type"] == "atomic-forces":
                forces = np.asarray(property["atomic-forces"]["forces"]["source-value"])
            elif property["type"] == "cauchy-stress":
                stress = np.asarray(property["cauchy-stress"]["stress"]["source-value"])

        stress = stress_to_voigt(stress)
        self = cls(
            cell,
            species,
            coords,
            PBC,
            energy,
            forces,
            stress,
            identifier=configuration_id,
            weight=weight,
        )
        self.metadata = {
            "data_object": data_object,
        }

        return self

    @classmethod
    def from_ase_atoms(
        cls,
        atoms: ase.Atoms,
        weight: Optional[Weight] = None,
        energy_key: str = "energy",
        forces_key: str = "forces",
    ):
        """
        Read configuration from ase.Atoms object.

        Args:
            atoms: ase.Atoms object.
            weight: an instance that computes the weight of the configuration in the loss
                function.
            energy_key: Name of the field in extxyz that stores the energy.
            forces_key: Name of the field in extxyz that stores the forces.
        """
        cell = atoms.get_cell()
        species = atoms.get_chemical_symbols()
        coords = atoms.get_positions()
        PBC = atoms.get_pbc()
        energy = atoms.info[energy_key]
        try:
            forces = atoms.arrays[forces_key]
        except KeyError:
            forces = None

        try:
            stress = atoms.get_stress(voigt=True)
        except RuntimeError:
            stress = None

        weight = weight

        self = cls(
            cell,
            species,
            coords,
            PBC,
            energy,
            forces,
            stress,
            weight,
        )
        return self

    def to_ase_atoms(self):
        """
        Convert the configuration to ase.Atoms object.

        Returns:
            ase.Atoms representation of the Configuration
        """
        atoms = ase.Atoms(
            symbols=self.species,
            positions=self.coords,
            cell=self.cell,
            pbc=self.PBC,
        )
        if self.energy is not None:
            atoms.info["energy"] = self.energy
        if self.forces is not None:
            atoms.set_array("forces", self.forces)
        if self.stress is not None:
            atoms.info["stress"] = stress_to_tensor(self.stress)
        return atoms

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

    @energy.setter
    def energy(self, energy: Union[float, None]) -> None:
        """
        Set the energy of the configuration.
        Args:
            energy: Potential energy of the configuration.
        """
        self._energy = energy

    @property
    def forces(self) -> np.ndarray:
        """
        Return a `Nx3` matrix of the forces on each atoms.
        """
        if self._forces is None:
            raise ConfigurationError("Configuration does not contain forces.")
        return self._forces

    @forces.setter
    def forces(self, forces: np.ndarray):
        """
        Set the forces of the configuration.
        Args:
            forces: Numpy array containing the forces.
        """
        self._forces = forces

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

    @stress.setter
    def stress(self, stress: Union[List[float], np.ndarray]):
        """
        Set the stress of the configuration. The stress is given in Voigt notation i.e
        :math:`\sigma=[\sigma_{xx},\sigma_{yy},\sigma_{zz},\sigma_{yz},\sigma_{xz},
        \sigma_{xy}]`.
        Args:
            stress: List containing the stress.
        """
        self._stress = stress

    @property
    def weight(self):
        """
        Get the weight class of the loss function.
        """
        return self._weight

    @weight.setter
    def weight(self, weight: Weight):
        """
        Set the weight of the configuration if the loss function.
        """
        self._weight = weight
        self._weight.compute_weight(self)

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
    def fingerprint(self):
        """
        Return the stored fingerprint of the configuration.
        """
        return self._fingerprint

    @fingerprint.setter
    def fingerprint(self, fingerprint):
        """
        Set the fingerprint of the configuration.
        Args:
         fingerprint: Object which is the fingerprint of the configuration.
        """
        self._fingerprint = fingerprint

    @property
    def path(self) -> Union[Path, None]:
        """
        Return the path of the file containing the configuration. If the configuration
        is not read from a file, return None.
        """
        return self._path

    @property
    def metadata(self) -> dict:
        """
        Return the metadata of the configuration.
        """
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: dict):
        """
        Set the metadata of the configuration.
        """
        self._metadata = metadata

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

    @staticmethod
    def _get_colabfit_property(
        database_client: "MongoDatabase",
        property_id: Union[List[str], str],
        property_name: str,
        property_type: str,
    ):
        """
        Returns colabfit-property. workaround till we get proper working get_property routine

        Args:
            database_client: Instance of connected MongoDatabase client, which can be used to
                fetch database from colabfit-tools dataset.
            property_id: colabfit ID of the property instance to be associated with
                current configuration.
            property_name: subfield of the property to fetch
            property_type: type of property to fetch

        Returns:
            Property: fetched value, None if query comes empty
        """
        pi_doc = database_client.property_instances.find_one(
            {"colabfit-id": {"$in": property_id}, "type": property_type}
        )
        if pi_doc:
            return pi_doc[property_type][property_name]["source-value"]
        else:
            return None


class Dataset:
    """
    A dataset of multiple configurations (:class:`~kliff.dataset.Configuration`).

    Args:
        configurations: A list of :class:`~kliff.dataset.Configuration` objects.
    """

    def __init__(self, configurations: Iterable = None):
        if configurations is None:
            self.configs = []
        elif isinstance(configurations, Iterable) and not isinstance(
            configurations, str
        ):
            self.configs = list(configurations)
        else:
            raise DatasetError(
                "configurations must be a iterable of Configuration objects."
            )

    @classmethod
    @requires(MongoDatabase is not None, "colabfit-tools is not installed")
    def from_colabfit(
        cls,
        colabfit_database: str,
        colabfit_dataset: str,
        colabfit_uri: str = "mongodb://localhost:27017",
        weight: Optional[Weight] = None,
    ) -> "Dataset":
        """
        Read configurations from colabfit database and initialize a dataset.

        Args:
            weight: an instance that computes the weight of the configuration in the loss
                function.
            colabfit_database: Name of the colabfit Mongo database to read from.
            colabfit_dataset: Name of the colabfit dataset instance to read from, usually
                it is of form, e.g., "DS_xxxxxxxxxxxx_0"
            colabfit_uri: connection URI of the colabfit Mongo database to read from.

        Returns:
            A dataset of configurations.
        """
        instance = cls()
        instance.add_from_colabfit(
            colabfit_database, colabfit_dataset, colabfit_uri, weight
        )
        return instance

    @staticmethod
    @requires(MongoDatabase is not None, "colabfit-tools is not installed")
    def _read_from_colabfit(
        database_client: MongoDatabase,
        colabfit_dataset: str,
        weight: Optional[Weight] = None,
    ) -> List[Configuration]:
        """
        Read configurations from colabfit database.

        Args:
            database_client: Instance of connected MongoDatabase client, which can be used to
                fetch database from colabfit-tools dataset.
            colabfit_dataset: Name of the colabfit dataset instance to read from.
            weight: an instance that computes the weight of the configuration in the loss
                function.

        Returns:
            A list of configurations.
        """
        # get configuration and property ID and send it to load configuration-first get Data Objects
        data_objects = database_client.data_objects.find(
            {"relationships.dataset": colabfit_dataset}
        )
        if not data_objects:
            logger.error(f"{colabfit_dataset} is either empty or does not exist")
            raise DatasetError(f"{colabfit_dataset} is either empty or does not exist")

        configs = []
        for data_object in data_objects:
            configs.append(
                Configuration.from_colabfit(database_client, data_object, weight)
            )

        if len(configs) <= 0:
            raise DatasetError(f"No dataset file with in {colabfit_dataset} dataset.")

        logger.info(f"{len(configs)} configurations read from {colabfit_dataset}")

        return configs

    @requires(MongoDatabase is not None, "colabfit-tools is not installed")
    def add_from_colabfit(
        self,
        colabfit_database: str,
        colabfit_dataset: str,
        colabfit_uri: str = "mongodb://localhost:27017",
        weight: Optional[Weight] = None,
    ):
        """
        Read configurations from colabfit database and add them to the dataset.

        Args:
            colabfit_database: Name of the colabfit Mongo database to read from.
            colabfit_dataset: Name of the colabfit dataset instance to read from (usually
                it is of form, e.g., "DS_xxxxxxxxxxxx_0")
            colabfit_uri: connection URI of the colabfit Mongo database to read from.
            weight: an instance that computes the weight of the configuration in the loss
                function.

        """
        # open link to the mongo
        mongo_client = MongoDatabase(colabfit_database, uri=colabfit_uri)
        configs = Dataset._read_from_colabfit(mongo_client, colabfit_dataset, weight)
        self.configs.extend(configs)

    @classmethod
    def from_path(
        cls,
        path: Union[Path, str],
        weight: Optional[Weight] = None,
        file_format: str = "xyz",
    ) -> "Dataset":
        """
        Read configurations from path and initialize a dataset using KLIFF's own parser.

        Args:
            path: Path the directory (or filename) storing the configurations.
            weight: an instance that computes the weight of the configuration in the loss
                function.
            file_format: Format of the file that stores the configuration, e.g. `xyz`.

        Returns:
            A dataset of configurations.
        """
        instance = cls()
        instance.add_from_path(path, weight, file_format)
        return instance

    @staticmethod
    def _read_from_path(
        path: Path, weight: Optional[Weight] = None, file_format: str = "xyz"
    ) -> List[Configuration]:
        """
        Read configurations from path.

        Args:
            path: Path of the directory storing the configurations in individual files.
                For single file with multiple configurations, use `_read_from_ase()` instead.
            weight: an instance that computes the weight of the configuration in the loss
                function.
            file_format: Format of the file that stores the configuration, e.g. `xyz`.

        Returns:
            A list of configurations.
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
            for root, dirs, files in os.walk(parent, followlinks=True):
                for f in files:
                    if f.endswith(extension):
                        all_files.append(to_path(root).joinpath(f))
            all_files = sorted(all_files)
        else:
            parent = path.parent
            all_files = [path]

        configs = [
            Configuration.from_file(f, copy.copy(weight), file_format)
            for f in all_files
        ]

        if len(configs) <= 0:
            raise DatasetError(
                f"No dataset file with file format `{file_format}` found at {parent}."
            )
        return configs

    def add_from_path(
        self,
        path: Union[Path, str],
        weight: Optional[Weight] = None,
        file_format: str = "xyz",
    ):
        """
        Read configurations from path and append them to dataset.

        Args:
            path: Path the directory (or filename) storing the configurations.
            weight: an instance that computes the weight of the configuration in the loss
                function.
            file_format: Format of the file that stores the configuration, e.g. `xyz`.
        """
        if isinstance(path, str):
            path = Path(path)
        configs = self._read_from_path(path, weight, file_format)
        self.configs.extend(configs)

    @classmethod
    def from_ase(
        cls,
        path: Union[Path, str] = None,
        ase_atoms_list: List[ase.Atoms] = None,
        weight: Optional[Weight] = None,
        energy_key: str = "energy",
        forces_key: str = "forces",
        slices: str = ":",
        file_format: str = "xyz",
    ) -> "Dataset":
        """
        Read configurations from ase.Atoms object and initialize a dataset. The expected
        inputs are either a pre-initialized list of ase.Atoms, or a path from which
        the dataset can be read from (usually an extxyz file). If the configurations
        are in a file, or a directory, it would use ~ase.io.read() to read the
        configurations. Therefore, it is expected that the file format is supported by
        ASE.

        Example:
            >>> from ase.build import bulk
            >>> from kliff.dataset import Dataset
            >>> ase_configs = [bulk("Al"), bulk("Al", cubic=True)]
            >>> dataset_from_list = Dataset.from_ase(ase_atoms_list=ase_configs)
            >>> dataset_from_file = Dataset.from_ase(path="configs.xyz", energy_key="Energy")

        Args:
            path: Path the directory (or filename) storing the configurations.
            ase_atoms_list: A list of ase.Atoms objects.
            weight: an instance that computes the weight of the configuration in the loss
                function.
            energy_key: Name of the field in extxyz/ase.Atoms that stores the energy.
            forces_key: Name of the field in extxyz/ase.Atoms that stores the forces.
            slices: Slice of the configurations to read. It is used only when `path` is
                a file.
            file_format: Format of the file that stores the configuration, e.g. `xyz`.

        Returns:
            A dataset of configurations.
        """
        instance = cls()
        instance.add_from_ase(
            path, ase_atoms_list, weight, energy_key, forces_key, slices, file_format
        )
        return instance

    @staticmethod
    def _read_from_ase(
        path: Path = None,
        ase_atoms_list: List[ase.Atoms] = None,
        weight: Optional[Weight] = None,
        energy_key: str = "energy",
        forces_key: str = "forces",
        slices: str = ":",
        file_format: str = "xyz",
    ) -> List[Configuration]:
        """
        Read configurations from ase.Atoms object. If the configurations are in a file,
        or a directory, it would use ~ase.io.read() to read the configurations.

        Args:
            path: Path the directory (or filename) storing the configurations.
            ase_atoms_list: A list of ase.Atoms objects.
            weight: an instance that computes the weight of the configuration in the loss
                function.
            energy_key: Name of the field in extxyz/ase.Atoms that stores the energy.
            forces_key: Name of the field in extxyz/ase.Atoms that stores the forces.
            slices: Slice of the configurations to read. It is used only when `path` is
                a file.
            file_format: Format of the file that stores the configuration, e.g. `xyz`.

        Returns:
            A list of configurations.
        """
        if ase_atoms_list is None and path is None:
            raise DatasetError(
                "Either list of ase.Atoms objects or a path must be provided."
            )

        if ase_atoms_list:
            configs = [
                Configuration.from_ase_atoms(
                    config,
                    weight=copy.copy(weight),
                    energy_key=energy_key,
                    forces_key=forces_key,
                )
                for config in ase_atoms_list
            ]
        else:
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
                for root, dirs, files in os.walk(parent, followlinks=True):
                    for f in files:
                        if f.endswith(extension):
                            all_files.append(to_path(root).joinpath(f))
                all_files = sorted(all_files)
            else:
                parent = path.parent
                all_files = [path]

            if len(all_files) == 1:  # single xyz file with multiple configs
                all_configs = ase.io.read(all_files[0], index=slices)
                configs = [
                    Configuration.from_ase_atoms(
                        config,
                        weight=copy.copy(weight),
                        energy_key=energy_key,
                        forces_key=forces_key,
                    )
                    for config in all_configs
                ]
            else:
                configs = [
                    Configuration.from_ase_atoms(
                        ase.io.read(f),
                        weight=copy.copy(weight),
                        energy_key=energy_key,
                        forces_key=forces_key,
                    )
                    for f in all_files
                ]

        if len(configs) <= 0:
            raise DatasetError(
                f"No dataset file with file format `{file_format}` found at {path}."
            )

        logger.info(f"{len(configs)} configurations loaded using ASE.")
        return configs

    def add_from_ase(
        self,
        path: Union[Path, str] = None,
        ase_atoms_list: List[ase.Atoms] = None,
        weight: Optional[Weight] = None,
        energy_key: str = "energy",
        forces_key: str = "forces",
        slices: str = ":",
        file_format: str = "xyz",
    ):
        """
        Read configurations from ase.Atoms object and append to a dataset. The expected
        inputs are either a pre-initialized list of ase.Atoms, or a path from which
        the dataset can be read from (usually an extxyz file). If the configurations
        are in a file, or a directory, it would use ~ase.io.read() to read the
        configurations. Therefore, it is expected that the file format is supported by
        ASE.

        Example:
            >>> from ase.build import bulk
            >>> from kliff.dataset import Dataset
            >>> ase_configs = [bulk("Al"), bulk("Al", cubic=True)]
            >>> dataset = Dataset()
            >>> dataset.add_from_ase(ase_atoms_list=ase_configs)
            >>> dataset.add_from_ase(path="configs.xyz", energy_key="Energy")

        Args:
            path: Path the directory (or filename) storing the configurations.
            ase_atoms_list: A list of ase.Atoms objects.
            weight: an instance that computes the weight of the configuration in the loss
            energy_key: Name of the field in extxyz/ase.Atoms that stores the energy.
            forces_key: Name of the field in extxyz/ase.Atoms that stores the forces.
            slices: Slice of the configurations to read. It is used only when `path` is
                a file.
            file_format: Format of the file that stores the configuration, e.g. `xyz`.
        """
        if isinstance(path, str):
            path = Path(path)
        configs = self._read_from_ase(
            path, ase_atoms_list, weight, energy_key, forces_key, slices
        )
        self.configs.extend(configs)

    def get_configs(self) -> List[Configuration]:
        """
        Get shallow copy of the configurations.
        """
        return self.configs[:]

    def __len__(self) -> int:
        """
        Get length of the dataset. It is needed to make dataset directly compatible
        with various dataloaders.

        Returns:
            Number of configurations in the dataset.
        """
        return len(self.configs)

    def __getitem__(self, idx:Union[int,np.ndarray, List]) -> Union[Configuration, "Dataset"]:
        """
        Get the configuration at index `idx`. If the index is a list, it returns a new
        dataset with the configurations at the indices.

        Args:
         idx: Index of the configuration to get or a list of indices.

        Returns:
            The configuration at index `idx` or a new dataset with the configurations at
            the indices.
        """
        if isinstance(idx, int):
            return self.configs[idx]
        else:
            configs = [self.configs[i] for i in idx]
            return Dataset(configs)


class ConfigurationError(Exception):
    def __init__(self, msg):
        super(ConfigurationError, self).__init__(msg)
        self.msg = msg


class DatasetError(Exception):
    def __init__(self, msg):
        super(DatasetError, self).__init__(msg)
        self.msg = msg
