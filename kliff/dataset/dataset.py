import copy
import hashlib
import importlib
import json
import os
import pickle as pkl
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
from loguru import logger
from monty.dev import requires

from kliff.atomic_data import atomic_species
from kliff.dataset.weight import Weight
from kliff.utils import str_to_numpy, to_path

from .configuration import Configuration, ConfigurationError

# For type checking
if TYPE_CHECKING:
    from colabfit.tools.database import MongoDatabase

# check if colabfit-tools is installed
try:
    from colabfit.tools.database import MongoDatabase
except ImportError:
    MongoDatabase = None

import ase
import ase.build.bulk
import ase.io

# map from file_format to file extension
SUPPORTED_FORMAT = {"xyz": ".xyz"}
SUPPORTED_PARSERS = ["ase"]


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

        self._metadata: dict = {}

    @classmethod
    @requires(MongoDatabase is not None, "colabfit-tools is not installed")
    def from_colabfit(
        cls,
        colabfit_database: str,
        colabfit_dataset: str,
        colabfit_uri: str = "mongodb://localhost:27017",
        weight: Optional[Union[Weight, Path]] = None,
        **kwargs,
    ) -> "Dataset":
        """
        Read configurations from colabfit database and initialize a dataset.

        Args:
            weight: an instance that computes the weight of the configuration in the loss
                function. If a path is provided, it is used to read the weight from the
                file.  The file must be a plain text file with 4 whitespace separated
                columns: config_weight, energy_weight, forces_weight, and stress_weight.
                Length of the file must be equal to the number of configurations, or 1
                (in which case the same weight is used for all configurations).
            colabfit_database: Name of the colabfit Mongo database to read from.
            colabfit_dataset: Name of the colabfit dataset instance to read from, usually
                it is of form, e.g., "DS_xxxxxxxxxxxx_0"
            colabfit_uri: connection URI of the colabfit Mongo database to read from.

        Returns:
            A dataset of configurations.
        """
        instance = cls()
        instance.add_from_colabfit(
            colabfit_database, colabfit_dataset, colabfit_uri, weight, **kwargs
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
        weight: Optional[Union[Weight, Path]] = None,
        **kwargs,
    ):
        """
        Read configurations from colabfit database and add them to the dataset.

        Args:
            colabfit_database: Name of the colabfit Mongo database to read from.
            colabfit_dataset: Name of the colabfit dataset instance to read from (usually
                it is of form, e.g., "DS_xxxxxxxxxxxx_0")
            colabfit_uri: connection URI of the colabfit Mongo database to read from.
            weight: an instance that computes the weight of the configuration in the loss
                function. If a path is provided, it is used to read the weight from the
                file.  The file must be a plain text file with 4 whitespace separated
                columns: config_weight, energy_weight, forces_weight, and stress_weight.
                Length of the file must be equal to the number of configurations, or 1
                (in which case the same weight is used for all configurations).

        """
        # open link to the mongo
        mongo_client = MongoDatabase(colabfit_database, uri=colabfit_uri, **kwargs)
        if isinstance(weight, Weight):
            configs = Dataset._read_from_colabfit(
                mongo_client, colabfit_dataset, weight
            )
        else:
            configs = Dataset._read_from_colabfit(mongo_client, colabfit_dataset, None)
            Dataset.add_weights(configs, weight)
        self.configs.extend(configs)

    @classmethod
    def from_path(
        cls,
        path: Union[Path, str],
        weight: Optional[Union[Path, Weight]] = None,
        file_format: str = "xyz",
    ) -> "Dataset":
        """
        Read configurations from path and initialize a dataset using KLIFF's own parser.

        Args:
            path: Path the directory (or filename) storing the configurations.
            weight: an instance that computes the weight of the configuration in the loss
                function. If a path is provided, it is used to read the weight from the
                file.  The file must be a plain text file with 4 whitespace separated
                columns: config_weight, energy_weight, forces_weight, and stress_weight.
                Length of the file must be equal to the number of configurations, or 1
                (in which case the same weight is used for all configurations).
            file_format: Format of the file that stores the configuration, e.g. `xyz`.

        Returns:
            A dataset of configurations.
        """
        instance = cls()
        instance.add_from_path(path, weight, file_format)
        return instance

    @staticmethod
    def _read_from_path(
        path: Path,
        weight: Optional[Weight] = None,
        file_format: str = "xyz",
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

        configs = [Configuration.from_file(f, weight, file_format) for f in all_files]

        if len(configs) <= 0:
            raise DatasetError(
                f"No dataset file with file format `{file_format}` found at {parent}."
            )
        return configs

    def add_from_path(
        self,
        path: Union[Path, str],
        weight: Optional[Union[Weight, Path]] = None,
        file_format: str = "xyz",
    ):
        """
        Read configurations from path and append them to dataset.

        Args:
            path: Path the directory (or filename) storing the configurations.
            weight: an instance that computes the weight of the configuration in the loss
                function. If a path is provided, it is used to read the weight from the
                file.  The file must be a plain text file with 4 whitespace separated
                columns: config_weight, energy_weight, forces_weight, and stress_weight.
                Length of the file must be equal to the number of configurations, or 1
                (in which case the same weight is used for all configurations).
            file_format: Format of the file that stores the configuration, e.g. `xyz`.
        """
        if isinstance(path, str):
            path = Path(path)

        if isinstance(weight, Weight):
            configs = self._read_from_path(path, weight, file_format)
        else:
            configs = self._read_from_path(path, None, file_format)
            Dataset.add_weights(configs, weight)
        self.configs.extend(configs)

    @classmethod
    def from_ase(
        cls,
        path: Union[Path, str] = None,
        ase_atoms_list: List[ase.Atoms] = None,
        weight: Optional[Union[Weight, Path]] = None,
        energy_key: str = "energy",
        forces_key: str = "forces",
        stress_key: str = "stress",
        slices: Union[slice, str] = ":",
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
                function. If a path is provided, it is used to read the weight from the
                file.  The file must be a plain text file with 4 whitespace separated
                columns: config_weight, energy_weight, forces_weight, and stress_weight.
                Length of the file must be equal to the number of configurations, or 1
                (in which case the same weight is used for all configurations).
            energy_key: Name of the field in extxyz/ase.Atoms that stores the energy.
            forces_key: Name of the field in extxyz/ase.Atoms that stores the forces.
            stress_key: Name of the field in extxyz/ase.Atoms that stores the stress.
            slices: Slice of the configurations to read. It is used only when `path` is
                a file.
            file_format: Format of the file that stores the configuration, e.g. `xyz`.

        Returns:
            A dataset of configurations.
        """
        instance = cls()
        instance.add_from_ase(
            path,
            ase_atoms_list,
            weight,
            energy_key,
            forces_key,
            stress_key,
            slices,
            file_format,
        )
        return instance

    @staticmethod
    def _read_from_ase(
        path: Path = None,
        ase_atoms_list: List[ase.Atoms] = None,
        weight: Optional[Weight] = None,
        energy_key: str = "energy",
        forces_key: str = "forces",
        stress_key: str = "stress",
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
            stress_key: Name of the field in extxyz/ase.Atoms that stores the stress.
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
                    weight=weight,
                    energy_key=energy_key,
                    forces_key=forces_key,
                    stress_key=stress_key,
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
                        weight=weight,
                        energy_key=energy_key,
                        forces_key=forces_key,
                        stress_key=stress_key,
                    )
                    for config in all_configs
                ]
            else:
                configs = [
                    Configuration.from_ase_atoms(
                        ase.io.read(f),
                        weight=weight,
                        energy_key=energy_key,
                        forces_key=forces_key,
                        stress_key=stress_key,
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
        weight: Optional[Union[Weight, Path]] = None,
        energy_key: str = "energy",
        forces_key: str = "forces",
        stress_key: str = "stress",
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
                function. If a path is provided, it is used to read the weight from the
                file.  The file must be a plain text file with 4 whitespace separated
                columns: config_weight, energy_weight, forces_weight, and stress_weight.
                Length of the file must be equal to the number of configurations, or 1
                (in which case the same weight is used for all configurations).
            energy_key: Name of the field in extxyz/ase.Atoms that stores the energy.
            forces_key: Name of the field in extxyz/ase.Atoms that stores the forces.
            stress_key: Name of the field in extxyz/ase.Atoms that stores the stress.
            slices: Slice of the configurations to read. It is used only when `path` is
                a file.
            file_format: Format of the file that stores the configuration, e.g. `xyz`.
        """
        if isinstance(path, str):
            path = Path(path)

        if isinstance(weight, Weight):
            configs = self._read_from_ase(
                path,
                ase_atoms_list,
                weight,
                energy_key,
                forces_key,
                stress_key,
                slices,
                file_format,
            )
        else:
            configs = self._read_from_ase(
                path,
                ase_atoms_list,
                None,
                energy_key,
                forces_key,
                stress_key,
                slices,
                file_format,
            )
            Dataset.add_weights(configs, weight)
        self.configs.extend(configs)

    @classmethod
    def from_lmdb(
        cls,
        lmdb_file: Path,
        n_configs: Optional[int] = None,
        config_key_prefix: Optional[str] = None,
        coords_key: str = "coords",
        species_key: str = "species",
        pbc_key: str = "PBC",
        cell_key: str = "cell",
        energy_key: str = "energy",
        forces_key: str = "forces",
        stress_key: str = "stress",
        config_weight_key: str = "config_weight",
        energy_weight_key: str = "energy_weight",
        forces_weight_key: str = "forces_weight",
        stress_weight_key: str = "stress_weight",
        metadata_keys: Optional[List[str]] = None,
        weight_file: Optional[Path] = None,
    ) -> "Dataset":
        """
        Load dataset from an LMDB file.

        Args:
            lmdb_file: Path to the LMDB file.
            n_configs: Number of configurations to load.
            config_key_prefix: KLIFF assumes that configurations can be loaded as "prefix{idx}"
                where idx is the index of the configuration in the LMDB file.
            coords_key: Key to get coordinates from the lmdb configuration.
            species_key: Key to get species from the lmdb configuration.
            pbc_key: Key to get PBC array from the lmdb configuration.
            cell_key: Key to get cell vectors from the lmdb configuration.
            energy_key: Key to get energy from the lmdb configuration.
            forces_key: Key to get forces from the lmdb configuration.
            stress_key: Key to get stress from the lmdb configuration.
            config_weight_key: Key to get config_weight from the lmdb configuration.
            energy_weight_key: Key to get energy_weight from the lmdb configuration.
            forces_weight_key: Key to get forces_weight from the lmdb configuration.
            stress_weight_key: Key to get stress_weight from the lmdb configuration.
            metadata_keys: List of keys to get all metadata from the lmdb configuration.
            weight_file: Path to the KLIFF weight file.

        Returns:
            Dataset object.

        """
        instance = cls()
        lmdb_file = str(lmdb_file)
        instance.add_from_lmdb(
            lmdb_file,
            n_configs,
            config_key_prefix,
            coords_key,
            species_key,
            pbc_key,
            cell_key,
            energy_key,
            forces_key,
            stress_key,
            config_weight_key,
            energy_weight_key,
            forces_weight_key,
            stress_weight_key,
            metadata_keys,
        )

        if weight_file is not None:
            logger.info(f"Loading weights from {weight_file}")
            Dataset.add_weights(instance.configs, weight_file)

        return instance

    def add_from_lmdb(
        self,
        lmdb_file,
        n_configs,
        config_key_prefix,
        coords_key,
        species_key,
        pbc_key,
        cell_key,
        energy_key,
        forces_key,
        stress_key,
        config_weight_key,
        energy_weight_key,
        forces_weight_key,
        stress_weight_key,
        metadata_keys,
    ):
        """
        Add configurations from an LMDB file.

        Args:
            lmdb_file: Path to the LMDB file.
            n_configs: Number of configurations to load.
            config_key_prefix: KLIFF assumes that configurations can be loaded as "prefix{idx}"
                where idx is the index of the configuration in the LMDB file.
            coords_key: Key to get coordinates from the lmdb configuration.
            species_key: Key to get species from the lmdb configuration.
            pbc_key: Key to get PBC array from the lmdb configuration.
            cell_key: Key to get cell vectors from the lmdb configuration.
            energy_key: Key to get energy from the lmdb configuration.
            forces_key: Key to get forces from the lmdb configuration.
            stress_key: Key to get stress from the lmdb configuration.
            config_weight_key: Key to get config_weight from the lmdb configuration.
            energy_weight_key: Key to get energy_weight from the lmdb configuration.
            forces_weight_key: Key to get forces_weight from the lmdb configuration.
            stress_weight_key: Key to get stress_weight from the lmdb configuration.
            metadata_keys: List of keys to get all metadata from the lmdb configuration.
        """
        configs = self._read_from_lmdb(
            lmdb_file,
            n_configs,
            config_key_prefix,
            coords_key,
            species_key,
            pbc_key,
            cell_key,
            energy_key,
            forces_key,
            stress_key,
            config_weight_key,
            energy_weight_key,
            forces_weight_key,
            stress_weight_key,
            metadata_keys,
        )
        self.configs.extend(configs)

    @staticmethod
    def _read_from_lmdb(
        lmdb_file,
        n_configs,
        config_key_prefix,
        coords_key,
        species_key,
        pbc_key,
        cell_key,
        energy_key,
        forces_key,
        stress_key,
        config_weight_key,
        energy_weight_key,
        forces_weight_key,
        stress_weight_key,
        metadata_keys,
    ) -> List[Configuration]:
        """
        Read configurations from an LMDB file.

        Args:
            lmdb_file: Path to the LMDB file.
            n_configs: Number of configurations to load.
            config_key_prefix: KLIFF assumes that configurations can be loaded as "prefix{idx}"
                where idx is the index of the configuration in the LMDB file.
            coords_key: Key to get coordinates from the lmdb configuration.
            species_key: Key to get species from the lmdb configuration.
            pbc_key: Key to get PBC array from the lmdb configuration.
            cell_key: Key to get cell vectors from the lmdb configuration.
            energy_key: Key to get energy from the lmdb configuration.
            forces_key: Key to get forces from the lmdb configuration.
            stress_key: Key to get stress from the lmdb configuration.
            config_weight_key: Key to get config_weight from the lmdb configuration.
            energy_weight_key: Key to get energy_weight from the lmdb configuration.
            forces_weight_key: Key to get forces_weight from the lmdb configuration.
            stress_weight_key: Key to get stress_weight from the lmdb configuration.
            metadata_keys: List of keys to get all metadata from the lmdb configuration.

        """
        try:
            # local import to avoid unnecessary dependency
            import lmdb
        except ImportError:
            raise DatasetError(
                "LMDB is needed for reading configurations from LMDB dataset, please do `pip install lmdb` first"
            )

        map_size = os.environ.get("KLIFF_LMDB_MAP_SIZE", 1e12)
        map_size = int(map_size)
        logger.info(
            f"Using lmdb map size={map_size}, to change it use KLIFF_LMDB_MAP_SIZE env variable"
        )

        env = lmdb.open(lmdb_file, map_size=map_size, subdir=False)
        txn = env.begin(write=False)
        n_configs_available = txn.stat()["entries"]
        if n_configs is None:
            n_configs = n_configs_available

        if n_configs_available < n_configs:
            raise DatasetError(
                f"LMDB file {lmdb_file} contains only {n_configs_available} configurations; asked to load {n_configs} configurations."
            )
        else:
            logger.info(f"Reading {n_configs} configurations from {lmdb_file}")

        configs = []

        for i in range(n_configs):
            config_key = (
                f"{i}".encode()
                if config_key_prefix is None
                else f"{config_key_prefix}{i}".encode()
            )
            config_dict: dict = pkl.loads(txn.get(config_key))

            coords = config_dict.get(coords_key, None)
            species = config_dict.get(species_key, None)
            pbc = config_dict.get(pbc_key, None)
            cell = config_dict.get(cell_key, None)
            energy = config_dict.get(energy_key, None)
            forces = config_dict.get(forces_key, None)
            stress = config_dict.get(stress_key, None)
            configuration_weight = config_dict.get(config_weight_key, None)
            energy_weight = config_dict.get(energy_weight_key, None)
            forces_weight = config_dict.get(forces_weight_key, None)
            stress_weight = config_dict.get(stress_weight_key, None)
            weight = Weight(
                config_weight=configuration_weight,
                energy_weight=energy_weight,
                forces_weight=forces_weight,
                stress_weight=stress_weight,
            )

            if not isinstance(species[0], str):
                species = [atomic_species[z] for z in species]

            metadata = {}
            if metadata_keys is not None:
                for keys in metadata_keys:
                    metadata[keys] = config_dict.get(keys, None)

            config = Configuration(
                cell, species, coords, pbc, energy, forces, stress, weight
            )
            config._metadata = metadata

            configs.append(config)

        env.close()
        return configs

    def to_lmdb(self, lmdb_file):
        try:
            # local import to avoid unnecessary dependency
            import lmdb
        except ImportError:
            raise DatasetError(
                "LMDB is needed for writing configurations to LMDB dataset,"
                "please do `pip install lmdb` first"
            )

        map_size = os.environ.get("KLIFF_LMDB_MAP_SIZE", 1e12)
        map_size = int(map_size)

        lmdb_file = str(lmdb_file)
        env = lmdb.open(lmdb_file, map_size=map_size, subdir=False)
        txn = env.begin(write=True)

        n_configs_available = txn.stat()["entries"]

        for i, config in enumerate(self.configs):
            key = f"{n_configs_available + i}".encode()
            config_dict = config.to_dict()
            txn.put(key, pkl.dumps(config_dict))

        txn.commit()
        env.close()

    @classmethod
    def from_huggingface(
        cls,
        hf_id: str,
        split: str = "train",
        n_configs: Optional[int] = None,
        coords_key: str = "positions",
        species_key: str = "atomic_numbers",
        pbc_key: str = "pbc",
        cell_key: str = "cell",
        energy_key: str = "energy",
        forces_key: str = "atomic_forces",
        stress_key: Optional[str] = None,
        weights_file: Optional[Union[str, Path]] = None,
        **load_kwargs,
    ) -> "Dataset":
        """
        Load dataset from a HuggingFace Hub dataset.

        Args:
            hf_id:         Huggingface id e.g. "colabfit/xxMD-CASSCF_train"
            split:         which split to load, e.g. "train"
            n_configs:     optionally limit to the first N configs
            *_key:         column names in the HF dataset
            load_kwargs:   passed through to `datasets.load_dataset`

        Returns:
            Dataset
        """
        instance = cls()
        instance.add_from_huggingface(
            hf_id,
            split,
            n_configs,
            coords_key,
            species_key,
            pbc_key,
            cell_key,
            energy_key,
            forces_key,
            stress_key,
            weights_file=weights_file,
            **load_kwargs,
        )
        return instance

    def add_from_huggingface(
        self,
        hf_id,
        split,
        n_configs,
        coords_key,
        species_key,
        pbc_key,
        cell_key,
        energy_key,
        forces_key,
        stress_key: Optional[str] = None,
        weights_file: Optional[Union[str, Path]] = None,
        **load_kwargs,
    ):
        """
        Add configurations from a HuggingFace Hub dataset.

        Args:
            hf_id:         Huggingface id e.g. "colabfit/xxMD-CASSCF_train"
            split:         which split to load, e.g. "train"
            n_configs:     optionally limit to the first N configs
            *_key:         column names in the HF dataset
            load_kwargs:   passed through to `datasets.load_dataset`

        """
        configs = self._read_from_huggingface(
            hf_id,
            split,
            n_configs,
            coords_key,
            species_key,
            pbc_key,
            cell_key,
            energy_key,
            forces_key,
            stress_key,
            weights_file=weights_file,
            **load_kwargs,
        )
        if weights_file is not None:
            Dataset.add_weights(configs, weights_file)
        self.configs.extend(configs)

    @staticmethod
    def _read_from_huggingface(
        hf_id,
        split,
        n_configs,
        coords_key,
        species_key,
        pbc_key,
        cell_key,
        energy_key,
        forces_key,
        stress_key,
        **load_kwargs,
    ) -> List[Configuration]:
        """
        read and process data from a HuggingFace dataset.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise DatasetError(
                "Please do `pip install datasets` first to load from HuggingFace"
            )

        # 1. pull the split
        logger.info(f"Loading dataset from HuggingFace: {hf_id} ...")
        ds = load_dataset(hf_id, split=split, **load_kwargs)

        # 2. optionally truncate
        if n_configs is not None:
            ds = ds.select(range(n_configs))

        # 3. keep only the columns we need
        cols = [coords_key, species_key, pbc_key, cell_key]
        if energy_key:
            cols.append(energy_key)
        if forces_key:
            cols.append(forces_key)
        if stress_key:
            cols.append(stress_key)
        ds = ds.select_columns(cols)

        # this is logged to mainly tell the user that something is going on, else for
        # large datasets it appears as hung
        logger.info(
            f"Loaded {len(ds)} configurations from {hf_id}; processing ... (it may take a little while)"
        )
        # 4) have HF convert ALL of them to NumPy arrays at once
        ds.set_format(type="numpy", columns=cols)

        # 5) extract the raw arrays
        coords_arr: np.ndarray = ds[coords_key]
        species_arr: np.ndarray = ds[species_key]
        pbc_arr: np.ndarray = ds[pbc_key]
        cell_arr: np.ndarray = ds[cell_key]
        if energy_key:
            energy_arr: np.ndarray = ds[energy_key]
        if forces_key:
            forces_arr: np.ndarray = ds[forces_key]
        if stress_key:
            stress_arr: np.ndarray = ds[stress_key]

        logger.info(f"Processed {len(coords_arr)} configurations from HuggingFace")
        configs = []
        for i in range(len(coords_arr)):
            configs.append(
                Configuration(
                    coords=coords_arr[i],
                    species=species_arr[i].tolist(),
                    PBC=pbc_arr[i].tolist(),
                    cell=cell_arr[i],
                    energy=float(energy_arr[i]) if energy_key else None,
                    forces=forces_arr[i] if forces_key else None,
                    stress=(stress_arr[i] if stress_key else None),
                )
            )
        return configs

    def to_path(self, path: Union[Path, str], prefix: Union[str, None] = None) -> None:
        """
        Save the dataset to a folder, as per the KLIFF xyz format. The folder will
        contain multiple files, each containing a configuration. Prefix is added to the
        filename of each configuration. Path is created if it does not exist.

        Args:
            path: Path to the directory to save the dataset.
            prefix: Prefix to add to the filename of each configuration.
        """
        path = to_path(path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        if prefix is None:
            prefix = "config_"

        for i, config in enumerate(self.configs):
            config.to_file(path.joinpath(f"{prefix}{i}.xyz"))

        logger.info(f"Dataset saved to {path}.")

    def to_ase(self, path: Union[Path, str]) -> None:
        """
        Save the dataset to a file in ASE format. The file will contain multiple
        configurations, each separated by a newline. The file will be saved in the
        specified path. The file format is determined by the extension of the path.

        Args:
            path: Path to the file to save the dataset.

        """
        path = to_path(path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        to_format = (
            "extxyz" if str(path.suffix[1:]).lower() == "xyz" else path.suffix[1:]
        )
        ase.io.write(
            path,
            self.to_ase_list(),
            format=to_format,
            append=True,
        )
        logger.info(f"Dataset saved to {path}.")

    def to_ase_list(self) -> List[ase.Atoms]:
        """
        Convert the dataset to a list of ase.Atoms objects.

        Returns:
            List of ase.Atoms objects.
        """
        return [config.to_ase_atoms() for config in self.configs]

    def to_colabfit(
        self,
        colabfit_database: str,
        colabfit_dataset: str,
        colabfit_uri: str = "mongodb://localhost:27017",
    ):
        """
        Save dataset to a colabfit database.
        Args:
            colabfit_database:
            colabfit_dataset:
            colabfit_uri:

        Returns:
        """
        raise NotImplementedError("Export to colabfit, not implemented yet.")

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

    def __getitem__(
        self, idx: Union[int, np.ndarray, List, slice]
    ) -> Union[Configuration, "Dataset"]:
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
        elif isinstance(idx, slice):
            return Dataset(self.configs[idx])
        else:
            configs = [self.configs[i] for i in idx]
            return Dataset(configs)

    def save_weights(self, path: Union[Path, str]):
        """
        Save the weights of the configurations to a file.

        Args:
            path: Path of the file to save the weights.
        """
        path = to_path(path)
        with path.open("w") as f:
            for config in self.configs:
                f.write(
                    f"{config.weight.config_weight} "
                    + f"{config.weight.energy_weight} "
                    + f"{config.weight.forces_weight} "
                    + f"{config.weight.stress_weight}\n"
                )

    @staticmethod
    def add_weights(
        configurations: Union[List[Configuration], "Dataset"],
        source: Union[Path, str, Weight],
    ):
        """
        Load weights from a text file/ Weight class. The text file should contain 1 to 4 columns,
        whitespace seperated, formatted as,
        ```
        Config Energy Forces Stress
        1.0    0.0    10.0   0.0
        ```
        ```{note}
        The column headers are case-insensitive, but should have same name as above.
        The weight of 0.0 will set respective weight as `None`. The length of column can
        be either 1 (all configs same weight) or n, where n is the number of configs in
        the dataset.
        ```
        Missing columns are treated as 0.0, i.e. above example file can also be written
        as
        ```
        Config Forces
        1.0    10.0
        ```

        It also now supports the yaml weight file. The yaml file should be formatted as,
        ```
        - config: [1.0, 1.0, 1.0]
          energy: 0.0
          forces: [1.0, 1.0, 1.0]
          stress: 0.0
        - config: [1.0, 1.0, 1.0]
          energy: 0.0
          forces: [1.0, 1.0, 1.0]
          stress: 0.0
        ```
        Any missing key is treated as 0.0. The weights are assumed to be in same order
        as the dataset configurations.

        TODO:
            - Add support for Index. Currently, it is assumed that the weights are in
                same order as the configurations. Ideally, it should be able to handle
                index based weights.

        Args:
            configurations: List of configurations to add weights to.
            source: Path to the configuration file

        """
        if source is None:
            logger.info("No explicit weights provided.")
            return
        elif isinstance(source, Weight):
            for config in configurations:
                config.weight = source
            logger.info("Weights set to the same value for all configurations.")
            return

        path = to_path(source)

        if path.suffix == ".yaml":
            with path.open("r") as f:
                weights_data = yaml.safe_load(f)
                for weight, config in zip(weights_data, configurations):
                    # check if config and forces have per atom weights
                    forces_weight = weight.get("forces", None)
                    if isinstance(forces_weight, list):
                        forces_weight = np.array(forces_weight).reshape(
                            -1, 1
                        )  # reshape to column vector for broadcasting

                    config.weight = Weight(
                        config_weight=weight.get("config", None),
                        energy_weight=weight.get("energy", None),
                        forces_weight=forces_weight,
                        stress_weight=weight.get("stress", None),
                    )
            logger.info(f"Weights loaded from YAML file: {path}")
        else:
            weights_data = np.genfromtxt(path, names=True)
            weights_col = weights_data.dtype.names

            # sanity checks
            if 1 > len(weights_col) > 4:
                raise DatasetError(
                    "Weights file contains improper number of cols,"
                    "there needs to be at least 1 col, and at most 4"
                )

            if not (weights_data.size == 1 or weights_data.size == len(configurations)):
                raise DatasetError(
                    "Weights file contains improper number of rows,"
                    "there can be either 1 row (all weights same), "
                    "or same number of rows as the configurations."
                )

            expected_cols = {"config", "energy", "forces", "stress"}
            missing_cols = expected_cols - set([col.lower() for col in weights_col])

            # missing weights are set to 0.0
            weights = {k.lower(): weights_data[k] for k in weights_col}
            for fields in missing_cols:
                weights[fields] = np.zeros_like(weights["config"])

            # if only one row, set same weight for all
            if weights_data.size == 1:
                weights = {
                    k: np.full(len(configurations), v) for k, v in weights.items()
                }

            # set weights
            for i, config in enumerate(configurations):
                config.weight = Weight(
                    config_weight=weights["config"][i],
                    energy_weight=weights["energy"][i],
                    forces_weight=weights["forces"][i],
                    stress_weight=weights["stress"][i],
                )
            logger.info(f"Weights loaded from {path}")

    def add_metadata(self, metadata: dict):
        """
        Add metadata to the dataset object.

        Args:
            metadata: A dictionary containing the metadata.
        """
        if not isinstance(metadata, dict):
            raise DatasetError("metadata must be a dictionary.")
        self._metadata.update(metadata)

    def get_metadata(self, key: str):
        """
        Get the metadata of the dataset.

        Args:
            key: Key of the metadata to get.

        Returns:
            Value of the metadata.
        """
        return self._metadata[key]

    @property
    def metadata(self):
        """
        Return the metadata of the dataset.
        """
        return self._metadata

    def check_properties_consistency(self, properties: List[str] = None):
        """
        Check which of the properties of the configurations are consistent. These
        consistent properties are saved a list which can be used to get the attributes
        from the configurations. "Consistent" in this context means that same property
        is available for all the configurations. A property is not considered consistent
        if it is None for any of the configurations.

        Args:
            properties: List of properties to check for consistency. If None, no
                properties are checked. All consistent properties are saved in the
                metadata.
        """
        if properties is None:
            logger.warning("No properties provided to check for consistency.")
            return

        # property_list = list(copy.deepcopy(properties))  # make it mutable, if not
        # for config in self.configs:
        #     for prop in property_list:
        #         try:
        #             getattr(config, prop)
        #         except ConfigurationError:
        #             property_list.remove(prop)
        property_list = []
        for prop in properties:
            for config in self.configs:
                try:
                    getattr(config, prop)
                except ConfigurationError:
                    break
            else:
                property_list.append(prop)

        self.add_metadata({"consistent_properties": tuple(property_list)})
        logger.info(
            f"Consistent properties: {property_list}, stored in metadata key: `consistent_properties`"
        )

    @staticmethod
    def get_manifest_checksum(
        dataset_manifest: dict[str, Any],
        transform_manifest: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Get the checksum of the dataset manifest.

        Args:
            dataset_manifest: Manifest of the dataset.
            transform_manifest: Manifest of the transformation.

        Returns:
            Checksum of the manifest.
        """
        dataset_str = json.dumps(dataset_manifest, sort_keys=True)
        if transform_manifest:
            transform_str = json.dumps(transform_manifest, sort_keys=True)
            dataset_str += transform_str
        return hashlib.md5(dataset_str.encode()).hexdigest()

    @staticmethod
    def get_dataset_from_manifest(dataset_manifest: dict) -> "Dataset":
        """
        Get a dataset from a manifest.

        Examples:
           1.  Manifest file for initializing dataset using ASE parser:
            ```yaml
            dataset:
                type: ase           # ase or path or colabfit
                path: Si.xyz        # Path to the dataset
                save: True          # Save processed dataset to a file
                save_path: /folder/to   # Save to this folder
                shuffle: False      # Shuffle the dataset
                weights: /path/to/weights.dat # or dictionary with weights
                keys:
                    energy: Energy  # Key for energy, if ase dataset is used
                    forces: forces  # Key for forces, if ase dataset is used
            ```

            2. Manifest file for initializing dataset using KLIFF extxyz parser:
            ```yaml
            dataset:
                type: path          # ase or path or colabfit
                path: /all/my/xyz   # Path to the dataset
                save: False         # Save processed dataset to a file
                shuffle: False      # Shuffle the dataset
                weights:            # same weight for all, or file with weights
                    config: 1.0
                    energy: 0.0
                    forces: 10.0
                    stress: 0.0
            ```

            3. Manifest file for initializing dataset using ColabFit parser:
            ```yaml
            dataset:
                type: colabfit      # ase or path or colabfit
                save: False         # Save processed dataset to a file
                shuffle: False      # Shuffle the dataset
                weights: None
                colabfit_dataset:
                    dataset_name:
                    database_name:
                    database_url:
            ```


        Args:
            dataset_manifest: List of configurations.

        Returns:
            A dataset of configurations.
        """
        dataset_type = dataset_manifest.get("type").lower()
        if (
            dataset_type != "ase"
            and dataset_type != "path"
            and dataset_type != "colabfit"
        ):
            raise DatasetError(
                f"Dataset type {dataset_type} not supported."
                "Supported types are: ase, path, colabfit"
            )
        weights = dataset_manifest.get("weights", None)
        if weights is not None:
            if isinstance(weights, str):
                weights = Path(weights)
            elif isinstance(weights, dict):
                weights = Weight(
                    config_weight=weights.get("config", None),
                    energy_weight=weights.get("energy", None),
                    forces_weight=weights.get("forces", None),
                    stress_weight=weights.get("stress", None),
                )
            else:
                raise DatasetError("Weights must be a path or a dictionary.")

        if dataset_type == "ase":
            dataset = Dataset.from_ase(
                path=dataset_manifest.get("path", "."),
                weight=weights,
                energy_key=dataset_manifest.get("keys", {}).get("energy", "energy"),
                forces_key=dataset_manifest.get("keys", {}).get("forces", "forces"),
            )
        elif dataset_type == "path":
            dataset = Dataset.from_path(
                path=dataset_manifest.get("path", "."),
                weight=weights,
            )
        elif dataset_type == "colabfit":
            try:
                colabfit_dataset = dataset_manifest.get("colabfit_dataset")
                colabfit_database = colabfit_dataset.database_name
            except KeyError:
                raise DatasetError("Colabfit dataset or database not provided.")
            colabfit_uri = dataset_manifest.get(
                "colabfit_uri", "mongodb://localhost:27017"
            )

            dataset = Dataset.from_colabfit(
                colabfit_database=colabfit_database,
                colabfit_dataset=colabfit_dataset,
                colabfit_uri=colabfit_uri,
                weight=weights,
            )
        else:
            # this should not happen
            raise DatasetError(f"Dataset type {dataset_type} not supported.")

        return dataset


class DatasetError(Exception):
    def __init__(self, msg):
        super(DatasetError, self).__init__(msg)
        self.msg = msg
