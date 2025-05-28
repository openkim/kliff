from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

from kliff.dataset.extxyz import read_extxyz, write_extxyz
from kliff.dataset.weight import Weight
from kliff.utils import stress_to_tensor, stress_to_voigt, to_path

# For type checking
if TYPE_CHECKING:
    from colabfit.tools.database import MongoDatabase

# check if colabfit-tools is installed
try:
    from colabfit.tools.database import MongoDatabase
except ImportError:
    MongoDatabase = None

import ase
import ase.io
import ase.build.bulk
from ase.calculators.singlepoint import (
    PropertyNotImplementedError,
    SinglePointCalculator,
)
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
            fetched_configuration = database_client.get_cleaned_configuration(
                configuration_id
            )
            fetched_properties = list(
                database_client.get_cleaned_property_instances(
                    data_object["relationships"][0]["property_instance"]
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
            "do-id": data_object["colabfit-id"],
            "co-id": fetched_configuration["colabfit-id"],
            "pi-ids": [pi["colabfit-id"] for pi in fetched_properties],
            "names": fetched_configuration["names"],
        }
        # Update self.metadata with information from metadata collection
        md_dict = database_client.get_metadata_from_do_doc(data_object)
        if md_dict:
            md_dict["md-id"] = md_dict["colabfit-id"]
            md_dict.pop("colabfit-id")
            self.metadata.update(md_dict)

        return self

    def to_colabfit(
        self,
        database_client: "MongoDatabase",
        data_object: dict,
        weight: Optional[Weight] = None,
    ):
        """
        Save configuration from colabfit database.

        Args:
            database_client:
            data_object:
            weight:

        Returns:
        """
        raise NotImplementedError("Export to colabfit is not supported yet.")

    @classmethod
    def from_ase_atoms(
        cls,
        atoms: ase.Atoms,
        weight: Optional[Weight] = None,
        energy_key: str = "energy",
        forces_key: str = "forces",
        stress_key: str = "stress",
    ):
        """
        Read configuration from ase.Atoms object.

        Args:
            atoms: ase.Atoms object.
            weight: an instance that computes the weight of the configuration in the loss
                function.
            energy_key: Name of the field in extxyz that stores the energy.
            forces_key: Name of the field in extxyz that stores the forces.
            stress_key: Name of the field in extxyz that stores the stress.
        """
        cell = atoms.get_cell()
        species = atoms.get_chemical_symbols()
        coords = atoms.get_positions()
        PBC = atoms.get_pbc()

        try:
            energy = (
                atoms.get_potential_energy()
            )  # calculator is attached with property
        except (PropertyNotImplementedError, RuntimeError):
            energy = atoms.info.get(energy_key, None)  # search the info dict, else None

        try:
            forces = atoms.get_forces()
        except (PropertyNotImplementedError, RuntimeError):
            forces = atoms.arrays.get(forces_key, None)

        try:
            stress = atoms.get_stress(voigt=True)
        except (PropertyNotImplementedError, RuntimeError):
            stress: Optional[np.ndarray] = atoms.arrays.get(stress_key, None)
            if stress is not None and stress.ndim == 2:
                stress: list = stress_to_voigt(stress)

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
        properties = {}
        try:
            properties["energy"] = self.energy
        except ConfigurationError:
            pass

        try:
            properties["forces"] = self.forces
        except ConfigurationError:
            pass

        try:
            properties["stress"] = self.stress
        except ConfigurationError:
            pass

        calc = SinglePointCalculator(atoms, **properties)
        atoms.calc = calc
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
        self._metadata |= metadata

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

    def to_dict(self) -> dict:
        cell = self.cell
        species = self.species
        coords = self.coords
        PBC = self.PBC
        try:
            energy = self.energy
        except ConfigurationError:
            energy = None
        try:
            forces = self.forces
        except ConfigurationError:
            forces = None
        try:
            stress = self.stress
        except ConfigurationError:
            stress = None

        weight = self.weight
        if weight is not None:
            energy_weight = weight.energy_weight
            forces_weight = weight.forces_weight
            stress_weight = weight.stress_weight
            config_weight = weight.config_weight
        else:
            energy_weight = None
            forces_weight = None
            stress_weight = None
            config_weight = None

        return {
            "cell": cell,
            "species": species,
            "coords": coords,
            "energy": energy,
            "forces": forces,
            "stress": stress,
            "PBC": PBC,
            "energy_weight": energy_weight,
            "forces_weight": forces_weight,
            "stress_weight": stress_weight,
            "config_weight": config_weight,
        }

    @classmethod
    def bulk(cls, **kwargs) -> "Configuration":
        """
        Transparent wrapper to get KLIFF configuration from bulk ASE atoms.

        Args:
            **kwargs:

        Returns:
            Configuration
        """
        atoms = ase.build.bulk(**kwargs)
        config = cls.from_ase_atoms(atoms)
        return config

    def get_supercell(self, nx: int=1, ny: int=1, nz: int=1) -> "Configuration":
        """
        Generate supercell from a configuration.

        Args:
            nx: repetition along x-axis
            ny: repetition along y-axis
            nz: repetition along z-axis

        Returns:
            Configuration

        """
        new_cell = np.diag([nx, ny, nz]) @ self.cell

        translations = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    translations.append(i * self.cell[0] + j * self.cell[1] + k * self.cell[2])

        translations = np.vstack(translations)
        new_coords = (self.coords[None, : , :] + translations[:, None, :]).reshape(-1, 3)
        new_species = self.species * (nx * ny * nz)

        try:
            new_energy = self.energy * nx * ny * nz
        except ConfigurationError:
            new_energy = None

        try:
            new_forces = np.tile(self.forces, (nx * ny * nz, 1))
        except ConfigurationError:
            new_forces = None

        try:
            new_stress = self.stress
        except ConfigurationError:
            new_stress = None

        return Configuration(
            cell=new_cell,
            species=new_species,
            coords=new_coords,
            energy=new_energy,
            forces=new_forces,
            stress=new_stress,
            PBC=self.PBC,
            weight=self.weight
        )


class ConfigurationError(Exception):
    def __init__(self, msg):
        super(ConfigurationError, self).__init__(msg)
        self.msg = msg
