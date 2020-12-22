import logging
import os
from collections import OrderedDict

import numpy as np

from kliff.log import log_entry
from kliff.dataset.extxyz import read_extxyz, write_extxyz


logger = logging.getLogger(__name__)


SUPPORTED_FORMAT = {"extxyz": ".xyz"}


class Configuration:
    r"""Class of atomic configuration.

    Parameters
    ----------

    filename: str
        Path to the file storing the atomic configuration. If `None`, atomic
        information is not read.

    format: str
        Format of the file that stores the configuration. Currently, supported format
        includes: `extxyz`.

    identifier: str
        A unique identifier of the configuration.

    order_by_species: bool
        If `True`, the atoms in the configuration will be ordered according to their
        species such that atoms with the same species will have contiguous indices.
    """

    def __init__(
        self, filename=None, format="extxyz", identifier=None, order_by_species=True
    ):
        self.format = format
        self.id = identifier
        self.do_order = order_by_species
        self.weight = 1.0
        self.natoms = None  # int
        self.cell = None  # ndarray of shape(3,3)
        self.PBC = None  # ndarray of shape(3,)
        self.energy = None  # float
        self.stress = None  # ndarray of shape(6,)
        self.species = None  # ndarray of shape(N,)
        self.coords = None  # ndarray of shape(N, 3)
        self.forces = None  # ndarray of shape(N, 3)
        self.natoms_by_species = None  # dict

        if filename is not None:
            self.read(filename)

    def read(self, path):
        r"""Read configuration stored in a file.

        Parameters
        ----------
        path: str
            Path to the file that stores the configuration.
        """
        (
            self.cell,
            self.PBC,
            self.species,
            self.coords,
            self.energy,
            self.forces,
            self.stress,
        ) = read_config(path, self.format)
        self.natoms = len(self.species)
        self.volume = abs(np.dot(np.cross(self.cell[0], self.cell[1]), self.cell[2]))

        if self.do_order:
            self.order_by_species()

    def order_by_species(self):
        r"""Order the atoms according to the species."""
        if self.forces is not None:
            species, coords, forces = zip(
                *sorted(
                    zip(self.species, self.coords, self.forces),
                    key=lambda pair: pair[0],
                )
            )
            self.species = np.asarray(species)
            self.coords = np.asarray(coords)
            self.forces = np.asarray(forces)
        else:
            species, coords = zip(
                *sorted(zip(self.species, self.coords), key=lambda pair: pair[0])
            )
            self.species = np.asarray(species)
            self.coords = np.asarray(coords)

    def count_atoms_by_species(self, symbols=None):
        r"""Count the number of atoms with species `symbols` in the configuration.

        Parameters
        ----------
        symbols: list
            A list of species string. If `None`, the species that are already in the
            configuration are used.

        Return
        ------
        Return an OrderedDict with `keys` the species string specified in `symbols`, and
        `values` the number of atoms with each species.
        """

        unique, counts = np.unique(self.species, return_counts=True)
        symbols = unique if symbols is None else symbols

        natoms_by_species = OrderedDict()
        for s in symbols:
            if s in unique:
                natoms_by_species[s] = counts[list(unique).index(s)]
            else:
                natoms_by_species[s] = 0

        self.natoms_by_species = natoms_by_species
        return natoms_by_species

    def get_identifier(self):
        r"""Return the identifier of the configuration, which is specified at the
        initialization of the class."""
        return self.id

    def get_number_of_atoms(self):
        r"""Return the total number of atoms in the configuration."""
        return self.natoms

    def get_number_of_atoms_by_species(self):
        r"""Return a dictionary of the number of atoms with each species."""
        return self.count_atoms_by_species()

    def get_cell(self):
        r"""Return a 3x3 matrix of the lattice vectors of the configurations.

        The first, second, and third rows are :math:`a_1`, :math:`a_2`, and :math:`a_3`,
        respectively.
        """
        return self.cell.copy()

    def get_volume(self):
        r"""Return the volume of the configuration."""
        return self.volume

    def get_PBC(self):
        r"""Return a list with 3 components indicating whether periodic boundary condition
        is used along the directions of the first, second, and third lattice vectors.
        """
        return self.PBC.copy()

    def get_species(self):
        r"""Return a list of species string of all atoms."""
        return self.species.copy()

    def get_coordinates(self):
        r"""Return a `Nx3` matrix of the Cartesian coordinates of all atoms."""
        return self.coords.copy()

    def get_energy(self):
        r"""Return the potential energy of the configuration."""
        if self.energy is None:
            raise DatasetError("Configuration does not contain forces.")
        return self.energy

    def get_forces(self):
        r"""Return a `Nx3` matrix of the forces on each atoms."""
        if self.forces is None:
            raise DatasetError("Configuration does not contain forces.")
        return self.forces.copy()

    def get_stress(self):
        r"""Return the stress of the configuration.

        It returns a list with 6 components in Voigt notation, i.e. it returns
        :math:`\sigma=[\sigma_{xx},\sigma_{yy},\sigma_{zz},\sigma_{yz},\sigma_{xz},
        \sigma_{xy}]`.

        See Also
        --------
            https://en.wikipedia.org/wiki/Voigt_notation
        """
        if self.stress is None:
            raise DatasetError("Configuration does not contain stress.")
        return self.stress.copy()

    def set_weight(self, weight):
        r"""Set the weight of the configuration if the loss function.

        Parameters
        ----------
        weight: float
            The weight of the configuration.
        """
        self.weight = weight

    def get_weight(self):
        r"""Get the weight of the configuration if the loss function."""
        return self.weight


class Dataset:
    r"""Dataset class to deal with multiple :class:`~kliff.dataset.Configuration`.

    Parameters
    ----------
    path: str
        Path of a file storing a configuration or path to a directory containing
        multiple files. If given a directory, all the files in this directory and its
        subdirectories with the extension corresponding to the specified format will
        be read.

    format: str
        Format of the file that stores the configuration. Currently, supported format
        includes: `extxyz`.

    order_by_species: bool
        If `True`, the atoms in each configuration will be ordered according to their
        species such that atoms with the same species will have contiguous indices.

    """

    def __init__(self, path=None, format="extxyz", order_by_species=True):
        self.order_by_species = order_by_species
        self.configs = []

        if path is not None:
            self.read(path, format)

        logger.info('"{}" instantiated.'.format(self.__class__.__name__))

    def read(self, path, format="extxyz"):
        r"""Read an atomic configuration.

        Parameters
        ----------
        path: str
            Path of a file storing a configuration or path to a directory containing
            multiple files. If given a directory, all the files in this directory and its
            subdirectories with the extension corresponding to the specified format will
            be read.

        format: str
            Format of the file that stores the configuration (e.g. 'extxyz').
        """
        try:
            extension = SUPPORTED_FORMAT[format]
        except KeyError:
            raise DatasetError(
                f"Expect data format to be one of {list(SUPPORTED_FORMAT.keys())}, "
                "But got unsupported one: {format}."
            )

        if os.path.isdir(path):
            dirpath = path
            all_files = []
            for root, dirs, files in os.walk(dirpath):
                for f in files:
                    if f.endswith(extension):
                        all_files.append(os.path.join(root, f))
            all_files = sorted(all_files)
        else:
            dirpath = os.path.dirname(path)
            all_files = [path]

        configs = [
            Configuration(
                f, format, identifier=f, order_by_species=self.order_by_species
            )
            for f in all_files
        ]

        if len(configs) <= 0:
            raise DatasetError(
                f"No dataset file with format `{format}` found at {dirpath}."
            )

        self.configs.extend(configs)

        if self.order_by_species:
            # find species present in all configurations
            all_species = []
            for conf in self.configs:
                conf_species = set(conf.get_species())
                all_species.extend(conf_species)
            all_species = set(all_species)
            # find occurrence of species in each configuration
            for conf in self.configs:
                conf.natoms_by_species = conf.count_atoms_by_species(all_species)

        msg = '{} configurations read from "{}"'.format(len(configs), path)
        log_entry(logger, msg, level="info")

    def get_configs(self):
        r"""Get the configurations.

        Return
        ------
            A list of :class:`~kliff.dataset.Configuration` instance.
        """
        return self.configs

    def get_num_configs(self):
        """Return the number of configurations in the dataset"""
        return len(self.configs)


def read_config(path, fmt="extxyz"):
    r"""Read configuration stored in a file.

    Parameters
    ----------
    path: str
        Path to the file that stores the configuration.

    fmt: str
        Format of the file that stores the configuration (e.g. `extxyz`).

    Returns
    -------
    cell: array
        A 3x3 matrix of the lattice vectors.  The first, second, and third rows are
        :math:`a_1`, :math:`a_2`, and :math:`a_3`, respectively.

    PBC: list
        A list with 3 components indicating whether periodic boundary condition is used
        along the directions of the first, second, and third lattice vectors.

    species: list
        A list of string with N component, where N is the number of atoms.

    coords: array
        A Nx3 matrix of the coordinates of the atoms, where N is the number of atoms.

    energy: float or None
        Potential energy of the configuration. If it is not provided in the file, return
        `None`.

    forces: array or None
        A Nx3 array of the forces on atoms, where N is the number of atoms. If the forces
        are not provided in the file, return `None`.

    stress: list or None
        A list with 6 components in Voigt notation, i.e. it returns
        :math:`\sigma=[\sigma_{xx},\sigma_{yy},\sigma_{zz},\sigma_{yz},\sigma_{xz},
        \sigma_{xy}]`. If the stresses are not provided in the file, return `None`.
    """

    if fmt == "extxyz":
        cell, PBC, species, coords, energy, forces, stress = read_extxyz(path)
    else:
        raise DatasetError(
            f"Expect data format to be one of {list(SUPPORTED_FORMAT.keys())}, "
            "But got unsupported one: {fmt}."
        )

    return cell, PBC, species, coords, energy, forces, stress


def write_config(
    path,
    cell,
    PBC,
    species,
    coords,
    energy=None,
    forces=None,
    stress=None,
    fmt="extxyz",
):
    r"""
    Write a configuration to a file in the specified format.

    Parameters
    ----------
    path: str
        Path to the file that stores the configuration.

    fmt: str
        Format of the file that stores the configuration (e.g. `extxyz`).

    cell: array
        A 3x3 matrix of the lattice vectors.  The first, second, and third rows are
        :math:`a_1`, :math:`a_2`, and :math:`a_3`, respectively.

    PBC: list
        A list with 3 components indicating whether periodic boundary condition is used
        along the directions of the first, second, and third lattice vectors.

    species: list
        A list of string with N component, where N is the number of atoms.

    coords: array
        A Nx3 matrix of the coordinates of the atoms, where N is the number of atoms.

    energy: float (optional)
        Potential energy of the configuration. If `None`, skip writing this information.

    forces: array (optional)
        A Nx3 array of the forces on atoms, where N is the number of atoms. If `None`,
        skip writing this information.

    stress: list (optional)
        A list with 6 components in Voigt notation, i.e. it returns
        :math:`\sigma=[\sigma_{xx},\sigma_{yy},\sigma_{zz},\sigma_{yz},\sigma_{xz},
        \sigma_{xy}]`. If `None`, skip writing this information.
    """

    if fmt not in SUPPORTED_FORMAT:
        raise DatasetError(
            f"Expect data format to be one of {list(SUPPORTED_FORMAT.keys())}, "
            "But got unsupported one: {fmt}."
        )

    dirname = os.path.dirname(os.path.abspath(path))
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    if fmt == "extxyz":
        write_extxyz(path, cell, PBC, species, coords, energy, forces, stress)


class DatasetError(Exception):
    def __init__(self, msg):
        super(DatasetError, self).__init__(msg)
        self.msg = msg
