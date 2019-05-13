import os
import numpy as np
import copy
from collections import OrderedDict
from kliff.error import SupportError
from kliff.dataset.extxyz import read_extxyz, write_extxyz


implemented_format = dict()
implemented_format['extxyz'] = '.xyz'


class Configuration:
    """Class of atomic configuration.

    Parameters
    ----------
    format: str
        Format of the file that stores the configuration. Currently, supported format
        includes: `extxyz`.

    identifer: str
        A unique identifier of the configuration.

    order_by_species: bool
        If `True`, the atoms in the configuration will be ordered according to their
        species such that atoms with the same species will have contiguous indices.
    """

    def __init__(self, format='extxyz', identifier=None, order_by_species=True):
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

    def read(self, path):
        """Read configuration stored in a file.

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
        """Order the atoms according to the species."""
        if self.forces is not None:
            species, coords, forces = zip(
                *sorted(
                    zip(self.species, self.coords, self.forces), key=lambda pair: pair[0]
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
        """Count the number of atoms with species `symbols` in the configuration.

        Parameters
        ----------
        symbols: list
            A list of species string. If `None`, the species that are already in the
            configuration are used.

        Return
        ------
        Return an OrderedDict with `keys` the species string specified in `symbols`,
        and `values` the number of atoms with each species.
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
        """Return the identifier of the configuration, which is specified at the
        initialization of the class."""
        return self.id

    def get_number_of_atoms(self):
        """Return the total number of atoms in the configuration."""
        return self.natoms

    def get_number_of_atoms_by_species(self):
        """Return a dictionary of the number of atoms with each species."""
        return self.count_atoms_by_species()

    def get_cell(self):
        """Return a 3x3 matrix of the lattice vectors of the configurations.

        The first, second, and third rows are :math:`a_1`, :math:`a_2`, and
        :math:`a_3`, respetively.
        """
        return self.cell.copy()

    def get_volume(self):
        """Return the volume of the configuration."""
        return self.volume

    def get_PBC(self):
        """Return a list with 3 components indicating whether periodic boundary
        condiction is used along the directions of the first, second, and third
        lattice vectors.
        """
        return self.PBC.copy()

    def get_species(self):
        """Return a list of species string of all atoms."""
        return self.species.copy()

    def get_coordinates(self):
        """Return a `Nx3` matrix of the Cartesian coordiantes of all atoms."""
        return self.coords.copy()

    def get_energy(self):
        """Return the potential energy of the configuration."""
        return self.energy

    def get_forces(self):
        """Return a `Nx3` matrix of the forces on each atoms."""
        if self.forces is not None:
            return self.forces.copy()
        else:
            return None

    def get_stress(self):
        """Return the stress of the configuration.

        It returns a list with 6 components in Voigt notation, i.e. it returns
        :math:`\sigma=[\sigma_{xx},\sigma_{yy},\sigma_{zz},\sigma_{yz},\sigma_{xz},
        \sigma_{xy}]`.

        .. seealso::
            https://en.wikipedia.org/wiki/Voigt_notation
        """
        if self.stress is not None:
            return self.stress.copy()
        else:
            return None

    def set_weight(self, weight):
        """Set the weight of the configuration if the loss function.

        Parameters
        ----------
        weight: float
            The weight of the configuration.
        """
        self.weight = weight

    def get_weight(self):
        """Get the weight of the configuration if the loss function.
        """
        return self.weight


class DataSet:
    """Dataset class to deal with multiple :class:`~kliff.dataset.Configuration`.

    Parameters
    ----------
    order_by_species: bool
        If `True`, the atoms in each configuration will be ordered according to their
        species such that atoms with the same species will have contiguous indices.

    """

    def __init__(self, order_by_species=True):
        self.do_order = order_by_species
        self.configs = []

    def read(self, path, format='extxyz'):
        """Read an atomic configuration.

        Parameters
        ----------

        path: str
            Path of a file storing a configuration or path to a directory containing
            multiple files. If given a directory, all the files in this directory and
            its subdirectories with the extension corresponding to the specified
            format will be read.

        format: str
            Format of the file that stores the configuration (e.g. 'extxyz').
        """
        try:
            extension = implemented_format[format]
        except KeyError as e:
            raise SupportError(
                '{}\nNot supported data file format "{}".'.format(e, format)
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

        for f in all_files:
            conf = Configuration(format, f, self.do_order)
            conf.read(f)
            self.configs.append(conf)

        size = len(self.configs)
        if size <= 0:
            raise InputError(
                'No dataset file with format "{}" found in directory: {}.'.format(
                    format, dirpath
                )
            )

        if self.do_order:
            # find species present in all configurations
            all_species = []
            for conf in self.configs:
                conf_species = set(conf.get_species())
                all_species.extend(conf_species)
            all_species = set(all_species)
            # find occurence of species in each configuration
            for conf in self.configs:
                conf.natoms_by_species = conf.count_atoms_by_species(all_species)

    def get_configs(self):
        """Get the configurations.

        Return
        ------
            A list of :class:`~kliff.dataset.Configuration` instance.
        """
        return self.configs

    def get_num_configs(self):
        """Return the number of configurations in the dataset"""
        return len(self.configs)


def read_config(path, format='extxyz'):
    """Read configuration stored in a file.

    Parameters
    ----------
    path: str
        Path to the file that stores the configuration.

    format: str
        Format of the file that stores the configuration (e.g. `extxyz`).

    Returns
    -------
    cell: array
        A 3x3 matrix of the lattice vectors.  The first, second, and third rows are
        :math:`a_1`, :math:`a_2`, and :math:`a_3`, respetively.

    PBC: list
        A list with 3 components indicating whether periodic boundary condiction is
        used along the directions of the first, second, and third lattice vectors.

    species: list
        A list of string with N componment, where N is the number of atoms.

    coords: array
        A Nx3 matrix of the coordinates of the atoms, where N is the number of atoms.

    energy: float or None
        Potential energy of the configuration. If it is not provided in the file,
        return `None`.

    forces: array or None
        A Nx3 array of the forces on atoms, where N is the number of atoms.
        If the forces are not provided in the file, return `None`.

    stress: list or None
        A list with 6 components in Voigt notation, i.e. it returns
        :math:`\sigma=[\sigma_{xx},\sigma_{yy},\sigma_{zz},\sigma_{yz},\sigma_{xz},
        \sigma_{xy}]`. If the stresses are not provided in the file, return `None`.
    """

    if format not in implemented_format:
        raise SupportError('Data file format "{}" not recognized.')

    if format == 'extxyz':
        cell, PBC, species, coords, energy, forces, stress = read_extxyz(path)

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
    format='extxyz',
):
    """
    Write a configuration to a file in the specified format.

    Parameters
    ----------
    path: str
        Path to the file that stores the configuration.

    format: str
        Format of the file that stores the configuration (e.g. `extxyz`).

    cell: array
        A 3x3 matrix of the lattice vectors.  The first, second, and third rows are
        :math:`a_1`, :math:`a_2`, and :math:`a_3`, respetively.

    PBC: list
        A list with 3 components indicating whether periodic boundary condiction is
        used along the directions of the first, second, and third lattice vectors.

    species: list
        A list of string with N componment, where N is the number of atoms.

    coords: array
        A Nx3 matrix of the coordinates of the atoms, where N is the number of atoms.

    energy: float (optional)
        Potential energy of the configuration. If `None`, skip writting this
        information.

    forces: array (optional)
        A Nx3 array of the forces on atoms, where N is the number of atoms.
        If `None`, skip writting this information.

    stress: list (optional)
        A list with 6 components in Voigt notation, i.e. it returns
        :math:`\sigma=[\sigma_{xx},\sigma_{yy},\sigma_{zz},\sigma_{yz},\sigma_{xz},
        \sigma_{xy}]`. If `None`, skip writting this information.
    """

    if format not in implemented_format:
        raise SupportError('Data file format "{}" not recognized.')

    if format == 'extxyz':
        write_extxyz(path, cell, PBC, species, coords, energy, forces, stress)
