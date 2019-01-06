import os
import numpy as np
import copy
from collections import OrderedDict
from kliff.error import SupportError
from .extxyz import read_extxyz


implemented_format = dict()
implemented_format['extxyz'] = '.xyz'


class Configuration(object):
    """ Class for one atomistic configuration.

    Parameters
    ----------

    identifer: str
      name of the configuration

    order_by_species: bool
      whether to order coords (, and forces if provided) by species
    """

    def __init__(self, format='extxyz', identifier=None, order_by_species=True):
        self.format = format
        self.id = identifier
        self.do_order = order_by_species
        self.weight = 1.0
        self.natoms = None   # int
        self.cell = None     # ndarray of shape(3,3)
        self.PBC = None      # ndarray of shape(3,)
        self.energy = None   # float
        self.stress = None   # ndarray of shape(6,)
        self.species = None  # ndarray of shape(N,)
        self.coords = None   # ndarray of shape(N, 3)
        self.forces = None   # ndarray of shape(N, 3)
        self.natoms_by_species = None   # dict

    def read(self, fname):
        if self.format not in implemented_format:
            raise SupportError('Data file format "{}" not recognized.')

        if self.format == 'extxyz':
            (self.cell, self.PBC, self.energy, self.stress, self.species,
             self.coords, self.forces) = read_extxyz(fname)
            self.natoms = len(self.species)
            self.volume = abs(np.dot(np.cross(self.cell[0], self.cell[1]), self.cell[2]))

        if self.do_order:
            self.order_by_species()

    def order_by_species(self):
        if self.forces is not None:
            species, coords, forces = zip(*sorted(
                zip(self.species, self.coords, self.forces), key=lambda pair: pair[0]))
            self.species = np.asarray(species)
            self.coords = np.asarray(coords)
            self.forces = np.asarray(forces)
        else:
            species, coords = zip(*sorted(
                zip(self.species, self.coords), key=lambda pair: pair[0]))
            self.species = np.asarray(species)
            self.coords = np.asarray(coords)

    def count_atoms_by_species(self, symbols=None):
        """Count the number of atoms with species `symbols' in the configuration.

        Parameters
        ----------

        symbols: list of str
          species of atoms to count
          If `None', the species already in the configuration are used.

        Return
        ------

        natoms_by_species: OrderedDict
            key: str, value: int
        """

        unique, counts = np.unique(self.species, return_counts=True)  # unique is sorted

        if symbols is None:
            symbols = unique

        natoms_by_species = OrderedDict()
        for s in symbols:
            if s in unique:
                natoms_by_species[s] = counts[list(unique).index(s)]
            else:
                natoms_by_species[s] = 0

        return natoms_by_species

    def get_identifier(self):
        return self.id

    def get_number_of_atoms(self):
        return self.natoms

    def get_number_of_atoms_by_species(self):
        return copy.deepcopy(self.natoms_by_species)

    def get_cell(self):
        return self.cell.copy()

    def get_volume(self):
        return self.volume

    def get_PBC(self):
        return self.PBC.copy()

    def get_species(self):
        return self.species.copy()

    def get_coordinates(self):
        return self.coords.copy()

    def get_energy(self):
        return self.energy

    def get_forces(self):
        if self.forces is not None:
            return self.forces.copy()
        else:
            return None

    def get_stress(self):
        if self.stress is not None:
            return self.stress.copy()
        else:
            return None

    def get_weight(self):
        return self.weight

    def set_weight(self, weight):
        self.weight = weight


class DataSet(object):
    """Data set class, to deal with multiple configurations.

    Argument
    --------

    order_by_species: bool
      whether to order coords (forces if provided) by species
    """

    def __init__(self, order_by_species=True):
        """"
        Parameters
        ---------

        order_by_species: bool
            whether to sort coords (forces if provided) by species
        """
        self.do_order = order_by_species
        self.configs = []

    def read(self, fname, format='extxyz'):
        """Read atomistic configurations.

        Parameters
        ----------

        fname: str
            File name or directory name where the configurations are stored. If given
            a directory, all the files in this directory and subdirectories with the
            extension corresponding to the specified format will be read.

        format: str
            The format in which the data is stored (e.g. 'extxyz').
        """
        try:
            extension = implemented_format[format]
        except KeyError as e:
            raise SupportError(
                '{}\nNot supported data file format "{}".'.format(e, format))

        if os.path.isdir(fname):
            dirpath = fname
            all_files = []
            for root, dirs, files in os.walk(dirpath):
                for f in files:
                    if f.endswith(extension):
                        all_files.append(os.path.join(root, f))
            all_files = sorted(all_files)
        else:
            dirpath = os.path.dirname(fname)
            all_files = [fname]

        for f in all_files:
            conf = Configuration(format, f, self.do_order)
            conf.read(f)
            self.configs.append(conf)

        size = len(self.configs)
        if size <= 0:
            raise InputError(
                'No dataset file with format "{}" found in directory: {}.'
                .format(format, dirpath))

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

    def get_configurations(self):
        """Get the configurations.

        Return
        ------
        a list of Configuration instance.
        """

        return self.configs
