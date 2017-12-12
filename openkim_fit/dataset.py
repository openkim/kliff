from __future__ import print_function
import os
import numpy as np
from collections import OrderedDict
from error import InputError

#TODO weight can be allowed to be read in from dataset
#TODO  if energy is to be fitted, allow read only coords, not forces
class Configuration:
  """ Class for one atomistic configuration.

  Parameters
  ----------

  identifer: str
    name of the configuration

  order_by_species: bool
    whether to order coords (, and forces if provided) by species
  """

  def __init__(self, identifier='id_not_provided', order_by_species=True):
    self.id = identifier
    self.weight = 1.0
    self.do_order = order_by_species
    self.natoms = None   # int
    self.cell = None     # 3 by 3 ndarray
    self.PBC = None      # 1 by 3 int
    self.energy = None   # float
    self.species = None  # 1 by N  ndarray (N: number of atoms)
    self.coords = None   # 1 by 3N ndarray (N: number of atoms)
    self.forces = None   # 1 by 3N ndarray (N: number of atoms)
    self.num_atoms_by_species = None   # dict

  def read_extxyz(self, fname):
    """Read atomic configuration stored in extended xyz format.

    Parameter
    ---------

    fname: str
      name of the extended xyz file
    """
    with open(fname, 'r') as fin:
      lines = fin.readlines()

      # number of atoms
      try:
        self.natoms = int(lines[0].split()[0])
      except ValueError as err:
        raise InputError('{}.\nCorrupted data at line 1 in xyz '
            'file: {}.'.format(err, fname))

      # lattice vector, PBC, and energy
      line = lines[1]
      self.cell = self._parse_key_value(line, 'Lattice', 'float', 9, fname)
      self.cell = np.reshape(self.cell, (3, 3))
      self.PBC = self._parse_key_value(line, 'PBC', 'int', 3, fname)
      self.energy = self._parse_key_value(line, 'Energy', 'float', 1, fname)[0]

      # body, species symbol, x, y, z (and fx, fy, fz if provided)
      species = []
      coords = []
      forces = []
      # is forces provided
      line = lines[2].strip().split()
      if len(line) == 4:
        has_forces = False
      elif len(line) == 7:
        has_forces = True
      else:
        raise InputError('\nCorrupted xyz file {} at line 3.'.format(fname))

      try:
        num_lines = 0
        for line in lines[2:]:
          num_lines += 1
          line = line.strip().split()
          if len(line) != 4 and len(line) != 7:
            raise InputError('\nCorrupted data at line {} in xyz '
                 'file: {}.'.format(num_lines+3, fname))
          if has_forces:
            symbol, x, y, z, fx, fy, fz = line
            species.append(symbol.lower().capitalize())
            coords.append([float(x), float(y), float(z)])
            forces.append([float(fx), float(fy), float(fz)])
          else:
            symbol, x, y, z = line
            species.append(symbol.lower().capitalize())
            coords.append([float(x), float(y), float(z)])
      except ValueError as err:
        raise InputError('{}.\nCorrupted data at line {} in xyz'
            'file {}.'.format(err, num_lines+3, fname))

      if num_lines != self.natoms:
        raise InputError('Corrupted xyz file: {}. Listed number of atoms = {}, while '
                 'number of data lines = {}.'.format(fname, self.natoms, num_lines))

      # order according to species
      if self.do_order:
        if has_forces:
          species,coords,forces = zip(*sorted(zip(species,coords,forces),
              key=lambda pair: pair[0]))
        else:
          species,coords = zip(*sorted(zip(species,coords), key=lambda pair: pair[0]))

      # make it numpy array
      self.species = np.array(species)
      self.coords = np.array(coords).ravel()
      if has_forces:
        self.forces = np.array(forces).ravel()
      else:
        self.forces = None
      # count atoms
      self.num_atoms_by_species = self.count_atoms_by_species()


  def _parse_key_value(self, line, key, dtype, size, fname):
    """Given key, parse a string like 'other stuff key = "value" other stuff'
    to get value.

    Parameters
    ----------

    line: str
      The sting line

    key: str
      keyword we want to parse

    dtype: str: {'int', 'float'}
      expected data type of value

    size: int
      expected size of value

    fname: str
      file name where the line comes from

    Return
    ------

    A list of values assocaited with key
    """

    if key not in line:
      raise InputError('"{}" not found at line 2 in file: {}.'.format(key, fname))
    value = line[line.index(key):]
    value = value[value.index('"')+1:]
    value = value[:value.index('"')]
    value = value.split()
    if len(value) != size:
      raise InputError('Incorrect size of "{}" at line 2 in file: {}.\nRequired: {}, '
               'provided: {}.'.format(key, fname, size, len(value)))

    try:
      if dtype == 'float':
        value = [float(i) for i in value]
      elif dtype == 'int':
        value = [int(i) for i in value]
    except ValueError as err:
      raise InputError('{}.\nCorrupted "{}" data at line 2 in file: {}.'.format(err, key, fname))
    return value


  def count_atoms_by_species(self, symbols=None):
    """Count the number of atoms with species `symbols' in the configuration.

    Parameter
    ---------

    symbols: list of str
      species of atoms to count
      If `None', the species already in the configuration are used.

    Return
    ------

    num_atoms_by_species: OrderedDict
      number of atoms for each species
    """

    unique, counts = np.unique(self.species, return_counts=True) # unique is sorted

    if symbols is None:
      symbols = unique

    num_atoms_by_species = OrderedDict()
    for s in symbols:
      if s in unique:
        num_atoms_by_species[s] = counts[list(unique).index(s)]
      else:
        num_atoms_by_species[s] = 0

    return num_atoms_by_species


  def write_extxyz(self, fname='./echo_config.xyz'):

    with open (fname, 'w') as fout:

      # first line (num of atoms)
      fout.write('{}\n'.format(self.natoms))

      # second line
      # lattice
      fout.write('Lattice="')
      for line in self.cell:
        for item in line:
          fout.write('{:.16g} '.format(item))
      fout.write('" ')
      # PBC
      fout.write('PBC="')
      for i in self.PBC:
          fout.write('{} '.format(int(i)))
      fout.write('" ')
      # properties
      fout.write('Properties="species:S:1:pos:R:3:vel:R:3" ')
      # energy
      fout.write('Energy="{:.16g}"\n'.format(self.energy))

      # body
      # species, coords, and forces
      for i in range(self.natoms):
        fout.write('{:3s}'.format(self.species[i]))
        fout.write('{:14.6e}'.format(self.coords[3*i+0]))
        fout.write('{:14.6e}'.format(self.coords[3*i+1]))
        fout.write('{:14.6e}'.format(self.coords[3*i+2]))
        if self.forces is not None:
          fout.write('{:14.6e}'.format(self.forces[3*i+0]))
          fout.write('{:14.6e}'.format(self.forces[3*i+1]))
          fout.write('{:14.6e}'.format(self.forces[3*i+2]))
        fout.write('\n')

  def get_id(self):
    return self.id

  def get_num_atoms(self):
    return self.natoms

  def get_cell(self):
    return self.cell

  def get_pbc(self):
    return self.PBC

  def get_energy(self):
    return self.energy

  def get_species(self):
    return self.species

  def get_coords(self):
    return self.coords

  def get_forces(self):
    return self.forces

  def get_weight(self):
    return self.weight

  def set_weight(self, weight):
    self.weight = weight


class DataSet:
  """Data set class, to deal with multiple configurations.

  Argument
  --------

  order_by_species: bool
    whether to order coords (, and forces if provided) by species
  """

  def __init__(self, order_by_species=True):
    self.do_order = order_by_species
    self.configs = []

#NOTE this could be moved to init
  def read(self, fname):
    """Read atomistic configurations stored in the extend xyz format.

    Parameter
    ---------

    fname: str
      file name or directory name where the configurations are stored. If given
      a directory, all the files in this directory and subdirectories with 'xyz'
      extension will be read.
    """

    if os.path.isdir(fname):
      dirpath = fname
      all_files = []
      for root, dirs, files in os.walk(dirpath):
        for f in files:
          if f.endswith('.xyz'):
            all_files.append(os.path.join(root, f))
      all_files = sorted(all_files)
    else:
      dirpath = os.path.dirname(fname)
      all_files = [fname]

    for f in all_files:
      conf = Configuration(f, self.do_order)
      conf.read_extxyz(f)
      self.configs.append(conf)

    size = len(self.configs)
    if size <= 0:
      raise InputError('No training set files (ends with .xyz) found '
               'in directory: {}/'.format(dirpath))

    print('Number of configurations in dataset:', size)

    if self.do_order:
      # find species present in all configurations
      all_species = []
      for conf in self.configs:
        conf_species = set(conf.get_species())
        all_species.extend(conf_species)
      all_species = set(all_species)

      # find occurence of species in each configuration
      for conf in self.configs:
        conf.num_atoms_by_species = conf.count_atoms_by_species(all_species)


  def get_configs(self):
    """Get the configurations.

    Return
    ------
    a list of Configuration instance.
    """

    return self.configs


