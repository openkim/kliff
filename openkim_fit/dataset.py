from __future__ import print_function
import os
import numpy as np
from error import InputError

class Configuration:
  """
  Class to read and store the information in one configuraiton in extented xyz format.
  """

  def __init__(self, identifier='id_not_provided', order_by_species=True):
    self.id = identifier
    self.do_order = order_by_species
    self.natoms = None   # int
    self.cell = None     # 3 by 3 ndarray
    self.PBC = None      # 1 by 3 int
    self.energy = None   # float
    self.species = None  # 1 by N  ndarray (N: number of atoms)
    self.coords = None   # 1 by 3N ndarray (N: number of atoms)
    self.forces = None   # 1 by 3N ndarray (N: number of atoms)

  def read_extxyz(self, fname):
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
      self.cell = self.parse_key_value(line, 'Lattice', 'float', 9, fname)
      self.cell = np.reshape(self.cell, (3, 3))
      self.PBC = self.parse_key_value(line, 'PBC', 'int', 3, fname)
      self.energy = self.parse_key_value(line, 'Energy', 'float', 1, fname)[0]

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


#NOTE not needed
#  def get_unique_species(self):
#    '''
#    Get a set of the species list.
#    '''
#    return list(set(self.species))
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

  def parse_key_value(self, line, key, dtype, size, fname):
    '''
    Given key, parse a string like 'other stuff key = "value" other stuff'
    to get value.

    Parameters:

    line: The sting line

    key: keyword we want to parse

    dtype: expected data type of value

    size: expected size of value

    fname: file name where the line comes from

    Returns:

    A list of valves assocaited with key
    '''
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
      # species, coords, and forces
      for i in range(self.natoms):
        fout.write('{:3s}'.format(self.species[i]))
        fout.write('{:14.6e}'.format(self.coords[3*i+0]))
        fout.write('{:14.6e}'.format(self.coords[3*i+1]))
        fout.write('{:14.6e}'.format(self.coords[3*i+2]))
        fout.write('{:14.6e}'.format(self.forces[3*i+0]))
        fout.write('{:14.6e}'.format(self.forces[3*i+1]))
        fout.write('{:14.6e}'.format(self.forces[3*i+2]))
        fout.write('\n')



class DataSet():
  '''
  Data set class, to deal with multiple configurations.
  '''
  def __init__(self, order_by_species=True):
    self.do_order = order_by_species
    self.size = 0
    self.configs = []

#NOTE this could be moved to init
  def read(self, fname):
    """
    Read atomistic configuration stored in the extend xyz format.

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
    self.size = len(self.configs)
    if self.size <= 0:
      raise InputError('No training set files (ends with .xyz) found '
               'in directory: {}/'.format(dirpath))

    print('Number of configurations in dataset:', self.size)


  def get_size(self):
    return self.size

  def get_configs(self):
    return self.configs


