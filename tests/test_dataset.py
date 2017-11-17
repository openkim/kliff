from __future__ import print_function
import sys
from openkim_fit.dataset import Configuration, DataSet


def test_dataset():

  # one configuration
  config = Configuration()
  config.read_extxyz('training_set/training_set_MoS2.xyz')
  config.write_extxyz('./echo.xyz')
  print ('training config written to: echo.xyz')
  print ('number of atoms by species', config.num_atoms_by_species)

  # multiple configuration
  Tset = DataSet()
  path = 'training_set/training_set_multi_small'
  Tset.read(path)
  configs = Tset.get_configs()
  print ('Number of configurations in "{}": {}'.format(path, len(configs)))
  for i,conf in enumerate(configs):
    conf.write_extxyz('echo{}.xyz'.format(i))
    print ('number of atoms by species', conf.num_atoms_by_species)


if __name__ == '__main__':
  test_dataset()
