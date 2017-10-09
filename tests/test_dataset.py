from __future__ import print_function
import sys
from openkim_fit.dataset import Configuration, DataSet


def test_dataset():

  # one configuration
  configs = Configuration()
  configs.read_extxyz('training_set/training_set_MoS2.xyz')
  configs.write_extxyz('./echo.xyz')
  print ('training config written to: echo.xyz')

  # multiple configuration
  Tset = DataSet()
  path = 'training_set/training_set_multi_small'
  Tset.read(path)
  print ('Number of configurations in "{}": {}'.format(path, Tset.get_size()))
  configs = Tset.get_configs()
  for i,conf in enumerate(configs):
    conf.write_extxyz('echo{}.xyz'.format(i))


if __name__ == '__main__':
  test_dataset()
