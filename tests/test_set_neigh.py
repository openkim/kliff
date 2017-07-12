from __future__ import division
import sys
sys.path.append('../openkim_fit')
from training import TrainingSet
from neighbor import SetNeigh
from utils import write_extxyz
import numpy as np

def test_neigh():
    # read config and reference data
    tset = TrainingSet()
    tset.read('training_set/training_set_graphene/bilayer_sep3.36_i0_j0.xyz')
    configs = tset.get_configs()

    rcut = {'C-C':2}
    nei = SetNeigh(configs[0], rcut)
    write_extxyz(nei.conf.get_cell(), nei.spec, nei.coords, fname='check_set_neigh.xyz')

    assert nei.natoms == 16
    assert nei.numneigh == [3, 3, 3, 3]
    assert nei.neighlist == [[1, 6, 8], [0, 10, 12], [3, 7, 9], [2, 11, 13]]

if __name__ == '__main__':
    test_neigh()
