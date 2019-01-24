from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import tensorflow as tf
from kliff.dataset import DataSet
from kliff.fingerprints import Fingerprints
from kliff.descriptors.symmetry_function import Set51

# training set
tset = DataSet()
tset.read('../configs_extxyz/Si_4')
configs = tset.get_configurations()

descriptor = Set51(cutvalue={'Si-Si': 6.0})

fps = Fingerprints(descriptor, normalize=True,
                   fit_forces=False, dtype=tf.float32)

fps.generate_train_tfrecords(configs, nprocs=1)
