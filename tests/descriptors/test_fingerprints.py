from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import tensorflow as tf
from kliff.dataset import DataSet
from kliff.fingerprints import Fingerprints
from kliff.descriptor.symmetryfunction import Set51

# training set
tset = DataSet()
tset.read('training_set/Si_T300_4')
configs = tset.get_configurations()

descriptor = Set51(cutvalue={'Si-Si': 6.0})

fps = Fingerprints(descriptor, normalize=True,
                   fit_forces=False, dtype=tf.float32)

fps.generate_train_tfrecords(configs, nprocs=1)
