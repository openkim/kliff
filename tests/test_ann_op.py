from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from openkim_fit.dataset import DataSet
from openkim_fit.descriptor import Descriptor
import os
import inspect
import tensorflow_op._int_pot_grad
path = os.path.dirname(inspect.getfile(tensorflow_op._int_pot_grad))
int_pot_module = tf.load_op_library(path+os.path.sep+'int_pot_op.so')
int_pot = int_pot_module.int_pot



# read config and reference data
tset = DataSet()
tset.read('./training_set/training_set_graphene/bilayer_sep3.36_i2_j3.xyz')
configs = tset.get_configs()
conf = configs[0]

# create Descriptor
cutfunc = 'cos'
cutvalue = {'C-C':5.}
desc_params = {'g1': None,
                 'g2': [{'eta':0.1, 'Rs':0.2},
                        {'eta':0.3, 'Rs':0.4}],
                 'g3': [{'kappa':0.1},
                        {'kappa':0.2},
                        {'kappa':0.3}],
                 'g4': [{'zeta':0.1, 'lambda':0.2, 'eta':0.01},
                        {'zeta':0.3, 'lambda':0.4, 'eta':0.02}],
                 'g5': [{'zeta':0.11, 'lambda':0.22, 'eta':0.011},
                        {'zeta':0.33, 'lambda':0.44, 'eta':0.022}]
                }

desc = Descriptor(cutfunc, cutvalue, desc_params)

zeta, dzetadr = desc.generate_generalized_coords(conf)
print('feed value from python')
print('zeta\n', zeta)
print('dzetadr\n', dzetadr)

with tf.Session() as sess:

  ftype = tf.double
  itype = tf.int32
  func_double = int_pot(
      coords=tf.constant(conf.get_coords(), dtype=ftype),
      zeta = tf.constant(zeta, dtype=ftype),
      dzetadr = tf.constant(dzetadr, dtype=ftype),
      )

  init_op = tf.global_variables_initializer()
  sess.run(init_op)

  gencoords, deriv = sess.run(func_double)

  print('\n output value from op')
  print('gencoords\n', gencoords)
  print('deriv\n', deriv)

