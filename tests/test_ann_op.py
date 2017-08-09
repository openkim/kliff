from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from openkim_fit.training import TrainingSet
from openkim_fit.descriptor import Descriptor
import os
import inspect
import tfop._int_pot_grad
path = os.path.dirname(inspect.getfile(tfop._int_pot_grad))
int_pot_module = tf.load_op_library(path+os.path.sep+'int_pot_op.so')
int_pot = int_pot_module.int_pot





# read config and reference data
tset = TrainingSet()
tset.read('./training_set/config_1x1.xyz')
configs = tset.get_configs()
conf = configs[0]

# create Descriptor
cutfunc = 'cos'
rcut = {'C-C':2}
desc_params = {'g2': [{'eta':0.1, 'Rs':0.2}, {'eta':0.3, 'Rs':0.4}],
               'g3': [{'kappa':0.5}]
              }
desc = Descriptor(cutfunc, rcut, desc_params)
zeta,dzetadr = desc.generate_generalized_coords(conf)
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

#  out = sess.run(func_float)
#  print('float')
#  for i in out:
#    print('{:.15f}'.format(i))
#
  gencoords, deriv = sess.run(func_double)

  print('gencoords\n', gencoords)
  print('deriv\n', deriv)
 # for i in gencoords:
 #   print(i)
 # print('deriv')
 # for i in deriv:
 #   print(i)
