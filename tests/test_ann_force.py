from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from tensorflow.contrib import learn
import tensorflow as tf
from openkim_fit.training import TrainingSet
from openkim_fit.descriptor import Descriptor
import openkim_fit.ann as ann

# set a global random seed
tf.set_random_seed(1)


if __name__ == "__main__":


  # read config and reference data
  tset = TrainingSet()
  tset.read('./training_set/config_1x1.xyz')
  #tset.read('./training_set/bilayer_registry_sep3.38_i0_j0.xyz')
  configs = tset.get_configs()

  # create descriptors
  cutfunc = 'cos'
  cutvalue = {'C-C': 5.}
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

  # create params (we need to share params among different config, so create first)
  num_desc = desc.get_num_descriptors()
  weights,biases = ann.parameters(num_desc, [20, 10, 1])


  # create graph
  subloss = []
  for conf in configs:

    in_layer, coords = ann.input_layer(conf, desc)
    dense1 = ann.nn_layer(in_layer, weights[0], biases[0], 'hidden1')
    dense2 = ann.nn_layer(dense1, weights[1], biases[1], 'hidden2')
    output = ann.output_layer(dense2, weights[2], biases[2], 'outlayer')

    # compute forces
    with tf.name_scope('forces'):
      forces = tf.gradients(output, coords)[0] # tf.gradients return a LIST of tensors

    subloss.append(tf.reduce_sum( tf.square( tf.subtract(forces, conf.get_forces()) ) ))

  # loss
  loss = tf.reduce_sum(subloss)

  # log loss
  tf.summary.scalar('loss_value', loss)


  with tf.name_scope('train_node'):
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)


  with tf.Session() as sess:

    # Merge all the summaries and write them out to /tmp/tensorflow/potential
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('/tmp/tensorflow/potential', sess.graph)

    # init global vars
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    for i in range(100): # run 100 steps
      sess.run(train)
      if i%10 == 0:
        out, summary = sess.run([loss, merged])
        train_writer.add_summary(summary, i)
        print ('i =',i, 'loss =', out)

    # output results to a KIM model
    w,b = sess.run([weights, biases])
    ann.write_kim_ann(desc, w, b, 'relu')


