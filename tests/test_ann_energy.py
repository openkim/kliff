from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from tensorflow.contrib import learn
import tensorflow as tf
from openkim_fit.training import TrainingSet
from openkim_fit.descriptor import Descriptor
from openkim_fit.write_ann import write_kim_ann
import os
import inspect
import tfop._int_pot_grad
path = os.path.dirname(inspect.getfile(tfop._int_pot_grad))
int_pot_module = tf.load_op_library(path+os.path.sep+'int_pot_op.so')
int_pot = int_pot_module.int_pot


# set a global random seed
tf.set_random_seed(1)


def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def weight_variable(input_dim, output_dim, dtype=tf.float32):
  """Create a weight variable with appropriate initialization."""
  with tf.name_scope('weights'):
    shape = [input_dim, output_dim]
    weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1), dtype=dtype)
    variable_summaries(weights)
    return weights


def bias_variable(output_dim, dtype=tf.float32):
  """Create a bias variable with appropriate initialization."""
  with tf.name_scope('biases'):
    shape = [output_dim]
    biases = tf.Variable(tf.constant(0.1, shape=shape), dtype=dtype)
    variable_summaries(biases)
    return biases


def parameters(num_descriptors, units):
  """Create all weights and biases."""
  weights = []
  biases = []
  # input layer to first nn layer
  w = weight_variable(num_descriptors, units[0])
  b = bias_variable(units[0])
  weights.append(w)
  biases.append(b)
  # nn layer to next till output
  nlayers = len(units)
  for i in range(1, nlayers):
    w = weight_variable(units[i-1], units[i])
    b = bias_variable(units[i])
    weights.append(w)
    biases.append(b)
  return weights, biases


def nn_layer(input_tensor, weights, biases, layer_name, act=tf.nn.relu):
  """Reusable code for making a simple neural net layer.

  It does a matrix multiply, bias add, and then uses relu to nonlinearize.
  It also sets up name scoping so that the resultant graph is easy to read,
  and adds a number of summary ops.
  """
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights) + biases
      tf.summary.histogram('pre_activations', preactivate)
    activations = act(preactivate, name='activation')
    tf.summary.histogram('activations', activations)
    return activations


def output_layer(input_tensor, weights, biases, layer_name):
  """Reusable code for making a simple neural net layer.

  It does a matrix multiply, bias add, no activation is used.
  """
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights) + biases
      tf.summary.histogram('linear_output', preactivate)
    return preactivate


def input_layer(config, descriptor, ftype=tf.float32):
  """Reusable code for making an input layer for a configuration."""

  layer_name = os.path.splitext(os.path.basename(conf.id))[0]
  with tf.name_scope(layer_name):
    zeta,dzetadr = descriptor.generate_generalized_coords(config)
    input, dummy = int_pot(coords = tf.constant(config.get_coords(), ftype),
         zeta=tf.constant(zeta,ftype), dzetadr= tf.constant(dzetadr,ftype))
    return input



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
  weights,biases = parameters(num_desc, [20, 10, 1])


  # create graph
  subloss = []
  for conf in configs:

    in_layer = input_layer(conf, desc)
    dense1 = nn_layer(in_layer, weights[0], biases[0], 'hidden1')
    dense2 = nn_layer(dense1, weights[1], biases[1], 'hidden2')
    output = output_layer(dense2, weights[2], biases[2], 'outlayer')

    subloss.append(tf.square(tf.reduce_sum(output) - conf.get_energy()))

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
    write_kim_ann(desc, w, b, 'relu')


