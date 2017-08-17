from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import inspect
import tfop._int_pot_grad
path = os.path.dirname(inspect.getfile(tfop._int_pot_grad))
int_pot_module = tf.load_op_library(path+os.path.sep+'int_pot_op.so')
int_pot = int_pot_module.int_pot


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
    weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1, dtype=dtype))
    variable_summaries(weights)
    return weights


def bias_variable(output_dim, dtype=tf.float32):
  """Create a bias variable with appropriate initialization."""
  with tf.name_scope('biases'):
    shape = [output_dim]
    #biases = tf.Variable(tf.constant(0.1, shape=shape, dtype=dtype))
    biases = tf.constant(0.1, shape=shape, dtype=dtype)
    variable_summaries(biases)
    return biases


def parameters(num_descriptors, units, dtype=tf.float32):
  """Create all weights and biases."""
  weights = []
  biases = []
  # input layer to first nn layer
  w = weight_variable(num_descriptors, units[0], dtype)
  b = bias_variable(units[0], dtype)
  weights.append(w)
  biases.append(b)
  # nn layer to next till output
  nlayers = len(units)
  for i in range(1, nlayers):
    w = weight_variable(units[i-1], units[i], dtype)
    b = bias_variable(units[i], dtype)
    weights.append(w)
    biases.append(b)
  return weights, biases


def nn_layer(input_tensor, weights, biases, layer_name='hidden_layer', act=tf.nn.relu):
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


def output_layer(input_tensor, weights, biases, layer_name='output_layer'):
  """Reusable code for making a simple neural net layer.

  It does a matrix multiply, bias add, no activation is used.
  """
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights) + biases
      tf.summary.histogram('linear_output', preactivate)
    return preactivate


def preprocess(configs, descriptor):
  """Preprocess the data to generate the generalized coords and its derivatives.

  Parameter
  ---------

  configs, list of Config objects.

  descriptor, object of Descriptor class.
  """

  all_zeta = []
  all_dzetadr = []
  for i,conf in enumerate(configs):
    print('Preprocessing configuration:', i)
    zeta,dzetadr = descriptor.generate_generalized_coords(conf)
    all_zeta.append(zeta)
    all_dzetadr.append(dzetadr)
  all_zeta_concatenated = np.concatenate(all_zeta)
  mean = np.mean(all_zeta_concatenated, axis=0)
  std = np.std(all_zeta_concatenated, axis=0)

  # centering and normalization
  all_zeta_processed = []
  all_dzetadr_processed = []
  for zeta in all_zeta:
    all_zeta_processed.append( (zeta - mean) / std )
  for dzetadr in all_dzetadr:
    all_dzetadr_processed.append( dzetadr / np.atleast_2d(std).T)

  return all_zeta_processed, all_dzetadr_processed


def input_layer_using_preprocessed(zeta, dzetadr, layer_name='input_layer',
    dtype=tf.float32):
  """Reusable code for making an input layer for a configuration."""

  # we only need some placeholder for coords, since it is not used
#TODO we'd better use the actual coords
  shape = [zeta.shape[0]*3]
  coords = tf.constant(0.1, dtype=dtype, shape=shape)
  with tf.name_scope(layer_name):
    input, dummy = int_pot(coords=coords, zeta=tf.constant(zeta, dtype),
        dzetadr=tf.constant(dzetadr, dtype))
    return input,coords


def input_layer(config, descriptor, dtype=tf.float32):
  """Reusable code for making an input layer for a configuration."""

  layer_name = os.path.splitext(os.path.basename(config.id))[0]
  # need to return a tensor of coords since we want to take derivaives w.r.t it
  coords = tf.constant(config.get_coords(), dtype)
  zeta,dzetadr = descriptor.generate_generalized_coords(config)

  with tf.name_scope(layer_name):
    input, dummy = int_pot(coords = coords, zeta=tf.constant(zeta, dtype),
        dzetadr=tf.constant(dzetadr, dtype))
    return input, coords


def write_kim_ann(descriptor, weights, biases, activation, mode='float',
    fname='ann_kim.params'):
  """Output ANN structure, parameters etc. in the format of the KIM ANN model.

  Parameter
  ---------

  descriptor, object of Descriptor class

  """

  with open(fname,'w') as fout:

    # cutoff
    cutname, rcut = descriptor.get_cutoff()
    maxrcut = max(rcut.values())
    fout.write('# cutoff    rcut\n')
    if mode == 'double':
      fout.write('{}    {:.15g}\n\n'.format(cutname, maxrcut))
    else:
      fout.write('{}    {:.7g}\n\n'.format(cutname, maxrcut))

    # symmetry functions
    # header
    fout.write('#' + '='*80 + '\n')
    fout.write('# symmetry functions\n')
    fout.write('#' + '='*80 + '\n\n')

    desc = descriptor.get_hyperparams()
    # num of descriptors
    num_desc = len(desc)
    fout.write('{}    #number of symmetry funtion types\n\n'.format(num_desc))

    # descriptor values
    fout.write('# sym_function    rows    cols\n')
    for name, values in desc.iteritems():
      if name == 'g1':
        fout.write('g1\n\n')
      else:
        rows = len(values)
        cols = len(values[0])
        fout.write('{}    {}    {}\n'.format(name, rows, cols))
        if name == 'g2':
          for val in values:
            if mode == 'double':
              fout.write('{:.15g} {:.15g}'.format(val[0], val[1]))
            else:
              fout.write('{:.7g} {:.7g}'.format(val[0], val[1]))
            fout.write('    # eta  Rs\n')
          fout.write('\n')
        elif name =='g3':
          for val in values:
            if mode == 'double':
              fout.write('{:.15g}'.format(val[0]))
            else:
              fout.write('{:.7g}'.format(val[0]))
            fout.write('    # kappa\n')
          fout.write('\n')
        elif name =='g4':
          for val in values:
            zeta = val[0]
            lam = val[1]
            eta = val[2]
            if mode == 'double':
              fout.write('{:.15g} {:.15g} {:.15g}'.format(zeta, lam, eta))
            else:
              fout.write('{:.7g} {:.7g} {:.7g}'.format(zeta, lam, eta))
            fout.write('    # zeta  lambda  eta\n')
          fout.write('\n')
        elif name =='g5':
          for val in values:
            zeta = val[0]
            lam = val[1]
            eta = val[2]
            if mode == 'double':
              fout.write('{:.15g} {:.15g} {:.15g}'.format(zeta, lam, eta))
            else:
              fout.write('{:.7g} {:.7g} {:.7g}'.format(zeta, lam, eta))
            fout.write('    # zeta  lambda  eta\n')
          fout.write('\n')


    # ann structure and parameters
    # header
    fout.write('#' + '='*80 + '\n')
    fout.write('# ANN structure and parameters\n')
    fout.write('#\n')
    fout.write('# Note that the ANN assumes each row of the input "X" is '
        'an observation, i.e.\n')
    fout.write('# the layer is implemented as\n')
    fout.write('# Y = activation(XW + b).\n')
    fout.write('# You need to transpose your weight matrix if each column of "X" '
        'is an observation.\n')
    fout.write('#' + '='*80 + '\n\n')

    # number of layers
    num_layers = len(weights)
    fout.write('{}    # number of layers (excluding input layer, including'
        'output layer)\n'.format(num_layers))
    # size of layers
    for b in biases:
      fout.write('{}  '.format(b.size))
    fout.write('  # size of each layer (last must be 1)\n')
    # activation function
    fout.write('{}    # activation function\n\n'.format(activation))

    # weights and biases
    for i, (w, b) in enumerate(zip(weights, biases)):

      # weight
      rows,cols = w.shape
      if i != num_layers-1:
        fout.write('# weight of hidden layer {} (shape({}, {}))\n'.format(i+1,rows,cols))
      else:
        fout.write('# weight of output layer (shape({}, {}))\n'.format(rows,cols))
      for line in w:
        for item in line:
          if mode == 'double':
            fout.write('{:23.15e}'.format(item))
          else:
            fout.write('{:15.7e}'.format(item))
        fout.write('\n')

      # bias
      if i != num_layers-1:
        fout.write('# bias of hidden layer {} (shape({}, {}))\n'.format(i+1,rows,cols))
      else:
        fout.write('# bias of output layer (shape({}, {}))\n'.format(rows,cols))
      for item in b:
        if mode == 'double':
          fout.write('{:23.15e}'.format(item))
        else:
          fout.write('{:15.7e}'.format(item))
      fout.write('\n\n')










