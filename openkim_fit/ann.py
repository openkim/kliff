from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import shutil
import inspect
import tensorflow_op._int_pot_grad
path = os.path.dirname(inspect.getfile(tensorflow_op._int_pot_grad))
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
    biases = tf.Variable(tf.constant(0.01, shape=shape, dtype=dtype))
    #biases = tf.constant(0.1, shape=shape, dtype=dtype)
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


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def convert_to_tfrecords(configs, descriptor, name, directory='/tmp/data',
    do_generate=True, do_normalize=True, do_record=False, dtype=tf.float32):
  """Preprocess the data to generate the generalized coords and its derivatives,
  and store them, together with coords, and label as tfRecord binary.

  Parameter
  ---------

  configs, list of Config objects.

  descriptor, object of Descriptor class.

  name and directory
    The TFRecords file is written to `directory/name.tfrecords'.
    The centering and normalizing mean and standard deviation data is written
    to `directory/mean_and_std_for_kim_ann'.

  do_generate, bool
    Whether to compute the generalized coords and its derivatives or not.

  do_normalize, bool
    Whether to center and normalize the data or not through:
      zeta_new = (zeta - mean(zeta)) / std(zeta)
    Effective only when `do_generate' is set to `True'.

  do_record, bool
    Whether to store the generalized coords `zeta' obtained when computing
    the mean and standard deviation in memory for later use. Effective
    only when `do_normalize' is set to `True'. This flag only affects the
    running speed, but not the results. Enabling it results in faster running
    speed but more memory consumption. For large dataset and limited memory,
    set it to `False'.
  """

  fname = os.path.join(directory, name+'.tfrecords')
  if not os.path.exists(fname):
    do_generate = True

  if do_generate:

    if not os.path.exists(directory):
      os.makedirs(directory)

    print('\nWrining tfRecords of "{}" data as: {}'.format(name, fname))
    writer = tf.python_io.TFRecordWriter(fname)

    # determine the data type
    if dtype == tf.float32:
      np_dtype = np.float32
    elif dtype == tf.float64:
      np_dtype = np.float64

    # compute mean and standard deviation of input data
    if do_normalize:
      print('\nCentering and normalizing the data...')
      # We use the online algorithm proposed by Welford to compute the variance.
      # see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
      # The naive method suffers from numerical instability for large dataset.

      # number of features
      conf = configs[0]
      zeta, dzetadr = descriptor.generate_generalized_coords(conf)
      size = zeta.shape[1]

      # starting Welford's method
      n = 0
      mean = np.zeros(size)
      M2 = np.zeros(size)
      if do_record:
        all_zeta = []
        all_dzetadr = []
      for i,conf in enumerate(configs):
        print('Processing configuration:', i)
        zeta, dzetadr = descriptor.generate_generalized_coords(conf)
        for row in zeta:
          n += 1
          delta =  row - mean
          mean += delta/n
          delta2 = row - mean
          M2 += delta*delta2
        if do_record:
          all_zeta.append(zeta)
          all_dzetadr.append(dzetadr)
      std = np.sqrt(M2/(n-1))
      std_3d = np.atleast_3d(std)

      # write mean and std to file, such that it can be used in the KIM ANN model
      with open(os.path.join(directory, 'mean_and_std_for_kim_ann'), 'w') as fout:
        fout.write('{}  # number of descriptors.\n'.format(len(mean)))
        fout.write('# mean\n')
        for i in mean:
          fout.write('{:24.16e}\n'.format(i))
        fout.write('# standard derivation\n')
        for i in std:
          fout.write('{:24.16e}\n'.format(i))
    else:
      # we write empty info to this file
      with open(os.path.join(directory, 'mean_and_std_for_kim_ann'), 'w') as fout:
        fout.write('False\n')


    # write data to TFRecords
    print('\nGenerating TFRecords data...')
    for i,conf in enumerate(configs):
      print('Processing configuration:', i)
      if do_normalize and do_record:
        zeta = all_zeta[i]
        dzetadr = all_dzetadr[i]
      else:
        zeta, dzetadr = descriptor.generate_generalized_coords(conf)

      num_atoms = conf.get_num_atoms()
      num_descriptors = descriptor.get_num_descriptors()
      # do the actual centering and normalization if needed
      if do_normalize:
        zeta = (zeta - mean) / std
        dzetadr = dzetadr / std_3d
      zeta_raw = zeta.astype(np_dtype).tostring()
      dzetadr_raw = dzetadr.astype(np_dtype).tostring()
      coords_raw = conf.get_coords().astype(np_dtype).tostring()
      energy = np.array(conf.get_energy()).astype(np_dtype).tostring()
      forces_raw = conf.get_forces().astype(np_dtype).tostring()

      example = tf.train.Example(features=tf.train.Features(feature={
        # meta data
        'num_atoms': _int64_feature(num_atoms),
        'num_descriptors': _int64_feature(num_descriptors),
        # input data
        'atomic_coords': _bytes_feature(coords_raw),
        'gen_coords': _bytes_feature(zeta_raw),
        'dgen_datomic_coords': _bytes_feature(dzetadr_raw),
        # labels
        'energy': _bytes_feature(energy),
        'forces': _bytes_feature(forces_raw)
      }))

      writer.write(example.SerializeToString())
    writer.close()

  return fname


def convert_raw_to_tfrecords(configs, descriptor, size_validation=None,
    directory='/tmp/data', do_generate=True, do_normalize=True, do_record=False,
    do_shuffle=False, dtype=tf.float32):
  """Preprocess the data to generate the generalized coords and its derivatives,
  and store them, together with coords, and label as tfRecord binary.

  Parameter
  ---------

  configs, list of Config objects.

  descriptor, object of Descriptor class.

  size_validation, int
    Validation set size.

  directory
    The TFRecords file is written to `directory/.tfrecords'.
    The centering and normalizing mean and standard deviation data is written
    to `directory/mean_and_std_for_kim_ann'.

  do_generate, bool
    Whether to compute the generalized coords and its derivatives or not.

  do_normalize, bool
    Whether to center and normalize the data or not through:
      zeta_new = (zeta - mean(zeta)) / std(zeta)
    Effective only when `do_generate' is set to `True'.

  do_record, bool
    Whether to store the generalized coords `zeta' obtained when computing
    the mean and standard deviation in memory for later use. Effective
    only when `do_normalize' is set to `True'. This flag only affects the
    running speed, but not the results. Enabling it results in faster running
    speed but more memory consumption. For large dataset and limited memory,
    set it to `False'.

  do_shuffle, bool
    Whether to shuffle the dataset before dividing into training set and
    validation set.

  """

  # split dataset into training set and validation set if necessary
  size_dataset = len(configs)
  if size_validation is not None:
    if size_validation > size_dataset:
      raise ValueError('Validation set size "{}" larger than dataset size "{}".'
        .format(size_validation, size_dataset))
    size_train = size_dataset - size_validation
    indices = np.arange(size_dataset)
    if do_shuffle:
      np.random.shuffle(indices)
    tr_indices = indices[:size_train].tolist()
    va_indices = indices[size_train:].tolist()
  else:  # only need training set, not validation set
    tr_indices = range(size_dataset)
    va_indices = []

  tr_name = os.path.join(directory, 'train.tfrecords')
  va_name = os.path.join(directory, 'validation.tfrecords')
  # if cannot find the tfrecords data, we need to generate it
  if not os.path.exists(tr_name):
    do_generate = True


  if do_generate:

    if not os.path.exists(directory):
      os.makedirs(directory)

    # determine the data type
    if dtype == tf.float32:
      np_dtype = np.float32
    elif dtype == tf.float64:
      np_dtype = np.float64

    # compute mean and standard deviation of training set
    if do_normalize:
      print('\nCentering and normalizing the data...')
      # We use the online algorithm proposed by Welford to compute the variance.
      # see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
      # The naive method suffers from numerical instability for large dataset.

      # number of features
      conf = configs[0]
      zeta, dzetadr = descriptor.generate_generalized_coords(conf)
      size = zeta.shape[1]

      # starting Welford's method
      n = 0
      mean = np.zeros(size)
      M2 = np.zeros(size)
      if do_record:
        all_zeta = []
        all_dzetadr = []
      for i,idx in enumerate(tr_indices):
        print('Processing training example:', i)
        conf = configs[idx]
        zeta, dzetadr = descriptor.generate_generalized_coords(conf)
        for row in zeta:
          n += 1
          delta =  row - mean
          mean += delta/n
          delta2 = row - mean
          M2 += delta*delta2
        if do_record:
          all_zeta.append(zeta)
          all_dzetadr.append(dzetadr)
      std = np.sqrt(M2/(n-1))
      std_3d = np.atleast_3d(std)

      # write mean and std to file, such that it can be used in the KIM ANN model
      with open(os.path.join(directory, 'mean_and_std_for_kim_ann'), 'w') as fout:
        fout.write('{}  # number of descriptors.\n'.format(len(mean)))
        fout.write('# mean\n')
        for i in mean:
          fout.write('{:24.16e}\n'.format(i))
        fout.write('# standard derivation\n')
        for i in std:
          fout.write('{:24.16e}\n'.format(i))
    else:
      # we write empty info to this file
      with open(os.path.join(directory, 'mean_and_std_for_kim_ann'), 'w') as fout:
        fout.write('False\n')


    # write training data to TFRecords
    print('\nWrining training tfRecords data as: {}'.format(tr_name))
    tr_writer = tf.python_io.TFRecordWriter(tr_name)
    for i,idx in enumerate(tr_indices):
      print('Processing configuration:', i)
      conf = configs[idx]

      if do_normalize and do_record:
        zeta = all_zeta[i]
        dzetadr = all_dzetadr[i]
      else:
        zeta, dzetadr = descriptor.generate_generalized_coords(conf)

      num_atoms = conf.get_num_atoms()
      num_descriptors = descriptor.get_num_descriptors()
      # do the actual centering and normalization if needed
      if do_normalize:
        zeta = (zeta - mean) / std
        dzetadr = dzetadr / std_3d
      zeta_raw = zeta.astype(np_dtype).tostring()
      dzetadr_raw = dzetadr.astype(np_dtype).tostring()
      coords_raw = conf.get_coords().astype(np_dtype).tostring()
      energy = np.array(conf.get_energy()).astype(np_dtype).tostring()
      forces_raw = conf.get_forces().astype(np_dtype).tostring()

      example = tf.train.Example(features=tf.train.Features(feature={
        # meta data
        'num_atoms': _int64_feature(num_atoms),
        'num_descriptors': _int64_feature(num_descriptors),
        # input data
        'atomic_coords': _bytes_feature(coords_raw),
        'gen_coords': _bytes_feature(zeta_raw),
        'dgen_datomic_coords': _bytes_feature(dzetadr_raw),
        # labels
        'energy': _bytes_feature(energy),
        'forces': _bytes_feature(forces_raw)
      }))

      tr_writer.write(example.SerializeToString())
    tr_writer.close()


    # write validation data to TFRecords
    if va_indices:
      print('\nWrining validation tfRecords data as: {}'.format(va_name))
      va_writer = tf.python_io.TFRecordWriter(va_name)
      for i,idx in enumerate(va_indices):
        print('Processing configuration:', i)
        conf = configs[idx]
        zeta, dzetadr = descriptor.generate_generalized_coords(conf)

        num_atoms = conf.get_num_atoms()
        num_descriptors = descriptor.get_num_descriptors()
        # do the actual centering and normalization if needed
        if do_normalize:
          zeta = (zeta - mean) / std
          dzetadr = dzetadr / std_3d
        zeta_raw = zeta.astype(np_dtype).tostring()
        dzetadr_raw = dzetadr.astype(np_dtype).tostring()
        coords_raw = conf.get_coords().astype(np_dtype).tostring()
        energy = np.array(conf.get_energy()).astype(np_dtype).tostring()
        forces_raw = conf.get_forces().astype(np_dtype).tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
          # meta data
          'num_atoms': _int64_feature(num_atoms),
          'num_descriptors': _int64_feature(num_descriptors),
          # input data
          'atomic_coords': _bytes_feature(coords_raw),
          'gen_coords': _bytes_feature(zeta_raw),
          'dgen_datomic_coords': _bytes_feature(dzetadr_raw),
          # labels
          'energy': _bytes_feature(energy),
          'forces': _bytes_feature(forces_raw)
        }))

        va_writer.write(example.SerializeToString())
      va_writer.close()

  return tr_name, va_name



def _parse_function(example_proto):
  """Transforms a scalar string `example_proto' (corresponding to an atomic
  configuration) into usable data.
  """
  features = {
      # meta data
      'num_atoms': tf.FixedLenFeature((), tf.int64),
      'num_descriptors': tf.FixedLenFeature((), tf.int64),
      # input data
      'atomic_coords': tf.FixedLenFeature((), tf.string),
      'gen_coords': tf.FixedLenFeature((), tf.string),
      'dgen_datomic_coords': tf.FixedLenFeature((), tf.string),
      # labels
      'energy': tf.FixedLenFeature((), tf.string),
      'forces': tf.FixedLenFeature((), tf.string)
      }
  parsed_features = tf.parse_single_example(example_proto, features)

  num_atoms = tf.cast(parsed_features['num_atoms'], tf.int32)
  num_descriptors = tf.cast(parsed_features['num_descriptors'], tf.int32)
  dtype = HACKED_DTYPE  # defined as a global variable in read_from_trrecords

  # shape of tensors
  DIM = 3
  shape1 = tf.stack([num_atoms*DIM])
  shape2 = tf.stack([num_atoms, num_descriptors])
  shape3 = tf.stack([num_atoms, num_descriptors, num_atoms*DIM])

  # input
  atomic_coords = tf.decode_raw(parsed_features['atomic_coords'], dtype)
  atomic_coords = tf.reshape(atomic_coords, shape1)
  gen_coords = tf.decode_raw(parsed_features['gen_coords'], dtype)
  gen_coords = tf.reshape(gen_coords, shape2)
  dgen_datomic_coords = tf.decode_raw(parsed_features['dgen_datomic_coords'], dtype)
  dgen_datomic_coords = tf.reshape(dgen_datomic_coords, shape3)
  # labels
  energy = tf.decode_raw(parsed_features['energy'], dtype)
  forces = tf.decode_raw(parsed_features['forces'], dtype)
  forces = tf.reshape(forces, shape1)

  return atomic_coords, gen_coords, dgen_datomic_coords, energy, forces


def read_from_tfrecords(fname, dtype=tf.float32):
  """Read preprocessed TFRecords data from `fname'.

  Parameter
  ---------

  fname, name of the TFRecords data file.

  Return
  ------

  Instance of tf.contrib.data.
  """

  dataset = tf.contrib.data.TFRecordDataset(fname)
#TODO not sure whether this global variable is efficient
  global HACKED_DTYPE
  HACKED_DTYPE=dtype
  dataset = dataset.map(_parse_function)

  # copy mean_and_std_for_kim_ann to current directoy
  fname2 = os.path.join(os.path.dirname(fname), 'mean_and_std_for_kim_ann')
  shutil.copy(fname2, os.getcwd())

  return dataset


# for feed_dictionary, probably ok
def input_layer_given_data(coords, zeta, dzetadr, layer_name='input_layer'):
  """Reusable code for making an input layer for a configuration."""

  with tf.name_scope(layer_name):
    input, dummy = int_pot(coords=coords, zeta=zeta, dzetadr=dzetadr)
    return input


#  The following three methods. to built data into graphDef, OK and fast for small data set
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

  # write mean and std to file, such that it can be used in the KIM ANN model
  with open('mean_and_std_for_kim_ann', 'w') as fout:
    fout.write('{}  # number of descriptors.\n'.format(len(mean)))
    fout.write('# mean\n')
    for i in mean:
      fout.write('{:24.16e}\n'.format(i))
    fout.write('# standard derivation\n')
    for i in std:
      fout.write('{:24.16e}\n'.format(i))

  # centering and normalization
  all_zeta_processed = []
  all_dzetadr_processed = []
  for zeta in all_zeta:
    all_zeta_processed.append( (zeta - mean) / std )
  for dzetadr in all_dzetadr:
    all_dzetadr_processed.append( dzetadr / np.atleast_3d(std))

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

  # write a file to inform that no centering and normaling is used
  with open('mean_and_std_for_kim_ann', 'w') as fout:
    fout.write('False\n')

  layer_name = os.path.splitext(os.path.basename(config.id))[0]
  # need to return a tensor of coords since we want to take derivaives w.r.t it
  coords = tf.constant(config.get_coords(), dtype)
  zeta,dzetadr = descriptor.generate_generalized_coords(config)

  with tf.name_scope(layer_name):
    input, dummy = int_pot(coords = coords, zeta=tf.constant(zeta, dtype),
        dzetadr=tf.constant(dzetadr, dtype))
    return input, coords


def write_kim_ann(descriptor, weights, biases, activation, dtype=tf.float32,
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
    if dtype == tf.float64:
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
            if dtype == tf.float64:
              fout.write('{:.15g} {:.15g}'.format(val[0], val[1]))
            else:
              fout.write('{:.7g} {:.7g}'.format(val[0], val[1]))
            fout.write('    # eta  Rs\n')
          fout.write('\n')
        elif name =='g3':
          for val in values:
            if dtype == tf.float64:
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
            if dtype == tf.float64:
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
            if dtype == tf.float64:
              fout.write('{:.15g} {:.15g} {:.15g}'.format(zeta, lam, eta))
            else:
              fout.write('{:.7g} {:.7g} {:.7g}'.format(zeta, lam, eta))
            fout.write('    # zeta  lambda  eta\n')
          fout.write('\n')


    # data centering and normalization
    # header
    fout.write('#' + '='*80 + '\n')
    fout.write('# Preprocessing data to center and normalize\n')
    fout.write('#' + '='*80 + '\n')
    # data
    fname = 'mean_and_std_for_kim_ann'
    with open(fname, 'r') as fin:
      lines = fin.readlines()
      if 'False' in lines[0]:
        fout.write('center_and_normalize  False\n')
      else:
        fout.write('center_and_normalize  True\n\n')
        for l in lines:
          fout.write(l)
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
    if activation == tf.nn.sigmoid:
      act_name = 'sigmoid'
    elif activation == tf.nn.tanh:
      act_name = 'tanh'
    elif activation == tf.nn.relu:
      act_name = 'relu'
    elif activation == tf.nn.elu:
      act_name = 'elu'
    else:
      raise ValueError('unsupported activation function for KIM ANN model.')

    fout.write('{}    # activation function\n\n'.format(act_name))

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
          if dtype == tf.float64:
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
        if dtype == tf.float64:
          fout.write('{:23.15e}'.format(item))
        else:
          fout.write('{:15.7e}'.format(item))
      fout.write('\n\n')



