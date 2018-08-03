from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers import variance_scaling_initializer
import numpy as np
import functools
import os
import sys
import shutil
import inspect
import multiprocessing as mp
import tensorflow_op._int_pot_grad
path = os.path.dirname(inspect.getfile(tensorflow_op._int_pot_grad))
int_pot_module = tf.load_op_library(os.path.join(path, 'int_pot_op.so'))
int_pot = int_pot_module.int_pot


# TODO Maybe don't use this, put the content directly in get_loss() of NeuralNetwork
class Input(object):
  """ Input layer.

  Parameters
  ----------

  fit_forces: bool
    whether to fit to forces

  """
  def __init__(self, descriptor, iterator, fit_forces=False):
    self.descriptor = descriptor
    self.iterator = iterator
    self.fit_forces = fit_forces
    self.num_units = self.descriptor.get_number_of_descriptors()
    self.num_atoms_by_species = None
    self.config_weight = None
    self.zeta = None
    self.energy_label = None
    self.atomic_coords = None
    self.deriv_zeta_atomic_coords = None
    self.force_label = None

  def build(self):

    if self.fit_forces:
      (self.config_name,
       self.num_atoms_by_species,
       self.config_weight,
       self.zeta,
       self.energy_label,
       self.atomic_coords,
       self.deriv_zeta_atomic_coords,
       self.force_label) = self.iterator.get_next()

      fingerprint, _ = int_pot(coords=self.atomic_coords, zeta=self.zeta,
          dzetadr=self.deriv_zeta_atomic_coords)
    else:
      (self.config_name,
       self.num_atoms_by_species,
       self.config_weight,
       self.zeta,
       self.energy_label) = self.iterator.get_next()
      fingerprint = self.zeta

    # set the static shape is needed to use high level api like
    # tf.contrib.layers.fully_connected
    fingerprint.set_shape((None, self.num_units))

    return fingerprint


class Dense(object):
  """ Fully connected layer.

  Parameters
  ----------

  num_units:


  """
  def __init__(self,
      num_units,
      activation_fn=tf.nn.tanh,
      normalizer_fn=None,
      normalizer_params=None,
      weights_initializer=xavier_initializer(uniform=False),
      weights_regularizer=None,
      biases_initializer=tf.zeros_initializer(),
      biases_regularizer=None,
      trainable=True):

    self.num_units = num_units
    self.activation_fn = activation_fn
    self.normalizer_fn = normalizer_fn
    self.normalizer_params = normalizer_params
    self.weights_initializer = weights_initializer
    self.weights_regularizer = weights_regularizer
    self.biases_initializer = biases_initializer
    self.biases_regularizer = biases_regularizer
    self.trainable = trainable

  def build(self, inputs, reuse, scope):

    layer = tf.contrib.layers.fully_connected(
        inputs,
        self.num_units,
        self.activation_fn,
        self.normalizer_fn,
        self.normalizer_params,
        self.weights_initializer,
        self.weights_regularizer,
        self.biases_initializer,
        self.biases_regularizer,
        reuse,
        None,
        None,
        self.trainable,
        scope)
    return layer


class Output(Dense):
  """ Final output layer for energy.

  Same as the Dense layer, expect that
  1. `num_units` is set to 1 explicitly, and do not accept the `num_units` arguments;
  2. default `activation_fn` is set to `None` to not use any activation function.
  """
  def __init__(self,
      activation_fn=None,
      normalizer_fn=None,
      normalizer_params=None,
      weights_initializer=xavier_initializer(uniform=False),
      weights_regularizer=None,
      biases_initializer=tf.zeros_initializer(),
      biases_regularizer=None,
      trainable=True):
    super(Output, self).__init__(
        1,
        activation_fn,
        normalizer_fn,
        normalizer_params,
        weights_initializer,
        weights_regularizer,
        biases_initializer,
        biases_regularizer,
        trainable)


class Dropout(object):
  """ Dropout layer.

  Parameters
  ----------

  """

  def __init__(self, keep_ratio):
    self.keep_ratio = keep_ratio

  def build(self, inputs, noise_shape):
    layer = tf.nn.dropout(inputs, self.keep_ratio, noise_shape=noise_shape)
    return layer







class NeuralNetwork(object):
  """ Neural Network class build upon tensorflow.

  Parameters
  ----------

  descriptor: instance of a Descriptor class that transforms atomic environment
    information into the fingerprints that are used as the input for the NN.

  seed: int
    random seed to be used by tf.set_random_seed
  """

  def __init__(self, fingerprints, seed=35):
    self.fingerprints = fingerprints
    self.seed = seed

    self.descriptor = self.fingerprints.get_descriptor()
    self.fit_forces = self.fingerprints.get_fit_forces()
    self.dtype = self.fingerprints.get_dtype()

    self.input_layer = None
    self.layers = []

    tf.set_random_seed(seed)



  def data_iterator(self, tfrecords_path, num_epoches, batch_size, num_parallel_calls=None):
    """Create tf.data from fingerprints in tfrecords format.

    Parameter
    ---------

    num_epoches: int
      number of interation epoches through the dataset

    batch_size: int
      size of the batch of data for each minimization step

    num_parallel_calls: int
      number of cores to be used when doing the map

    """
    self.batch_size = batch_size


    dataset = tf.data.TFRecordDataset(tfrecords_path)
    global HACKED_DTYPE
    HACKED_DTYPE=self.dtype
    if self.fit_forces:
      dataset = dataset.map(_parse_energy_and_force, num_parallel_calls=num_parallel_calls)
    else:
      dataset = dataset.map(_parse_energy, num_parallel_calls=num_parallel_calls)

    dataset = dataset.repeat(num_epoches)
    dataset = dataset.prefetch(batch_size)
    iterator = dataset.make_one_shot_iterator()

    self.iterator = iterator
    return iterator


  def add_layer(self, layer):
    """Add a layer to the network.

    Parameters
    ----------

    layer: instance of layer class
      Options are `Dense`, `Dropout`, and `Output`.

    """
    layer_type = layer.__class__.__name__
    if layer_type == 'Dropout':
      if self.layers and self.layers[-1]['type'] == 'Dropout':
        raise Exception('Error adding layers: two adjacent "Drouput" layers.')
    scope = 'layer' + str(len(self.layers))
    current_layer = {'instance': layer, 'type': layer_type, 'scope': scope}
    self.layers.append(current_layer)


  def check_output_layer(self):
    output_layer = self.layers[-1]
    if output_layer['type'] == 'Dropout':
      raise Exception('The last layer cannot be a "Dropout" layer.')
    else:
      if output_layer['instance'].num_units != 1:
        raise Exception('Number of units in the last layer needs to be 1.')


  def get_loss(self, batch_size, forces_weight=0.1):
    """ Create the nueral network based on the added layers and dataset batch size.

    Parameters
    ----------

    force_weight:
    """

    input_layer = Input(self.descriptor, self.iterator, self.fit_forces)

    subloss_e = []
    subloss_f = []
    for i in range(batch_size):

      # reuse flag
      if i==0:
        reuse = False
      else:
        reuse = True

      # input layer
      output = input_layer.build()
      shape2 = input_layer.num_units

      # other layers
      for j,layer in enumerate(self.layers):
        li = layer['instance']
        lt = layer['type']
        ls = layer['scope']
        if lt == 'Dense' or lt == 'Output':
          output = li.build(output, reuse, ls)
          shape2 = li.num_units
        elif lt == 'Dropout':
          noise_shape = [1, shape2]
          output = li.build(output, noise_shape)
        else:
          raise Exception('Layer "{}" not supported.'.format(str(layer)))


      natoms = tf.cast(tf.reduce_sum(input_layer.num_atoms_by_species), self.dtype)
      energy = tf.reduce_sum(output)

      # energy loss of this configuration
      epart = input_layer.config_weight * tf.square((energy - input_layer.energy_label)/natoms)
      subloss_e.append(epart)

      # forces loss of this configuration
      if self.fit_forces:
        # tf.gradients(y,x) computes grad(sum(y))/grad(x), and returns a LIST of tensors
        forces = - tf.gradients(output, input_layer.atomic_coords)[0]
        fpart = input_layer.config_weight * tf.reduce_sum(tf.square((forces - input_layer.forces_label))) /natoms**2
        subloss_f.append(fpart)

    # loss function
    if self.fit_forces:
      loss_e = tf.reduce_sum(subloss_e)/batch_size
      loss_f = tf.reduce_sum(subloss_f)/batch_size
      loss = loss_e + forces_weight * loss_f
    else:
      loss_e = tf.reduce_sum(subloss_e)/batch_size
      loss = loss_e

    return loss



  def group_layers(self):
    """Divide all the layers into groups.

    The first group is either an empty list or a `Dropout` layer for the input layer.
    For other groups, each group contains one or two layers. The first is a layer
    with parameters (`Dense` or `Output`), and optionally (depending on the input
    layers) the second is a `Dropout` layer.


    Returns
    -------
    groups: list of list of layers
    """

    groups = []
    new_group = []
    for i,layer in enumerate(self.layers):
      if layer['type'] in ['Dense', 'Output']:  # layer with parameters
        groups.append(new_group)
        new_group = []
      new_group.append(layer)
    groups.append(new_group)

    return groups


  def get_parameter_layer_scopes(self):
    """Get the scope of layers with parameters.

    Returns
    -------

    scopes: list of str
    """
    groups = self.group_layers()
    scopes = []
    for i,g in enumerate(groups):
      if i != 0:
        scopes.append(g[0]['scope'])

    return scopes


  def get_activations(self):
    """Get the activations for layers with parameters.

    Returns
    -------

    activations: a list of activation function
    """
    groups = self.group_layers()
    activations = []
    for i,g in enumerate(groups):
      if i != 0:
        activations.append(g[0]['instance'].activation_fn)
    return activations


  def get_keep_ratios(self):
    """Get the keep_ratio of each `Dropout` layer.

    Returns
    -------
    keep_ratios: a list of floats
    """
    groups = self.group_layers()
    keep_ratios = []
    for i,g in enumerate(groups):
      if i == 0:
        if g:
          keep_ratios.append(g[0]['instance'].keep_ratio)
        else:
          keep_ratios.append(1.0)
      elif i == len(groups)-1:
        pass
      else:
        if len(g) == 2:
          keep_ratios.append(g[1]['instance'].keep_ratio)
        else:
          keep_ratios.append(1.0)

    return keep_ratios



  def write_kim_ann(self, sess, fname='ann_kim.params'):

    """Output ANN structure, parameters etc. in the format of the KIM ANN model.

    Parameter
    ---------

    sess: a tf session


    """

    fingerprints = self.fingerprints
    descriptor = self.descriptor
    weights, biases = self.get_weights_and_biases(self.get_parameter_layer_scopes())
    activations = self.get_activations()
    keep_ratios = self.get_keep_ratios()
    dtype = self.dtype

    weights, biases = sess.run([weights, biases])


    with open(fname,'w') as fout:
      fout.write('#' + '='*80 + '\n')
      fout.write('# KIM ANN potential parameters, generated by `kliff` fitting program.\n')
      fout.write('#' + '='*80 + '\n\n')

      # cutoff
      cutname, rcut, rcut_samelayer = descriptor.get_cutoff()
      if rcut_samelayer is None:
        rcut_samelayer  = rcut
      maxrcut = max(rcut.values())
      maxrcut_samelayer = max(rcut_samelayer.values())

      fout.write('# cutoff    rcut\n')
      if dtype == tf.float64:
        fout.write('{}  {:.15g}  {:.15g}\n\n'.format(cutname, maxrcut, maxrcut_samelayer))
      else:
        fout.write('{}  {:.7g}  {:.7g}\n\n'.format(cutname, maxrcut, maxrcut_samelayer))

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
      for name, values in desc.items():
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


      # mean and stdev
      mean = fingerprints.get_mean()
      stdev = fingerprints.get_stdev()
      if mean is None and stdev is None:
        fout.write('center_and_normalize  False\n')
      else:
        fout.write('center_and_normalize  True\n\n')

        fout.write('# mean\n')
        for i in mean:
          if dtype == tf.float64:
            fout.write('{:23.15e}\n'.format(i))
          else:
            fout.write('{:15.7e}\n'.format(i))
        fout.write('\n# standard deviation\n')
        for i in stdev:
          if dtype == tf.float64:
            fout.write('{:23.15e}\n'.format(i))
          else:
            fout.write('{:15.7e}\n'.format(i))
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
      fout.write('{}    # number of layers (excluding input layer, including '
          'output layer)\n'.format(num_layers))

      # size of layers
      for b in biases:
        fout.write('{}  '.format(b.size))
      fout.write('  # size of each layer (last must be 1)\n')

      # activation function
      # TODO enable writing different activations for each layer
      activation = activations[0]
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

      fout.write('{}    # activation function\n'.format(act_name))

      # keep probability
      for i in keep_ratios:
        fout.write('{:.15g}  '.format(i))
      fout.write('  # keep probability of input for each layer\n\n')

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



  @staticmethod
  def get_weights_and_biases(layer_names):
    """Get the weights and biases of all layers.

      If variable_scope is used, is should be prepended to the name.
      The element order matters of layer_names, since the returned weights and
      biases have the same order as layer_names.

    Parameters
    ----------

    layer_names: list of str
      The names of the layers (e.g. the scope of fully_connected()).

    Return
    ------
      weights: list of tensors
      biases: lsit of tensors

    """
    weight_names = [lm.rstrip('/')+'/weights' for lm in layer_names]
    bias_names = [lm.rstrip('/')+'/biases' for lm in layer_names]
    weights = []
    biases = []

    all_vars = tf.global_variables()

    for name in weight_names:
      for v in all_vars:
        if v.name.startswith(name):
          weights.append(v)
          break
    for name in bias_names:
      for v in all_vars:
        if v.name.startswith(name):
          biases.append(v)
          break

    if len(weight_names) != len(weights):
      name = weight_names[len(weights)+1]
      raise KeyError('{} cannot be found in global variables', name)
    if len(bias_names) != len(biases):
      name = bias_names[len(biases)+1]
      raise KeyError('{} cannot be found in global variables', name)

    return weights, biases


def _parse_energy(example_proto):
  """Transforms a scalar string `example_proto' (corresponding to an atomic
  configuration) into usable data.
  """
  features = {
      # meta data
      'num_descriptors': tf.FixedLenFeature((), tf.int64),
      'num_species': tf.FixedLenFeature((), tf.int64),
      # input data
      'name': tf.FixedLenFeature((), tf.string),
      'num_atoms_by_species': tf.FixedLenFeature((), tf.string),
      'weight': tf.FixedLenFeature((), tf.string),
      'gen_coords': tf.FixedLenFeature((), tf.string),
      # labels
      'energy': tf.FixedLenFeature((), tf.string),
      }
  parsed_features = tf.parse_single_example(example_proto, features)

  # meta
  num_descriptors = tf.cast(parsed_features['num_descriptors'], tf.int64)
  num_species = tf.cast(parsed_features['num_species'], tf.int64)

  dtype = HACKED_DTYPE  # defined as a global variable in read_from_trrecords

  ## input
  name = parsed_features['name']

  num_atoms_by_species = tf.decode_raw(parsed_features['num_atoms_by_species'], tf.int64)

  shape = tf.cast([num_species], tf.int64)  # tf.cast is necessary, otherwise tf will report error
                                            # or else, can use num_species as tf.int32
  num_atoms_by_species = tf.reshape(num_atoms_by_species, shape)

  weight = tf.decode_raw(parsed_features['weight'], dtype)[0]

  num_atoms = tf.reduce_sum(num_atoms_by_species)
  shape = tf.cast([num_atoms, num_descriptors], tf.int64)
  gen_coords = tf.decode_raw(parsed_features['gen_coords'], dtype)
  gen_coords = tf.reshape(gen_coords, shape)

  # labels
  energy = tf.decode_raw(parsed_features['energy'], dtype)[0]

  return  name, num_atoms_by_species, weight, gen_coords, energy


def _parse_energy_and_force(example_proto):
  """Transforms a scalar string `example_proto' (corresponding to an atomic
  configuration) into usable data.
  """
  features = {
      # meta data
      'num_descriptors': tf.FixedLenFeature((), tf.int64),
      'num_species': tf.FixedLenFeature((), tf.int64),

      # input data
      'name': tf.FixedLenFeature((), tf.string),
      'num_atoms_by_species': tf.FixedLenFeature((), tf.string),
      'weight': tf.FixedLenFeature((), tf.string),
      'atomic_coords': tf.FixedLenFeature((), tf.string),
      'gen_coords': tf.FixedLenFeature((), tf.string),
      'dgen_datomic_coords': tf.FixedLenFeature((), tf.string),

      # labels
      'energy': tf.FixedLenFeature((), tf.string),
      'forces': tf.FixedLenFeature((), tf.string)
      }
  parsed_features = tf.parse_single_example(example_proto, features)

  # meta
  num_descriptors = tf.cast(parsed_features['num_descriptors'], tf.int64)
  num_species = tf.cast(parsed_features['num_species'], tf.int64)

  dtype = HACKED_DTYPE  # defined as a global variable in read_from_trrecords

  # input
  name = parsed_features['name']
  num_atoms_by_species = tf.decode_raw(parsed_features['num_atoms_by_species'], tf.int64)
  shape = tf.cast([num_species], tf.int64)  # tf.cast is necessary, otherwise tf will report error
  num_atoms_by_species = tf.reshape(num_atoms_by_species, shape)
  weight = tf.decode_raw(parsed_features['weight'], dtype)[0]

  # shapes
  num_atoms = tf.reduce_sum(num_atoms_by_species)
  DIM = 3
  shape1 = tf.cast([num_atoms*DIM], tf.int64)
  shape2 = tf.cast([num_atoms, num_descriptors], tf.int64)
  shape3 = tf.cast([num_atoms, num_descriptors, num_atoms*DIM], tf.int64)

  atomic_coords = tf.decode_raw(parsed_features['atomic_coords'], dtype)
  atomic_coords = tf.reshape(atomic_coords, shape1)
  gen_coords = tf.decode_raw(parsed_features['gen_coords'], dtype)
  gen_coords = tf.reshape(gen_coords, shape2)
  dgen_datomic_coords = tf.decode_raw(parsed_features['dgen_datomic_coords'], dtype)
  dgen_datomic_coords = tf.reshape(dgen_datomic_coords, shape3)

  # labels
  energy = tf.decode_raw(parsed_features['energy'], dtype)[0]
  forces = tf.decode_raw(parsed_features['forces'], dtype)
  forces = tf.reshape(forces, shape1)

  return  name, num_atoms_by_species, weight, gen_coords, energy, atomic_coords, dgen_datomic_coords, forces




class ANNCalculator(object):
  """ A neural network calculator.

  Parameters
  ----------

  model: instance of `NeuralNetwork` class
  """

  def __init__(self, model, num_epoches, batch_size):
    self.model = model
    self.num_epoches = num_epoches
    self.batch_size = batch_size

    self.configs = None
    self.use_energy = None
    self.use_forces = None


  def create(self, configs, use_energy=True, use_forces=True):
    self.configs = configs
    self.use_energy = use_energy
    self.use_forces = use_forces

    self.model.check_output_layer()
    self.model.fingerprints.generate_train_tfrecords(self.configs, nprocs=1)
    tfrecords_path = self.model.fingerprints.get_train_tfrecords_path()
    self.model.data_iterator(tfrecords_path, self.num_epoches, self.batch_size, num_parallel_calls=1)

  def get_loss(self):
    return self.model.get_loss(self.batch_size)
