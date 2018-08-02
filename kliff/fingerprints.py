from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import sys
import multiprocessing as mp
from . import parallel


class Fingerprints(object):
  """ Preprocess the training set to generate fingerprints using a descriptor.

  Parameters
  ----------

  descriptor: instance of a Descriptor class that transforms atomic environment
    information into the fingerprints that are used as the input for the NN.

   normalize: bool
     Whether or not to center and normalize the fingerprints through:
       zeta_new = (zeta - mean(zeta)) / stdev(zeta)

  fit_forces: bool
      Whether or not to fit to forces.

  dtype: {tf.float32, tf.float64}
    Data type.
  """

  def __init__(self, descriptor, normalize=True, fit_forces=False, dtype=tf.float32):
    self.descriptor = descriptor
    self.normalize = normalize
    self.fit_forces = fit_forces
    self.dtype = dtype

    self.mean = None
    self.stdev = None


  def generate_train_tfrecords(self, configs, fname='fingerprints/train.tfrecords',
      welford=False, nprocs=mp.cpu_count()):
    """Convert training set to fingerprints.

    Parameters
    ----------

    configs: list of Configuration instances

    fname: str
      path for the generated tfrecords file

    welford: bool
      Wehther or not to use Welford's online method to compute mean and standard (if
      normalize == True).

    nprocs: int
      Number of processes to be used.
    """

    all_zeta = None
    all_dzetadr = None

    if self.normalize:
      if welford:
        self.mean,self.stdev = self.welford_mean_and_stdev(configs, self.descriptor)
      else:
        self.mean,self.stdev,all_zeta,all_dzetadr = self.numpy_mean_and_stdev(configs,
            self.descriptor, self.fit_forces, nprocs)
    else:
      self.mean = None
      self.stdev = None

    self._generate_tfrecords(configs, fname, all_zeta, all_dzetadr, nprocs=nprocs)


  def generate_test_tfrecords(self, configs, fname='fingerprints/train.tfrecords',
      nprocs=mp.cpu_count()):
    """Convert test set to fingerprints.

    This should be called after `generate_train_tfrecords()`, because it needs to
    use the mean and stdandard deviation of the fingerprints (when `normalize` is
    required) generated there.

    Parameters
    ----------

    configs: list of Configuration instances

    fname: str
      path for the generated tfrecords file

    nprocs: int
      Number of processes to be used.
    """

    if self.normalize is None:
      raise Exception('"generate_train_trrecords()" should be called before this.')

    self._generate_tfrecords(configs, fname, None, None, nprocs=nprocs)

  def get_mean(self):
    return self.mean.copy()

  def get_stdev(self):
    return self.stdev.copy()

  def write_mean_and_stdev(self, fname=None):
    mean = self.mean
    stdev = self.stdev

    if fname is None:
      fname = 'fingerprints/fingerprints_mean_and_stdev'

    with open(fname, 'w') as fout:
      if mean is not None and stdev is not None:
        fout.write('{}  # number of descriptors.\n'.format(len(mean)))
        fout.write('# mean\n')
        for i in mean:
          fout.write('{:24.16e}\n'.format(i))
        fout.write('# standard derivation\n')
        for i in stdev:
          fout.write('{:24.16e}\n'.format(i))

      else:
        fout.write('False\n')



  def _generate_tfrecords(self, configs, fname, all_zeta, all_dzetadr, structure='bulk',
      nprocs=mp.cpu_count()):

    print('\nWriting tfRecords data to: {}'.format(fname))

    dirname = os.path.dirname(fname)
    if not os.path.exists(dirname):
      os.makedirs(dirname)

    if self.dtype == tf.float32:
      np_dtype = np.float32
    elif self.dtype == tf.float64:
      np_dtype = np.float64
    else:
      raise ValueError('Unsupported data type "{}".'.format(dtype))

    writer = tf.python_io.TFRecordWriter(fname)
    for i,conf in enumerate(configs):
      if i%100 == 0:
        print('Processing configuration:', i)
        sys.stdout.flush()
      if all_zeta is not None:
        zeta = all_zeta[i]
      if all_dzetadr is not None and self.fit_forces:
        dzetadr = all_dzetadr[i]
      if self.fit_forces:
        self._write_tfrecord_energy_and_force(writer,conf,self.descriptor,
            self.normalize,np_dtype,self.mean,self.stdev,zeta,dzetadr,structure)
      else:
        self._write_tfrecord_energy(writer,conf,self.descriptor,
            self.normalize,np_dtype,self.mean,self.stdev,zeta,structure)
    writer.close()

    print('Processing {} configurations finished.\n'.format(len(configs)))


  @staticmethod
  def _write_tfrecord_energy(writer, conf, descriptor, normalize, np_dtype,
      mean, stdev, zeta=None, structure='bulk'):
    """ Write data to tfrecord format."""

    # descriptor features
    num_descriptors = descriptor.get_number_of_descriptors()
    if zeta is None:
      zeta, _ = descriptor.generate_generalized_coords(conf, fit_forces=False,
          structure=structure)

    # do centering and normalization if needed
    if normalize:
      zeta = (zeta - mean) / stdev
    zeta = np.asarray(zeta, np_dtype).tostring()

    # configuration features
    name = conf.get_identifier()
    d = conf.num_atoms_by_species
    num_atoms_by_species = [d[k] for k in d]
    num_species = len(num_atoms_by_species)
    num_atoms_by_species = np.asarray(num_atoms_by_species, np.int64).tostring()
    weight = np.asarray(conf.get_weight(), np_dtype).tostring()
    energy = np.asarray(conf.get_energy(), np_dtype).tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
      # meta data
      'num_descriptors': Fingerprints._int64_feature(num_descriptors),
      'num_species': Fingerprints._int64_feature(num_species),
      # input data
      'name': Fingerprints._bytes_feature(name),
      'num_atoms_by_species': Fingerprints._bytes_feature(num_atoms_by_species),
      'weight': Fingerprints._bytes_feature(weight),
      'gen_coords': Fingerprints._bytes_feature(zeta),
      # labels
      'energy': Fingerprints._bytes_feature(energy),
    }))

    writer.write(example.SerializeToString())


  @staticmethod
  def _write_tfrecord_energy_and_force(writer, conf, descriptor, normalize, np_dtype,
      mean, stdev, zeta=None, dzetadr=None, structure='bulk'):
    """ Write data to tfrecord format."""

    # descriptor features
    num_descriptors = descriptor.get_number_of_descriptors()
    if zeta is None or dzetadr is None:
      zeta, dzetadr = descriptor.generate_generalized_coords(conf, fit_forces=True,
          structure=structure)

    # do centering and normalization if needed
    if normalize:
      stdev_3d = np.atleast_3d(stdev)
      zeta = (zeta - mean) / stdev
      dzetadr = dzetadr / stdev_3d
    zeta = np.asarray(zeta, np_dtype).tostring()
    dzetadr = np.asarray(dzetadr, np_dtype).tostring()

    # configuration features
    name = conf.get_idtifier()
    d = conf.num_atoms_by_species
    num_atoms_by_species = [d[k] for k in d]
    num_species = len(num_atoms_by_species)
    num_atoms_by_species = np.asarray(num_atoms_by_species, np.int64).tostring()
    coords = np.asarray(conf.get_coords(), np_dtype).tostring()
    weight = np.asarray(conf.get_weight(), np_dtype).tostring()
    energy = np.asarray(conf.get_energy(), np_dtype).tostring()
    forces = np.asarray(conf.get_forces(), np_dtype).tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
      # meta data
      'num_descriptors': Fingerprints._int64_feature(num_descriptors),
      'num_species': Fingerprints._int64_feature(num_species),
      # input data
      'name': Fingerprints._bytes_feature(name),
      'num_atoms_by_species': Fingerprints._bytes_feature(num_atoms_by_species),
      'weight': Fingerprints._bytes_feature(weight),
      'atomic_coords': Fingerprints._bytes_feature(coords),
      'gen_coords': Fingerprints._bytes_feature(zeta),
      'dgen_datomic_coords': Fingerprints._bytes_feature(dzetadr),
      # labels
      'energy': Fingerprints._bytes_feature(energy),
      'forces': Fingerprints._bytes_feature(forces)
    }))

    writer.write(example.SerializeToString())


  @staticmethod
  def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  @staticmethod
  def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

  @staticmethod
  def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


  @staticmethod
  def welford_mean_and_stdev(configs, descriptor):
    """Compute the mean and standard deviation of fingerprints.

    This running mean and standard method proposed by Welford is memory-efficient.
    Besides, it outperforms the naive method from suffering numerical instability
    for large dataset.

    see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """
    structure = 'bulk'

    # number of features
    conf = configs[0]
    zeta,_ = descriptor.generate_generalized_coords(conf, fit_forces=False,
        structure=structure)
    size = zeta.shape[1]

    # starting Welford's method
    n = 0
    mean = np.zeros(size)
    M2 = np.zeros(size)
    for i,conf in enumerate(configs):
      zeta,_ = descriptor.generate_generalized_coords(conf, fit_forces=False,
          structure=structure)
      for row in zeta:
        n += 1
        delta =  row - mean
        mean += delta/n
        delta2 = row - mean
        M2 += delta*delta2
      if i%100 == 0:
        print('Processing training example:', i)
        sys.stdout.flush()
    stdev = np.sqrt(M2/(n-1))
    print('Processing {} configurations finished.\n'.format(i+1))

    return mean, stdev


  @staticmethod
  def numpy_mean_and_stdev(configs, descriptor, fit_forces=False, nprocs=mp.cpu_count()):
    """Compute the mean and standard deviation of fingerprints."""

    structure = 'bulk'
    try:
      rslt = parallel.parmap(descriptor.generate_generalized_coords, configs,
          nprocs, fit_forces, structure)
      all_zeta = [pair[0] for pair in rslt]
      all_dzetadr = [pair[1] for pair in rslt]
      stacked = np.concatenate(all_zeta)
      mean = np.mean(stacked, axis=0)
      stdev = np.std(stacked, axis=0)
    except MemoryError:
      raise MemoryError('Out of memory while computing mean and standard deviation. '
          'Try the memory-efficient `welford` method instead.')

    #TODO delete debug
#  with open('debug_descriptor_after_normalization.txt', 'w') as fout:
#    for zeta,conf in zip(all_zeta, configs):
#      zeta_norm = (zeta - mean) / stdev
#      fout.write('\n\n'+'#'+'='*80+'\n')
#      fout.write('# configure name: {}\n'.format(conf.id))
#      fout.write('# atom id    descriptor values ...\n\n')
#      for i,line in enumerate(zeta_norm):
#        fout.write('{}    '.format(i))
#        for j in line:
#          fout.write('{:.15g} '.format(j))
#        fout.write('\n')

    return mean, stdev, all_zeta, all_dzetadr

