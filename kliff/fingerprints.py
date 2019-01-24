import os
import sys
import pickle
import numpy as np
import multiprocessing as mp
import tensorflow as tf
from kliff import parallel


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

    def __init__(self, descriptor, normalize=True, fit_forces=False, dtype=None):

        self.descriptor = descriptor
        self.normalize = normalize
        self.fit_forces = fit_forces
        self.dtype = dtype if dtype is not None else tf.float32

        self.train_tfrecords = None
        self.test_tfrecords = None
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
        self.train_tfrecords = fname

        all_zeta = None
        all_dzetadr = None

        if self.normalize:
            if welford:
                self.mean, self.stdev = self.welford_mean_and_stdev(
                    configs, self.descriptor)
            else:
                self.mean, self.stdev, all_zeta, all_dzetadr = self.numpy_mean_and_stdev(
                    configs, self.descriptor, self.fit_forces, nprocs)
        else:
            self.mean = None
            self.stdev = None

        self._generate_tfrecords(configs, fname, all_zeta, all_dzetadr, nprocs=nprocs)
        self.dump_pickle(configs, fname, all_zeta, all_dzetadr, nprocs=nprocs)

        mean_std_name = os.path.join(os.path.dirname(
            fname), 'fingerprints_mean_and_stdev.txt')
        self.write_mean_and_stdev(mean_std_name)

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
            raise Exception(
                '"generate_train_trrecords()" should be called before this.')

        self.test_tfrecords = fname
        self._generate_tfrecords(configs, fname, None, None, nprocs=nprocs)

    def get_train_tfrecords_path(self):
        return self.train_tfrecords

    def get_test_tfrecords_path(self):
        return self.test_tfrecords

    def get_mean(self):
        return self.mean.copy()

    def get_stdev(self):
        return self.stdev.copy()

    def get_descriptor(self):
        return self.descriptor

    def get_fit_forces(self):
        return self.fit_forces

    def get_dtype(self):
        return self.dtype

    def write_mean_and_stdev(self, fname='fingerprints/mean_and_stdev.txt'):
        mean = self.mean
        stdev = self.stdev
        with open(fname, 'w') as fout:
            if mean is not None and stdev is not None:
                fout.write('{}  # number of descriptors.\n'.format(len(mean)))
                fout.write('# mean\n')
                for i in mean:
                    fout.write('{:24.16e}\n'.format(i))
                fout.write('\n# standard derivation\n')
                for i in stdev:
                    fout.write('{:24.16e}\n'.format(i))
            else:
                fout.write('False\n')

    def _generate_tfrecords(self, configs, fname, all_zeta, all_dzetadr,
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
        for i, conf in enumerate(configs):
            if i % 100 == 0:
                print('Processing configuration:', i)
                sys.stdout.flush()
            if all_zeta is not None:
                zeta = all_zeta[i]
            else:
                zeta = None
            if all_dzetadr is not None and self.fit_forces:
                dzetadr = all_dzetadr[i]
            else:
                dzetadr = None
            if self.fit_forces:
                self._write_tfrecord_energy_and_force(
                    writer, conf, self.descriptor, self.normalize, np_dtype,
                    self.mean, self.stdev, zeta, dzetadr)
            else:
                self._write_tfrecord_energy(
                    writer, conf, self.descriptor, self.normalize, np_dtype,
                    self.mean, self.stdev, zeta)
        writer.close()

    @staticmethod
    def _write_tfrecord_energy(writer, conf, descriptor, normalize, np_dtype,
                               mean, stdev, zeta=None):
        """ Write data to tfrecord format."""

        # descriptor features
        num_descriptors = descriptor.get_number_of_descriptors()
        if zeta is None:
            zeta, _ = descriptor.generate_generalized_coords(
                conf, fit_forces=False)

        # do centering and normalzation if needed
        if normalize:
            zeta = (zeta - mean) / stdev
        zeta = np.asarray(zeta, np_dtype).tostring()

        # configuration features
        name = conf.get_identifier()
        d = conf.get_number_of_atoms_by_species()
        num_atoms_by_species = [d[k] for k in d]
        num_species = len(num_atoms_by_species)
        num_atoms_by_species = np.asarray(
            num_atoms_by_species, np.int64).tostring()
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
                                         mean, stdev, zeta=None, dzetadr=None):
        """ Write data to tfrecord format."""

        # descriptor features
        num_descriptors = descriptor.get_number_of_descriptors()
        if zeta is None or dzetadr is None:
            zeta, dzetadr = descriptor.generate_generalized_coords(
                conf, fit_forces=True)

        # do centering and normalization if needed
        if normalize:
            stdev_3d = np.atleast_3d(stdev)
            zeta = (zeta - mean) / stdev
            dzetadr = dzetadr / stdev_3d
        zeta = np.asarray(zeta, np_dtype).tostring()
        dzetadr = np.asarray(dzetadr, np_dtype).tostring()

        # configuration features
        name = conf.get_identifier().encode()
        d = conf.get_number_of_atoms_by_species()
        num_atoms_by_species = [d[k] for k in d]
        num_species = len(num_atoms_by_species)
        num_atoms_by_species = np.asarray(
            num_atoms_by_species, np.int64).tostring()
        coords = np.asarray(conf.get_coordinates(), np_dtype).tostring()
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

    def dump_pickle(self, configs, fname, all_zeta, all_dzetadr,
                    nprocs=mp.cpu_count()):

        fname = 'fingerprints/train.pkl'
        if os.path.exists(fname):
            print('Found existing image data: {}'.format(fname))
            return

        print('Pickle images to: {}'.format(fname))

        dirname = os.path.dirname(fname)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        if self.dtype == tf.float32:
            np_dtype = np.float32
        elif self.dtype == tf.float64:
            np_dtype = np.float64
        else:
            raise ValueError('Unsupported data type "{}".'.format(dtype))

        writer = open(fname, 'ab')
        for i, conf in enumerate(configs):
            if i % 100 == 0:
                print('Processing configuration:', i)
                sys.stdout.flush()
            if all_zeta is not None:
                zeta = all_zeta[i]
            else:
                zeta = None
            if all_dzetadr is not None and self.fit_forces:
                dzetadr = all_dzetadr[i]
            else:
                dzetadr = None
            if self.fit_forces:
                self.pickle_energy_and_force(
                    writer, conf, self.descriptor, self.normalize, np_dtype,
                    self.mean, self.stdev, zeta, dzetadr)
            else:
                self.pickle_energy(
                    writer, conf, self.descriptor, self.normalize, np_dtype,
                    self.mean, self.stdev, zeta)
        writer.close()

        print('Processing {} configurations finished.\n'.format(len(configs)))

    @staticmethod
    def pickle_energy(writer, conf, descriptor, normalize, np_dtype, mean, stdev, zeta=None):
        """ Write data to tfrecord format."""

        # descriptor features
        num_descriptors = descriptor.get_number_of_descriptors()
        if zeta is None:
            zeta, _ = descriptor.generate_generalized_coords(conf, fit_forces=False)

        # do centering and normalzation if needed
        if normalize:
            zeta = (zeta - mean) / stdev
        zeta = np.asarray(zeta, np_dtype)

        # configuration features
        name = conf.get_identifier()
        d = conf.get_number_of_atoms_by_species()
        num_atoms_by_species = [d[k] for k in d]
        num_species = len(num_atoms_by_species)
        weight = np.asarray(conf.get_weight(), np_dtype)
        energy = np.asarray(conf.get_energy(), np_dtype)

        example = {
            # meta data
            'num_descriptors': num_descriptors,
            'num_species': num_species,
            # input data
            'name': name,
            'num_atoms_by_species': num_atoms_by_species,
            'weight': weight,
            'gen_coords': zeta,
            # labels
            'energy': energy}

        pickle.dump(example, writer)

    @staticmethod
    def pickle_energy_and_force(writer, conf, descriptor, normalize, np_dtype,
                                mean, stdev, zeta=None, dzetadr=None):
        """ Write data to tfrecord format."""

        # descriptor features
        num_descriptors = descriptor.get_number_of_descriptors()
        if zeta is None or dzetadr is None:
            zeta, dzetadr = descriptor.generate_generalized_coords(
                conf, fit_forces=True)

        # do centering and normalization if needed
        if normalize:
            stdev_3d = np.atleast_3d(stdev)
            zeta = (zeta - mean) / stdev
            dzetadr = dzetadr / stdev_3d
        zeta = np.asarray(zeta, np_dtype)
        dzetadr = np.asarray(dzetadr, np_dtype)

        # configuration features
        name = conf.get_identifier().encode()
        d = conf.get_number_of_atoms_by_species()
        num_atoms_by_species = [d[k] for k in d]
        num_species = len(num_atoms_by_species)
        coords = np.asarray(conf.get_coordinates(), np_dtype)
        weight = np.asarray(conf.get_weight(), np_dtype)
        energy = np.asarray(conf.get_energy(), np_dtype)
        forces = np.asarray(conf.get_forces(), np_dtype)

        example = {
            # meta data
            'num_descriptors': num_descriptors,
            'num_species': num_species,
            # input data
            'name': name,
            'num_atoms_by_species': num_atoms_by_species,
            'weight': weight,
            'atomic_coords': coords,
            'gen_coords': zeta,
            'dgen_datomic_coords': dzetadr,
            # labels
            'energy': energy,
            'forces': forces}

        pickle.dump(example, writer)

    @staticmethod
    def welford_mean_and_stdev(configs, descriptor):
        """Compute the mean and standard deviation of fingerprints.

        This running mean and standard method proposed by Welford is memory-efficient.
        Besides, it outperforms the naive method from suffering numerical instability
        for large dataset.

        see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        """

        # number of features
        conf = configs[0]
        zeta, _ = descriptor.generate_generalized_coords(
            conf, fit_forces=False)
        size = zeta.shape[1]

        # starting Welford's method
        n = 0
        mean = np.zeros(size)
        M2 = np.zeros(size)
        for i, conf in enumerate(configs):
            zeta, _ = descriptor.generate_generalized_coords(
                conf, fit_forces=False)
            for row in zeta:
                n += 1
                delta = row - mean
                mean += delta/n
                delta2 = row - mean
                M2 += delta*delta2
            if i % 100 == 0:
                sys.stdout.flush()
        stdev = np.sqrt(M2/(n-1))

        return mean, stdev

    @staticmethod
    def numpy_mean_and_stdev(configs, descriptor, fit_forces=False, nprocs=mp.cpu_count()):
        """Compute the mean and standard deviation of fingerprints."""

        try:
            rslt = parallel.parmap(descriptor.generate_generalized_coords,
                                   configs, nprocs, fit_forces)
            all_zeta = [pair[0] for pair in rslt]
            all_dzetadr = [pair[1] for pair in rslt]
            stacked = np.concatenate(all_zeta)
            mean = np.mean(stacked, axis=0)
            stdev = np.std(stacked, axis=0)
        except MemoryError:
            raise MemoryError('Out of memory while computing mean and standard deviation. '
                              'Try the memory-efficient `welford` method instead.')

        # TODO delete debug
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


def read_tfrecords(fnames, fit_forces=False, num_parallel_calls=None, dtype=tf.float32):
    """Read preprocessed TFRecord data from `fname'.

    Parameter
    ---------

    fnames, str or list of str
      names of the TFRecords data file.

    fit_forces, bool
      wheter to fit to forces.

    Return
    ------

    Instance of tf.data.
    """

    dataset = tf.data.TFRecordDataset(fnames)
    global HACKED_DTYPE
    HACKED_DTYPE = dtype
    if fit_forces:
        dataset = dataset.map(_parse_energy_and_force,
                              num_parallel_calls=num_parallel_calls)
    else:
        dataset = dataset.map(
            _parse_energy, num_parallel_calls=num_parallel_calls)
    return dataset


def load_pickle(fnames, fit_forces=False, num_parallel_calls=None, dtype=tf.float32):
    """Read preprocessed TFRecord data from `fname'.

    Parameter
    ---------

    fnames, str or list of str
      names of the TFRecords data file.

    fit_forces, bool
      wheter to fit to forces.

    Return
    ------

    Instance of tf.data.
    """
    data = []
    with open(fnames, 'rb') as f:
        try:
            while True:
                x = pickle.load(f)
                data.append(x)
        except EOFError:
            pass
    return data


def tfrecords_to_text(tfrecords_path, text_path, fit_forces=False, gradient=False,
                      nprocs=mp.cpu_count(), dtype=tf.float32):
    """Convert tfrecords data file of generalized coords (gc) to text file.

    Parameters
    ----------

    tfrecords_path: str
      File name of the tfrecords data file.

    text_path: str
      File name of the outout text data file.

    fit_forces: bool
      Is forces included in the tfrecords data.

    gradient: bool
      Wheter to write out gradient of fingerprints w.r.t. atomic coords.
    """

    if gradient and not fit_forces:
        raise Exception("Set `fit_forces` to `True` to use `gradient`.")

    dataset = read_tfrecords(tfrecords_path, fit_forces, nprocs, dtype)
    iterator = dataset.make_one_shot_iterator()

    if fit_forces:
        name, num_atoms_by_species, weight, gen_coords, energy_label, atomic_coords, \
            dgen_datomic_coords, forces_label = iterator.get_next()
    else:
        name, num_atoms_by_species, weight, gen_coords, energy_label = iterator.get_next()

    with open(text_path, 'w') as fout:
        fout.write('# Generalized coordinates for all configurations.\n')
        nconf = 0

        with tf.Session() as sess:
            while True:
                try:
                    if gradient:
                        nm, gc, grad_gc = sess.run(
                            [name, gen_coords, dgen_datomic_coords])
                    else:
                        nm, gc = sess.run([name, gen_coords])

                    fout.write('\n\n#'+'='*80+'\n')
                    fout.write('# configuration: {}\n'.format(nm))
                    fout.write('# atom id    descriptor values ...\n\n')
                    for i, line in enumerate(gc):
                        fout.write('{}    '.format(i))
                        for j in line:
                            fout.write('{:.15g} '.format(j))
                        fout.write('\n')

                    if gradient:
                        fout.write('\n')
                        fout.write('#'+'='*40+'\n')
                        fout.write(
                            '# atom id\n# gc id    atom 3i+0, 3i+1, 3i+2 ...\n\n')
                        for at, page in enumerate(grad_gc):
                            fout.write('{}\n'.format(at))
                            for i, line in enumerate(page):
                                fout.write('{:4d} '.format(i))
                                for j in line:
                                    fout.write('{:23.15e} '.format(j))
                                fout.write('\n')

                    nconf += 1

                except tf.errors.OutOfRangeError:
                    break

        fout.write('\n\n# Total number of configurations: {}.'.format(nconf))


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

    dtype = HACKED_DTYPE  # should be defined as a global variable by callee

    # input
    name = parsed_features['name']

    num_atoms_by_species = tf.decode_raw(
        parsed_features['num_atoms_by_species'], tf.int64)

    # tf.cast is necessary, otherwise tf will report error
    shape = tf.cast([num_species], tf.int64)
    # or else, can use num_species as tf.int32
    num_atoms_by_species = tf.reshape(num_atoms_by_species, shape)

    weight = tf.decode_raw(parsed_features['weight'], dtype)[0]

    num_atoms = tf.reduce_sum(num_atoms_by_species)
    shape = tf.cast([num_atoms, num_descriptors], tf.int64)
    gen_coords = tf.decode_raw(parsed_features['gen_coords'], dtype)
    gen_coords = tf.reshape(gen_coords, shape)

    # labels
    energy = tf.decode_raw(parsed_features['energy'], dtype)[0]

    return name, num_atoms_by_species, weight, gen_coords, energy


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

    dtype = HACKED_DTYPE  # should be defined as a global variable from callee

    # input
    name = parsed_features['name']
    num_atoms_by_species = tf.decode_raw(
        parsed_features['num_atoms_by_species'], tf.int64)
    # tf.cast is necessary, otherwise tf will report error
    shape = tf.cast([num_species], tf.int64)
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
    dgen_datomic_coords = tf.decode_raw(
        parsed_features['dgen_datomic_coords'], dtype)
    dgen_datomic_coords = tf.reshape(dgen_datomic_coords, shape3)

    # labels
    energy = tf.decode_raw(parsed_features['energy'], dtype)[0]
    forces = tf.decode_raw(parsed_features['forces'], dtype)
    forces = tf.reshape(forces, shape1)

    return name, num_atoms_by_species, weight, gen_coords, energy, atomic_coords, dgen_datomic_coords, forces
