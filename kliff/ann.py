from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import functools
import os
import sys
import shutil
import inspect
import multiprocessing as mp
from kliff import parallel


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
        weights = tf.Variable(tf.truncated_normal(
            shape, stddev=0.1, dtype=dtype))
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


def weight_decorator(func):
    """A decorator for weight initializer to output summaries."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        weights = func(*args, **kwargs)
        variable_summaries(weights)
        return weights
    return wrapper


def layer_decorator(func):
    """A decorator for layer to output activations histogram."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        activations = func(*args, **kwargs)
        tf.summary.histogram('activations', activations)
        return activations
    return wrapper


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _write_tfrecord_energy(writer, conf, descriptor, do_normalize, np_dtype,
                           mean, std, zeta=None, structure='bulk'):
    """ Write data to tfrecord format."""

    # descriptor features
    num_descriptors = descriptor.get_number_of_descriptors()
    if zeta is None:
        zeta, _ = descriptor.generate_generalized_coords(
            conf, fit_forces=False, structure=structure)

    # do centering and normalization if needed
    if do_normalize:
        zeta = (zeta - mean) / std
    zeta = zeta.astype(np_dtype).tostring()

    # configuration features
    name = conf.get_id()
    d = conf.num_atoms_by_species
    num_atoms_by_species = [d[k] for k in d]
    num_species = len(num_atoms_by_species)
    num_atoms_by_species = np.array(num_atoms_by_species, np.int64).tostring()
    weight = np.array(conf.get_weight(), np_dtype).tostring()
    energy = np.array(conf.get_energy(), np_dtype).tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        # meta data
        'num_descriptors': _int64_feature(num_descriptors),
        'num_species': _int64_feature(num_species),
        # input data
        'name': _bytes_feature(name),
        'num_atoms_by_species': _bytes_feature(num_atoms_by_species),
        'weight': _bytes_feature(weight),
        'gen_coords': _bytes_feature(zeta),
        # labels
        'energy': _bytes_feature(energy),
    }))

    writer.write(example.SerializeToString())


def _write_tfrecord_energy_and_force(writer, conf, descriptor, do_normalize, np_dtype,
                                     mean, std, zeta=None, dzetadr=None, structure='bulk'):
    """ Write data to tfrecord format."""

    # descriptor features
    num_descriptors = descriptor.get_number_of_descriptors()
    if zeta is None or dzetadr is None:
        zeta, dzetadr = descriptor.generate_generalized_coords(
            conf, fit_forces=True, structure=structure)

    # do centering and normalization if needed
    if do_normalize:
        std_3d = np.atleast_3d(std)
        zeta = (zeta - mean) / std
        dzetadr = dzetadr / std_3d
    zeta = zeta.astype(np_dtype).tostring()
    dzetadr = dzetadr.astype(np_dtype).tostring()

    # configuration features
    name = conf.get_identifier()
    d = conf.num_atoms_by_species
    num_atoms_by_species = [d[k] for k in d]
    num_species = len(num_atoms_by_species)
    num_atoms_by_species = np.array(num_atoms_by_species, np.int64).tostring()
    coords = conf.get_coordinates().astype(np_dtype).tostring()
    weight = np.array(conf.get_weight(), np_dtype).tostring()
    energy = np.array(conf.get_energy()).astype(np_dtype).tostring()
    forces = conf.get_forces().astype(np_dtype).tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        # meta data
        'num_descriptors': _int64_feature(num_descriptors),
        'num_species': _int64_feature(num_species),
        # input data
        'name': _bytes_feature(name),
        'num_atoms_by_species': _bytes_feature(num_atoms_by_species),
        'weight': _bytes_feature(weight),
        'atomic_coords': _bytes_feature(coords),
        'gen_coords': _bytes_feature(zeta),
        'dgen_datomic_coords': _bytes_feature(dzetadr),
        # labels
        'energy': _bytes_feature(energy),
        'forces': _bytes_feature(forces)
    }))

    writer.write(example.SerializeToString())


def welford_mean_and_std(configs, descriptor, structure='bulk'):
    """Compute the mean and standard deviation of generalized coords.

    This running mean and standard method proposed by Welford is memory-efficient.
    Besides, it outperforms the naive method from suffering numerical instability
    for large dataset.

    see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """
# TODO parallelize this (Chan's algorithm is criticized to be unstable)

    # number of features
    conf = configs[0]
    zeta, _ = descriptor.generate_generalized_coords(
        conf, fit_forces=False, structure=structure)
    size = zeta.shape[1]

    # starting Welford's method
    n = 0
    mean = np.zeros(size)
    M2 = np.zeros(size)
    for i, conf in enumerate(configs):
        zeta, _ = descriptor.generate_generalized_coords(
            conf, fit_forces=False, structure=structure)
        for row in zeta:
            n += 1
            delta = row - mean
            mean += delta/n
            delta2 = row - mean
            M2 += delta*delta2
        if i % 100 == 0:
            print('Processing training example:', i)
            sys.stdout.flush()
    std = np.sqrt(M2/(n-1))
    print('Processing {} configurations finished.\n'.format(i+1))

    return mean, std


def numpy_mean_and_std(configs, descriptor, fit_forces, structure='bulk', nprocs=mp.cpu_count()):
    """Compute the mean and standard deviation of generalized coords."""

    try:
        rslt = parallel.parmap(descriptor.generate_generalized_coords, configs,
                               nprocs, fit_forces, structure)
        all_zeta = [pair[0] for pair in rslt]
        all_dzetadr = [pair[1] for pair in rslt]
        stacked = np.concatenate(all_zeta)
        mean = np.mean(stacked, axis=0)
        std = np.std(stacked, axis=0)
    except MemoryError:
        raise MemoryError('Out of memory while computing mean and standard deviation. '
                          'Try the memory-efficient welford method instead.')

    # TODO delete debug
#  with open('debug_descriptor_after_normalization.txt', 'w') as fout:
#    for zeta,conf in zip(all_zeta, configs):
#      zeta_norm = (zeta - mean) / std
#      fout.write('\n\n'+'#'+'='*80+'\n')
#      fout.write('# configure name: {}\n'.format(conf.id))
#      fout.write('# atom id    descriptor values ...\n\n')
#      for i,line in enumerate(zeta_norm):
#        fout.write('{}    '.format(i))
#        for j in line:
#          fout.write('{:.15g} '.format(j))
#        fout.write('\n')

    return mean, std, all_zeta, all_dzetadr


def convert_to_tfrecord(configs, descriptor, size_validation=0,
                        directory='/tmp/data', do_generate=False, do_normalize=True, do_shuffle=True,
                        use_welford=False, fit_forces=False, structure='bulk', nprocs=mp.cpu_count(),
                        dtype=tf.float32):
    """Preprocess the data to generate the generalized coords and its derivatives,
    and store them, together with coords, and label as tfRecord binary.

    Parameter
    ---------

    configs, list of Config objects.

    descriptor, object of Descriptor class.

    size_validation, int
      Validation set size.

    directory
      The TFRecords file for training set is written to `directory/train.tfrecord'.
      If `size_validation' is not 0, the validation set is written to
      `directory/validation.tfrecord'.
      The centering and normalizing mean and standard deviation data is written
      to `directory/mean_and_std_for_kim_ann'.

    do_generate, bool
      Whether to compute the generalized coords and its derivatives or not.

    do_normalize, bool
      Whether to center and normalize the data or not through:
        zeta_new = (zeta - mean(zeta)) / std(zeta)
      Effective only when `do_generate' is set to `True'.

    do_shuffle, bool
      Whether to shuffle the dataset before dividing into training set and
      validation set.

    use_welford, bool
      Wehther to use Welford's online method to compute mean and standard (if
      do_normalize == True).

    fit_forces, bool
      Wheter to fit to forces.
    """

    # split dataset into training set and validation set if necessary
    configs = np.array(configs)
    size_dataset = configs.size

    indices = np.arange(size_dataset)
    if do_shuffle:
        np.random.shuffle(indices)

    if size_validation != 0:
        if size_validation > size_dataset:
            raise ValueError('Validation set size "{}" larger than dataset size "{}".'
                             .format(size_validation, size_dataset))
        size_train = size_dataset - size_validation
        tr_indices = indices[:size_train]
        va_indices = indices[size_train:]
        tr_configs = configs[tr_indices]
        va_configs = configs[va_indices]
    else:  # only need training set, not validation set
        tr_configs = configs[indices]
        va_configs = np.array([])

    tr_name = os.path.join(directory, 'train.tfrecord')
    va_name = os.path.join(directory, 'validation.tfrecord')

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
        else:
            raise ValueError('Unsupported data type "{}".'.format(dtype))

        # compute mean and standard deviation of training set
        if do_normalize:
            print('\nCentering and normalizing the data...')
            print('Size of training: {}, size of validation {}.'.format(
                tr_configs.size, va_configs.size))

            # compute mean and standard deviation of each feature of training set
            if use_welford:
                mean, std = welford_mean_and_std(
                    tr_configs, descriptor, structure)
            else:
                mean, std, all_zeta, all_dzetadr = numpy_mean_and_std(tr_configs, descriptor,
                                                                      fit_forces, structure, nprocs)

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
            mean = None
            std = None

            # we write empty info to this file
            with open(os.path.join(directory, 'mean_and_std_for_kim_ann'), 'w') as fout:
                fout.write('False\n')

        # write training data to TFRecords
        print('\nWriting training tfRecords data as: {}'.format(tr_name))
        tr_writer = tf.python_io.TFRecordWriter(tr_name)
        for i, conf in enumerate(tr_configs):
            if i % 100 == 0:
                print('Processing configuration:', i)
                sys.stdout.flush()
            if do_normalize and not use_welford:
                zeta = all_zeta[i]
                if fit_forces:
                    dzetadr = all_dzetadr[i]
            else:
                zeta = None
                dzetadr = None
            if fit_forces:
                _write_tfrecord_energy_and_force(tr_writer, conf, descriptor, do_normalize,
                                                 np_dtype, mean, std, zeta, dzetadr, structure)
            else:
                _write_tfrecord_energy(tr_writer, conf, descriptor, do_normalize,
                                       np_dtype, mean, std, zeta, structure)

        tr_writer.close()
        print('Processing {} configurations finished.\n'.format(tr_configs.size))

        # write validation data to TFRecords
        if va_configs.size != 0:
            print('\nWriting validation tfRecords data as: {}'.format(va_name))
            va_writer = tf.python_io.TFRecordWriter(va_name)
            for i, conf in enumerate(va_configs):
                if i % 100 == 0:
                    print('Processing configuration:', i)
                    sys.stdout.flush()
                if fit_forces:
                    _write_tfrecord_energy_and_force(va_writer, conf, descriptor,
                                                     do_normalize, np_dtype, mean, std, None, None, structure)
                else:
                    _write_tfrecord_energy(va_writer, conf, descriptor,
                                           do_normalize, np_dtype, mean, std, None, structure)
            va_writer.close()
            print('Processing {} configurations finished.\n'.format(va_configs.size))

    return tr_name, va_name


def convert_to_tfrecord_testset(configs, descriptor,
                                directory='/tmp/data', do_generate=True, do_normalize=True, do_shuffle=False,
                                fit_forces=False, structure='bulk', dtype=tf.float32):
    """Preprocess the testset data to generate the generalized coords and its
    derivatives, and store them, together with coords and label as tfRecord binary.

    This is for testset, so if do_normalize = True, the `mean_and_std_for_kim_ann'
    file should be in `directory', which is generated by `convert_to_tfrecords'.

    Parameter
    ---------

    configs: list of Config objects.

    descriptor: object of Descriptor class.

    directory
      The TFRecords file is written to `directory/test.tfrecord'.

    do_generate: bool
      Whether to compute the generalized coords and its derivatives or not.

    do_normalize: bool
      Whether to center and normalize the data or not through:
        zeta_new = (zeta - mean) / std
      where mean and std are read from `mean_and_std_for_kim_ann' in `directory'.

    do_shuffle, bool
      Wheter to shuffle data.

    fit_forces, bool
     Wheter to fit forces.
    """

    te_configs = np.array(configs)
    size_dataset = te_configs.size
    if do_shuffle:
        np.random.shuffle(te_configs)

    te_name = os.path.join(directory, 'test.tfrecord')

    # if cannot find the tfrecords data, we need to generate it
    if not os.path.exists(te_name):
        do_generate = True

    if do_generate:

        if not os.path.exists(directory):
            os.makedirs(directory)

        # determine the data type
        if dtype == tf.float32:
            np_dtype = np.float32
        elif dtype == tf.float64:
            np_dtype = np.float64
        else:
            raise ValueError('Unsupported data type "{}".'.format(dtype))

        # read mean and standard deviation of training set
        if do_normalize:

            size = None
            data = []
            fname = os.path.join(directory, 'mean_and_std_for_kim_ann')
            with open(fname, 'r') as fin:
                for line in fin:
                    if '#' in line:
                        line = line[:line.index('#')]
                    line = line.strip()
                    if not line:  # if empty
                        continue
                    if size is None:
                        size = int(line)
                    else:
                        data.append(float(line))
            mean = np.array(data[:size])
            std = np.array(data[size:])

            # expected number of features
            conf = configs[0]
            zeta, dzetadr = descriptor.generate_generalized_coords(
                conf, fit_forces, structure)
            expected_size = zeta.shape[1]

            if size != expected_size:
                raise Exception("Wrong mean and stdandard info file from directoy `{0}'. "
                                "Expected number of descriptors is {1}, while it is {2} from "
                                "`{0}/mean_and_standard_for_kim_ann'.".format(directory,
                                                                              expected_size, size))
        else:
            mean = None
            std = None

        # write test data to TFRecords
        print('\nWriting test tfRecords data as: {}'.format(te_name))
        te_writer = tf.python_io.TFRecordWriter(te_name)
        for i, conf in enumerate(te_configs):
            if fit_forces:
                _write_tfrecord_energy_and_force(te_writer, conf, descriptor,
                                                 do_normalize, np_dtype, mean, std, None, None, structure)
            else:
                _write_tfrecord_energy(te_writer, conf, descriptor,
                                       do_normalize, np_dtype, mean, std, None, structure)

            if i % 100 == 0:
                print('Processing configuration:', i)
        print('Processing {} configurations finished.\n'.format(te_configs.size))
        te_writer.close()

    return te_name


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

    dtype = HACKED_DTYPE  # defined as a global variable in read_from_trrecords

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


def read_tfrecord(fnames, fit_forces=False, num_parallel_calls=None, dtype=tf.float32):
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


# for feed_dictionary, probably ok
def input_layer_given_data(coords, zeta, dzetadr, num_descriptor,
                           layer_name='input_layer'):
    """Reusable code for making an input layer for a configuration."""

    with tf.name_scope(layer_name):
        input, dummy = int_pot(coords=coords, zeta=zeta, dzetadr=dzetadr)

        # set the static shape is needed to use high level api like
        # tf.contrib.layers.fully_connected
        input.set_shape((None, int(num_descriptor)))

        return input


#  The following three methods. to built data into graphDef, OK and fast for small data set
def preprocess(configs, descriptor, fit_forces=True):
    """Preprocess the data to generate the generalized coords and its derivatives.

    Parameter
    ---------

    configs, list of Config objects.

    descriptor, object of Descriptor class.
    """

    all_zeta = []
    all_dzetadr = []
    for i, conf in enumerate(configs):
        print('Preprocessing configuration:', i)
        zeta, dzetadr = descriptor.generate_generalized_coords(
            conf, fit_forces)
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
        all_zeta_processed.append((zeta - mean) / std)
    for dzetadr in all_dzetadr:
        all_dzetadr_processed.append(dzetadr / np.atleast_3d(std))

    return all_zeta_processed, all_dzetadr_processed


def input_layer_using_preprocessed(zeta, dzetadr, layer_name='input_layer',
                                   dtype=tf.float32):
    """Reusable code for making an input layer for a configuration."""

    # we only need some placeholder for coords, since it is not used
# TODO we'd better use the actual coords
    shape = [zeta.shape[0]*3]
    coords = tf.constant(0.1, dtype=dtype, shape=shape)
    with tf.name_scope(layer_name):
        input, dummy = int_pot(coords=coords, zeta=tf.constant(zeta, dtype),
                               dzetadr=tf.constant(dzetadr, dtype))
        return input, coords


def input_layer(config, descriptor, fit_forces=True, dtype=tf.float32):
    """Reusable code for making an input layer for a configuration."""

    # write a file to inform that no centering and normaling is used
    with open('mean_and_std_for_kim_ann', 'w') as fout:
        fout.write('False\n')

    layer_name = os.path.splitext(os.path.basename(config.id))[0]
    # need to return a tensor of coords since we want to take derivaives w.r.t it
    coords = tf.constant(config.get_coordinates(), dtype)
    zeta, dzetadr = descriptor.generate_generalized_coords(config, fit_forces)

    with tf.name_scope(layer_name):
        input, dummy = int_pot(coords=coords, zeta=tf.constant(zeta, dtype),
                               dzetadr=tf.constant(dzetadr, dtype))
        return input, coords


def get_weights_and_biases(layer_names):
    """Get the weights and biases of all layers.

      If variable_scope is used, is should be prepended to the name.
      The element order matters of layer_names, since the returned weights and
      biases have the same order as layer_names.


    Parameter
    ---------

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


def tfrecord_to_text(tfrecord_name, text_name, fit_forces=False, do_grad_gc=False, dtype=tf.float32):
    """ convert tfrecord data file of generalized coords (gc) to text file.

    Parameters
    ----------

    tfrecord_name: str
      File name of the tfrecord data file.

    text_name: str
      File name of the outout text data file.

    fit_forces: bool
      Is forces included in the tfrecord data.

    do_grad_gc: bool
      Wheter to write out gradient of gc w.r.t. atomic coords.
    """

    if do_grad_gc and not fit_forces:
        raise Exception("Set `fit_forces` to `True` to use `do_grad_gc`.")

    dataset = read_tfrecord(tfrecord_name, fit_forces, dtype=dtype)
    iterator = dataset.make_one_shot_iterator()

    if fit_forces:
        name, num_atoms_by_species, weight, gen_coords, energy_label, atomic_coords, \
            dgen_datomic_coords, forces_label = iterator.get_next()
    else:
        name, num_atoms_by_species, weight, gen_coords, energy_label = iterator.get_next()

    with open(text_name, 'w') as fout:
        fout.write('# Generalized coordinates for all configurations.\n')
        nconf = 0

        with tf.Session() as sess:
            while True:
                try:
                    if do_grad_gc:
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

                    if do_grad_gc:
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


def write_kim_ann(descriptor, weights, biases, activation, mean_and_std_name,
                  keep_prob=None, dtype=tf.float32, fname='ann_kim.params'):
    """Output ANN structure, parameters etc. in the format of the KIM ANN model.

    Parameter
    ---------

    descriptor, object of Descriptor class

    """

    if keep_prob is None:
        keep_prob = np.ones(len(weights))

    with open(fname, 'w') as fout:

        # cutoff
        cutname, rcut, rcut_samelayer = descriptor.get_cutoff()
        if rcut_samelayer is None:
            rcut_samelayer = rcut
        maxrcut = max(rcut.values())
        maxrcut_samelayer = max(rcut_samelayer.values())

        fout.write('# cutoff    rcut\n')
        if dtype == tf.float64:
            fout.write('{}  {:.15g}  {:.15g}\n\n'.format(
                cutname, maxrcut, maxrcut_samelayer))
        else:
            fout.write('{}  {:.7g}  {:.7g}\n\n'.format(
                cutname, maxrcut, maxrcut_samelayer))

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
                            fout.write(
                                '{:.15g} {:.15g}'.format(val[0], val[1]))
                        else:
                            fout.write('{:.7g} {:.7g}'.format(val[0], val[1]))
                        fout.write('    # eta  Rs\n')
                    fout.write('\n')
                elif name == 'g3':
                    for val in values:
                        if dtype == tf.float64:
                            fout.write('{:.15g}'.format(val[0]))
                        else:
                            fout.write('{:.7g}'.format(val[0]))
                        fout.write('    # kappa\n')
                    fout.write('\n')
                elif name == 'g4':
                    for val in values:
                        zeta = val[0]
                        lam = val[1]
                        eta = val[2]
                        if dtype == tf.float64:
                            fout.write(
                                '{:.15g} {:.15g} {:.15g}'.format(zeta, lam, eta))
                        else:
                            fout.write(
                                '{:.7g} {:.7g} {:.7g}'.format(zeta, lam, eta))
                        fout.write('    # zeta  lambda  eta\n')
                    fout.write('\n')
                elif name == 'g5':
                    for val in values:
                        zeta = val[0]
                        lam = val[1]
                        eta = val[2]
                        if dtype == tf.float64:
                            fout.write(
                                '{:.15g} {:.15g} {:.15g}'.format(zeta, lam, eta))
                        else:
                            fout.write(
                                '{:.7g} {:.7g} {:.7g}'.format(zeta, lam, eta))
                        fout.write('    # zeta  lambda  eta\n')
                    fout.write('\n')

        # data centering and normalization
        # header
        fout.write('#' + '='*80 + '\n')
        fout.write('# Preprocessing data to center and normalize\n')
        fout.write('#' + '='*80 + '\n')

        # data
        shutil.copy(mean_and_std_name, os.getcwd())
        with open(mean_and_std_name, 'r') as fin:
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
            raise ValueError(
                'unsupported activation function for KIM ANN model.')

        fout.write('{}    # activation function\n'.format(act_name))

        # keep probability
        for i in keep_prob:
            fout.write('{:.15g}  '.format(i))
        fout.write('  # keep probability of input for each layer\n\n')

        # weights and biases
        for i, (w, b) in enumerate(zip(weights, biases)):

            # weight
            rows, cols = w.shape
            if i != num_layers-1:
                fout.write(
                    '# weight of hidden layer {} (shape({}, {}))\n'.format(i+1, rows, cols))
            else:
                fout.write(
                    '# weight of output layer (shape({}, {}))\n'.format(rows, cols))
            for line in w:
                for item in line:
                    if dtype == tf.float64:
                        fout.write('{:23.15e}'.format(item))
                    else:
                        fout.write('{:15.7e}'.format(item))
                fout.write('\n')

            # bias
            if i != num_layers-1:
                fout.write(
                    '# bias of hidden layer {} (shape({}, {}))\n'.format(i+1, rows, cols))
            else:
                fout.write(
                    '# bias of output layer (shape({}, {}))\n'.format(rows, cols))
            for item in b:
                if dtype == tf.float64:
                    fout.write('{:23.15e}'.format(item))
                else:
                    fout.write('{:15.7e}'.format(item))
            fout.write('\n\n')
