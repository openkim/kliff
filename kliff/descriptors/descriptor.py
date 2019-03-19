import os
import sys
import logging
import pickle
import numpy as np
import multiprocessing as mp
import kliff
from kliff import parallel
from kliff.atomic_data import atomic_number
from kliff.error import InputError

logger = kliff.logger.get_logger(__name__)


class Descriptor(object):
    """Base class for all atomic enviroment descriptors.

    Preprocess dataset to generate fingerprints.

    Example
    -------


    Attributes
    ----------

    mean
    stdev


    """

    def __init__(self, normalize=True, dtype=np.float32):
        """


        Parameters
        ----------

        normalize: bool (optional)
            Whether to center and normalize the fingerprints by:
                zeta = (zeta - mean(zeta)) / stdev(zeta)

        """

        self.normalize = normalize
        self.dtype = dtype

        self.mean = None
        self.stdev = None

    def generate_train_fingerprints(self, configs, grad=False, reuse=False,
                                    prefix='fingerprints', nprocs=mp.cpu_count()):
        """Convert training set to fingerprints.

        Parameters
        ----------

        configs: list of Configuration objects

        grad: bool (optional)
            Whether to compute the gradient of fingerprints w.r.t. atomic coordinates.

        reuse: bool
            Whether to reuse the fingerprints if one is found at: {prefix}/train.pkl.

        prefix: str
            Directory name where the generated fingerprints are stored.
            Path to the generated fingerprints is: is: {prefix}/train.pkl

        nprocs: int
          Number of processors to be used.
        """

        fname = os.path.join(prefix, 'train.pkl')
        mean_stdev_name = os.path.join(prefix, 'mean_stdev.pkl')

        # TODO need to do additional check for the loaded data
        # restore data
        if os.path.exists(fname):
            logger.info('Found existing fingerprints: %s.', fname)
            if not reuse:
                os.remove(fname)
                logger.info(
                    'Delete existing fingerprints: %s, new ones is genereated.', fname)
            else:
                if self.normalize:
                    logger.info('Restore mean and stdev to: %s.', mean_stdev_name)
                    self.mean, self.stdev = load_mean_stdev(mean_stdev_name)
                return fname

        # generate data
        all_zeta, all_dzetadr = self.calc_zeta_dzetadr(configs, grad, nprocs)
        if self.normalize:
            if all_zeta is not None:
                stacked = np.concatenate(all_zeta)
                self.mean = np.mean(stacked, axis=0)
                self.stdev = np.std(stacked, axis=0)
            else:
                self.mean, self.stdev = self.welford_mean_and_stdev(configs, grad)
            dump_mean_stdev(self.mean, self.stdev, mean_stdev_name)
        self.dump_fingerprints(configs, fname, all_zeta, all_dzetadr, grad, nprocs)

        return fname

    def generate_test_fingerprints(self, configs, grad=False, reuse=False,
                                   prefix='fingerprints', train_prefix=None,
                                   nprocs=mp.cpu_count()):
        """Convert test set to fingerprints.

        Parameters
        ----------

        configs: list of Configuration objects

        grad: bool (optional)
            Whether to compute the gradient of fingerprints w.r.t. atomic coordinates.

        reuse: bool
            Whether to reuse the fingerprints if one is found at: {prefix}/test.pkl.

        prefix: str
            Directory name where the generated fingerprints are stored.
            Path to the generated fingerprints is: is: {prefix}/test.pkl

        train_prefix: str (optional)
            Directory name where the generated training fingerprints are stored.
            Defaults to `prefix`.
            This effects only when:
                1) `normalize` is `True` in the initializer; and
                2) self.genereate_train_fingerprints() has not been called,
            because in this case we need to find the mean and stdev of the fingerprints
            of the training set, which is saved as a file name `mean_stdev.pkl` in the
            derectory specified by `train_prefix`.

        nprocs: int
          Number of processors to be used.
        """

        if train_prefix is None:
            train_prefix = prefix

        fname = os.path.join(prefix, 'test.pkl')
        mean_stdev_name = os.path.join(train_prefix, 'mean_stdev.pkl')

        # TODO need to do additional check for the loaded data
        # restore data
        if os.path.exists(fname):
            logger.info('Found existing fingerprints: %s.', fname)
            if not reuse:
                os.remove(fname)
                logger.info(
                    'Delete existing fingerprints: %s, new ones is genereated.', fname)
            else:
                if self.normalize and (self.mean is None or self.stdev is None):
                    logger.info('Restore mean and stdev to: %s.', mean_stdev_name)
                    self.mean, self.stdev = load_mean_stdev(mean_stdev_name)
                return fname

        # generate data
        if self.normalize and (self.mean is None or self.stdev is None):
            raise DescriptorError(
                'Cannot proceed to genereate fingerprints for test set without '
                'that for training set generated when normalization is required. '
                'Try generating training set fingerprints first by calling: '
                'genereate_train_fingerprints().')
        all_zeta, all_dzetadr = self.calc_zeta_dzetadr(configs, grad, nprocs)
        self.dump_fingerprints(configs, fname, all_zeta, all_dzetadr, grad, nprocs)

        return fname

    def dump_fingerprints(self, configs, fname, all_zeta, all_dzetadr, grad,
                          nprocs=mp.cpu_count()):

        logger.info('Pickle fingerprint images to: %s.', fname)

        dirname = os.path.dirname(fname)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        with open(fname, 'ab') as f:
            for i, conf in enumerate(configs):
                if i % 100 == 0:
                    logger.info('Processing configuration: %d.', i)
                zeta = all_zeta[i]
                dzetadr = all_dzetadr[i]
                if zeta is None:
                    zeta, dzetadr = self.transform(conf, grad=grad)

                # reshape dzeta from a 4D to a 3D array (combining the last two dims)
                if grad:
                    new_shape = (dzetadr.shape[0], dzetadr.shape[1], -1)
                    dzetadr = dzetadr.reshape(new_shape)

                # centering and normalization
                if self.normalize:
                    zeta = (zeta - self.mean) / self.stdev
                    if grad:
                        stdev_3d = np.atleast_3d(self.stdev)
                        dzetadr = dzetadr / stdev_3d

                # pickling data
                name = conf.get_identifier()
                species = conf.get_species()
                species = np.asarray([atomic_number[i] for i in species], np.intc)
                weight = np.asarray(conf.get_weight(), self.dtype)
                zeta = np.asarray(zeta, self.dtype)
                energy = np.asarray(conf.get_energy(), self.dtype)
                if grad:
                    dzetadr = np.asarray(dzetadr, self.dtype)
                    forces = np.asarray(conf.get_forces(), self.dtype)

                # TODO maybe change num atoms by species to species list or even conf
                example = {'name': name,
                           'species': species,
                           'weight': weight,
                           'zeta': zeta,
                           'energy': energy}
                if grad:
                    example['dzeta_dr'] = dzetadr
                    example['forces'] = forces
                pickle.dump(example, f)

        logger.info('Processing %d configurations finished.', len(configs))

    def calc_zeta_dzetadr(self, configs, grad, nprocs=mp.cpu_count()):
        try:
            rslt = parallel.parmap(self.transform, configs, nprocs, grad)
            all_zeta = [pair[0] for pair in rslt]
            all_dzetadr = [pair[1] for pair in rslt]
        except MemoryError as e:
            logger.info('%s. Occurs in calculating fingerprints using parallel. '
                        'Will fallback to use a serial version.', str(e))
            all_zeta = [None for _ in len(configs)]
            all_dzetadr = [None for _ in len(configs)]
        return all_zeta, all_dzetadr

    def welford_mean_and_stdev(self, configs, grad):
        """Compute the mean and standard deviation of fingerprints.

        This running mean and standard method proposed by Welford is memory-efficient.
        Besides, it outperforms the naive method from suffering numerical instability
        for large dataset.

        see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        """

        # number of features
        conf = configs[0]
        zeta, _ = self.transform(conf, grad=grad)
        size = zeta.shape[1]

        # starting Welford's method
        n = 0
        mean = np.zeros(size)
        M2 = np.zeros(size)
        for i, conf in enumerate(configs):
            zeta, _ = transform(conf, grad=grad)
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

    def transform(self, conf, grad=False):
        raise NotImplementedError(
            'Method "transform" not implemented; it has to be added by any "Descriptor" '
            'subclass.')

    def get_mean(self):
        return self.mean.copy()

    def get_stdev(self):
        return self.stdev.copy()

    def get_dtype(self):
        return self.dtype


def dump_mean_stdev(mean, stdev, fname):
    dirname = os.path.dirname(fname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    data = {'mean': mean, 'stdev': stdev}
    with open(fname, 'wb') as f:
        pickle.dump(data, f)


def load_mean_stdev(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
        mean = data['mean']
        stdev = data['stdev']
    return mean, stdev


def load_fingerprints(fname):
    """Read preprocessed data from `fname'.

    Parameter
    ---------

    fname, str
      names of the pickled data file.

    fit_forces, bool
      wheter to fit to forces.

    Return
    ------

    Instance of tf.data.
    """
    data = []
    with open(fname, 'rb') as f:
        try:
            while True:
                x = pickle.load(f)
                data.append(x)
        except EOFError:
            pass
    return data


class DescriptorError(Exception):
    def __init__(self, msg):
        super(DescriptorError, self).__init__(msg)
        self.msg = msg

    def __str__(self):
        return self.msg
