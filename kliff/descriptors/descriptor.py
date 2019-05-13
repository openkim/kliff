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


class Descriptor:
    """Base class of atomic enviroment descriptors.

    Preprocess dataset to generate fingerprints. This is the base class for all
    descriptors, so it should not be used directly. Instead, descriptors built on top
    of this such as :class:`~kliff.descriptors.SymmetryFunction` and
    :class:`~kliff.descriptors.Bispectrum` can be used to transform the atomic
    enviroment information into fingerprints.

    Parameters
    ----------
    cut_dists: dict
        Cutoff distances, with key of the form ``A-B`` where ``A`` and ``B`` are
        atomic species string, and value should be a float.

    cut_name: str
        Name of the cutoff function, such as ``cos``, ``P3``, ``P7``.

    hyperparams: dict
        A dictionary of the hyperparams of the descriptor.

    normalize: bool (optional)
        If ``True``, the fingerprints is centered and normalized according to:
        ``zeta = (zeta - mean(zeta)) / stdev(zeta)``

    dtype: np.dtype
        Data type for the generated fingerprints, such as ``np.float32`` and
        ``np.float64``.

    Attributes
    ----------
    size: int
        Lengh of the fingerprint vector.

    mean: list
        Mean of the fingerprints.

    stdev: list
        Standard deviation of the fingerprints.
    """

    def __init__(
        self, cut_dists, cut_name, hyperparams, normalize=True, dtype=np.float32
    ):

        self.cut_dists = cut_dists
        self.cut_name = cut_name
        self.hyperparams = hyperparams
        self.normalize = normalize
        self.dtype = dtype

        self.size = None
        self.mean = None
        self.stdev = None

    #    def set_cutoff(self, name, values):
    #        """Set the cutoff used in the descriptor.
    #
    #        Parameters
    #        ----------
    #        name: str
    #            Name of the cutoff, such as ``cos``, ``P3``, ``P7``.
    #
    #        values: dict
    #            Values for the cutoff, with key of the form ``A-B`` where ``A`` and ``B``
    #            are atomic species, and value should be a float.
    #
    #        Example
    #        -------
    #        >>> desc = Descriptor()
    #        >>> name = 'cos'
    #        >>> values = {'C-C':4.5,'H-H':3.0,'C-H':4.0}
    #        >>> desc.set_cutoff(name, values)
    #        """
    #
    #        self.cutname = name
    #        self.cutoff = generate_full_cutoff(values)
    #        self.species_code = dict()
    #
    #        species = get_species_from_cutoff(values)
    #        num_species = len(species)
    #
    #        rcutsym = np.zeros([num_species, num_species], dtype=np.double)
    #        try:
    #            for i, si in enumerate(species):
    #                self.species_code[si] = i
    #                for j, sj in enumerate(species):
    #                    rcutsym[i][j] = self.cutoff[si+'-'+sj]
    #        except KeyError as e:
    #            raise InputError('Cutoff for "{}" not provided.'.format(e))
    #
    #    def set_hyperparams(self, hyperparams):
    #        """Set the hyperparameters that the descriptor needs.
    #
    #        Parameters
    #        ----------
    #        hyperparams: dict
    #            Hyperparameters of the descriptor.
    #        """
    #        self.hyperparams = hyperparams

    def generate_train_fingerprints(
        self,
        configs,
        grad=False,
        reuse=False,
        prefix='fingerprints',
        nprocs=mp.cpu_count(),
    ):
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
                    'Delete existing fingerprints: %s, new ones is genereated.', fname
                )
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

    def generate_test_fingerprints(
        self,
        configs,
        grad=False,
        reuse=False,
        prefix='fingerprints',
        train_prefix=None,
        nprocs=mp.cpu_count(),
    ):
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
                    'Delete existing fingerprints: %s, new ones is genereated.', fname
                )
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
                'genereate_train_fingerprints().'
            )
        all_zeta, all_dzetadr = self.calc_zeta_dzetadr(configs, grad, nprocs)
        self.dump_fingerprints(configs, fname, all_zeta, all_dzetadr, grad, nprocs)

        return fname

    def dump_fingerprints(
        self, configs, fname, all_zeta, all_dzetadr, grad, nprocs=mp.cpu_count()
    ):

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
                example = {
                    'name': name,
                    'species': species,
                    'weight': weight,
                    'zeta': zeta,
                    'energy': energy,
                }
                if grad:
                    example['dzeta_dr'] = dzetadr
                    example['forces'] = forces
                pickle.dump(example, f)

        logger.info('Processing %d configurations finished.', len(configs))

    def calc_zeta_dzetadr(self, configs, grad, nprocs=mp.cpu_count()):
        try:
            rslt = parallel.parmap1(self.transform, configs, grad, nprocs=nprocs)
            all_zeta = [pair[0] for pair in rslt]
            all_dzetadr = [pair[1] for pair in rslt]
        except MemoryError as e:
            logger.info(
                '%s. Occurs in calculating fingerprints using parallel. '
                'Will fallback to use a serial version.',
                str(e),
            )
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
                mean += delta / n
                delta2 = row - mean
                M2 += delta * delta2
            if i % 100 == 0:
                sys.stdout.flush()
        stdev = np.sqrt(M2 / (n - 1))

        return mean, stdev

    def transform(self, conf, grad=False):
        """Transform atomic coords to atomic enviroment descriptor values.

        Parameters
        ----------
        conf: :class:`~kliff.dataset.Configuration` object
            A configuration of atoms.

        grad: bool (optional)
            Whether to compute the gradient of descriptor values w.r.t. atomic
            coordinates.

        Returns
        -------
        zeta: 2D array
            Descriptor values, each row for one atom.
            zeta has shape (num_atoms, num_descriptors), where num_atoms is the
            number of atoms in the configuration, and num_descriptors is the size
            of the descriptor vector (depending on the the choice of hyper-parameters).

        dzeta_dr: 4D array if grad is ``True``, otherwise ``None``
            Gradient of descriptor values w.r.t. atomic coordinates.
            dzeta_dr has shape (num_atoms, num_descriptors, num_atoms, DIM), where
            num_atoms and num_descriptors has the same meanings as described in zeta.
            DIM = 3 denotes three Cartesian coordinates.
        """

        raise NotImplementedError(
            'Method "transform" not implemented; it has to '
            'be needes to be added by any subclass.'
        )

    def get_size(self):
        """Retrun the size of the descritpor vector."""
        return self.size

    def get_mean(self):
        """Return a list of the mean of the fingerprints."""
        return self.mean.copy()

    def get_stdev(self):
        """Return a list of the standard deviation of the fingerprints."""
        return self.stdev.copy()

    def get_dtype(self):
        """Return the data type of the fingerprints."""
        return self.dtype

    def get_cutoff(self):
        """Return the name and values of cutoff. """
        return self.cut_name, self.cut_dists

    def get_hyperparams(self):
        """Return the hyperparameters of descriptors. """
        return self.hyperparams


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
    """Read preprocessed data.

    Parameters
    ----------
    fname: str
        Names of the pickled data file.

    fit_forces: bool
        Whether to fit to forces.

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


def generate_full_cutoff(cutoff):
    """Generate a full binary cutoff dictionary.

    For species pair `S1-S2` in the ``cutoff`` dictionary, add key `S2-S1` to it,
    whih the same value as `S1-S2`.

    Parameters
    ----------
    cutoff: dict
        Cutoff dictionary with key of the form ``A-B`` where ``A`` and ``B``
        are atomic species, and value should be a float.

    Return
    ------
    dict
        A dictionary with all combination of species as keys.

    Example
    -------
    >>> cutoff = {'C-C': 4.0, 'C-H':3.5}
    >>> generate_full_cutoff(cutoff)
        {'C-C': 4.0, 'C-H':3.5, 'H-C':3.5}
    """
    rcut2 = dict()
    for key, val in cutoff.items():
        s1, s2 = key.split('-')
        if s1 != s2:
            rcut2[s2 + '-' + s1] = val
    # merge
    rcut2.update(cutoff)

    return rcut2


def generate_species_code(cutoff):
    """Genereate species code info from cutoff dictionary.

    Parameters
    ----------
    cutoff: dict
        Cutoff dictionary with key of the form ``A-B`` where ``A`` and ``B``
        are atomic species, and value should be a float.

    Return
    ------
    species: dict
        A dictionary of species and the integer code, with keys the species in
        ``cutoff`` keys, and values integer code for species.

    Example
    -------
    >>> cutoff = {'C-C': 4.0, 'C-H':3.5}
    >>> generate_species_code(cutoff)
        {'C':0, 'H':1}
    """
    species = set()
    for key in cutoff:
        s1, s2 = key.split('-')
        species.update([s1, s2])
    species = list(species)
    species_code = {s: i for i, s in enumerate(species)}
    return species_code


class DescriptorError(Exception):
    def __init__(self, msg):
        super(DescriptorError, self).__init__(msg)
        self.msg = msg

    def __str__(self):
        return self.msg
