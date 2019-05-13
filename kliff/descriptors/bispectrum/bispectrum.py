import sys
import logging
import numpy as np
import kliff
from kliff.descriptors.descriptor import Descriptor
from kliff.descriptors.descriptor import generate_full_cutoff, generate_species_code
from kliff.neighbor import NeighborList
from . import bs

logger = kliff.logger.get_logger(__name__)


class Bispectrum(Descriptor):
    """Bispectrum descriptor.

    Preprocess dataset to generate fingerprints using the Bispectrum descriptor
    as discussed in [Bartok2010]_ and [Thompson2015]_.

    Parameters
    ----------
    cut_dists: dict
        Cutoff distances, with key of the form ``A-B`` where ``A`` and ``B`` are
        atomic species string, and value should be a float.

    cut_name: str
        Name of the cutoff function.

    hyperparams: dict
        A dictionary of the hyperparams of the descriptor.

    normalize: bool (optional)
        If ``True``, the fingerprints is centered and normalized according to:
        ``zeta = (zeta - mean(zeta)) / stdev(zeta)``

    dtype: np.dtype
        Data type for the generated fingerprints, such as ``np.float32`` and
        ``np.float64``.

    Example
    -------
    >>> cut_name = 'cos'
    >>> cut_dists = {'C-C': 5.0, 'C-H': 4.5, 'H-H': 4.0}
    >>> hyperparams = {'jmax': 4, 'weight': {'C':1.0, 'H':1.0}}
    >>> desc = Bispectrum(cut_dists, cut_name, hyperparams)

    References
    ----------
    .. [Bartok2010] Bartók, Albert P., Mike C. Payne, Risi Kondor, and Gábor Csányi.
       "Gaussian approximation potentials: The accuracy of quantum mechanics, without
       the electrons." Physical review letters 104, no. 13 (2010): 136403.
    .. [Thompson2015] Thompson, Aidan P., Laura P. Swiler, Christian R. Trott,
       Stephen M. Foiles, and Garritt J. Tucker. "Spectral neighbor analysis method
       for automated generation of quantum-accurate interatomic potentials." Journal
       of Computational Physics 285 (2015): 316-330.
    """

    def __init__(
        self, cut_dists, cut_name=None, hyperparams=None, normalize=True, dtype=np.float32
    ):
        super(Bispectrum, self).__init__(
            cut_dists, cut_name, hyperparams, normalize, dtype
        )

        self.update_hyperparams(self.hyperparams)

        # init cdesc
        rfac0 = self.hyperparams['rfac0']
        jmax = self.hyperparams['jmax']
        diagonalstyle = self.hyperparams['diagonalstyle']
        rmin0 = self.hyperparams['rmin0']
        switch_flag = self.hyperparams['switch_flag']
        bzero_flag = self.hyperparams['bzero_flag']
        use_shared_arrays = 0
        self._cdesc = bs.Bispectrum(
            rfac0,
            2 * jmax,
            diagonalstyle,
            use_shared_arrays,
            rmin0,
            switch_flag,
            bzero_flag,
        )

        self._set_cutoff()
        self._set_hyperparams()

    def transform(self, conf, grad=False):

        # neighbor list
        infl_dist = max(self.cutoff.values())
        nei = NeighborList(conf, infl_dist, padding_need_neigh=False)

        coords = nei.coords
        image = nei.image
        species = np.asarray([self.species_code[i] for i in nei.species], dtype=np.intc)
        numneigh, neighlist = nei.get_numneigh_and_neighlist_1D()

        Natoms = len(coords)
        Ncontrib = conf.get_number_of_atoms()
        Ndesc = self.get_size()

        if grad:
            zeta, dzeta_dr = self._cdesc.compute_zeta_and_dzeta_dr(
                coords, species, neighlist, numneigh, image, Natoms, Ncontrib, Ndesc
            )
            # reshape to 4D array
            dzeta_dr = dzeta_dr.reshape(Ncontrib, Ndesc, Ncontrib, 3)
        else:
            zeta = self._cdesc.compute_zeta(
                coords, species, neighlist, numneigh, image, Natoms, Ncontrib, Ndesc
            )
            dzeta_dr = None

        if logger.getEffectiveLevel() == logging.DEBUG:
            logger.debug(
                '\n' + '=' * 25 + 'descriptor values (no normalization)' + '=' * 25
            )
            logger.debug('\nconfiguration name: %s', conf.get_identifier())
            logger.debug('\natom id    descriptor values ...')
            for i, line in enumerate(zeta):
                s = '\n{}    '.format(i)
                for j in line:
                    s += '{:.15g} '.format(j)
                logger.debug(s)

        return zeta, dzeta_dr

    def update_hyperparams(self, params):
        """Update the hyperparameters based on the input at initialization.
        """
        default_hyperparams = {
            'jmax': 4,
            'rfac0': 0.99363,
            'diagonalstyle': 3,
            'rmin0': 0,
            'switch_flag': 1,
            'bzero_flag': 0,
            'weight': None,
        }

        if params is not None:
            for key, value in params.items():
                if key not in default_hyperparams:
                    raise BispectrumError(
                        'Hyperparameter "{}" not supported by this descirptor.'.format(
                            key
                        )
                    )
                else:
                    default_hyperparams[key] = value
        self.hyperparams = default_hyperparams

    def _set_cutoff(self):
        supported = ['cos']
        if self.cut_name is None:
            self.cut_name = supported[0]
        if self.cut_name not in supported:
            spd = ['"{}", '.format(s) for s in supported]
            raise BispectrumError(
                'Cutoff "{}" not supported by this descriptor. Use {}.'.format(
                    self.cut_name, spd
                )
            )

        self.cutoff = generate_full_cutoff(self.cut_dists)
        self.species_code = generate_species_code(self.cut_dists)
        num_species = len(self.species_code)

        rcutsym = np.zeros([num_species, num_species], dtype=np.double)
        for si, i in self.species_code.items():
            for sj, j in self.species_code.items():
                rcutsym[i][j] = self.cutoff[si + '-' + sj]

        self._cdesc.set_cutoff(self.cut_name, rcutsym)

    def _set_hyperparams(self):
        weight_in = self.hyperparams['weight']
        if weight_in is None:
            weight = np.ones(len(self.species_code), dtype=np.double)
        else:
            weight = np.zeros(len(self.species_code), dtype=np.double)
            for spec, code in self.species_code.items():
                try:
                    weight[code] = weight_in[spec]
                except KeyError:
                    raise BispectrumError(
                        '"weight" for species "{}" not provided.'.format(spec)
                    )
        self._cdesc.set_weight(weight)

    def get_size(self):
        """Return the size of descriptor.
        """
        diagonal = self.hyperparams['diagonalstyle']
        twojmax = int(2 * self.hyperparams['jmax'])

        N = 0
        for j1 in range(0, twojmax + 1):
            if diagonal == 2:
                N += 1
            elif diagonal == 1:
                for j in range(0, min(twojmax, 2 * j1) + 1, 2):
                    N += 1
            elif diagonal == 0:
                for j2 in range(0, j1 + 1):
                    for j in range(j1 - j2, min(twojmax, j1 + j2) + 1, 2):
                        N += 1
            elif diagonal == 3:
                for j2 in range(0, j1 + 1):
                    for j in range(j1 - j2, min(twojmax, j1 + j2) + 1, 2):
                        if j >= j1:
                            N += 1
        return N


class BispectrumError(Exception):
    def __init__(self, msg):
        super(BispectrumError, self).__init__(msg)
        self.msg = msg

    def __expr__(self):
        return self.msg
