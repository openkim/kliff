import sys
import numpy as np
import multiprocessing as mp
from kliff.error import InputError
from kliff.neighbor import NeighborList
from kliff.dataset import Configuration
from kliff.descriptors.descriptor import Descriptor


from numpy import sqrt, exp
from kliff.atomic_data import atomic_number
from ase.calculators.calculator import Parameters


# class Bispectrum(object):
#    """Class that calculates spherical harmonic bispectrum fingerprints.
#
#    Parameters
#    ----------
#    cutoff : object or float
#        Cutoff function, typically from amp.descriptor.cutoffs.  Can be also
#        fed as a float representing the radius above which neighbor
#        interactions are ignored; in this case a cosine cutoff function will be
#        employed.  Default is a 6.5-Angstrom cosine cutoff.
#
#    Gs : dict
#        Dictionary of symbols and dictionaries for making fingerprints.  Either
#        auto-genetrated, or given in the following form, for example:
#
#               >>> Gs = {"Au": {"Au": 3., "O": 2.}, "O": {"Au": 5., "O": 10.}}
#
#    jmax : integer or half-integer or dict
#        Maximum degree of spherical harmonics that will be included in the
#        fingerprint vector. Can be also fed as a dictionary with chemical
#        species as keys.
#
#    dblabel : str
#        Optional separate prefix/location for database files, including
#        fingerprints, fingerprint derivatives, and neighborlists. This file
#        location can be shared between calculator instances to avoid
#        re-calculating redundant information. If not supplied, just uses the
#        value from label.
#
#    elements : list
#        List of allowed elements present in the system. If not provided, will
#        be found automatically.
#
#    """
#
#    def __init__(self, cutoff=5., Gs=None, jmax=5, dblabel=None, elements=None):
#
#        # If the cutoff is provided as a number, Cosine function will be used
#        # by default.
#        if isinstance(cutoff, int) or isinstance(cutoff, float):
#            cutoff = Cosine(cutoff)
#        # If the cutoff is provided as a dictionary, assume we need to load it
#        # with dict2cutoff.
#        if type(cutoff) is dict:
#            cutoff = dict2cutoff(cutoff)
#
#        # The parameters dictionary contains the minimum information
#        # to produce a compatible descriptor; that is, one that gives
#        # an identical fingerprint when fed an ASE image.
#        p = self.parameters = Parameters(
#            {'importname': '.descriptor.bispectrum.Bispectrum'})
#        p.cutoff = cutoff.todict()
#        p.Gs = Gs
#        p.jmax = jmax
#        p.elements = elements
#
#        self.dblabel = dblabel
#        self.parent = None  # Can hold a reference to main Amp instance.
#
#    def calculate(self, configs, nprocs=mp.cpu_count(), derivatives=False):
#        """Calculates the fingerpints of the images, for the ones not already
#        done.
#
#        Parameters
#        ----------
#        images : list or str
#            List of ASE atoms objects with positions, symbols, energies, and
#            forces in ASE format. This is the training set of data. This can
#            also be the path to an ASE trajectory (.traj) or database (.db)
#            file. Energies can be obtained from any reference, e.g. DFT
#            calculations.
#
#        nprocs: int
#            Number of processors
#
#        derivatives : bool
#            Decides whether or not fingerprintprimes should also be calculated.
#        """
#        if derivatives is True:
#            import warnings
#            warnings.warn('Zernike descriptor cannot train forces yet. '
#                          'Force training automatically turnned off. ')
#            derivatives = False
#
#        if (self.dblabel is None) and hasattr(self.parent, 'dblabel'):
#            self.dblabel = self.parent.dblabel
#        self.dblabel = 'amp-data' if self.dblabel is None else self.dblabel
#
#        p = self.parameters
#
#        if p.elements is None:
#            p.elements = set([conf.get_species() for conf in configs])
#        p.elements = sorted(p.elements)
#
#        if p.Gs is None:
#            p.Gs = generate_coefficients(p.elements)
#
#        # compute neighbor list
#        if not hasattr(self, 'neighborlist'):
#            calc = NeighborlistCalculator(cutoff=p.cutoff['kwargs']['Rc'])
#            self.neighborlist = Data(filename='%s-neighborlists'
#                                     % self.dblabel,
#                                     calculator=calc)
#        self.neighborlist.calculate_items(images, parallel=parallel, log=log)
#
#        # compute fingerprints
#        if not hasattr(self, 'fingerprints'):
#            calc = FingerprintCalculator(neighborlist=self.neighborlist,
#                                         Gs=p.Gs,
#                                         jmax=p.jmax,
#                                         cutoff=p.cutoff,)
#            self.fingerprints = Data(filename='%s-fingerprints'
#                                     % self.dblabel,
#                                     calculator=calc)
#        self.fingerprints.calculate_items(images, parallel=parallel, log=log)
#
#    def get_number_of_descriptors(self):
#        # Counts the number of descriptors for each element.
#        no_of_descriptors = {}
#        for element in p.elements:
#            count = 0
#            if isinstance(p.jmax, dict):
#                for _2j1 in range(int(2 * p.jmax[element]) + 1):
#                    for j in range(int(min(_2j1, p.jmax[element])) + 1):
#                        count += 1
#            else:
#                for _2j1 in range(int(2 * p.jmax) + 1):
#                    for j in range(int(min(_2j1, p.jmax)) + 1):
#                        count += 1
#            no_of_descriptors[element] = count
#
#        log('Number of descriptors for each element:')
#        for element in p.elements:
#            log(' %2s: %d' % (element, no_of_descriptors.pop(element)))
#

class Bispectrum(Descriptor):
    """For integration with .utilities.Data
    """

    def __init__(self, jmax=5, cutoff=None, Rc=5., *args, **kwargs):
        super(Bispectrum, self).__init__(*args, **kwargs)
        """
        Parameter
        ---------
        jmax: int

        cutoff: dict
            For example cutoff = {'Si-Si':5}

        Rc: float
        """
        if 'grad' in kwargs and kwargs['grad']:
            raise NotImplementedError(
                'Support for gradients of descriptors not implemented.')

        self.globals = Parameters({'cutoff': cutoff,
                                   'jmax': jmax})
        self.Rc = Rc

        self.factorial = [1]
        for _ in range(1, int(3. * jmax) + 2):
            self.factorial.append(_ * self.factorial[_ - 1])

        self._rcut = self.generate_full_cutoff(cutoff)

    def transform(self, conf, grad=False):
        """Makes a list of fingerprints, one per atom, for the fed config.

        Parameters
        ----------

        conf: Configuration object.

        Returns
        -------

        fingerprints: 2D array
            Each row is a fingerprints for an atom in the configuration.
        """

        cutoff = max(self._rcut.values())
        nei = NeighborList(conf, cutoff, padding_need_neigh=True)
        coords = nei.coords.reshape(-1, 3)  # coords of contribing and padding
        species = nei.species

        Gs = generate_coefficients(set(species))
        self.globals['Gs'] = Gs

        fingerprints = []
        for i in range(conf.get_number_of_atoms()):
            neighbors, _, _ = nei.get_neigh(i)
            indexfp = self.get_fingerprint(i, coords, species, neighbors)
            fingerprints.append(indexfp[1])
        return np.asarray(fingerprints), None

    def get_fingerprint(self, index, coords, species, neighbors):
        """Returns the fingerprint of bispectrum for atom
        specified by its index and symbol.

        n_symbols and Rs are lists of
        neighbors' symbols and Cartesian positions, respectively.

        Parameters
        ----------
        index: int
            Index of the center atom.

        symbol : str
            Symbol of the center atom.

        coords: coordinates of all atoms

        n_symbols : list of str
            List of neighbors' symbols.

        Rs : list of list of float
            List of Cartesian atomic positions of neighbors.

        Returns
        -------
        symbols, fingerprints : list of float
            fingerprints for atom specified by its index and symbol.
        """

        home = coords[index]
        symbol = species[index]
        n_symbols = [species[i] for i in neighbors]

        cutoff = self.globals.cutoff
        Rc = self.Rc
        cutoff = max(self._rcut.values())
        jmax = self.globals.jmax

        cutoff_fxn = Cosine(Rc)

        rs = []
        psis = []
        thetas = []
        phis = []
        for nei in neighbors:
            x = coords[nei][0] - home[0]
            y = coords[nei][1] - home[1]
            z = coords[nei][2] - home[2]
            r = np.linalg.norm(coords[nei] - home)
            if r > 10.**(-10.):

                psi = np.arcsin(r / Rc)

                theta = np.arccos(z / r)
                if abs((z / r) - 1.0) < 10.**(-8.):
                    theta = 0.0
                elif abs((z / r) + 1.0) < 10.**(-8.):
                    theta = np.pi

                if x < 0.:
                    phi = np.pi + np.arctan(y / x)
                elif 0. < x and y < 0.:
                    phi = 2 * np.pi + np.arctan(y / x)
                elif 0. < x and 0. <= y:
                    phi = np.arctan(y / x)
                elif x == 0. and 0. < y:
                    phi = 0.5 * np.pi
                elif x == 0. and y < 0.:
                    phi = 1.5 * np.pi
                else:
                    phi = 0.

                rs += [r]
                psis += [psi]
                thetas += [theta]
                phis += [phi]

        fingerprint = []
        for _2j1 in range(int(2 * jmax) + 1):
            j1 = 0.5 * _2j1
            j2 = 0.5 * _2j1
            for j in range(int(min(_2j1, jmax)) + 1):
                value = calculate_B(j1, j2, 1.0 * j, self.globals.Gs[symbol],
                                    Rc, 'Cosine',
                                    self.factorial, n_symbols,
                                    rs, psis, thetas, phis)
                value = value.real
                fingerprint.append(value)

        return symbol, fingerprint

    @staticmethod
    def generate_full_cutoff(rcut):
        """Generate a full binary cutoff dictionary.

        e.g. for input `rcut = {'C-C':1.42, 'C-H':1.0, 'H-H':0.8}`, the output would
        be `rcut = {'C-C':1.42, 'C-H':1.0, 'H-C':1.0, 'H-H':0.8}`.
        """
        rcut2 = dict()
        for key, val in rcut.items():
            spec1, spec2 = key.split('-')
            if spec1 != spec2:
                rcut2[spec2+'-'+spec1] = val
        # merge
        rcut2.update(rcut)
        return rcut2


###############################################################################
# The following auxiliary functions are from AMP https://amp.readthedocs.io
###############################################################################


def calculate_B(j1, j2, j, G_element, cutoff, cutofffn, factorial, n_symbols,
                rs, psis, thetas, phis):
    """Calculates bi-spectrum B_{j1, j2, j} according to Eq. (5) of "Gaussian
    Approximation Potentials: The Accuracy of Quantum Mechanics, without the
    Electrons", Phys. Rev. Lett. 104, 136403.
    """

    mvals = m_values(j)
    B = 0.
    for m in mvals:
        for mp in mvals:
            c = calculate_c(j, mp, m, G_element, cutoff, cutofffn, factorial,
                            n_symbols, rs, psis, thetas, phis)
            m1bound = min(j1, m + j2)
            mp1bound = min(j1, mp + j2)
            m1 = max(-j1, m - j2)
            while m1 < (m1bound + 0.5):
                mp1 = max(-j1, mp - j2)
                while mp1 < (mp1bound + 0.5):
                    c1 = calculate_c(j1, mp1, m1, G_element, cutoff, cutofffn,
                                     factorial, n_symbols, rs, psis, thetas,
                                     phis)
                    c2 = calculate_c(j2, mp - mp1, m - m1, G_element, cutoff,
                                     cutofffn, factorial, n_symbols, rs, psis,
                                     thetas, phis)
                    B += CG(j1, m1, j2, m - m1, j, m, factorial) * \
                        CG(j1, mp1, j2, mp - mp1, j, mp, factorial) * \
                        np.conjugate(c) * c1 * c2
                    mp1 += 1.
                m1 += 1.

    return B


def calculate_c(j, mp, m, G_element, cutoff, cutofffn, factorial, n_symbols,
                rs, psis, thetas, phis):
    """Calculates c^{j}_{m'm} according to Eq. (4) of "Gaussian Approximation
    Potentials: The Accuracy of Quantum Mechanics, without the Electrons",
    Phys. Rev. Lett. 104, 136403
    """

    if cutofffn is 'Cosine':
        cutoff_fxn = Cosine(cutoff)
    elif cutofffn is 'Polynomial':
        # cutoff_fxn = Polynomial(cutoff)
        raise NotImplementedError

    value = 0.
    for n_symbol, r, psi, theta, phi in zip(n_symbols, rs, psis, thetas, phis):

        value += G_element[n_symbol] * \
            np.conjugate(U(j, m, mp, psi, theta, phi, factorial)) * \
            cutoff_fxn(r)

    return value


def m_values(j):
    """Returns a list of m values for a given j."""

    assert j >= 0, '2*j should be a non-negative integer.'

    return [j - i for i in range(int(2 * j + 1))]


def binomial(n, k, factorial):
    """Returns C(n,k) = n!/(k!(n-k)!)."""

    assert n >= 0 and k >= 0 and n >= k, \
        'n and k should be non-negative integers with n >= k.'
    c = factorial[int(n)] / (factorial[int(k)] * factorial[int(n - k)])
    return c


def WignerD(j, m, mp, alpha, beta, gamma, factorial):
    """Returns the Wigner-D matrix. alpha, beta, and gamma are the Euler
    angles."""

    result = 0
    if abs(beta - np.pi / 2.) < 10.**(-10.):
        # Varshalovich Eq. (5), Section 4.16, Page 113.
        # j, m, and mp here are J, M, and M', respectively, in Eq. (5).
        for k in range(int(2 * j + 1)):
            if k > j + mp or k > j - m:
                break
            elif k < mp - m:
                continue
            result += (-1)**k * binomial(j + mp, k, factorial) * \
                binomial(j - mp, k + m - mp, factorial)

        result *= (-1)**(m - mp) * \
            sqrt(float(factorial[int(j + m)] * factorial[int(j - m)]) /
                 float((factorial[int(j + mp)] * factorial[int(j - mp)]))) / \
            2.**j
        result *= exp(-1j * m * alpha) * exp(-1j * mp * gamma)

    else:
        # Varshalovich Eq. (10), Section 4.16, Page 113.
        # m, mpp, and mp here are M, m, and M', respectively, in Eq. (10).
        mvals = m_values(j)
        for mpp in mvals:
            # temp1 = WignerD(j, m, mpp, 0, np.pi/2, 0) = d(j, m, mpp, np.pi/2)
            temp1 = 0.
            for k in range(int(2 * j + 1)):
                if k > j + mpp or k > j - m:
                    break
                elif k < mpp - m:
                    continue
                temp1 += (-1)**k * binomial(j + mpp, k, factorial) * \
                    binomial(j - mpp, k + m - mpp, factorial)
            temp1 *= (-1)**(m - mpp) * \
                sqrt(float(factorial[int(j + m)] * factorial[int(j - m)]) /
                     float((factorial[int(j + mpp)] *
                            factorial[int(j - mpp)]))) / 2.**j

            # temp2 = WignerD(j, mpp, mp, 0, np.pi/2, 0) = d(j, mpp, mp,
            # np.pi/2)
            temp2 = 0.
            for k in range(int(2 * j + 1)):
                if k > j - mp or k > j - mpp:
                    break
                elif k < - mp - mpp:
                    continue
                temp2 += (-1)**k * binomial(j - mp, k, factorial) * \
                    binomial(j + mp, k + mpp + mp, factorial)
            temp2 *= (-1)**(mpp + mp) * \
                sqrt(float(factorial[int(j + mpp)] * factorial[int(j - mpp)]) /
                     float((factorial[int(j - mp)] *
                            factorial[int(j + mp)]))) / 2.**j

            result += temp1 * exp(-1j * mpp * beta) * temp2

        # Empirical normalization factor so results match Varshalovich
        # Tables 4.3-4.12
        # Note that this exact normalization does not follow from the
        # above equations
        result *= (1j**(2 * j - m - mp)) * ((-1)**(2 * m))
        result *= exp(-1j * m * alpha) * exp(-1j * mp * gamma)

    return result


def U(j, m, mp, omega, theta, phi, factorial):
    """Calculates rotation matrix U_{MM'}^{J} in terms of rotation angle omega as
    well as rotation axis angles theta and phi, according to Varshalovich,
    Eq. (3), Section 4.5, Page 81. j, m, mp, and mpp here are J, M, M', and M''
    in Eq. (3).
    """

    result = 0.
    mvals = m_values(j)
    for mpp in mvals:
        result += WignerD(j, m, mpp, phi, theta, -phi, factorial) * \
            exp(- 1j * mpp * omega) * \
            WignerD(j, mpp, mp, phi, -theta, -phi, factorial)
    return result


def CG(a, alpha, b, beta, c, gamma, factorial):
    """Clebsch-Gordan coefficient C_{a alpha b beta}^{c gamma} is calculated
    acoording to the expression given in Varshalovich Eq. (3), Section 8.2,
    Page 238."""

    if int(2. * a) != 2. * a or int(2. * b) != 2. * b or int(2. * c) != 2. * c:
        raise ValueError("j values must be integer or half integer")
    if int(2. * alpha) != 2. * alpha or int(2. * beta) != 2. * beta or \
            int(2. * gamma) != 2. * gamma:
        raise ValueError("m values must be integer or half integer")

    if alpha + beta - gamma != 0.:
        return 0.
    else:
        minimum = min(a + b - c, a - b + c, -a + b + c, a + b + c + 1.,
                      a - abs(alpha), b - abs(beta), c - abs(gamma))
        if minimum < 0.:
            return 0.
        else:
            sqrtarg = \
                factorial[int(a + alpha)] * \
                factorial[int(a - alpha)] * \
                factorial[int(b + beta)] * \
                factorial[int(b - beta)] * \
                factorial[int(c + gamma)] * \
                factorial[int(c - gamma)] * \
                (2. * c + 1.) * \
                factorial[int(a + b - c)] * \
                factorial[int(a - b + c)] * \
                factorial[int(-a + b + c)] / \
                factorial[int(a + b + c + 1.)]

            sqrtres = sqrt(sqrtarg)

            zmin = max(a + beta - c, b - alpha - c, 0.)
            zmax = min(b + beta, a - alpha, a + b - c)
            sumres = 0.
            for z in range(int(zmin), int(zmax) + 1):
                value = \
                    factorial[int(z)] * \
                    factorial[int(a + b - c - z)] * \
                    factorial[int(a - alpha - z)] * \
                    factorial[int(b + beta - z)] * \
                    factorial[int(c - b + alpha + z)] * \
                    factorial[int(c - a - beta + z)]
                sumres += (-1.)**z / value

            result = sqrtres * sumres

            return result


def generate_coefficients(elements):
    """Automatically generates coefficients if not given by the user.

    Parameters
    ----------
    elements : list of str
        List of symbols of all atoms.

    Returns
    -------
    G : dict of dicts
    """
    _G = {}
    for element in elements:
        _G[element] = atomic_number[element]
    G = {}
    for element in elements:
        G[element] = _G
    return G


class Cosine(object):
    """Cosine functional form suggested by Behler.

    Parameters
    ---------
    Rc : float
        Radius above which neighbor interactions are ignored.
    """

    def __init__(self, Rc):

        self.Rc = Rc

    def __call__(self, Rij):
        """
        Parameters
        ----------
        Rij : float
            Distance between pair atoms.

        Returns
        -------
        float
            The value of the cutoff function.
        """
        if Rij > self.Rc:
            return 0.
        else:
            return 0.5 * (np.cos(np.pi * Rij / self.Rc) + 1.)

    def prime(self, Rij):
        """Derivative (dfc_dRij) of the Cosine cutoff function with respect to Rij.

        Parameters
        ----------
        Rij : float
            Distance between pair atoms.

        Returns
        -------
        float
            The value of derivative of the cutoff function.
        """
        if Rij > self.Rc:
            return 0.
        else:
            return -0.5 * np.pi / self.Rc * np.sin(np.pi * Rij / self.Rc)

    def todict(self):
        return {'name': 'Cosine',
                'kwargs': {'Rc': self.Rc}}

    def __repr__(self):
        return ('<Cosine cutoff with Rc=%.3f from amp.descriptor.cutoffs>'
                % self.Rc)
