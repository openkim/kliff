import logging
import os
from collections import OrderedDict

import numpy as np

import kliff

from ...log import log_entry
from ...neighbor import NeighborList
from ..descriptor import (
    Descriptor,
    generate_full_cutoff,
    generate_species_code,
    generate_unique_cutoff_pairs,
)
from . import sf

logger = logging.getLogger(__name__)


class SymmetryFunction(Descriptor):
    r"""Atom-centered symmetry functions descriptor as discussed in [Behler2011]_.

    Parameters
    ----------
    cut_dists: dict
        Cutoff distances, with key of the form ``A-B`` where ``A`` and ``B`` are
        atomic species string, and value should be a float.

    cut_name: str
        Name of the cutoff function.

    hyperparams: dict or str
        A dictionary of the hyper parameters of that define the descriptor. We provide two
        sets of hyperparams that can be used by setting ``hyperparams='set51'`` or
        ``hyperparams='set30'``, which are taken from [Artrith2012]_ and [Artrith2013]_,
        respectively. To see what they are, one can do:

        >>> cut_name = 'cos'  # just for init purpose
        >>> cut_dists = {'C-C': 5.}  # just for init purpose
        >>> hyperparams = 'set51'
        >>> desc = SymmetryFunction(cut_dists, cut_name, hyperparams)
        >>> desc.get_hyperparams()

    normalize: bool (optional)
        If ``True``, the fingerprints is centered and normalized according to:
        ``zeta = (zeta - mean(zeta)) / stdev(zeta)``

    dtype: np.dtype (optional)
        Data type for the generated fingerprints, such as ``np.float32`` and
        ``np.float64``.

    Example
    -------

    If ``set51`` or ``set30`` hyperparams are used, the cutoff distances should be
    given in ``Angstrom``.

    >>> cut_name = 'cos'
    >>> cut_dists = {'C-C': 5., 'C-H': 4.5, 'H-H': 4.0}
    >>> hyperparams = 'set51'
    >>> desc = SymmetryFunction(cut_dists, cut_name, hyperparams)

    You can provide your own hyperparams as a dictionary:

    >>> cut_name = 'cos'
    >>> cut_dists = {'C-C': 5., 'C-H': 4.5, 'H-H': 4.0}
    >>> hyperparams = {'g1': None,
    >>>                'g2': [{'eta':0.1, 'Rs':0.2}, {'eta':0.3, 'Rs':0.4}],
    >>>                'g3': [{'kappa':0.1}, {'kappa':0.2}, {'kappa':0.3}]}
    >>> desc = SymmetryFunction(cut_dists, cut_name, hyperparams)

    References
    ----------
    .. [Behler2011] J. Behler, "Atom-centered symmetry functions for constructing
       high-dimensional neural network potentials," J. Chem. Phys. 134, 074106
       (2011).
    .. [Artrith2012] N. Artrith and J. Behler. "High-dimensional neural network
       potentials for metal surfaces: A prototype study for copper." Physical Review
       B 85, no. 4 (2012): 045439.
    .. [Artrith2013] N. Artrith, B. Hiller, and J. Behler. "Neural network potentials
       for metals and oxides–First applications to copper clusters at zinc oxide."
       physica status solidi (b) 250, no. 6 (2013): 1191-1203.
    """

    def __init__(
        self, cut_dists, cut_name, hyperparams, normalize=True, dtype=np.float32
    ):
        super(SymmetryFunction, self).__init__(
            cut_dists, cut_name, hyperparams, normalize, dtype
        )

        self._desc = OrderedDict()

        self._cdesc = sf.Descriptor()
        self._set_cutoff()
        self._set_hyperparams()
        self.size = self.get_size()

        msg = '"{}" descriptor initialized.'.format(self.__class__.__name__)
        log_entry(logger, msg, level="info")

    def transform(self, conf, fit_forces=False, fit_stress=False):
        r"""Transform atomic coords to atomic environment descriptor values.

        Parameters
        ----------
        conf: :class:`~kliff.dataset.Configuration` object
            A configuration of atoms.


        fit_forces: bool (optional)
            Whether to compute the gradient of descriptor values w.r.t. atomic
            coordinates so as to compute forces.

        fit_stress: bool (optional)
            Whether to compute the gradient of descriptor values w.r.t. atomic
            coordinates so as to compute stress.

        Returns
        -------
        zeta: 2D array
            Descriptor values, each row for one atom.
            zeta has shape (num_atoms, num_descriptors), where num_atoms is the
            number of atoms in the configuration, and num_descriptors is the size
            of the descriptor vector (depending on the the choice of hyper-parameters).

        dzetadr_forces: 3D array if fit_forces is ``True``, otherwise ``None``
            Gradient of descriptor values w.r.t. atomic coordinates for forces
            computation.
            dzetadr_forces has shape (num_atoms, num_descriptors, num_atoms*DIM), where
            num_atoms and num_descriptors has the same meanings as described in zeta.
            DIM = 3 denotes three Cartesian coordinates.

        dzetadr_stress: 3D array if fit_stress is ``True``, otherwise ``None``
            Gradient of descriptor values w.r.t. atomic coordinates for stress computation.
            dzetadr_stress has shape (num_atoms, num_descriptors, 6), where
            num_atoms and num_descriptors has the same meanings as described in zeta.
            The last dimension is the 6 component associated with virial stress in the
            order of 11, 22, 33, 23, 31, 12.
        """

        # create neighbor list
        infl_dist = max(self.cutoff.values())
        nei = NeighborList(conf, infl_dist, padding_need_neigh=False)

        coords = nei.coords
        image = nei.image
        species = np.asarray([self.species_code[i] for i in nei.species], dtype=np.intc)

        Ncontrib = conf.get_number_of_atoms()
        Ndesc = len(self)

        grad = fit_forces or fit_stress

        zeta_config = []
        dzetadr_forces_config = []
        dzetadr_stress_config = []

        for i in range(Ncontrib):
            neigh_indices, _, _ = nei.get_neigh(i)
            neighlist = np.asarray(neigh_indices, dtype=np.intc)
            zeta, dzetadr = self._cdesc.generate_one_atom(
                i, coords, species, neighlist, grad
            )

            zeta_config.append(zeta)

            if grad:
                # last 3 elements dzetadr is associated with atom i
                atom_ids = np.concatenate((neigh_indices, [i]))
                dzetadr = dzetadr.reshape(Ndesc, -1, 3)

            if fit_forces:
                dzetadr_forces = np.zeros((Ndesc, Ncontrib, 3))
                for ii, idx in enumerate(atom_ids):
                    org_idx = image[idx]
                    dzetadr_forces[:, org_idx, :] += dzetadr[:, ii, :]
                dzetadr_forces_config.append(dzetadr_forces.reshape(Ndesc, -1))

            if fit_stress:
                dzetadr_stress = np.zeros((Ndesc, 6))
                for ii, idx in enumerate(atom_ids):
                    dzetadr_stress[:, 0] += dzetadr[:, ii, 0] * coords[idx][0]
                    dzetadr_stress[:, 1] += dzetadr[:, ii, 1] * coords[idx][1]
                    dzetadr_stress[:, 2] += dzetadr[:, ii, 2] * coords[idx][2]
                    dzetadr_stress[:, 3] += dzetadr[:, ii, 1] * coords[idx][2]
                    dzetadr_stress[:, 4] += dzetadr[:, ii, 2] * coords[idx][0]
                    dzetadr_stress[:, 5] += dzetadr[:, ii, 0] * coords[idx][1]
                dzetadr_stress_config.append(dzetadr_stress)

        zeta_config = np.asarray(zeta_config)
        if fit_forces:
            dzetadr_forces_config = np.asarray(dzetadr_forces_config)
        else:
            dzetadr_forces_config = None
        if fit_stress:
            dzetadr_stress_config = np.asarray(dzetadr_stress_config)
        else:
            dzetadr_stress_config = None

        if logger.getEffectiveLevel() == logging.DEBUG:
            msg = (
                "=" * 25
                + "descriptor values (no normalization)"
                + "=" * 25
                + "\nconfiguration name: {}".format(conf.get_identifier())
                + "\natom id    descriptor values ..."
            )
            log_entry(logger, msg, level="debug")

            for i, line in enumerate(zeta_config):
                s = "\n{}    ".format(i)
                for j in line:
                    s += "{:.15g} ".format(j)
                log_entry(logger, s, level="debug")

        return zeta_config, dzetadr_forces_config, dzetadr_stress_config

    def _set_cutoff(self):
        supported = ["cos"]
        if self.cut_name is None:
            self.cut_name = supported[0]
        if self.cut_name not in supported:
            spd = ['"{}", '.format(s) for s in supported]
            raise SymmetryFunctionError(
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
                rcutsym[i][j] = self.cutoff[si + "-" + sj]
        self._cdesc.set_cutoff(self.cut_name, rcutsym)

    def _set_hyperparams(self):
        if isinstance(self.hyperparams, str):
            name = self.hyperparams.lower()
            if name == "set51":
                self.hyperparams = get_set51()
            elif name == "set30":
                self.hyperparams = get_set30()
            else:
                raise SymmetryFunctionError(
                    'hyperparams "{}" unrecognized.'.format(name)
                )
        if not isinstance(self.hyperparams, OrderedDict):
            self.hyperparams = OrderedDict(self.hyperparams)

        # hyperparams of descriptors
        for key, values in self.hyperparams.items():
            if key.lower() not in ["g1", "g2", "g3", "g4", "g5"]:
                raise SymmetryFunctionError(
                    'Symmetry function "{}" unrecognized.'.format(key)
                )

            # g1 needs no hyperparams, put a placeholder
            name = key.lower()
            if name == "g1":
                # it has no hyperparams, zeros([1,1]) for placeholder
                params = np.zeros([1, 1], dtype=np.double)
            else:
                rows = len(values)
                cols = len(values[0])
                params = np.zeros([rows, cols], dtype=np.double)
                for i, line in enumerate(values):
                    if name == "g2":
                        params[i][0] = line["eta"]
                        params[i][1] = line["Rs"]
                    elif name == "g3":
                        params[i][0] = line["kappa"]
                    elif key == "g4":
                        params[i][0] = line["zeta"]
                        params[i][1] = line["lambda"]
                        params[i][2] = line["eta"]
                    elif key == "g5":
                        params[i][0] = line["zeta"]
                        params[i][1] = line["lambda"]
                        params[i][2] = line["eta"]

            # store cutoff values in both this python and cpp class
            self._desc[name] = params
            self._cdesc.add_descriptor(name, params)

    def write_kim_params(self, path, fname="descriptor.params"):

        with open(os.path.join(path, fname), "w") as fout:

            if self.dtype == np.float64:
                fmt = "{:.15e} "
            else:
                fmt = "{:.7e} "

            # header
            fout.write("#" + "=" * 80 + "\n")
            fout.write("# Descriptor parameters file generated by KLIFF.\n")
            fout.write("#" + "=" * 80 + "\n\n")

            # cutoff and species
            cutname, rcut = self.get_cutoff()
            unique_pairs = generate_unique_cutoff_pairs(rcut)
            species = generate_species_code(rcut)

            fout.write("{}  # cutoff type\n\n".format(cutname))
            fout.write("{}  # number of species\n\n".format(len(species)))
            fout.write("# species 1    species 2    cutoff\n")
            for key, value in unique_pairs.items():
                s1, s2 = key.split("-")
                fout.write(("{}  {}  " + fmt + "\n").format(s1, s2, value))
            fout.write("\n")

            #
            # symmetry functions
            #

            # header
            fout.write("#" + "=" * 80 + "\n")
            fout.write("# symmetry functions\n")
            fout.write("#" + "=" * 80 + "\n\n")

            desc = self.get_hyperparams()
            num_desc = len(desc)
            fout.write("{}  # number of symmetry functions types\n\n".format(num_desc))

            # descriptor values
            fout.write("# sym_function    rows    cols\n")
            for name, values in desc.items():
                if name == "g1":
                    fout.write("g1\n\n")
                else:
                    rows = len(values)
                    cols = len(values[0])
                    fout.write("{}    {}    {}\n".format(name, rows, cols))
                    if name == "g2":
                        for val in values:
                            fout.write((fmt * 2).format(val[0], val[1]))
                            fout.write("    # eta  Rs\n")
                        fout.write("\n")
                    elif name == "g3":
                        for val in values:
                            fout.write((fmt).format(val[0]))
                            fout.write("    # kappa\n")
                        fout.write("\n")
                    elif name == "g4":
                        for val in values:
                            zeta = val[0]
                            lam = val[1]
                            eta = val[2]
                            fout.write((fmt * 3).format(zeta, lam, eta))
                            fout.write("    # zeta  lambda  eta\n")
                        fout.write("\n")
                    elif name == "g5":
                        for val in values:
                            zeta = val[0]
                            lam = val[1]
                            eta = val[2]
                            fout.write((fmt * 3).format(zeta, lam, eta))
                            fout.write("    # zeta  lambda  eta\n")
                        fout.write("\n")

            #
            # data centering and normalization
            #

            # header
            fout.write("#" + "=" * 80 + "\n")
            fout.write("# Preprocessing data to center and normalize\n")
            fout.write("#" + "=" * 80 + "\n")

            # mean and stdev
            mean = self.get_mean()
            stdev = self.get_stdev()
            if mean is None and stdev is None:
                fout.write("center_and_normalize  False\n")
            else:
                fout.write("center_and_normalize  True\n\n")

                fout.write("{}   # descriptor size\n".format(self.get_size()))

                fout.write("# mean\n")
                for i in mean:
                    fout.write((fmt + "\n").format(i))
                fout.write("\n# standard deviation\n")
                for i in stdev:
                    fout.write((fmt + "\n").format(i))
                fout.write("\n")

    def get_size(self):
        return len(self)

    def get_hyperparams(self):
        return self._desc

    def __len__(self):
        N = 0
        for key in self._desc:
            N += len(self._desc[key])
        return N


def get_set51():
    r"""Hyperparameters for symmetry functions, as discussed in:
    Nongnuch Artrith and Jorg Behler. "High-dimensional neural network potentials for
    metal surfaces: A prototype study for copper." Physical Review B 85, no. 4 (2012):
    045439.
    """

    params = OrderedDict()

    params["g2"] = [
        {"eta": 0.001, "Rs": 0.0},
        {"eta": 0.01, "Rs": 0.0},
        {"eta": 0.02, "Rs": 0.0},
        {"eta": 0.035, "Rs": 0.0},
        {"eta": 0.06, "Rs": 0.0},
        {"eta": 0.1, "Rs": 0.0},
        {"eta": 0.2, "Rs": 0.0},
        {"eta": 0.4, "Rs": 0.0},
    ]

    params["g4"] = [
        {"zeta": 1, "lambda": -1, "eta": 0.0001},
        {"zeta": 1, "lambda": 1, "eta": 0.0001},
        {"zeta": 2, "lambda": -1, "eta": 0.0001},
        {"zeta": 2, "lambda": 1, "eta": 0.0001},
        {"zeta": 1, "lambda": -1, "eta": 0.003},
        {"zeta": 1, "lambda": 1, "eta": 0.003},
        {"zeta": 2, "lambda": -1, "eta": 0.003},
        {"zeta": 2, "lambda": 1, "eta": 0.003},
        {"zeta": 1, "lambda": -1, "eta": 0.008},
        {"zeta": 1, "lambda": 1, "eta": 0.008},
        {"zeta": 2, "lambda": -1, "eta": 0.008},
        {"zeta": 2, "lambda": 1, "eta": 0.008},
        {"zeta": 1, "lambda": -1, "eta": 0.015},
        {"zeta": 1, "lambda": 1, "eta": 0.015},
        {"zeta": 2, "lambda": -1, "eta": 0.015},
        {"zeta": 2, "lambda": 1, "eta": 0.015},
        {"zeta": 4, "lambda": -1, "eta": 0.015},
        {"zeta": 4, "lambda": 1, "eta": 0.015},
        {"zeta": 16, "lambda": -1, "eta": 0.015},
        {"zeta": 16, "lambda": 1, "eta": 0.015},
        {"zeta": 1, "lambda": -1, "eta": 0.025},
        {"zeta": 1, "lambda": 1, "eta": 0.025},
        {"zeta": 2, "lambda": -1, "eta": 0.025},
        {"zeta": 2, "lambda": 1, "eta": 0.025},
        {"zeta": 4, "lambda": -1, "eta": 0.025},
        {"zeta": 4, "lambda": 1, "eta": 0.025},
        {"zeta": 16, "lambda": -1, "eta": 0.025},
        {"zeta": 16, "lambda": 1, "eta": 0.025},
        {"zeta": 1, "lambda": -1, "eta": 0.045},
        {"zeta": 1, "lambda": 1, "eta": 0.045},
        {"zeta": 2, "lambda": -1, "eta": 0.045},
        {"zeta": 2, "lambda": 1, "eta": 0.045},
        {"zeta": 4, "lambda": -1, "eta": 0.045},
        {"zeta": 4, "lambda": 1, "eta": 0.045},
        {"zeta": 16, "lambda": -1, "eta": 0.045},
        {"zeta": 16, "lambda": 1, "eta": 0.045},
        {"zeta": 1, "lambda": -1, "eta": 0.08},
        {"zeta": 1, "lambda": 1, "eta": 0.08},
        {"zeta": 2, "lambda": -1, "eta": 0.08},
        {"zeta": 2, "lambda": 1, "eta": 0.08},
        {"zeta": 4, "lambda": -1, "eta": 0.08},
        {"zeta": 4, "lambda": 1, "eta": 0.08},
        # {'zeta':16,  'lambda':-1,  'eta':0.08 },
        {"zeta": 16, "lambda": 1, "eta": 0.08},
    ]

    # transfer units from bohr to angstrom
    bhor2ang = 0.529177
    for key, values in params.items():
        for val in values:
            if key == "g2":
                val["eta"] /= bhor2ang ** 2
            elif key == "g4":
                val["eta"] /= bhor2ang ** 2

    return params


def get_set30():
    r"""Hyperparameters for symmetry functions, as discussed in:
    Artrith, N., Hiller, B. and Behler, J., 2013. Neural network potentials for metals and
    oxides–First applications to copper clusters at zinc oxide. physica status solidi (b),
    250(6), pp.1191-1203.
    """

    params = OrderedDict()

    params["g2"] = [
        {"eta": 0.0009, "Rs": 0.0},
        {"eta": 0.01, "Rs": 0.0},
        {"eta": 0.02, "Rs": 0.0},
        {"eta": 0.035, "Rs": 0.0},
        {"eta": 0.06, "Rs": 0.0},
        {"eta": 0.1, "Rs": 0.0},
        {"eta": 0.2, "Rs": 0.0},
        {"eta": 0.4, "Rs": 0.0},
    ]

    params["g4"] = [
        {"zeta": 1, "lambda": -1, "eta": 0.0001},
        {"zeta": 1, "lambda": 1, "eta": 0.0001},
        {"zeta": 2, "lambda": -1, "eta": 0.0001},
        {"zeta": 2, "lambda": 1, "eta": 0.0001},
        {"zeta": 1, "lambda": -1, "eta": 0.003},
        {"zeta": 1, "lambda": 1, "eta": 0.003},
        {"zeta": 2, "lambda": -1, "eta": 0.003},
        {"zeta": 2, "lambda": 1, "eta": 0.003},
        {"zeta": 1, "lambda": 1, "eta": 0.008},
        {"zeta": 2, "lambda": 1, "eta": 0.008},
        {"zeta": 1, "lambda": 1, "eta": 0.015},
        {"zeta": 2, "lambda": 1, "eta": 0.015},
        {"zeta": 4, "lambda": 1, "eta": 0.015},
        {"zeta": 16, "lambda": 1, "eta": 0.015},
        {"zeta": 1, "lambda": 1, "eta": 0.025},
        {"zeta": 2, "lambda": 1, "eta": 0.025},
        {"zeta": 4, "lambda": 1, "eta": 0.025},
        {"zeta": 16, "lambda": 1, "eta": 0.025},
        {"zeta": 1, "lambda": 1, "eta": 0.045},
        {"zeta": 2, "lambda": 1, "eta": 0.045},
        {"zeta": 4, "lambda": 1, "eta": 0.045},
        {"zeta": 16, "lambda": 1, "eta": 0.045},
    ]

    # transfer units from bohr to angstrom
    bhor2ang = 0.529177
    for key, values in params.items():
        for val in values:
            if key == "g2":
                val["eta"] /= bhor2ang ** 2
            elif key == "g4":
                val["eta"] /= bhor2ang ** 2

    return params


class SymmetryFunctionError(Exception):
    def __init__(self, msg):
        super(SymmetryFunctionError, self).__init__(msg)
        self.msg = msg

    def __expr__(self):
        return self.msg
