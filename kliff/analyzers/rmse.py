import logging
import os
import sys
from collections.abc import Iterable

import numpy as np

from ..dataset import write_config
from ..log import log_entry
from ..utils import split_string

logger = logging.getLogger(__name__)


class EnergyForcesRMSE:
    r"""Analyzer to compute the root-mean-square error (RMSE) for energy and forces.

    The `energy difference norm` for a configuration is defined as:

    .. math::
        e_\text{norm} = |e_\text{pred} - e_\text{ref}| / N,

    where :math:`e_\text{pred}` is the prediction of the total energy from the model,
    :math:`e_\text{ref}` is the corresponding reference energy, and :math:`N` is the
    number of atoms in the configuration. The division by :math:`N` is applied only when
    ``normalize = True`` in ``run``.
    Similarly, the `forces difference norm` for a configuration is defined as:

    .. math::
        f_\text{norm} = || \bm f_\text{pred} - \bm f_\text{ref}|| / N,

    where :math:`f_\text{pred}` is the prediction of the forces on atoms from the model
    and :math:`f_\text{ref}` is the corresponding reference forces, :math:`N` is the
    number of atoms in the configuration. The division by :math:`N` is applied only when
    ``normalize = True`` in ``run``.

    The RMSEs for energy and forces are defined as:

    .. math::
        e_\text{RMSE} = \sqrt{ \frac{\sum_{m=1}^M e_\text{norm}^2}{M}}

    and

    .. math::
        f_\text{RMSE} = \sqrt{ \frac{\sum_{m=1}^M f_\text{norm}^2}{M}},

    in which :math:`M` is the total number of configurations in the dataset.
    """

    def __init__(self, calculator, energy=True, forces=True):
        self.calculator = calculator
        self.compute_energy = energy
        self.compute_forces = forces

    def run(self, normalize=True, sort=None, path=None, verbose=1):
        r"""Run the RMSE analyzer.

        Parameters
        ----------
        normalize: bool
            Whether to normalize the energy (forces) by the number of atoms in a
            configuration.

        sort: str (optional)
            Sort per configuration information according to `energy` or `forces`.
            If `None`, no sort. This works only when per configuration information is
            requested, i.e. ``verbose > 0``.

        path: str (optional)
            Path to write out the results. If `None`, write to stdout, otherwise, write to
            the file specified by `path`.
            Note, if ``verbose==3``, the difference of energy and forces will be written
            to a directory named `energy_forces_RMSE-difference`.

        verbose: int (optional)
            Verbose level of the output info. Available values are: 0, 1, 2.
            If ``verbose=0``, only output the energy and forces RMSEs for the dataset.
            If ``verbose==1``, output the norms of the energy and forces for each
            configuration additionally.
            If ``verbose==2``, output the difference of the energy and forces for each
            atom, and the information is written to extended XYZ files with the location
            specified by ``path``.
        """

        msg = "Start analyzing energy and forces RMSE."
        log_entry(logger, msg, level="info")

        cas = self.calculator.get_compute_arguments()

        all_enorm = []
        all_fnorm = []
        all_identifier = []

        # common path of dataset
        ids = [_get_config(ca).get_identifier() for ca in cas]
        common = _get_common_path(ids)

        for i, ca in enumerate(cas):
            if i % 100 == 0:
                msg = "Processing configuration {}.".format(i)
                log_entry(logger, msg, level="info")
            prefix = "analysis_energy_forces_RMSE-difference"
            enorm, fnorm = self._compute_single_config(
                ca, normalize, verbose, common, prefix
            )
            all_enorm.append(enorm)
            all_fnorm.append(fnorm)
            all_identifier.append(_get_config(ca).get_identifier())
        all_enorm = np.asarray(all_enorm)
        all_fnorm = np.asarray(all_fnorm)
        all_identifier = np.asarray(all_identifier)

        if sort == "energy":
            if self.compute_energy:
                order = all_enorm.argsort()
                all_enorm = all_enorm[order]
                all_fnorm = all_fnorm[order]
                all_identifier = all_identifier[order]
        elif sort == "forces":
            if self.compute_forces:
                order = all_fnorm.argsort()
                all_enorm = all_enorm[order]
                all_fnorm = all_fnorm[order]
                all_identifier = all_identifier[order]

        if path is not None:
            fout = open(path, "w")
        else:
            fout = sys.stdout

        # header
        print("#" * 80, file=fout)
        print("#", file=fout)
        print("# Root-mean-square errors for energy and forces", file=fout)
        print("#", file=fout)
        msg = (
            'Values reported is per atom quantify if "normalize=True". For example, '
            '"eV/atom" for energy and "(eV/Angstrom)/atom" if "eV" is the units for '
            'energy and "Angstrom" is the units for forces.'
        )
        print(split_string(msg, length=80, starter="#"), file=fout)
        print("#", file=fout)
        print(
            "# See (TODO insert url of doc) for the meaning of the reported values.",
            file=fout,
        )
        print("#" * 80 + "\n", file=fout)

        # norms of each config
        if verbose >= 1:
            print("#" * 80, file=fout)
            print("Per configuration quantify\n", file=fout)
            print("# config", end=" " * 4, file=fout)
            if self.compute_energy:
                print("energy difference norm", end=" " * 4, file=fout)
            if self.compute_forces:
                print("forces difference norm", end=" " * 4, file=fout)
            print("config identifier", file=fout)

            for i, (enorm, fnorm, identifier) in enumerate(
                zip(all_enorm, all_fnorm, all_identifier)
            ):
                print("{:<10d}".format(i), end=" " * 4, file=fout)
                if self.compute_energy:
                    print("{:.10e}".format(enorm), end=" " * 10, file=fout)
                if self.compute_forces:
                    print("{:.10e}".format(fnorm), end=" " * 10, file=fout)
                print(identifier, file=fout)
            print("\n", file=fout)

        # RMSE of all configs
        print("#" * 80, file=fout)
        print("RMSE for the dataset (all configurations).", file=fout)
        if self.compute_energy:
            e_rmse = np.linalg.norm(all_enorm) / len(all_enorm) ** 0.5
            print("{:.10e}    # energy RMSE".format(e_rmse), file=fout)
        if self.compute_forces:
            f_rmse = np.linalg.norm(all_fnorm) / len(all_fnorm) ** 0.5
            print("{:.10e}    # forces RMSE".format(f_rmse), file=fout)
        print("\n", file=fout)

        # difference of each atom
        if verbose >= 2:
            print("#" * 80, file=fout)
            msg = (
                "The differences of energy and forces are written to the directory "
                '"energy_forces_RMSE-difference" in extended XYZ format.'
            )
            print(split_string(msg, length=80, starter="#"), file=fout)
            print("\n", file=fout)

        msg = "Finish analyzing energy and forces RMSE."
        log_entry(logger, msg, level="info")

    def _compute_single_config(self, ca, normalize, verbose, common_path, prefix):

        self.calculator.compute(ca)
        conf = _get_config(ca)
        identifier = os.path.abspath(conf.get_identifier())
        natoms = conf.get_number_of_atoms()

        if self.compute_energy:
            pred_e = self.calculator.get_energy(ca)
            pred_e = _to_numpy(pred_e, ca)
            ref_e = conf.get_energy()
            ediff = pred_e - ref_e
            enorm = abs(ediff)
            if normalize:
                enorm /= natoms
        else:
            ediff = None
            enorm = None

        if self.compute_forces:
            pred_f = self.calculator.get_forces(ca)
            pred_f = _to_numpy(pred_f, ca).reshape(-1, 3)
            ref_f = conf.get_forces().reshape(-1, 3)
            fdiff = pred_f - ref_f
            fnorm = np.linalg.norm(fdiff)
            if normalize:
                fnorm /= natoms
        else:
            fdiff = None
            fnorm = None

        # write the difference to extxyz files
        if verbose >= 2:
            if identifier.startswith(common_path):
                base = identifier[len(common_path) :]
            else:
                raise AnalyzerError(
                    'identifier "{}" not start with common_path "{}".'.format(
                        identifier, common_path
                    )
                )

            path = os.path.join(prefix, base)
            cell = conf.get_cell()
            PBC = conf.get_PBC()
            species = conf.get_species()
            coords = conf.get_coordinates()
            write_config(
                path,
                cell,
                PBC,
                species,
                coords,
                energy=ediff,
                forces=fdiff,
                stress=None,
                fmt="extxyz",
            )

        return enorm, fnorm


def _get_config(compute_argument):
    """Get the configuration attached to a compute argument.

    For KIM model and Torch model, the way is different. It would be better to unify these
    two. The method here is very vulnerable.
    """
    if isinstance(compute_argument, Iterable):
        # compute argument from Torch dataset; [0] because it is a batch of 1 element
        conf = compute_argument[0]["configuration"]
    else:
        # For KIM and built-in models, it is a compute argument class
        conf = compute_argument.conf

    return conf


def _to_numpy(x, compute_argument):
    """Convert to a numpy array from a tensor.

    `compute_argument` is needed to determine whether ``x`` is a list of tensor of a numpy
    array.
    """
    if isinstance(compute_argument, Iterable):
        return x[0].detach().numpy()
    else:
        return x


def _get_common_path(paths):
    """Find the common path of a list of paths.

    For example, given paths = ['/A/B/c.x', '/A/B/D/e.x'], the returns `/A/B/`.
    """
    paths = [os.path.abspath(p) for p in paths]
    common = ""

    i = 0
    while True:
        if i < len(paths[0]):
            c = paths[0][i]
        else:
            break
        not_same = False
        for p in paths:
            if not p[i] == c:
                not_same = True
                break
        if not_same:
            break
        common += c
        i += 1

    return common


class AnalyzerError(Exception):
    def __init__(self, msg):
        super(AnalyzerError, self).__init__(msg)
        self.msg = msg

    def __expr__(self):
        return self.msg
