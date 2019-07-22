import os
import sys
import numpy as np
from ..dataset import write_config
from ..utils import split_string

import kliff


logger = kliff.logger.get_logger(__name__)


class energy_forces_RMSE:
    def __init__(self, calculator, energy=True, forces=True):
        self.calculator = calculator
        self.compute_energy = energy
        self.compute_forces = forces

    def run(self, normalize=True, verbose=1, sort=None, path=None):
        """Run the RMSE analyzer.

        Parameters
        ----------
        normalize: bool
            Whether to normalize the energy (forces) by the number of atoms in a
            configuration.

        verbose: int (optional)
            Verbose level of the output info. Available values are: 0, 1, 2.

        sort: str (optional)
            Sort per configuration information according to `energy` or `forces`.
            If `None`, no sort. This works only when per configuration information is
            requested, i.e. ``verbose > 0``.

        path: str (optional)
            Path to write out the results. If `None`, write to stdout, otherwise, write to
            the file specified by `path`.

        """
        cas = self.calculator.get_compute_arguments()

        all_enorm = []
        all_fnorm = []
        all_identifier = []
        for ca in cas:
            enorm, fnorm = self.compute_single_config(ca, normalize, verbose)
            all_enorm.append(enorm)
            all_fnorm.append(fnorm)
            all_identifier.append(ca.conf.get_identifier())
        all_enorm = np.asarray(all_enorm)
        all_fnorm = np.asarray(all_fnorm)
        all_identifier = np.asarray(all_identifier)

        if sort == 'energy':
            if self.compute_energy:
                order = all_enorm.argsort()
                all_enorm = all_enorm[order]
                all_fnorm = all_fnorm[order]
                all_identifier = all_identifier[order]
        elif sort == 'forces':
            if self.compute_forces:
                order = all_fnorm.argsort()
                all_enorm = all_enorm[order]
                all_fnorm = all_fnorm[order]
                all_identifier = all_identifier[order]
        # else silently ignore

        if path is not None:
            fout = open(path, 'w')
        else:
            fout = sys.stdout

        # header
        print('#' * 80, file=fout)
        print('#', file=fout)
        print('# Root-mean-square errors for energy and forces', file=fout)
        print('#', file=fout)
        if normalize:
            msg = (
                'Values reported is per atom quantify, e.g. "eV/atom" for energy and '
                '"(eV/Angstrom)/atom" if "eV" is the units for energy and "Angstrom" '
                'is the units for forces.'
            )
            print(split_string(msg, length=80, starter='#'), file=fout)
            print('#', file=fout)
        print('#' * 80 + '\n', file=fout)

        if verbose >= 1:

            # header
            print('#' * 80, file=fout)
            print('Per configuration quantify\n', file=fout)
            print('# config     ', end=' ', file=fout)
            if self.compute_energy:
                print('energy RMSE     ', end=' ', file=fout)
            if self.compute_forces:
                print('forces RMSE     ', end=' ', file=fout)
            print('config identifier', file=fout)

            for i, (enorm, fnorm, identifier) in enumerate(
                zip(all_enorm, all_fnorm, all_identifier)
            ):
                print('{:<10d}  '.format(i), end='', file=fout)
                if self.compute_energy:
                    print('{:.10e}  '.format(enorm), end='', file=fout)
                if self.compute_forces:
                    print('{:.10e}  '.format(fnorm), end='', file=fout)
                print(identifier, file=fout)
            print('\n', file=fout)

        print('#' * 80, file=fout)
        print('RMSE for the dataset (all configurations).', file=fout)
        if self.compute_energy:
            e_rmse = np.linalg.norm(all_enorm) / len(all_enorm) ** 0.5
            print('{:.10e}    # energy RMSE'.format(e_rmse), file=fout)
        if self.compute_forces:
            f_rmse = np.linalg.norm(all_fnorm) / len(all_fnorm) ** 0.5
            print('{:.10e}    # forces RMSE'.format(f_rmse), file=fout)
        print('\n', file=fout)

    def compute_single_config(self, ca, normalize, verbose):

        self.calculator.compute(ca)
        conf = ca.conf
        identifier = conf.get_identifier()
        natoms = conf.get_number_of_atoms()

        if self.compute_energy:
            pred_e = self.calculator.get_energy(ca)
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
            ref_f = conf.get_forces()
            fdiff = pred_f - ref_f
            fnorm = np.linalg.norm(fdiff)
            if normalize:
                fnorm /= natoms
        else:
            fdiff = None
            fnorm = None

        # write the difference to extxyz files
        if verbose >= 2:
            path = os.path.join(
                'energy_forces_RMSE-difference', os.path.basename(identifier)
            )
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
                fmt='extxyz',
            )

        return enorm, fnorm


class AnalyzerError(Exception):
    def __init__(self, msg):
        super(AnalyzerError, self).__init__(msg)
        self.msg = msg

    def __expr__(self):
        return self.msg
