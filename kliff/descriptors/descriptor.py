import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from kliff import parallel
from kliff.dataset import Configuration
from loguru import logger


class Descriptor:
    """
    Base class of atomic environment descriptors.

    Process dataset to generate fingerprints. This is the base class for all descriptors,
    so it should not be used directly. Instead, descriptors built on top of this such as
    :class:`~kliff.descriptors.SymmetryFunction` and
    :class:`~kliff.descriptors.Bispectrum` can be used to transform the atomic environment
    information into fingerprints.

    Args:
        cut_dists: Cutoff distances, with key of the form `A-B` where `A` and `B` are
            species string, and value should be a float.
            Example: `cut_dists = {'C-C': 5.0}`
        cut_name: Name of the cutoff function, such as `cos`, `P3`, and `P7`.
        hyperparams: A dictionary of the hyperparams of the descriptor or a string to
            select the predefined hyperparams.
        normalize: If `True`, the fingerprints is centered and normalized:
            `zeta = (zeta - mean(zeta)) / stdev(zeta)`
        dtype: np.dtype
            Data type of the generated fingerprints, such as `np.float32` and `np.float64`.

    Attributes:
        size: int
            Length of the fingerprint vector.
        mean: list
            Mean of the fingerprints.
        stdev: list
            Standard deviation of the fingerprints.
    """

    def __init__(
        self,
        cut_dists: Dict[str, float],
        cut_name: str,
        hyperparams: Union[Dict, str],
        normalize: bool = True,
        dtype=np.float32,
    ):

        self.cut_dists = cut_dists
        self.cut_name = cut_name
        self.hyperparams = hyperparams
        self.normalize = normalize
        self.dtype = dtype

        self.size = None
        self.mean = None
        self.stdev = None

    def generate_fingerprints(
        self,
        configs: List[Configuration],
        fit_forces: bool = False,
        fit_stress: bool = False,
        reuse: bool = False,
        fingerprints_path: Optional[Union[Path, str]] = None,
        fingerprints_mean_and_stdev_path: Optional[Union[Path, str]] = None,
        use_welford_method: bool = False,
        nprocs: int = 1,
    ):
        """
        Convert dataset to fingerprints.

        Args:
            configs: Dataset configurations
            fit_forces: Whether to compute the gradient of fingerprints w.r.t. atomic
                coordinates so as to compute forces.
            fit_stress: Whether to compute the gradient of fingerprints w.r.t. atomic
                coordinates so as to compute stress.
            reuse: Whether to reuse the previous generated fingerprints if one is found.
            fingerprints_path: Path to fingerprints. Default to `fingerprints.pkl`.
            fingerprints_mean_and_stdev_path: Path to the mean and standard deviation of
            fingerprints. If `None`, default to `fingerpints_mean_and_stdev.pkl`.
                Only needed if `normalize = True` at instantiation.
            use_welford_method: Whether to compute mean and standard deviation using the
                Welford method, which is memory efficient. See
                https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
            nprocs: Number of processes used to generate the fingerprints. If `1`, run
                in serial mode, otherwise `nprocs` processes will be forked via
                multiprocessing to do the work.
        """

        if fingerprints_path is None:
            fname = "fingerprints.pkl"
        else:
            fname = fingerprints_path

        if fingerprints_mean_and_stdev_path is None:
            mean_stdev_name = "fingerprints_mean_and_stdev.pkl"
            mean_stdev_provided = False
        else:
            mean_stdev_name = fingerprints_mean_and_stdev_path
            mean_stdev_provided = True

        def restore_mean_and_stdev():
            self.load_mean_stdev(mean_stdev_name)
            logger.info(f"Restore mean and stdev from {mean_stdev_name}.")

        # restore data
        if os.path.exists(fname):
            msg = 'Found existing fingerprints "{}".'.format(fname)
            logger.info(msg)
            if not reuse:
                os.remove(fname)
                msg = 'Delete existing fingerprints "{}"'.format(fname)
                logger.info(msg)
            else:
                msg = "Reuse existing fingerprints."
                logger.info(msg)
                if self.normalize:
                    restore_mean_and_stdev()
                return fname

        # generate data
        msg = "Start generating fingerprints."
        logger.info(msg)

        if use_welford_method:
            all_zeta, all_dzetadr_forces, all_dzetadr_stress = None, None, None

            if self.normalize:
                if mean_stdev_provided:
                    restore_mean_and_stdev()
                else:
                    self.mean, self.stdev = self.welford_mean_and_stdev(configs)
                    self.dump_mean_stdev(mean_stdev_name)
                    msg = "Calculating mean and stdev."
                    logger.info(msg)
        else:
            all_zeta, all_dzetadr_forces, all_dzetadr_stress = self._calc_zeta_dzetadr(
                configs, fit_forces, fit_stress, nprocs
            )

            if self.normalize:
                if mean_stdev_provided:
                    restore_mean_and_stdev()
                else:
                    if all_zeta is not None:
                        stacked = np.concatenate(all_zeta)
                        self.mean = np.mean(stacked, axis=0)
                        self.stdev = np.std(stacked, axis=0)
                    else:
                        self.mean, self.stdev = self.welford_mean_and_stdev(configs)
                    self.dump_mean_stdev(mean_stdev_name)
                    msg = "Calculating mean and stdev."
                    logger.info(msg)

        self.dump_fingerprints(
            configs,
            fname,
            all_zeta,
            all_dzetadr_forces,
            all_dzetadr_stress,
            fit_forces,
            fit_stress,
        )

        msg = "Finish generating fingerprints."
        logger.info(msg)

        return fname

    def dump_fingerprints(
        self,
        configs,
        fname,
        all_zeta,
        all_dzetadr_forces,
        all_dzetadr_stress,
        fit_forces,
        fit_stress,
    ):

        msg = 'Pickling fingerprints to "{}"'.format(fname)
        logger.info(msg)

        dirname = os.path.dirname(os.path.abspath(fname))
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        with open(fname, "ab") as f:
            for i, conf in enumerate(configs):
                if i % 100 == 0:
                    msg = "Processing configuration: {}.".format(i)
                    logger.info(msg)

                if all_zeta is None:
                    zeta, dzetadr_f, dzetadr_s = self.transform(
                        conf, fit_forces, fit_stress
                    )
                else:
                    zeta = all_zeta[i]
                    dzetadr_f = all_dzetadr_forces[i]
                    dzetadr_s = all_dzetadr_stress[i]

                # centering and normalization
                if self.normalize:
                    zeta = (zeta - self.mean) / self.stdev
                    if fit_forces or fit_stress:
                        stdev_3d = np.atleast_3d(self.stdev)
                    if fit_forces:
                        dzetadr_f = dzetadr_f / stdev_3d
                    if fit_stress:
                        dzetadr_s = dzetadr_s / stdev_3d

                # pickling data
                zeta = np.asarray(zeta, self.dtype)
                energy = np.asarray(conf.energy, self.dtype)
                if fit_forces:
                    dzetadr_f = np.asarray(dzetadr_f, self.dtype)
                    forces = np.asarray(conf.forces, self.dtype)
                if fit_stress:
                    dzetadr_s = np.asarray(dzetadr_s, self.dtype)
                    stress = np.asarray(conf.stress, self.dtype)
                    volume = np.asarray(conf.get_volume(), self.dtype)

                example = {"configuration": conf, "zeta": zeta, "energy": energy}
                if fit_forces:
                    example["dzetadr_forces"] = dzetadr_f
                    example["forces"] = forces
                if fit_stress:
                    example["dzetadr_stress"] = dzetadr_s
                    example["stress"] = stress
                    example["volume"] = volume

                pickle.dump(example, f)

        msg = "Pickle {} configurations finished.".format(len(configs))
        logger.info(msg)

    def _calc_zeta_dzetadr(self, configs, fit_forces, fit_stress, nprocs=1):
        """
        Calculate the fingerprints and maybe its gradients w.r.t the atomic coords.
        """
        try:
            rslt = parallel.parmap1(
                self.transform, configs, fit_forces, fit_stress, nprocs=nprocs
            )
            zeta = [pair[0] for pair in rslt]
            dzetadr_forces = [pair[1] for pair in rslt]
            dzetadr_stress = [pair[2] for pair in rslt]
        except MemoryError as e:
            msg = (
                f"{e}. MemoryError occurs in calculating fingerprints using parallel. "
                "Fallback to use a serial version."
            )
            logger.info(msg)

            zeta = None
            dzetadr_forces = None
            dzetadr_stress = None
        return zeta, dzetadr_forces, dzetadr_stress

    def welford_mean_and_stdev(self, configs: List[Configuration]):
        """
        Compute the mean and standard deviation of fingerprints.

        This running mean and standard method proposed by Welford is memory-efficient.
        Besides, it outperforms the naive method from suffering numerical instability for
        large dataset.

        Args:
            configs: Dataset class:`~kliff.dataset.Configuration`.

        See Also:
            https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        """
        msg = "Welford method to calculate mean and standard deviation."
        logger.info(msg)

        # number of features
        conf = configs[0]
        zeta, _, _ = self.transform(conf, fit_forces=False, fit_stress=False)
        size = zeta.shape[1]

        # starting Welford's method
        n = 0
        mean = np.zeros(size)
        M2 = np.zeros(size)
        for i, conf in enumerate(configs):
            zeta, _, _ = self.transform(conf, fit_forces=False, fit_stress=False)
            for row in zeta:
                n += 1
                delta = row - mean
                mean += delta / n
                delta2 = row - mean
                M2 += delta * delta2

            if i % 100 == 0:
                msg = "Processing configuration: {}.".format(i)
                logger.info(msg)
                sys.stdout.flush()

        stdev = np.sqrt(M2 / n)  # not the unbiased

        msg = "Finish mean and standard deviation calculation using Welford method."
        logger.info(msg)

        return mean, stdev

    def transform(
        self, conf: Configuration, fit_forces: bool = False, fit_stress: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Transform atomic coords to atomic environment descriptor values.

        Args:
            conf: atomic configuration
            fit_forces: Whether to fit forces, so as to compute gradients of fingerprints
                w.r.t. coords
            fit_stress: Whether to fit stress, so as to compute gradients of fingerprints
                w.r.t. coords

        Returns:
            zeta: Descriptor values. 2D array with shape (num_atoms, num_descriptors),
                where num_atoms is the number of atoms in the configuration, and
                num_descriptors is the size of the descriptor vector (depending on the
                choice of the hyperparameters).
            dzeta_dr: Gradient of the descriptor w.r.t. atomic coordinates. 4D array if
                grad is `True`, otherwise `None`. Shape: (num_atoms, num_descriptors,
                num_atoms, 3), where num_atoms and num_descriptors has the same meanings
                as described in zeta, and 3 denotes the 3D space for the Cartesian
                coordinates.
            dzeta_ds: Gradient of the descriptor w.r.t. virial stress component. 2D
                array of shape (num_atoms, num_descriptors, 6), where num_atoms and
                num_descriptors has the same meanings as described in zeta,
                and 6 denote the virial stress component in Voigt notation, see
                https://en.wikipedia.org/wiki/Voigt_notation
        """
        raise NotImplementedError

    def write_kim_params(
        self, path: Union[Path, str], fname: str = "descriptor.params"
    ):
        """
        Write descriptor info for KIM model.

        Args:
            path: Directory Path to write the file.
            fname: Name of the file.
        """
        raise NotImplementedError

    def get_size(self):
        """Return the size of the descriptor vector."""
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
        """Return the name and values of cutoff."""
        return self.cut_name, self.cut_dists

    def get_hyperparams(self):
        """Return the hyperparameters of descriptors."""
        return self.hyperparams

    def dump_mean_stdev(self, path):
        """
        Save mean and stdev info to a file.
        """
        dirname = os.path.dirname(os.path.abspath(path))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        data = {"mean": self.mean, "stdev": self.stdev}
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load_mean_stdev(self, path):
        """
        Load mean and stdev info from a file.
        """
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
                mean = data["mean"]
                stdev = data["stdev"]
        except Exception as e:
            msg = 'Cannot load mean and standard data from "{}". {}'.format(
                path, str(e)
            )
            raise DescriptorError(msg)

        if len(mean.shape) != 1 or mean.shape[0] != self.get_size():
            msg = 'Corrupted mean data from "{}".'.format(path)
            raise DescriptorError(msg)
        if len(stdev.shape) != 1 or stdev.shape[0] != self.get_size():
            msg = 'Corrupted standard deviation data from "{}".'.format(path)
            raise DescriptorError(msg)

        self.mean, self.stdev = mean, stdev

        return mean, stdev


def load_fingerprints(path: Union[Path, str]):
    """
    Read preprocessed fingerprints from file.

    This is the reverse operation of Descriptor.dump_fingerprints.

    Args:
        path: Path to the pickled data file.

    Returns:
        Fingerprints
    """
    data = []
    with open(path, "rb") as f:
        try:
            while True:
                x = pickle.load(f)
                data.append(x)
        except EOFError:
            pass
        except Exception as e:
            msg = 'Cannot fingerprints from "{}". {}'.format(path, str(e))
            raise DescriptorError(msg)

    return data


def generate_full_cutoff(cutoff):
    """
    Generate a full binary cutoff dictionary.

    For species pair `S1-S2` in the ``cutoff`` dictionary, add key `S2-S1` to it, with
    the same value as `S1-S2`.

    Args:
        cutoff: Cutoff dictionary with key of the form ``A-B`` where ``A`` and ``B``
            are atomic species, and value should be a float.

    Returns:
        A dictionary with all combination of species as keys.

    Example:
    >>> cutoff = {'C-C': 4.0, 'C-H':3.5}
    >>> generate_full_cutoff(cutoff)
        {'C-C': 4.0, 'C-H':3.5, 'H-C':3.5}
    """
    rcut2 = dict()
    for key, val in cutoff.items():
        s1, s2 = key.split("-")
        if s1 != s2:
            reverse_key = s2 + "-" + s1
            if reverse_key in cutoff and cutoff[reverse_key] != val:
                raise Exception(
                    'Corrupted cutoff dictionary. cutoff["{0}-{1}"] != '
                    'cutoff["{1}-{0}"].'.format(s1, s2)
                )
            else:
                rcut2[reverse_key] = val
    # merge
    rcut2.update(cutoff)

    return rcut2


def generate_unique_cutoff_pairs(cutoff):
    """
    Generate a full binary cutoff dictionary.

    For species pair `S1-S2` in the ``cutoff`` dictionary, remove key `S2-S1` from it if
    `S1` is different from `S2`.

    Args:
        cutoff: Cutoff dictionary with key of the form ``A-B`` where ``A`` and ``B``
            are atomic species, and value should be a float.

    Returns:
        A dictionary with unique species pair as keys.

    Example:
    >>> cutoff = {'C-C': 4.0, 'C-H':3.5, 'H-C':3.5}
    >>> generate_unique_cutoff_pairs(cutoff)
        {'C-C': 4.0, 'C-H':3.5}
    """
    rcut2 = dict()
    for key, val in cutoff.items():
        s1, s2 = key.split("-")
        reverse_key = s2 + "-" + s1
        if key not in rcut2 and reverse_key not in rcut2:
            rcut2[key] = val

    return rcut2


def generate_species_code(cutoff):
    """
    Generate species code info from cutoff dictionary.

    Args:
        cutoff: Cutoff dictionary with key of the form ``A-B`` where ``A`` and ``B``
            are atomic species, and value should be a float.

    Returns:
        species_code: A dictionary of species and the integer code (starting from 0),
            with keys the species in ``cutoff`` keys, and values integer code for species.

    Example:
    >>> cutoff = {'C-C': 4.0, 'C-H':3.5}
    >>> generate_species_code(cutoff)
        {'C':0, 'H':1}
    """
    species = set()
    for key in cutoff:
        s1, s2 = key.split("-")
        species.update([s1, s2])
    species = list(species)
    species_code = {s: i for i, s in enumerate(species)}
    return species_code


class DescriptorError(Exception):
    def __init__(self, msg):
        super(DescriptorError, self).__init__(msg)
        self.msg = msg
