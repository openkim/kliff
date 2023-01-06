import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger

from kliff import parallel
from kliff.dataset import Configuration
from kliff.utils import create_directory, pickle_dump, to_path


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

        # size, mean, and stdev of fingerprints; mean and stdev will be used only when
        # `normalize=True`
        self.size = None
        self.mean = None
        self.stdev = None

    def generate_fingerprints(
        self,
        configs: List[Configuration],
        fit_forces: bool = False,
        fit_stress: bool = False,
        fingerprints_filename: Union[Path, str] = "fingerprints.pkl",
        fingerprints_mean_stdev_filename: Optional[Union[Path, str]] = None,
        use_welford_method: bool = False,
        nprocs: int = 1,
    ):
        """
        Convert all configurations to their fingerprints.

        Args:
            configs: Dataset configurations
            fit_forces: Whether to compute the gradient of fingerprints w.r.t. atomic
                coordinates so as to compute forces.
            fit_stress: Whether to compute the gradient of fingerprints w.r.t. atomic
                coordinates so as to compute stress.
            use_welford_method: Whether to compute mean and standard deviation using the
                Welford method, which is memory efficient. See
                https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
            fingerprints_filename: Path to dump fingerprints to a pickle file.
            fingerprints_mean_stdev_filename: Path to dump the mean and standard
                deviation of the fingerprints as a pickle file. If `normalize=False`
                for the descriptor, this is ignored.

            nprocs: Number of processes used to generate the fingerprints. If `1`, run
                in serial mode, otherwise `nprocs` processes will be forked via
                multiprocessing to do the work.
        """
        if self.mean is not None and self.stdev is not None:
            has_mean_stdev = True
        else:
            has_mean_stdev = False

        # compute mean and stdev
        if self.normalize and not has_mean_stdev:
            logger.info("Start computing mean and stdev of fingerprints.")

            if use_welford_method:
                logger.info(
                    "Computing fingerprint mean and stdev using the Welford method."
                )
                self.mean, self.stdev = self._welford_mean_and_stdev(configs)
                zeta, dzetadr_forces, dzetadr_stress = None, None, None

            else:
                zeta, dzetadr_forces, dzetadr_stress = self._calc_zeta_dzetadr(
                    configs, fit_forces, fit_stress, nprocs
                )
                stacked = np.concatenate(zeta)
                self.mean = np.mean(stacked, axis=0)
                self.stdev = np.std(stacked, axis=0)

            logger.info("Finish computing mean and stdev of fingerprints.")

            # save to a pickle file
            if fingerprints_mean_stdev_filename is None:
                fingerprints_mean_stdev_filename = "fingerprints_mean_and_stdev.pkl"
            state_dict = self.state_dict()
            pickle_dump(state_dict, fingerprints_mean_stdev_filename)

            logger.info(
                "Fingerprints mean and stdev saved to "
                f"`{fingerprints_mean_stdev_filename}`."
            )

        else:
            zeta, dzetadr_forces, dzetadr_stress = None, None, None

        # generate fingerprints
        self._dump_fingerprints(
            configs,
            fingerprints_filename,
            zeta,
            dzetadr_forces,
            dzetadr_stress,
            fit_forces,
            fit_stress,
        )

        return fingerprints_filename

    def _dump_fingerprints(
        self,
        configs,
        fname,
        all_zeta,
        all_dzetadr_forces,
        all_dzetadr_stress,
        fit_forces,
        fit_stress,
    ):
        """
        Dump fingerprints to a pickle file.
        """

        logger.info(f"Pickling fingerprints to `{fname}`")

        create_directory(fname, is_directory=False)

        # remove it, because we use append mode for the file below
        fname = to_path(fname)
        if fname.exists():
            fname.unlink()

        with open(fname, "ab") as f:
            for i, conf in enumerate(configs):
                if i % 100 == 0:
                    logger.info(f"Processing configuration: {i}.")

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

        logger.info(f"Pickle {len(configs)} configurations finished.")

    def _calc_zeta_dzetadr(self, configs, fit_forces, fit_stress, nprocs=1):
        """
        Calculate the fingerprints and maybe its gradients w.r.t the atomic coords.
        """
        if nprocs == 1:
            zeta = []
            dzetadr_forces = []
            dzetadr_stress = []
            for conf in configs:
                z, dzdr_f, dzdr_s = self.transform(conf, fit_forces, fit_stress)
                zeta.append(z)
                dzetadr_forces.append(dzdr_f)
                dzetadr_stress.append(dzdr_s)
        else:
            rslt = parallel.parmap1(
                self.transform, configs, fit_forces, fit_stress, nprocs=nprocs
            )
            zeta = [pair[0] for pair in rslt]
            dzetadr_forces = [pair[1] for pair in rslt]
            dzetadr_stress = [pair[2] for pair in rslt]

        return zeta, dzetadr_forces, dzetadr_stress

    def _welford_mean_and_stdev(self, configs: List[Configuration]):
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

        # number of features
        conf = configs[0]
        zeta, _, _ = self.transform(conf, fit_forces=False, fit_stress=False)
        size = zeta.shape[1]

        # Welford's method
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
                logger.info("Processing configuration: {i}.")
                sys.stdout.flush()

        stdev = np.sqrt(M2 / n)  # not the unbiased

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
        if self.mean is not None:
            size = len(self.mean)
        else:
            size = self.size
        return size

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

    def state_dict(self) -> Dict[str, Any]:
        """
        Return the state dict of the descriptor.
        """
        data = {"mean": self.mean, "stdev": self.stdev, "size": self.size}

        return data

    def load_state_dict(self, data: Dict[str, Any]):
        """
        Load state dict of a descriptor.

        Args:
            data: state dict to load.
        """
        try:
            mean = data["mean"]
            stdev = data["stdev"]
            size = data["size"]
        except Exception as e:
            raise DescriptorError(f"Corrupted state dict for descriptor: {str(e)}")

        # more checks on data integrity
        if mean is not None and stdev is not None and size is not None:
            if len(mean.shape) != 1 or mean.shape[0] != size:
                raise DescriptorError(f"Corrupted descriptor mean.")

            if len(stdev.shape) != 1 or stdev.shape[0] != size:
                raise DescriptorError("Corrupted descriptor standard deviation.")

        self.mean = mean
        self.stdev = stdev
        self.size = size


def load_fingerprints(path: Union[Path, str]):
    """
    Read preprocessed fingerprints from file.

    This is the reverse operation of Descriptor._dump_fingerprints.

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
            raise DescriptorError(f"Cannot load fingerprints from `{path}`. {str(e)}")

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
