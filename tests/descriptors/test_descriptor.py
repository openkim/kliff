import itertools
import os

import numpy as np

from kliff.dataset import Configuration
from kliff.descriptors.descriptor import Descriptor, DescriptorError, load_fingerprints

# make up some data
num_atoms = 4
num_desc = 5
dim = 3
_zeta = np.arange(num_atoms * num_desc).reshape(num_atoms, num_desc)
_dzetadr_forces = np.arange(num_atoms * num_desc * num_atoms * dim).reshape(
    num_atoms, num_desc, num_atoms * dim
)
_dzetadr_stress = np.arange(num_atoms * num_desc * 6).reshape(num_atoms, num_desc, 6)

_mean = np.mean(_zeta, axis=0)
_stdev = np.std(_zeta, axis=0)
_normalized_zeta = (_zeta - _mean) / _stdev
_normalized_dzetadr_forces = _dzetadr_forces / np.atleast_3d(_stdev)
_normalized_dzetadr_stress = _dzetadr_stress / np.atleast_3d(_stdev)


def delete_file(fname):
    if os.path.exists(fname):
        os.remove(fname)


def assert_mean_stdev(mean, stdev, target_mean, target_stdev):
    if mean is None:
        assert target_mean is None
    else:
        assert np.allclose(mean, target_mean)
    if stdev is None:
        assert target_stdev is None
    else:
        assert np.allclose(stdev, target_stdev)


def assert_zeta_dzetadr(
    zeta, dzetadr_f, dzetadr_s, target_zeta, target_dzetadr_f, target_dzetadr_s
):
    assert np.allclose(zeta, target_zeta)

    if dzetadr_f is None:
        assert target_dzetadr_f is None
    else:
        assert np.allclose(dzetadr_f, target_dzetadr_f)

    if dzetadr_s is None:
        assert target_dzetadr_s is None
    else:
        assert np.allclose(dzetadr_s, target_dzetadr_s)


class ExampleDescriptor(Descriptor):
    def __init__(self, normalize):
        cutvalues = None
        cutname = None
        hyperparams = None
        super(ExampleDescriptor, self).__init__(
            cutvalues, cutname, hyperparams, normalize
        )

    def transform(self, conf, fit_forces=False, fit_stress=False):
        zeta = _zeta
        if fit_forces:
            dzetadr_forces = _dzetadr_forces
        else:
            dzetadr_forces = None
        if fit_stress:
            dzetadr_stress = _dzetadr_stress
        else:
            dzetadr_stress = None
        return zeta, dzetadr_forces, dzetadr_stress

    def get_size(self):
        return num_desc


def test_descriptor():
    fname = "./configs_extxyz/Si.xyz"
    conf = Configuration(format="extxyz", identifier=fname)
    conf.read(fname)
    configs = [conf, conf]

    # reuse should be the last and `True` should be after `False` so as to test reuse for
    # each case of normalize, fit_forces, and fit_stress
    for normalize, fit_forces, fit_stress, serial, reuse in itertools.product(
        [False, True], [False, True], [False, True], [False, True], [False, True]
    ):
        desc = ExampleDescriptor(normalize)
        desc.generate_fingerprints(
            configs, fit_forces, fit_stress, reuse=reuse, serial=serial
        )
        data = load_fingerprints("fingerprints.pkl")[0]

        if normalize:
            assert_mean_stdev(desc.mean, desc.stdev, _mean, _stdev)
            assert np.allclose(data["zeta"], _normalized_zeta)
            if fit_forces:
                assert np.allclose(data["dzetadr_forces"], _normalized_dzetadr_forces)
            if fit_stress:
                assert np.allclose(data["dzetadr_stress"], _normalized_dzetadr_stress)
        else:
            assert_mean_stdev(desc.mean, desc.stdev, None, None)
            assert np.allclose(data["zeta"], _zeta)
            if fit_forces:
                assert np.allclose(data["dzetadr_forces"], _dzetadr_forces)
            if fit_stress:
                assert np.allclose(data["dzetadr_stress"], _dzetadr_stress)

    # check when normalize is True, if mean and stdev is provided by user, it has to be
    # correct.
    for normalize, fp_path, mean_std_path in itertools.product(
        [False, True], [None, "fp.pkl"], [None, "ms.pkl"]
    ):

        delete_file("fingerprints.pkl")
        delete_file("fingerprints_mean_and_stdev.pkl")
        delete_file("fp.pkl")
        delete_file("ms.pkl")

        desc = ExampleDescriptor(normalize)

        if normalize and mean_std_path is not None:
            # will raise exception because the file mean_std_path does not exist
            try:
                desc.generate_fingerprints(
                    configs,
                    fingerprints_path=fp_path,
                    fingerprints_mean_and_stdev_path=mean_std_path,
                )
            except DescriptorError:
                pass

        else:
            desc.generate_fingerprints(
                configs,
                fingerprints_path=fp_path,
                fingerprints_mean_and_stdev_path=mean_std_path,
            )

            if fp_path is None:
                fp_path = "fingerprints.pkl"
            data = load_fingerprints(fp_path)[0]

            if normalize:
                assert_mean_stdev(desc.mean, desc.stdev, _mean, _stdev)
                assert np.allclose(data["zeta"], _normalized_zeta)
            else:
                assert_mean_stdev(desc.mean, desc.stdev, None, None)
                assert np.allclose(data["zeta"], _zeta)


if __name__ == "__main__":
    test_descriptor()
