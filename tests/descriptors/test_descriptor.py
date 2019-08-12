import numpy as np
import itertools
from kliff.dataset import Configuration
from kliff.descriptors.descriptor import Descriptor
from kliff.descriptors.descriptor import load_fingerprints
from kliff.descriptors.descriptor import DescriptorError

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


def test_descriptor():
    fname = './configs_extxyz/Si.xyz'
    conf = Configuration(format='extxyz', identifier=fname)
    conf.read(fname)
    configs = [conf, conf]

    # reuse should be the last and `True` should be after `False` so as to test reuse for
    # each case of normalize, fit_forces, and fit_stress
    for normalize, fit_forces, fit_stress, reuse in itertools.product(
        [False, True], [False, True], [False, True], [False, True]
    ):
        desc = ExampleDescriptor(normalize)
        desc.generate_train_fingerprints(configs, fit_forces, fit_stress, reuse)
        data = load_fingerprints('fingerprints/train.pkl')[0]

        if normalize:
            assert np.allclose(data['zeta'], _normalized_zeta)
            assert_mean_stdev(desc.mean, desc.stdev, _mean, _stdev)
            if fit_forces:
                assert np.allclose(data['dzetadr_forces'], _normalized_dzetadr_forces)
            if fit_stress:
                assert np.allclose(data['dzetadr_stress'], _normalized_dzetadr_stress)
        else:
            assert np.allclose(data['zeta'], _zeta)
            assert_mean_stdev(desc.mean, desc.stdev, None, None)
            if fit_forces:
                assert np.allclose(data['dzetadr_forces'], _dzetadr_forces)
            if fit_stress:
                assert np.allclose(data['dzetadr_stress'], _dzetadr_stress)

    # TODO we are planning to change generate_train_fingerprints to generate_fingerprints
    # and allow proving mean and stdev. Update below once that is done.

    ## try use generate_test_fingerprints() before genereate_train_fingerprints()
    ## case 1 (should work, since we do not require normalize)
    # desc = ExampleDescriptor(normalize=False)
    # desc.generate_test_fingerprints(configs, grad=True)
    # data = load_fingerprints('fingerprints/test.pkl')[0]
    # assert_mean_stdev(desc.mean, desc.stdev, None, None)
    # assert_zeta_dzetadr(data['zeta'], data['dzetadr_forces'], _zeta, _dzetadr_forces)

    ## case 2 (should not work, since we do need normalize)
    # desc = ExampleDescriptor(normalize=True)
    # try:
    #    desc.generate_test_fingerprints(configs, grad=True)
    # except DescriptorError as e:
    #    assert e.__class__ == DescriptorError


if __name__ == '__main__':
    test_descriptor()
