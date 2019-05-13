import numpy as np
from kliff.dataset import Configuration
from kliff.descriptors.descriptor import Descriptor
from kliff.descriptors.descriptor import load_fingerprints
from kliff.descriptors.descriptor import DescriptorError

# make up some data
num_atoms = 4
num_desc = 5
dim = 3
_zeta = np.arange(num_atoms * num_desc).reshape(num_atoms, num_desc)
_dzeta_dr = np.arange(num_atoms * num_desc * num_atoms * dim)
_dzeta_dr = _dzeta_dr.reshape(num_atoms, num_desc, num_atoms * dim)
_mean = np.mean(_zeta, axis=0)
_stdev = np.std(_zeta, axis=0)
_normalized_zeta = (_zeta - _mean) / _stdev
_normalized_dzeta_dr = _dzeta_dr / np.atleast_3d(_stdev)


def assert_mean_stdev(mean, stdev, target_mean, target_stdev):
    if mean is None:
        assert target_mean is None
    else:
        assert np.allclose(mean, target_mean)
    if stdev is None:
        assert target_stdev is None
    else:
        assert np.allclose(stdev, target_stdev)


def assert_zeta_dzeta_dr(zeta, dzeta_dr, target_zeta, target_dzeta_dr):
    assert np.allclose(zeta, target_zeta)
    if dzeta_dr is None:
        assert target_dzeta_dr is None
    else:
        assert np.allclose(dzeta_dr, target_dzeta_dr)


class ExampleDescriptor(Descriptor):
    def __init__(self, normalize):
        cutvalues = None
        cutname = None
        hyperparams = None
        super(ExampleDescriptor, self).__init__(
            cutvalues, cutname, hyperparams, normalize
        )

    def transform(self, conf, grad):
        if grad:
            return _zeta, _dzeta_dr
        else:
            return _zeta, None


def test_descriptor():
    fname = '../configs_extxyz/Si.xyz'
    conf = Configuration(format='extxyz', identifier=fname)
    conf.read(fname)
    configs = [conf, conf]

    # case 1
    desc = ExampleDescriptor(normalize=False)
    grad = False
    # train set
    desc.generate_train_fingerprints(configs, grad=grad)
    data = load_fingerprints('fingerprints/train.pkl')[0]
    assert_mean_stdev(desc.mean, desc.stdev, None, None)
    assert_zeta_dzeta_dr(data['zeta'], None, _zeta, None)
    # test set
    desc.generate_test_fingerprints(configs, grad=grad)
    data = load_fingerprints('fingerprints/test.pkl')[0]
    assert_mean_stdev(desc.mean, desc.stdev, None, None)
    assert_zeta_dzeta_dr(data['zeta'], None, _zeta, None)

    # case 2
    desc = ExampleDescriptor(normalize=False)
    grad = True
    # train set
    desc.generate_train_fingerprints(configs, grad=grad)
    data = load_fingerprints('fingerprints/train.pkl')[0]
    assert_mean_stdev(desc.mean, desc.stdev, None, None)
    assert_zeta_dzeta_dr(data['zeta'], data['dzeta_dr'], _zeta, _dzeta_dr)
    # test set
    desc.generate_test_fingerprints(configs, grad=grad)
    data = load_fingerprints('fingerprints/test.pkl')[0]
    assert_mean_stdev(desc.mean, desc.stdev, None, None)
    assert_zeta_dzeta_dr(data['zeta'], data['dzeta_dr'], _zeta, _dzeta_dr)

    # case 3
    desc = ExampleDescriptor(normalize=True)
    grad = False
    # train set
    desc.generate_train_fingerprints(configs, grad=grad)
    data = load_fingerprints('fingerprints/train.pkl')[0]
    assert_mean_stdev(desc.mean, desc.stdev, _mean, _stdev)
    assert_zeta_dzeta_dr(data['zeta'], None, _normalized_zeta, None)
    # test set
    desc.generate_test_fingerprints(configs, grad=grad)
    data = load_fingerprints('fingerprints/test.pkl')[0]
    assert_mean_stdev(desc.mean, desc.stdev, _mean, _stdev)
    assert_zeta_dzeta_dr(data['zeta'], None, _normalized_zeta, None)

    # case 4
    desc = ExampleDescriptor(normalize=True)
    grad = True
    # train set
    desc.generate_train_fingerprints(configs, grad=grad)
    data = load_fingerprints('fingerprints/train.pkl')[0]
    assert_mean_stdev(desc.mean, desc.stdev, _mean, _stdev)
    assert_zeta_dzeta_dr(
        data['zeta'], data['dzeta_dr'], _normalized_zeta, _normalized_dzeta_dr
    )
    # test set
    desc.generate_test_fingerprints(configs, grad=grad)
    data = load_fingerprints('fingerprints/test.pkl')[0]
    assert_mean_stdev(desc.mean, desc.stdev, _mean, _stdev)
    assert_zeta_dzeta_dr(
        data['zeta'], data['dzeta_dr'], _normalized_zeta, _normalized_dzeta_dr
    )

    # case 5 allow reuse
    desc = ExampleDescriptor(normalize=True)
    grad = True
    # train set
    desc.generate_train_fingerprints(configs, grad=grad, reuse=True)
    data = load_fingerprints('fingerprints/train.pkl')[0]
    assert_mean_stdev(desc.mean, desc.stdev, _mean, _stdev)
    assert_zeta_dzeta_dr(
        data['zeta'], data['dzeta_dr'], _normalized_zeta, _normalized_dzeta_dr
    )
    # test set
    desc.generate_test_fingerprints(configs, grad=grad, reuse=True)
    data = load_fingerprints('fingerprints/test.pkl')[0]
    assert_mean_stdev(desc.mean, desc.stdev, _mean, _stdev)
    assert_zeta_dzeta_dr(
        data['zeta'], data['dzeta_dr'], _normalized_zeta, _normalized_dzeta_dr
    )

    # try use generate_test_fingerprints() before genereate_train_fingerprints()
    # case 1 (should work, since we do not require normalize)
    desc = ExampleDescriptor(normalize=False)
    desc.generate_test_fingerprints(configs, grad=True)
    data = load_fingerprints('fingerprints/test.pkl')[0]
    assert_mean_stdev(desc.mean, desc.stdev, None, None)
    assert_zeta_dzeta_dr(data['zeta'], data['dzeta_dr'], _zeta, _dzeta_dr)

    # case 2 (should not work, since we do need normalize)
    desc = ExampleDescriptor(normalize=True)
    try:
        desc.generate_test_fingerprints(configs, grad=True)
    except DescriptorError as e:
        assert e.__class__ == DescriptorError


if __name__ == '__main__':
    test_descriptor()
