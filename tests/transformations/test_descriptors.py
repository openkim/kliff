import numpy as np
import pytest

from kliff.dataset import Configuration
from kliff.legacy.descriptors import Bispectrum, SymmetryFunction
from kliff.transforms.configuration_transforms.descriptors import (
    Descriptor,
    bispectrum_default,
    symmetry_functions_set30,
    symmetry_functions_set51,
)


def test_symmetry_functions(test_data_dir):
    config = Configuration.from_file(
        test_data_dir / "configs" / "Si_4" / "Si_T300_step_0.xyz"
    )

    # Forward pass
    # initialize kliff older desc
    desc_30 = SymmetryFunction(
        cut_dists={"Si-Si": 4.5},
        cut_name="cos",
        hyperparams=symmetry_functions_set30(),
        normalize=False,
    )
    descriptor_old_30 = desc_30.transform(config, fit_forces=True, fit_stress=False)

    # initialize kliff new desc
    libdesc_30 = Descriptor(
        cutoff=4.5,
        species=["Si"],
        descriptor="SymmetryFunctions",
        hyperparameters=symmetry_functions_set30(),
    )
    descriptor_new_30 = libdesc_30.forward(config)

    assert np.allclose(descriptor_old_30[0], descriptor_new_30)

    # Backward pass
    libdesc_jvp_30 = libdesc_30.backward(config, np.ones_like(descriptor_new_30))
    desc_jvp_30 = np.tensordot(
        np.ones_like(descriptor_old_30[0]), descriptor_old_30[1], ([0, 1], [0, 1])
    ).reshape(-1, 3)
    assert np.allclose(desc_jvp_30, libdesc_jvp_30)

    # Set 51
    desc_51 = SymmetryFunction(
        cut_dists={"Si-Si": 4.5},
        cut_name="cos",
        hyperparams=symmetry_functions_set51(),
        normalize=False,
    )
    descriptor_old_51 = desc_51.transform(config, fit_forces=True, fit_stress=False)

    # initialize kliff new desc
    libdesc_51 = Descriptor(
        cutoff=4.5,
        species=["Si"],
        descriptor="SymmetryFunctions",
        hyperparameters=symmetry_functions_set51(),
    )
    descriptor_new_51 = libdesc_51.forward(config)

    assert np.allclose(descriptor_old_51[0], descriptor_new_51)

    # Backward pass
    libdesc_jvp_51 = libdesc_51.backward(config, np.ones_like(descriptor_new_51))
    desc_jvp_51 = np.tensordot(
        np.ones_like(descriptor_old_51[0]), descriptor_old_51[1], ([0, 1], [0, 1])
    ).reshape(-1, 3)
    assert np.allclose(desc_jvp_51, libdesc_jvp_51)


def test_bispectrum(test_data_dir):
    config = Configuration.from_file(
        test_data_dir / "configs" / "Si_4" / "Si_T300_step_0.xyz"
    )

    # Forward pass
    # initialize kliff older desc
    hyperparams = {
        "jmax": 4,
        "rfac0": 0.99363,
        "diagonalstyle": 3,
        "rmin0": 0,
        "switch_flag": 1,
        "bzero_flag": 0,
    }

    desc = Bispectrum(cut_dists={"Si-Si": 4.5}, cut_name="cos", hyperparams=hyperparams)

    descriptor_old = desc.transform(config, grad=False)

    # initialize kliff new desc
    libdesc = Descriptor(
        cutoff=4.5,
        species=["Si"],
        descriptor="Bispectrum",
        hyperparameters=bispectrum_default(),
    )
    descriptor_new = libdesc.forward(config)

    assert np.allclose(descriptor_old[0], descriptor_new)
    # Current Bispectrum implementation has wrong grads. So, we can't compare grads


def test_implicit_copying(test_data_dir):
    config = Configuration.from_file(
        test_data_dir / "configs" / "Si_4" / "Si_T300_step_0.xyz"
    )

    libdesc = Descriptor(
        cutoff=4.5,
        species=["Si"],
        descriptor="SymmetryFunctions",
        hyperparameters=symmetry_functions_set30(),
        copy_to_config=True,
    )
    desc = libdesc(config)
    assert np.allclose(desc, libdesc.forward(config))
    assert desc is config.fingerprint
