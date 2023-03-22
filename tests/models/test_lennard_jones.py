import os
import warnings
from pathlib import Path

import numpy as np
import pytest

from kliff.dataset import Configuration
from kliff.models.lennard_jones import LennardJones, LJComputeArguments


def write_tmp_params(fname):
    with open(fname, "w") as fout:
        fout.write("sigma\n")
        fout.write("1.1  fix\n")
        fout.write("epsilon\n")
        fout.write("2.1  None  3.\n")


def delete_tmp_params(fname):
    os.remove(fname)


def energy_forces_stress(
    model, config, use_energy=False, use_forces=False, use_stress=False
):
    pred_energy = -53.14480791226465
    pred_forces = [
        [2.39376242e-02, 1.28483509e-03, 3.31975197e-04],
        [-1.88510118e-02, -3.10094633e-03, -1.50128135e-02],
        [2.08312503e-04, -2.23804534e-03, -3.67359255e-01],
        [8.84446196e-04, 1.70668544e-04, -3.46205093e-01],
        [-1.79631714e-03, 6.38953162e-03, 3.47102707e-01],
        [-9.07755780e-03, 1.05143562e-02, 3.81969022e-01],
    ]

    pred_stress = [
        3.99807132e-03,
        4.02212566e-03,
        4.23288249e-03,
        -3.59750079e-06,
        -1.35922778e-06,
        -3.71951100e-06,
    ]

    ref_energy = -5.302666
    ref_forces = [
        [-0.425324, 0.295866, -0.065479],
        [0.245043, -0.061658, 0.264104],
        [0.010127, 0.041539, 0.301571],
        [0.079468, -0.072558, -0.340646],
        [0.126296, -0.152711, 0.313083],
        [0.224514, -0.233065, -0.737724],
    ]
    ref_stress = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]

    ca = LJComputeArguments(
        config,
        supported_species=model.supported_species,
        influence_distance=model.get_influence_distance(),
        compute_energy=use_energy,
        compute_forces=use_forces,
        compute_stress=use_stress,
    )

    assert ca.get_compute_flag("energy") == use_energy
    assert ca.get_compute_flag("forces") == use_forces
    assert ca.get_compute_flag("stress") == use_stress

    ca.compute(model.get_model_params())

    energy = ca.get_energy()
    if use_energy:
        assert energy == pytest.approx(pred_energy, 1e-6)
    else:
        assert energy == None

    forces = ca.get_forces()
    if use_forces:
        assert np.allclose(forces[:6], pred_forces)
    else:
        assert forces == None

    stress = ca.get_stress()
    if use_stress:
        assert np.allclose(stress, pred_stress)
    else:
        assert stress == None

    pred = ca.get_prediction()
    ref = ca.get_reference()

    if use_energy:
        assert pred[0] == pytest.approx(pred_energy, 1e-6)
        assert ref[0] == pytest.approx(ref_energy, 1e-6)
    if use_forces:
        if use_energy:
            assert np.allclose(pred[1 : 1 + 3 * 6], np.ravel(pred_forces))
            assert np.allclose(ref[1 : 1 + 3 * 6], np.ravel(ref_forces))
        else:
            assert np.allclose(pred[: 3 * 6], np.ravel(pred_forces))
            assert np.allclose(ref[: 3 * 6], np.ravel(ref_forces))

    if use_stress:
        assert np.allclose(pred[-6:], pred_stress)
        assert np.allclose(ref[-6:], ref_stress)


def test_lj():
    model = LennardJones(species=["Mo", "S"])

    # set optimizing parameters
    model.set_opt_params(
        sigma=[[1.1, "fix"], [1.1], [1.1]],
        epsilon=[[2.1, None, 3.0], [2.1], [2.1]],
        cutoff=[[4.0, "fix"], [4.0, "fix"], [4.0, "fix"]],
    )

    model.echo_model_params()
    model.echo_opt_params()

    with warnings.catch_warnings():  # context manager to ignore warning
        warnings.simplefilter("ignore")

        # set model_params by reading from file (the same as set model_params directly)
        # filename = "tmp_lj_params.txt"
        # write_tmp_params(filename)
        # model.read_opt_params(filename)
        # delete_tmp_params(filename)

        model.echo_model_params()
        model.echo_opt_params()

    path = (
        Path(__file__)
        .parents[1]
        .joinpath("configs_extxyz/MoS2/MoS2_energy_forces_stress.xyz")
    )
    config = Configuration.from_file(path)

    # energy_forces_stress(model, config, True, False, False)
    # energy_forces_stress(model, config, True, True, False)
    energy_forces_stress(model, config, True, True, True)
