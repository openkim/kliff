import numpy as np
import os
import pytest
from kliff.calculator import Calculator
from kliff.models import LennardJones
from kliff.error import ModelError
from kliff.dataset import DataSet


def params_relation(params):
    sigma = params.get_value('sigma')
    epsilon = params.get_value('epsilon')
    epsilon[0] = 2 * sigma[0]
    params.set_value('epsilon', epsilon)


def write_tmp_params(fname):
    with open(fname, 'w') as fout:
        fout.write('sigma\n')
        fout.write('1.1  fix\n')
        fout.write('epsilon\n')
        fout.write('2.1  None  3.\n')


def delete_tmp_params(fname):
    os.remove(fname)


def energy_forces_stress(
    calc, configs, use_energy=False, use_forces=False, use_stress=False
):

    pred_energy = -56.08345486063805
    pred_forces = [
        [2.41107532e-02, 1.29215347e-03, 2.89182619e-04],
        [-2.13103656e-02, -7.23000337e-03, -1.28954011e-02],
        [3.89036609e-04, -2.07124189e-03, -3.83185414e-01],
        [7.03144295e-04, 4.09004596e-04, -3.62012614e-01],
        [-1.84061312e-03, 6.56687434e-03, 3.62871504e-01],
        [-6.79944094e-03, 6.50195868e-03, 3.95977434e-01],
    ]
    pred_stress = [
        -4.24791582e-03,
        -4.26804209e-03,
        -4.41900170e-03,
        3.25993799e-06,
        2.05334688e-06,
        2.82003088e-06,
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

    calc.create(configs, use_energy, use_forces, use_stress)
    cas = calc.get_compute_arguments()
    ca = cas[0]
    calc.compute(ca)

    assert ca.get_compute_flag('energy') == use_energy
    assert ca.get_compute_flag('forces') == use_forces
    assert ca.get_compute_flag('stress') == use_stress

    try:
        energy = calc.get_energy(ca)
        assert energy == pytest.approx(pred_energy, 1e-6)
    except ModelError as e:
        if use_energy:
            raise ModelError(e)
        else:
            pass

    try:
        forces = calc.get_forces(ca)
        assert np.allclose(forces[:6], pred_forces)
    except ModelError as e:
        if use_forces:
            raise ModelError(e)
        else:
            pass

    try:
        stress = calc.get_stress(ca)
        assert np.allclose(stress, pred_stress)
    except ModelError as e:
        if use_stress:
            raise ModelError(e)
        else:
            pass

    pred = calc.get_prediction(ca)
    ref = calc.get_reference(ca)

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
    model = LennardJones()

    # set params directly
    model.set_fitting_params(sigma=[[1.1, 'fix']], epsilon=[[2.1, None, 3.0]])

    model.update_model_params()
    # model.echo_model_params()
    # model.echo_fitting_params()

    # set params by reading from file (the same as set params directly)
    # fname = 'tmp_lj.params'
    # write_tmp_params(fname)
    # model.read_fitting_params(fname)
    # delete_tmp_params(fname)

    calc = Calculator(model)

    dset = DataSet(order_by_species=False)
    fname = '../configs_extxyz/MoS2/MoS2_energy_forces_stress.xyz'
    dset.read(fname)
    configs = dset.get_configs()

    energy_forces_stress(calc, configs, True, False, False)
    energy_forces_stress(calc, configs, True, True, False)
    energy_forces_stress(calc, configs, True, True, True)

    # params relation callback
    model.set_params_relation_callback(params_relation)
    x0 = calc.get_opt_params()
    calc.update_opt_params(x0)
    sigma = model.get_model_params('sigma')
    epsilon = model.get_model_params('epsilon')
    assert np.allclose(sigma * 2, epsilon)


if __name__ == '__main__':
    test_lj()
