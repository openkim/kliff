import numpy as np

from kliff.calculators import Calculator
from kliff.dataset import Dataset
from kliff.loss import Loss
from kliff.models import KIM


def residual_fn(identifier, natoms, weight, prediction, reference, data):
    assert len(prediction) == 3 * natoms + 1
    return weight * (prediction - reference)


def init():
    model = KIM(model_name="SW_StillingerWeber_1985_Si__MO_405512056662_005")

    # Cannot set them all by calling this function only once, because the assertion
    # depends on order
    model.set_fitting_params(A=[[5.0]])
    model.set_fitting_params(B=[["default"]])
    model.set_fitting_params(sigma=[[2.0951, "fix"]])
    model.set_fitting_params(gamma=[[1.5]])

    dataset_name = "./configs_extxyz/Si_4"
    tset = Dataset()
    tset.read(dataset_name)
    configs = tset.get_configs()

    calc = Calculator(model)
    calc.create(configs, use_energy=True, use_forces=True)

    loss = Loss(calc, residual_fn=residual_fn, nprocs=1)

    return loss


def optimize(method, reference):
    loss = init()
    steps = 3
    result = loss.minimize(method, options={"disp": False, "maxiter": steps})
    assert np.allclose(result.x, reference)


def least_squares(method, reference):
    loss = init()
    steps = 3
    result = loss.minimize(method, max_nfev=steps, verbose=0)
    assert np.allclose(result.x, reference)


def test_lbfgsb():
    optimize("L-BFGS-B", [4.81469314, 1.52682165, 1.50169278])


def test_bfgs():
    optimize("BFGS", [4.33953474, 1.53585151, 2.36509375])


def test_cg():
    optimize("CG", [4.54044991, 1.562142, 5.16457532])


def test_powell():
    optimize("Powell", [0.09323494, 1.11123083, 2.10211015])


def test_lm():
    least_squares("lm", [6.42665872, 1.81078954, 1.93466249])


def test_trf():
    least_squares("trf", [6.42575061, 1.54254653, 2.13551637])


def test_dogbox():
    least_squares("dogbox", [6.42575107, 1.54254652, 2.13551639])


if __name__ == "__main__":
    test_lbfgsb()
    test_bfgs()
    test_cg()
    test_powell()
    test_lm()
    test_trf()
    test_dogbox()
