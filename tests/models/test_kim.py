import numpy as np
import pytest
from kliff.dataset import Dataset
from kliff.models.kim import KIMComputeArguments, KIMModel

ref_energies = [-277.409737571, -275.597759276, -276.528342759, -275.482988187]


ref_forces = [
    [
        [-1.15948917e-14, -1.15948917e-14, -1.16018306e-14],
        [-1.16989751e-14, -3.93232367e-15, -3.93232367e-15],
        [-3.92538477e-15, -1.17128529e-14, -3.92538477e-15],
    ],
    [
        [1.2676956, 0.1687802, -0.83520474],
        [-1.2536342, 1.19394052, -0.75371034],
        [-0.91847129, -0.26861574, -0.49488973],
    ],
    [
        [0.18042083, -0.11940541, 0.01800594],
        [0.50030564, 0.09165797, -0.09694234],
        [-0.23992404, -0.43625564, 0.14952855],
    ],
    [
        [-1.1114163, 0.21071302, 0.55246303],
        [1.51947195, -1.21522541, 0.7844481],
        [0.54684859, 0.01018317, 0.5062204],
    ],
]


def test_main():

    # training set
    data = Dataset("./configs_extxyz/Si_4")
    configs = data.get_configs()

    # model
    modelname = "SW_StillingerWeber_1985_Si__MO_405512056662_005"
    model = KIMModel(modelname)

    # compute arguments
    compute_arguments = []
    for conf in configs:
        ca = model.create_a_kim_compute_argument()
        compute_arguments.append(
            KIMComputeArguments(
                ca,
                conf,
                supported_species=model.get_supported_species(),
                influence_distance=model.get_influence_distance(),
            )
        )
    for i, ca in enumerate(compute_arguments):
        ca.compute(model.kim_model)
        energy = ca.get_energy()
        forces = ca.get_forces()[:3]

        assert energy == pytest.approx(ref_energies[i], 1e-6)
        assert np.allclose(forces, ref_forces[i])

    # Cannot set them all by calling this function only once, because the assertion
    # depends on order
    model.set_opt_params(sigma=[["default"]])
    model.set_opt_params(A=[["default", "fix"]])
    model.set_opt_params(B=[["default"]])

    # update params
    x0 = model.get_opt_params()
    x1 = [i + 0.1 for i in x0]
    model.update_model_params(x1)

    params = model.get_kim_model_params()
    assert np.allclose(params["sigma"].value, [x1[0]])
    assert np.allclose(params["B"].value, [x1[1]])
