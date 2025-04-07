from copy import deepcopy

import numpy as np
import pytest

from kliff.dataset import Dataset
from kliff.models.kim import KIMComputeArguments, KIMModel
from kliff.transforms.parameter_transforms import LogParameterTransform

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


def test_compute(test_data_dir):
    # model
    modelname = "SW_StillingerWeber_1985_Si__MO_405512056662_006"
    model = KIMModel(modelname)

    # training set
    data = Dataset.from_path(test_data_dir / "configs/Si_4")
    configs = data.get_configs()

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


def test_functional_call(test_data_dir):
    # model
    modelname = "SW_StillingerWeber_1985_Si__MO_405512056662_006"
    model = KIMModel(modelname)

    # training set
    data = Dataset.from_path(test_data_dir / "configs/Si_4")
    configs = data.get_configs()

    for i, conf in enumerate(configs):
        output_dict = model(conf)
        energy = output_dict["energy"]
        forces = output_dict["forces"][:3]

        assert energy == pytest.approx(ref_energies[i], 1e-6)
        assert np.allclose(forces, ref_forces[i])


def test_set_one_param():
    modelname = "SW_StillingerWeber_1985_Si__MO_405512056662_006"
    model = KIMModel(modelname)

    # parameters
    params = model.get_model_params()

    # keep the original copy
    sigma = deepcopy(params["sigma"])

    model.set_one_opt_param(name="sigma", settings=[[sigma + 0.1]])
    params = model.get_model_params()

    assert params["sigma"] == sigma + 0.1

    # internal kim params
    kim_params = model.get_kim_model_params()
    assert kim_params["sigma"][0] == sigma + 0.1


def test_set_opt_params():
    modelname = "SW_StillingerWeber_1985_Si__MO_405512056662_006"
    model = KIMModel(modelname)

    # parameters
    params = model.get_model_params()
    sigma = deepcopy(params["sigma"][0])
    A = deepcopy(params["A"][0])

    model.set_opt_params(sigma=[[sigma + 0.1]], A=[[A + 0.1]])

    assert params["sigma"][0] == sigma + 0.1
    assert params["A"][0] == A + 0.1

    # internal kim params
    kim_params = model.get_kim_model_params()
    assert kim_params["sigma"][0] == sigma + 0.1
    assert kim_params["A"][0] == A + 0.1


def test_get_update_params():
    modelname = "SW_StillingerWeber_1985_Si__MO_405512056662_006"

    model = KIMModel(modelname)

    # parameters
    params = model.get_model_params()
    sigma = deepcopy(params["sigma"])
    A = deepcopy(params["A"])

    # optimizing parameters
    # B will not be optimized, only providing initial guess
    model.set_opt_params(sigma=[["default"]], B=[["default", "fix"]], A=[["default"]])

    x0 = model.get_opt_params()

    assert x0[0] == sigma
    assert x0[1] == A
    assert len(x0) == 2
    assert model.get_num_opt_params() == 2

    x1 = [i + 0.1 for i in x0]
    model.update_model_params(x1)

    assert params["sigma"][0] == sigma + 0.1
    assert params["A"][0] == A + 0.1

    # internal kim params
    kim_params = model.get_kim_model_params()
    assert kim_params["sigma"][0] == sigma + 0.1
    assert kim_params["A"][0] == A + 0.1


def test_params_transform():
    modelname = "SW_StillingerWeber_1985_Si__MO_405512056662_006"
    model = KIMModel(modelname)

    # reference values in KIM model
    sigma = 2.0951
    A = 15.2848479197914
    B = 0.6022245584

    # Check forward transform (all in original space)
    assert model.model_params["sigma"] == sigma
    assert model.model_params["A"] == A
    assert model.model_params["B"] == B

    # Transformed params in log space
    # No log for B since it is not asked to transform
    transform = LogParameterTransform()
    model.set_params_mutable(["sigma", "A", "B"])
    params = model.named_parameters()
    params["sigma"].add_transform(transform)
    params["A"].add_transform(transform)
    # none for B

    assert params["sigma"] == np.log(sigma)
    assert params["A"] == np.log(A)
    assert params["B"] == B

    # optimizing parameters, provided in log space
    # B will not be optimized, only providing initial guess
    v1 = 2.0
    v2 = 3.0
    model.set_opt_params(sigma=[[v1]], B=[[B, "fix"]], A=[[v2]])
    transformed_params = model.named_parameters()

    assert params["sigma"] == v1
    assert params["A"] == v2
    assert params["B"] == B

    assert np.allclose(model.get_opt_params(), [v1, v2])

    model.update_model_params([v1, v2])

    # Check inverse transform
    kim_params = model.get_kim_model_params()

    # temp fix to avoid ufunc numpy errors
    assert np.abs(kim_params["sigma"] - np.exp(v1)) < 1e-12
    assert np.abs(kim_params["A"] - np.exp(v2)) < 1e-12
    assert np.abs(kim_params["B"] - B) < 1e-12
