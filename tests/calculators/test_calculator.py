from pathlib import Path

import numpy as np
import pytest
from kliff.calculators import Calculator
from kliff.dataset import Dataset
from kliff.models import KIMModel

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


class TestCalculator:
    def test_compute(self):
        test_file_path = Path(__file__).parents[1].joinpath("configs_extxyz")
        tset = Dataset(test_file_path.joinpath("Si_4"))
        configs = tset.get_configs()

        modelname = "SW_StillingerWeber_1985_Si__MO_405512056662_005"
        model = KIMModel(modelname)

        # calculator
        calc = Calculator(model)
        compute_arguments = calc.create(configs)

        for i, ca in enumerate(compute_arguments):
            calc.compute(ca)
            energy = calc.get_energy(ca)
            forces = calc.get_forces(ca)[:3]

            assert energy == pytest.approx(ref_energies[i], 1e-6)
            assert np.allclose(forces, ref_forces[i])

    def test_parameter(self):
        modelname = "SW_StillingerWeber_1985_Si__MO_405512056662_005"
        model = KIMModel(modelname)

        # parameters
        params = model.get_model_params()
        sigma = params["sigma"][0]
        A = params["A"][0]

        # optimizing parameters
        # B will not be optimized, only providing initial guess
        model.set_opt_params(
            sigma=[["default"]], B=[["default", "fix"]], A=[["default"]]
        )

        calc = Calculator(model)

        x0 = calc.get_opt_params()
        assert x0[0] == sigma
        assert x0[1] == A
        assert len(x0) == 2
        assert model.get_num_opt_params() == 2

        x1 = [i + 0.1 for i in x0]
        calc.update_model_params(x1)

        assert params["sigma"][0] == sigma + 0.1
        assert params["A"][0] == A + 0.1
