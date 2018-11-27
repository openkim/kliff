from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from kliff.modelparameters import ModelParameters
from kliff.kimcalculator import KIMCalculator
from kliff.loss import Loss


def test_main():

    # KIM model parameters
    model = 'Three_Body_Stillinger_Weber_Si__MO_405512056662_004'
    params = ModelParameters(model)
    params.echo_avail_params()
    params.set_param(("A", [16.0]))
    params.set_param(("sigma", [1.5]))
    params.set_param(("cutoff", [3.5, 'fix']))

    # calculator
    calc = KIMCalculator(model)

    # before update
    A, error = calc.kim_model.get_parameter_double(0, 0)
    sigma, error = calc.kim_model.get_parameter_double(4, 0)
    cutoff = calc.get_cutoff()
    assert A == 15.2848479197914
    assert sigma == 2.0951
    assert calc.get_cutoff() == 3.77118

    # loss
    loss = Loss(params, calc)

    # after update
    A, error = calc.kim_model.get_parameter_double(0, 0)
    sigma, error = calc.kim_model.get_parameter_double(4, 0)
    cutoff = calc.get_cutoff()
    assert A == 16.0
    assert sigma == 1.5
    assert calc.get_cutoff() == 3.5


if __name__ == '__main__':
    test_main()
