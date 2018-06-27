from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import pytest
from openkim_fit.dataset import DataSet
from openkim_fit.modelparameters import ModelParameters
from openkim_fit.kimcalculator import KIMCalculator

ref_energies = [
  -277.409737571,
  -275.597759276,
  -276.528342759,
  -275.482988187
]


ref_forces = [
  [[ -1.15948917e-14,  -1.15948917e-14,  -1.16018306e-14],
   [ -1.16989751e-14,  -3.93232367e-15,  -3.93232367e-15],
   [ -3.92538477e-15,  -1.17128529e-14,  -3.92538477e-15]],
  [[ 1.2676956,   0.1687802,   -0.83520474],
   [-1.2536342,   1.19394052,  -0.75371034],
   [-0.91847129, -0.26861574,  -0.49488973]],
  [[ 0.18042083, -0.11940541,   0.01800594],
   [ 0.50030564,  0.09165797,  -0.09694234],
   [-0.23992404, -0.43625564,   0.14952855]],
  [[-1.1114163,   0.21071302,   0.55246303],
   [ 1.51947195, -1.21522541,   0.7844481 ],
   [ 0.54684859,  0.01018317,   0.5062204 ]]
]


def assert_2D_array(a, b, tol=1.e-6):
  a = np.asarray(a)
  b = np.asarray(b)
  # size
  for i,j in zip(a.shape, b.shape):
    assert i == j
  # content
  for i in range(a.shape[0]):
    for j in range(a.shape[1]):
      assert a[i][j] == pytest.approx(b[i][j], tol)


def test_main():

  # model
  modelname = 'Three_Body_Stillinger_Weber_Si__MO_000000111111_000'

  params = ModelParameters(modelname)
  #fname = 'input/Si_SW_init_guess.txt'
  #params.read(fname)

  # training set
  tset = DataSet()
  tset.read('training_set/Si_T300_4')
  configs = tset.get_configurations()

  # calculator
  calc = KIMCalculator(params)

  kim_in_out_data = calc.create(configs)
  for i,data in enumerate(kim_in_out_data):
    compute_arguments = data.get_compute_arguments()
    calc.compute(compute_arguments)
    energy = data.get_energy()
    forces = data.get_forces()[:3]

    tol = 1e-6
    assert energy == pytest.approx(ref_energies[i], tol)
    assert_2D_array(forces, ref_forces[i], tol)


if __name__ == '__main__':
    test_main()
