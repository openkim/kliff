import sys
import os
import pytest
from kliff.modelparameters import ModelParameters


# callback function to specify the relations between parameters
def param_relations(params):
  A = params.get_value('A')
  B = params.get_value('B')
  B[0] = 2 * A[0]
  params.set_value('B', B)


def test_main():

  modelname = 'Three_Body_Stillinger_Weber_Si__MO_405512056662_004'

  # create a tmp input file
  fname = 'test_params.txt'
  with open(fname, 'w') as fout:
    fout.write('A\n')
    fout.write('kim 0 20\n')
    fout.write('p\n')
    fout.write('4  1.0  5.0\n')

  params = ModelParameters(modelname, debug=True)
  params.echo_avail_params()
  params.read(fname)


  # change param values
  param_A = ['A', [2.0]]
  params.set_param(param_A)

  param_B = ('B', (3.0, 'fix'))
  params.set_param(param_B)
  params.echo_params()

  A0 = params.get_value('A')[0]
  B0 = params.get_value('B')[0]
  assert A0 == 2.0
  assert B0 == 3.0

  # register callback
  params.register_param_relations_callback(param_relations)

  # update params (where, the above registerd callback is called)
  x0 = params.get_x0()
  params.update_params(x0)

  A0 = params.get_value('A')[0]
  B0 = params.get_value('B')[0]
  assert B0 == 2*A0


  # remove the file we generate
  os.remove(fname)

if __name__ == '__main__':
    test_main()


