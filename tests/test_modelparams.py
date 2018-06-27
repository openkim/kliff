import sys
import os
import pytest
from tipp.modelparameters import ModelParameters


def test_main():

    modelname = 'Three_Body_Stillinger_Weber_Si__MO_000000111111_000'
    #modelname = 'Three_Body_Stillinger_Weber_CdTeZnSeHgS__MO_000000111111_000'

    # create a tmp input file
    fname = 'test_params.txt'
    with open(fname, 'w') as fout:
      fout.write('A\n')
      fout.write('kim 0 20\n')
      fout.write('p\n')
      fout.write('2.0  1.0  3.0\n')

    params = ModelParameters(modelname, debug=True)
    params.echo_avail_params()
    params.read(fname)


    # change param values
    param_A = ['A', [2.0, 'fix']]
    params.set_param(param_A)

    param_B = ('B', (2.0, 'fix'))
    params.set_param(param_B)
    params.echo_params()

    print params.get_value('A')
    print params.get_size('A')

    assert params.get_value('A')[0] == 2.0


    # remove the file we generate
    os.remove(fname)

if __name__ == '__main__':
    test_main()


