import sys
sys.path.append('../openkim_fit')
from modelparams import ModelParams


def test_modelparams():
    #modelname = 'Pair_Lennard_Jones_Truncated_Nguyen_Ar__MO_398194508715_000'
    #modelname = 'EDIP_BOP_Bazant_Kaxiras_Si__MO_958932894036_001'
    modelname = 'Three_Body_Stillinger_Weber_MoS__MO_000000111111_001'

    # create a tmp input file
    lines=['PARAM_FREE_A']
    lines.append('kim 0 20')
    lines.append('2.0 fix')
    lines.append('2.0 fix')
    lines.append('PARAM_FREE_p')
    lines.append('kim 0 20')
    lines.append('2.0  1.0  3.0')
    lines.append('2.0 fix')
    fname = 'test_params.txt'
    with open(fname, 'w') as fout:
        for line in lines:
            fout.write(line+'\n')

    att_params = ModelParams(modelname)
    att_params.echo_avail_params()
    att_params.read(fname)


    # change param values
    param_A = ['PARAM_FREE_A',
               ['kim', 0, 20],
               [2.0, 'fix'],
               [2.2, 1.1, 3.3]]
    att_params.set_param(param_A)

    param_B = ('PARAM_FREE_B',
               ('kim', 0, 20),
               (2.0, 'fix'),
               (2.2, 1.1, 3.3))
    att_params.set_param(param_B)
    att_params.echo_params()

    print att_params.get_value('PARAM_FREE_A')
    print att_params.get_size('PARAM_FREE_A')


    assert att_params.get_value('PARAM_FREE_A')[1] - 2.0 <= 1e-10

if __name__ == '__main__':
    test_modelparams()
