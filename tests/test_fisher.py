from fisher import Fisher
from kimcalculator import KIMcalculator
from modelparams import ModelParams
from dataset import DataSet
from __future__ import print_function
import numpy as np
import sys

sys.path.append('../openkim_fit')


def test_fisher(relative_variance=False):

    # temperature
    T = 150

    # KIM model parameters
    modelname = 'Three_Body_Stillinger_Weber_MoS__MO_000000111111_000'
    params = ModelParams(modelname)
    params.echo_avail_params()
    fname = 'input/fitted_params_T150.txt'
    # fname = '/media/sf_share/mos2_fitted-T{}_interval4'.format(T)
    params.read(fname)
    params.echo_params()

    # read config and reference data
    tset = DataSet()
    fname = 'training_set/training_set_multi_small'
    # fname = 'training_set/training_set_multi_large'
    # fname = '/media/sf_share/xyz_interval4/'
    # fname = '/media/sf_share/training_set_T{}_interval4'.format(T)
    # fname = '/media/sf_share/T150_training_1000.xyz'
    tset.read(fname)
    configs = tset.get_configs()

    # prediction
    KIMobjs = []
    for i in range(len(configs)):
        obj = KIMcalculator(modelname, params, configs[i])
        obj.initialize()
        KIMobjs.append(obj)

    print('Hello there, started computing Fisher information matrix. It may take a')
    print('while, so have a cup of coffee.')
    fisher = Fisher(KIMobjs, params)
    Fij, Fij_std = fisher.compute()

    # take a look at the SW potential paper
    kB = 8.61733034e-5
    gamma = 0.02
    Fij = Fij / (2.0 * kB * T * gamma)

    # relative variance of Fij
    param_values = params.get_x0()

    # relative variance
    if relative_variance:
        Fij = np.dot(np.dot(np.diag(param_values), Fij), np.diag(param_values))
        Fij_std = np.dot(np.dot(np.diag(param_values), Fij_std), np.diag(param_values))

    with open('Fij', 'w') as fout:
        for line in Fij:
            for i in line:
                fout.write('{:24.16e} '.format(i))
            fout.write('\n')

    with open('Fij_std', 'w') as fout:
        for line in Fij_std:
            for i in line:
                fout.write('{:24.16e} '.format(i))
            fout.write('\n')

    Fij_diag = np.diag(Fij)
    with open('Fij_diag', 'w') as fout:
        for line in Fij_diag:
            fout.write('{:13.5e}\n'.format(line))

    # inverse
    Fij_inv = np.linalg.inv(Fij)
    with open('Fij_inv', 'w') as fout:
        for line in Fij_inv:
            for i in line:
                fout.write('{:24.16e} '.format(i))
            fout.write('\n')

    # inverse_diag
    Fij_inv_diag = np.diag(Fij_inv)
    with open('Fij_inv_diag', 'w') as fout:
        for line in Fij_inv_diag:
            fout.write('{:13.5e}\n'.format(line))

    # eiven analysis
    w, v = np.linalg.eig(Fij)
    with open('eigenVec', 'w') as fout:
        for row in v:
            for item in row:
                fout.write('{:13.5e}'.format(float(item)))
            fout.write('\n')

    with open('eigenVal', 'w') as fout:
        for item in w:
            fout.write('{:13.5e}\n'.format(float(item)))


if __name__ == '__main__':
    # test_fisher(relative_variance=True)
    test_fisher(relative_variance=False)
