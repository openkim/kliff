import sys
sys.path.append('../openkim_fit')
sys.path.append('../libs/geodesicLMv1.1/pythonInterface')
from dataset import DataSet
from modelparams import ModelParams
from kimcalculator import KIMcalculator
from cost import Cost
from fixparams import alternate_fix_params
from geodesiclm import geodesiclm
from scipy.optimize import least_squares
import numpy as np
import time


modelname = 'RDP_Kolmogorov_Crespi_graphite__MO_000000111111_000'
fixparamsfiles = ['input/graphene_init_guess.txt',
                  'input/graphene_init_guess_1.txt',
                  'input/graphene_init_guess_2.txt']

@alternate_fix_params(modelname, fixparamsfiles)
def test_fixparams():

    # KIM model parameters
    modelname = 'RDP_Kolmogorov_Crespi_graphite__MO_000000111111_000'
    params = ModelParams(modelname)

    # read initial guess of parameter values
    fname = 'input/graphene_init_guess.txt'
    params.read(fname)
    params.echo_params()
    x0 = params.get_x0()

    # read config and referene data
    tset = DataSet()
    tset.read('training_set/training_set_graphene/')
    configs = tset.get_configs()

    # prediction
    KIMobjs=[]
    for i in range(len(configs)):
        obj = KIMcalculator(modelname, params, configs[i])
        obj.initialize()
        KIMobjs.append(obj)

    # reference
    refs = []
    for i in range(len(configs)):
        f = configs[i].get_forces()
        refs.append(f)

    # cost function
    cst = Cost(params, nprocs=2)
    for i in range(len(configs)):
        cst.add(KIMobjs[i], refs[i])


    # optimize
    func = cst.get_residual
    #method = 'scipy-lm'
    method = 'geodesiclm'
    if method == 'geodesiclm':
        xf = geodesiclm(func, x0, h1=1e-5, h2=0.1, factoraccept=5, factorreject=2,
                              imethod=2, initialfactor=1, damp_model=0, print_level=2,
                              maxiters=10000, Cgoal=0.5e-7, artol=1.E-8, gtol=1.5e-8, xtol=1e-6,
                              xrtol=1.5e-6, ftol=-1, frtol=1.5e-6)
        print 'fitted params', xf

    elif method == 'scipy-lm':
        # test scipy.optimize minimization method
        res_1 = least_squares(func, x0, method='trf',verbose = 2)
        print res_1

    params.echo_params()
    params.echo_params(fname='fitted_params.txt')




if __name__ == '__main__':
    start_time = time.time()

    test_fixparams()

    # print running time
    print"--- running time: {} seconds ---".format(time.time() - start_time)

