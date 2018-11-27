from scipy.optimize import least_squares
from geodesiclm import geodesiclm
from cost import Cost
from kimcalculator import KIMcalculator
from modelparams import ModelParams
from keywords import InputKeywords
from dataset import DataSet
from lmplatconst import lmp_lat_const
from wrapcalculator import WrapCalculator
import sys
sys.path.append('../openkim_fit')
sys.path.append('../libs/geodesicLMv1.1/pythonInterface')


def test_endparse():

    # KIM model parameters
    modelname = 'Three_Body_Stillinger_Weber_MoS__MO_000000111111_001'
    params = ModelParams(modelname)
    fname = 'input/mos2_init_guess.txt'
    params.read(fname)
    params.echo_params()

    fname = 'input/data_4x4.edn'
    keys = ['unrelaxed-periodic-cell-vector-1',
            'unrelaxed-configuration-positions',
            'species']
    # key not in file
    #keys = [ 'haha']
    # list of list
    #keys = ['unrelaxed-configuration-positions']
    # list of one value
    #keys = ['unrelaxed-configuration-forces']
    # bare value
    #keys = ['unrelaxed-potential-energy']

    def func(arg):
        pass
    test_parse = WrapCalculator(params, fname, keys, func, modelname)
    rslt = test_parse.get_prediction()
    print rslt


# do a real fitting
def test_run_fitting():
    # KIM model parameters
    modelname = 'Three_Body_Stillinger_Weber_MoS__MO_000000111111_001'
    params = ModelParams(modelname)
    fname = 'input/mos2_init_guess.txt'
    params.read(fname)
    params.echo_params()

    # initial guess of params
    x0 = params.get_x0()
    print 'x0', x0

    # read config and reference data
    tset = DataSet()
    tset.read('training_set/training_set_MoS2.xyz')
    configs = tset.get_configs()

    # predictor
    func = lmp_lat_const
    fname = 'lattice_const.edn'
    keys = ['lattice-const']
    predictor = WrapCalculator(params, fname, keys, func, modelname)
    #rslt = predictor.get_prediction()
    # print 'init prediction', rslt

    # reference
    refs = [3.3]

    # cost function
    cst = Cost(params, nprocs=1)
    cst.add(predictor, refs, weight=1)

    cost = cst.get_cost(x0)
    print 'init cost', cost

    # optimize
    func = cst.get_residual
    #method = 'scipy-lm'
    method = 'geodesiclm'
    if method == 'geodesiclm':
        xf = geodesiclm(func, x0, h1=1e-5, h2=0.1, factoraccept=5, factorreject=2,
                        imethod=2, initialfactor=1, damp_model=0, print_level=2,
                        maxiters=10000, Cgoal=0.5e-7, artol=1.E-8, gtol=1.5e-10, xtol=1e-6,
                        xrtol=1.5e-6, ftol=-1, frtol=1.5e-6)
        print 'fitted params', xf

    elif method == 'scipy-lm':
        # test scipy.optimize minimization method
        res_1 = least_squares(func, x0, method='lm')
        print res_1

    params.echo_params()


if __name__ == '__main__':
    # test_endparse()
    test_run_fitting()
