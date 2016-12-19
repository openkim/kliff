import sys
sys.path.append('../openkim_fit')
sys.path.append('../libs/geodesicLMv1.1/pythonInterface')
sys.path.append('../libs/geodesicLMv1.1/pythonInterface')
from training import TrainingSet
from keywords import InputKeywords
from modelparams import ModelParams
from kimcalculator import KIMcalculator
from cost import Cost
from geodesiclm import geodesiclm
from scipy.optimize import least_squares
import numpy as np


# KIM model parameters
#modelname = 'Pair_Lennard_Jones_Truncated_Nguyen_Ar__MO_398194508715_000'
#modelname = 'EDIP_BOP_Bazant_Kaxiras_Si__MO_958932894036_001'
modelname = 'Three_Body_Stillinger_Weber_MoS__MO_000000111111_001'
params = ModelParams(modelname)
params.echo_avail_params()
#fname = '../tests/test_params.txt'
#params.read(fname)
#param_A = ('PARAM_FREE_A',
#           ('kim', 0, 20),
#           ('kim', 'fix'),
#           ('kim', 0.1, 9.9))
#params.set_param(param_A)
#params.echo_params()

fname = '../tests/mos2_init_guess.txt'
params.read(fname)
params.echo_params()

# initial guess of params
x0 = params.get_x0()
#print 'type(x0)',type(x0)
#print 'len(x0)',len(x0)
print 'x0', x0


# read config and reference data
tset = TrainingSet()
#tset.read('../tests/config.txt_20x20')
#tset.read('../tests/T150_training_1000.xyz')
tset.read('training_set/training_set_multi_small/')
configs = tset.get_configs()

# prediction
KIMobjs=[]
for i in range(len(configs)):
    obj = KIMcalculator(modelname, params, configs[i])
    obj.initialize()
    #obj.update_params()
    #obj.compute()
    KIMobjs.append(obj)

# reference
refs = []
for i in range(len(configs)):
    f = configs[i].get_forces()
    refs.append(f)

# cost function
cst = Cost(params, nprocs=1)
for i in range(len(configs)):
    cst.add(KIMobjs[i], refs[i], weight=1)



resid = cst.get_residual(x0)
print resid[:3]
cost = cst.get_cost(x0)


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
    res_1 = least_squares(func, x0, method='lm')
    print res_1

params.echo_params()


