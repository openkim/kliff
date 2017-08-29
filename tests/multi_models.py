import sys
sys.path.append('../openkim_fit')
sys.path.append('../libs/geodesicLMv1.1/pythonInterface')
from dataset import DataSet
from keywords import InputKeywords
from modelparams import ModelParams, WrapperModelParams
from kimcalculator import KIMcalculator
from cost import Cost
from geodesiclm import geodesiclm
from scipy.optimize import least_squares
import numpy as np
import time

# start timing
start_time = time.time()


# read config and reference data
tset = DataSet()
tset.read('training_set/training_set_multi_small/')
configs = tset.get_configs()
# reference
refs = []
for i in range(len(configs)):
    f = configs[i].get_forces()
    refs.append(f)


# KIM model parameters
modelnames = ['Three_Body_Stillinger_Weber_MoS__MO_000000111111_000',
              'Three_Body_Stillinger_Weber_MoS__MO_000000111111_001']

params = []
for j,name in enumerate(modelnames):
    params.append(ModelParams(name))
    params[j].echo_avail_params()
    fname = 'input/mos2_init_guess.txt'
    params[j].read(fname)
    params[j].echo_params()

# wrap multiple objects
wrapper_params = WrapperModelParams(params)
# initial guess of params
x0 = wrapper_params.get_x0()
print 'x0', x0

# predictor
KIMobjs=[]
for j in range(len(modelnames)):
    KIMobjs.append([])
    for i in range(len(configs)):
        obj = KIMcalculator(modelnames[j], params[j], configs[i])
        obj.initialize()
        KIMobjs[j].append(obj)


# cost function
cst = Cost(wrapper_params, nprocs=2)
for j in range(len(modelnames)):
    for i in range(len(configs)):
        cst.add(KIMobjs[j][i], refs[i], weight=1)


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


params[0].echo_params()

print"--- running time: {} seconds ---".format(time.time() - start_time)

