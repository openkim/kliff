from training import TrainingSet
from keywords import InputKeywords
from modelparams import ModelParams
#from kimobject import init_KIMobjects
from kimcalculator import KIMcalculator
import sys
sys.path.append('../lib/geodesicLMv1.1/pythonInterface')
from geodesiclm import geodesiclm
from traincost import Cost 
from scipy.optimize import least_squares
import numpy as np


# KIM model parameters
modelname = 'Pair_Lennard_Jones_Truncated_Nguyen_Ar__MO_398194508715_000'
modelname = 'EDIP_BOP_Bazant_Kaxiras_Si__MO_958932894036_001'
modelname = 'Three_Body_Stillinger_Weber_MoS__MO_000000111111_000'
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
#params.set_cutoff(6.0)

fname = '../tests/mos2_init_guess.txt' 
params.read(fname)
params.echo_params()    

# initial guess of params
x0 = params.get_x0()
print 'type(x0)',type(x0)
print 'len(x0)',len(x0)
print 'x0', x0 


# read config and reference data
tset = TrainingSet()
#tset.read('../tests/config.txt_20x20')
#tset.read('../tests/T150_training_1000.xyz')
tset.read('../tests/training_set/')
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
cst = Cost(params)
for i in range(len(configs)):
    cst.add(KIMobjs[i], refs[i], weight=1)



resid = cst.get_residual(x0)
print resid[:3]
cost = cst.get_cost(x0)
print 'cost' , cost 


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


## read keywords
#keywords = InputKeywords()
#keywords.read()
#keywords.echo()
#
## read training set
#training_set_file_name = keywords.get_value('trainingset')
#Tset = TrainingSet()
#Tset.read(training_set_file_name)
#configs = Tset.get_configs()
#
## read model paramters
#input_params = keywords.get_block('modelparameters')
#modelname = keywords.get_value('modelname')
#att_params = ModelParams(input_params, modelname)
##att_params.get_avail_params()
#if keywords.get_value('echo_avail_params'):
#    att_params.echo_avail_params()
#att_params.read()
#att_params.echo_opt_params()    
#
#
## KIM objects
#kim_objects = init_KIMobjects(modelname, configs, att_params)
#print kim_objects[0].get_NBC_method()
#
## optimize
#x0 = kim_objects[0].get_opt_x0()
#
#
#func = get_residual
#method = 'scipy-lm'
#if method == 'geodesiclm': 
#    xf, info = geodesiclm(func, x0, args=(kim_objects, configs),
#                          full_output=1, print_level=5, iaccel=1, maxiters=10000,
#                          artol=-1.0, xtol=-1, ftol=-1, avmax = 2.0)
#    print info
#    print xf
#
#elif method == 'scipy-lm':
#    # test scipy.optimize minimization method
#    res_1 = least_squares(func, x0, args=(kim_objects,configs), method='lm')
#    print res_1
#

