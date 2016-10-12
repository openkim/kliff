from openkim_fit.training import TrainingSet
from openkim_fit.modelparams import ModelParams
from openkim_fit.kimcalculator import KIMcalculator
from geolm.geodesiclm import geodesiclm
from openkim_fit.traincost import Cost
from scipy.optimize import least_squares
import numpy as np

from openkim_fit.wrapcalculator import WrapCalculator
from openkim_fit.lmplatconst import lmp_lat_const


# KIM model parameters
modelname = 'Three_Body_Stillinger_Weber_MoS__MO_000000111111_000'
params = ModelParams(modelname)
params.echo_avail_params()

fname = '../tests/mos2_init_guess.txt'
params.read(fname)
params.echo_params()

# initial guess of params
x0 = params.get_x0()
print 'initiai guess x0', x0

# lammps lattice constant predictor
outname = 'lattice_const.edn'
runner_lat_const = WrapCalculator(lmp_lat_const, outname, ['lattice-const'])
# reference
refs = [3.2]

# cost function
cst = Cost(params)
cst.add(runner_lat_const, refs, weight=1)



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

