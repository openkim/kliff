from openkim_fit.modelparams import ModelParams
from openkim_fit.kimcalculator import KIMcalculator
from openkim_fit.training import TrainingSet
import os

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

# read config and reference data
tset = TrainingSet()
tset.read('../tests/config.txt_20x20')
configs = tset.get_configs()

# prediction 
kimobj = KIMcalculator(modelname, params, configs[0])
kimobj.initialize()



