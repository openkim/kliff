from training import TrainingSet
from keywords import InputKeywords
from modelparams import ModelParams
from kimobject import KIMobject

# read keywords
keywords = InputKeywords()
keywords.read()
keywords.echo()

# read training set
training_set_file_name = keywords.get_value('trainingset')
Tset = TrainingSet()
Tset.read(training_set_file_name)

# read model paramters
input_params = keywords.get_block('modelparameters')
modelname = keywords.get_value('modelname')
att_params = ModelParams(input_params, modelname)
att_params.get_avail_params()
#att_params.echo_avail_params()
att_params.read()
att_params.echo_opt_params()    

# KIM objects
configs = Tset.get_configs()
KIMobj = KIMobject(modelname, configs[0])
KIMobj.initialize()
KIMobj.compute()

print KIMobj.get_NBC_method()
print KIMobj.get_potential_energy()

KIMobj.map_opt_index(att_params)
x0 = KIMobj.get_opt_x0()
print 'opt_x0 = ',  x0
#KIMobj.publish_params()
KIMobj.compute()
print KIMobj.get_NBC_method()
print KIMobj.get_potential_energy()






