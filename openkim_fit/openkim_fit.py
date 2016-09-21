from training import TrainingSet
from keywords import InputKeywords
from modelparams import ModelParams
from kimobject import init_KIMobjects
#from residual import compute_residual
from optimize import fit


# read keywords
keywords = InputKeywords()
keywords.read()
keywords.echo()

# read training set
training_set_file_name = keywords.get_value('trainingset')
Tset = TrainingSet()
Tset.read(training_set_file_name)
configs = Tset.get_configs()

# read model paramters
input_params = keywords.get_block('modelparameters')
modelname = keywords.get_value('modelname')
att_params = ModelParams(input_params, modelname)
att_params.get_avail_params()
if keywords.get_value('echo_avail_params'):
    att_params.echo_avail_params()
att_params.read()
att_params.echo_opt_params()    


# KIM objects
kim_objects = init_KIMobjects(modelname, configs, att_params)
x0 = kim_objects[0].get_opt_x0()

print 'opt_x0 = ',  x0
#KIMobj.publish_params()
print kim_objects[0].get_NBC_method()


fit(x0, kim_objects, configs, method='geodesiclm')

#residual = compute_residual(kim_objects, configs)


#print residual[0:4]
#print residual[len(residual)/4: len(residual)/4+4]
#print residual[len(residual)/2: len(residual)/2+4]
#
