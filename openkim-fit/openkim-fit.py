from training import TrainingSet
from keywords import InputKeywords
from modelparams import ModelParams

# read keywords
keywords = InputKeywords()
keywords.read()
keywords.echo()

# read training set
training_set_file_name = keywords.get_value('trainingset')
Tset = TrainingSet()
Tset.read(training_set_file_name)


# test FreeParam class
#free_params.echo()


modelname = keywords.get_value('modelname')
input_params = keywords.get_block('modelparameters')
att_params = ModelParams(input_params, modelname)
att_params.get_avail_params()
#att_params.echo_avail_params()
att_params.read()
att_params.echo_opt_params()    
#print lines



