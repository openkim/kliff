from training import TrainingSet
from keywords import InputKeywords
from modelparams import FreeParams
from modelparams import AttemptedParams

# read keywords
keywords = InputKeywords()
keywords.read()
keywords.echo()

# read training set
training_set_file_name = keywords.get_value('trainingset')
Tset = TrainingSet()
Tset.read(training_set_file_name)


# test FreeParam class
free_params = FreeParams()
free_params.inquire_free_params()
#free_params.echo()


input_params = keywords.get_block('modelparameters')
att_params = AttemptedParams(input_params, free_params)
att_params.read()
att_params.echo()    
#print lines



