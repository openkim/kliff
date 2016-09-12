from training import TrainingSet
from keywords import InputKeywords


# read keywords
keywords = InputKeywords()
keywords.read()
keywords.echo_readin()

# read training set
training_set_file_name = keywords.get_value('file_training')
Tset = TrainingSet()
Tset.read(training_set_file_name)
