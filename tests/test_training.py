import sys
sys.path.append('../openkim_fit')
from training import TrainingSet


def test_training():
#    configs = Config()
#    configs.read_extxyz('./training_set/T150_training_1000.xyz')
#    configs.write_extxyz('./echo.xyz')


    Tset = TrainingSet()
    #Tset.read('./training_set')
    Tset.read('training_set/training_set_multi_small')
    print 'num of configurations', Tset.get_size()
    configs = Tset.get_configs()
    for i,conf in enumerate(configs):
        conf.write_extxyz('echo{}.xyz'.format(i))

if __name__ == '__main__':
    test_training()

