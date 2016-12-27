import sys
sys.path.append('../openkim_fit')
from training import Config, TrainingSet


def test_training():
    # one configuration
    configs = Config()
    configs.read_extxyz('training_set/training_set_MoS2.xyz')
    configs.write_extxyz('./echo.xyz')
    print 'training config written to: echo.xyz'

    # multiple configuration
    Tset = TrainingSet()
    Tset.read('training_set/training_set_multi_small')
    #Tset.read('training_set/training_set_multi_large')
    #Tset.read('/media/sf_share/xyz_interval4/')
    print 'num of configurations', Tset.get_size()
    configs = Tset.get_configs()
    for i,conf in enumerate(configs):
        conf.write_extxyz('echo{}.xyz'.format(i))


if __name__ == '__main__':
    test_training()
