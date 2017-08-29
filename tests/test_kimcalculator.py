import sys
sys.path.append('../openkim_fit')
from dataset import DataSet
from modelparams import ModelParams
from kimcalculator import KIMcalculator


def test_kimcalculator():

    # model
    modelname = 'Three_Body_Stillinger_Weber_MoS__MO_000000111111_000'
    optparams = ModelParams(modelname)
    fname = 'input/mos2_init_guess.txt'
    optparams.read(fname)

    # training set
    tset = DataSet()
    tset.read('training_set/training_set_MoS2.xyz')
    configs = tset.get_configs()

    # calculator
    KIMobj = KIMcalculator(modelname, optparams, configs[0])
    KIMobj.initialize()
    KIMobj.compute()
    energy = KIMobj.get_energy()
    forces = KIMobj.get_forces()
    print 'energy', energy
    print 'forces', forces[:3]
    print 'get prediction', KIMobj.get_prediction()[:4]
    print KIMobj.get_NBC_method()

    assert abs(energy-(-863.95)) < 0.01
    assert abs(forces[0]-(-0.349)) < 0.001

if __name__ == '__main__':
    test_kimcalculator()
