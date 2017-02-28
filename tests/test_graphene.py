import sys
sys.path.append('../openkim_fit')
from training import TrainingSet
from modelparams import ModelParams
from kimcalculator import KIMcalculator


def test_kimcalculator():

    # model
    modelname = 'RDP_Kolmogorov_Crespi_graphite__MO_000000111111_000'
    optparams = ModelParams(modelname)
    #fname = 'input/mos2_init_guess.txt'
    #optparams.read(fname)

    # training set
    tset = TrainingSet()
    tset.read('training_set/graphene.xyz')
    configs = tset.get_configs()

    # calculator
    KIMobj = KIMcalculator(modelname, optparams, configs[0])
    KIMobj.initialize()
    KIMobj.compute()
    energy = KIMobj.get_energy()
    forces = KIMobj.get_forces()
    print 'energy', energy
    #print 'forces', forces
    print 'forces'
    for i,f in enumerate(forces):
      print "{:13.5e} " .format(f),
      if i%3 == 2:
        print

    print 'get prediction', KIMobj.get_prediction()[:3]
    print KIMobj.get_NBC_method()

if __name__ == '__main__':
    test_kimcalculator()
