import sys
sys.path.append('../openkim_fit')
from training import TrainingSet
from modelparams import ModelParams
from kimcalculator import KIMcalculator

def test_kimcalculator():

    # test generate_kimstr()
    tset = TrainingSet()

    tset.read('training_set/training_set_MoS2.xyz')
    modelname = 'Three_Body_Stillinger_Weber_MoS__MO_000000111111_000'
    # read model params that will be optimized
    optparams = ModelParams(modelname)
    fname = 'mos2_init_guess.txt'
    optparams.read(fname)
    configs = tset.get_configs()


    #tset.read('training_set/training_set_Si.xyz')
    #modelname = 'EDIP_BOP_Bazant_Kaxiras_Si__MO_958932894036_001'


    # calculator
    KIMobj = KIMcalculator(modelname, optparams, configs[0] )
    KIMobj.initialize()
    KIMobj.compute()
    print 'energy', KIMobj.get_energy()
    print 'forces', KIMobj.get_forces()[:3]
    print 'get prediction', KIMobj.get_prediction()[:4]
    print KIMobj.get_NBC_method()


#    ks.KIM_API_print(KIMobj.pkim)

if __name__ == '__main__':
    test_kimcalculator()
