from kliff.dataset import DataSet
from kliff.calculators.kim import KIM


tset = DataSet()
tset.read('training_set/Si_T300_4')
configs = tset.get_configurations()

calc = KIM(model_name='Three_Body_Stillinger_Weber_Si__MO_405512056662_004')
calc.echo_model_params()
calc.save_model_params('kim_params.yml')
calc.restore_model_params('kim_params.yml')

calc.read_fitting_params('input/Si_SW_init_guess.txt')
calc.echo_fitting_params()

#
# calc.create(configs)
#compute_arguments = calc.get_compute_arguments()
#ca1 = compute_arguments[0]
# calc.compute(ca1)
#
#energy = calc.get_energy(ca1)
#forces = calc.get_forces(ca1)
#print('Energy:', energy)
#print('Forces:', forces[:3])
