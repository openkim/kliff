from kliff.dataset import DataSet
from kliff.calculators.lennard_jones import LennardJones


tset = DataSet()
tset.read('training_set/Si_T300_4')
configs = tset.get_configurations()

calc = LennardJones()
calc.echo_model_params()
calc.save_model_params('lj_params.yml')
calc.restore_model_params('lj_params.yml')

calc.create(configs)
compute_arguments = calc.get_compute_arguments()
ca1 = compute_arguments[0]
calc.compute(ca1)

energy = calc.get_energy(ca1)
forces = calc.get_forces(ca1)
print('Energy:', energy)
print('Forces:', forces)
