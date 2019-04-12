from kliff.dataset import Configuration
from kliff.descriptors.bispectrum import Bispectrum


fname = '../configs_extxyz/Si.xyz'
conf = Configuration(format='extxyz', identifier=fname)
conf.read(fname)


cut_func = 'cos'
cut_values = {'Si-Si': 5.0}

desc = Bispectrum(cut_func, cut_values)
zeta, dzeta_dr = desc.transform(conf, grad=True)
print('natoms=', conf.get_number_of_atoms())
print('ndescs=', desc.get_size())
print(zeta.shape)
print(dzeta_dr.shape)
