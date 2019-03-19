from kliff.dataset import Configuration
from kliff.descriptors.bispectrum import Bispectrum


fname = '../configs_extxyz/Si.xyz'
conf = Configuration(format='extxyz', identifier=fname)
conf.read(fname)


jmax = 2
cutoff = {'Si-Si': 4}
desc = Bispectrum(jmax, cutoff, normalize=True)
desc.generate_train_fingerprints([conf], grad=False)
