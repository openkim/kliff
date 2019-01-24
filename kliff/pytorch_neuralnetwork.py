import numpy as np
import multiprocessing as mp
from collections import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from kliff.fingerprints import load_pickle


class FingerprintsDataset(Dataset):
    """Atomic environment fingerprints dataset."""

    def __init__(self, fname, transform=None):
        """
        Parameters
        ----------
        fname: string
            Name of the pickle file.

        transform: callable (optional):
            Optional transform to be applied on a sample.
        """
        self.fp = load_pickle(fname)
        self.transform = transform

    def __len__(self):
        return len(self.fp)

    def __getitem__(self, index):
        sample = self.fp[index]
        if self.transform:
            sample = self.transform(sample)
        return sample


class FingerprintsDataLoader(DataLoader):
    """ A dataset loader that support number of epochs for the next element method."""

    def __init__(self, num_epochs=1, *args, **kwargs):
        super(FingerprintsDataLoader, self).__init__(*args, **kwargs)
        self.num_epochs = num_epochs
        self.epoch = 0
        self.iterable = None

    def make_iterable(self):
        iterable = iter(self)
        return iterable

    def next_element(self):
        if self.iterable is None:
            self.iterable = self.make_iterable()
        try:
            element = self.iterable.next()
        except StopIteration:
            self.epoch += 1
            if self.epoch == self.num_epochs:
                raise StopIteration
            else:
                self.iterable = self.make_iterable()
                element = self.next_element()
        return element


class NeuralNetwork(nn.Module):
    def __init__(self, fingerprints, seed=35):
        super(NeuralNetwork, self).__init__()
        self.fingerprints = fingerprints
        self.seed = seed

        self.descriptor = self.fingerprints.get_descriptor()
        self.fit_forces = self.fingerprints.get_fit_forces()
        self.dtype = self.fingerprints.get_dtype()

        self.layer_count = 0
        self.layers = []

        torch.manual_seed(seed)

    def add_layer(self, layer):
        """Add a layer to the network.

        Parameters
        ----------
        layer: layer object
            Options are `Dense`, `Dropout`, and `Output`.

        """
        layer_type = layer.__class__.__name__
        scope = 'layer' + str(len(self.layers))
        current_layer = {'instance': layer, 'type': layer_type, 'scope': scope}
        self.layers.append(current_layer)

        # set layer as an attribute so that the parameters are automatically registered
        setattr(self, 'layer_{}'.format(self.layer_count), layer)
        self.layer_count += 1

    def forward(self, x):
        for j, layer in enumerate(self.layers):
            li = layer['instance']
            lt = layer['type']
            ls = layer['scope']
            x = li(x)
        return x


class PytorchANNCalculator(object):
    """ A neural network calculator.

    Parameters
    ----------

    model: instance of `NeuralNetwork` class
    """

    def __init__(self, model, batch_size=100, num_epochs=100):
        self.model = model
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.configs = None
        self.use_energy = None
        self.use_forces = None

        self.data_loader = None

        # self.descriptor = self.fingerprints.get_descriptor()
        self.fit_forces = self.model.fingerprints.get_fit_forces()
        self.dtype = self.model.fingerprints.get_dtype()

    def create(self, configs):
        """Preprocess configs into fingerprints, and create data loader for them.
        """
        self.configs = configs

        # generate pickled fingerprints
        # TODO temporary not generate it
        self.model.fingerprints.generate_train_tfrecords(configs, nprocs=mp.cpu_count())
        # create dataloader
        fname = 'fingerprints/train.pkl'
        fp = FingerprintsDataset(fname)
        self.data_loader = FingerprintsDataLoader(dataset=fp, num_epochs=self.num_epochs)

    def get_loss(self, forces_weight=1.):
        """
        """
        loss = 0
        for _ in range(self.batch_size):
            # raise StopIteration error if out of bounds; This will ignore the last
            # chunk of data whose size is smaller than `batch_size`
            x = self.data_loader.next_element()
            # [0] because data_loader make it a batch with 1 element
            zeta = x['gen_coords'][0]
            energy = x['energy'][0]
            natoms = torch.sum(x['num_atoms_by_species'][0])
            if self.fit_forces:
                zeta.requires_grad = True
            y = self.model(zeta)
            pred_energy = y.sum()
            if self.fit_forces:
                dzeta_dr = x['dgen_datomic_coords'][0]
                forces = self.compute_forces(pred_energy, zeta, dzeta_dr)
                zeta.requires_grad = False
            c = cost_single_config(pred_energy, energy)/natoms**2
            loss += c
            # TODO add forces cost
        loss /= self.batch_size
        return loss

    @staticmethod
    def compute_forces(energy, zeta, dzeta_dr):
        denergy_dzeta = torch.autograd.grad(energy, zeta, create_graph=True)[0]
        forces = -torch.tensordot(denergy_dzeta, dzeta_dr, dims=([0, 1], [0, 1]))


def cost_single_config(pred_energy, energy=None, forces=None):
    cost = (pred_energy - energy)**2
    return cost


def NN_model(data_loader, batch_size=2):
    n = 0
    eof = False
    while True:

        cost = 0
        for _ in range(batch_size):
            try:
                x = data_loader.next_element()
                n += 1
            except StopIteration:
                eof = True
                break
            zeta = x['gen_coords']
            deriv_zeta = x['dgen_datomic_coords']
            energy = x['energy']
            forces = x['forces']
            y = cost_single_config(zeta, energy)
            cost += y

        if (not eof) or (eof and (not n % batch_size == 0)):
            print('@@@cost', cost)

        if eof:
            break


if __name__ == '__main__':

    fname = 'fingerprints/train.pkl'
    fp = FingerprintsDataset(fname)

    fp_loader = FingerprintsDataLoader(dataset=fp, num_epochs=3)
    NN_model(fp_loader)
