import numpy as np
import multiprocessing as mp
from collections import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from kliff.descriptors.descriptor import load_fingerprints
from kliff.error import InputError


class FingerprintsDataset(Dataset):
    """Atomic environment fingerprints dataset."""

    def __init__(self, fname, transform=None):
        """
        Parameters
        ----------
        fname: string
            Name of the fingerprints file.

        transform: callable (optional):
            Optional transform to be applied on a sample.
        """
        self.fp = load_fingerprints(fname)
        self.transform = transform

    def __len__(self):
        return len(self.fp)

    def __getitem__(self, index):
        sample = self.fp[index]
        if self.transform:
            sample = self.transform(sample)
        return sample


# TODO implement here GPU options
class FingerprintsDataLoader(DataLoader):
    """A dataset loader that incorporate the support the number of epochs.

    The dataset loader will load an element from the next batch if a batch is fully
    iterarated. This, in effect, looks like concatenating the dataset the number of
    epochs times.
    """

    def __init__(self, num_epochs=1, *args, **kwargs):
        """
        Parameters
        ----------
        num_epochs: int
            Number of epochs to iterate through the dataset.
        """
        super(FingerprintsDataLoader, self).__init__(*args, **kwargs)
        self.num_epochs = num_epochs
        self.epoch = 0
        self.iterable = None

    def next_element(self):
        """ Get the next data element.
        """
        if self.iterable is None:
            self.iterable = self._make_iterable()
        try:
            element = self.iterable.next()
        except StopIteration:
            self.epoch += 1
            if self.epoch == self.num_epochs:
                raise StopIteration
            else:
                self.iterable = self._make_iterable()
                element = self.next_element()
        return element

    def _make_iterable(self):
        iterable = iter(self)
        return iterable


class NeuralNetwork(nn.Module):
    """ Neural Network class build upon PyTorch.

    Attributes
    -----------

    """

    def __init__(self, descriptor, seed=35):
        """

        Parameters
        ----------

        descriptor: descriptor object
            An instance of a descriptor that transforms atomic environment information
            to the fingerprints that are used as the input for the NN.

        seed: int (optional)
          random seed to be used by torch.manual_seed()
        """
        super(NeuralNetwork, self).__init__()
        self.descriptor = descriptor
        self.seed = seed

        self.grad = self.descriptor.get_grad()
        dtype = self.descriptor.get_dtype()
        if dtype == 'np.float32':
            self.dtype = torch.float32
        elif dtype == 'np.float64':
            self.dtype = torch.float64

        self.layers = None

        torch.manual_seed(seed)

    # TODO maybe remove layer['type'], just add a warning saying that this type of
    # layer is not supported be converted to KIM yet
    def add_layers(self, *layers):
        """Add layers to the sequential model.

        Parameters
        ----------
        layers: torch.nn layers
            torch.nn layers that are used to build a sequential model.
            Available ones including: torch.nn.Linear, torch.nn.Dropout, and
            torch.nn.Sigmoid among others. See https://pytorch.org/docs/stable/nn.html
            for a full list of torch.nn layers.
        """
        if self.layers is not None:
            raise NeuralNetworkError(
                '"add_layers" called multiple times. It should be called only once.')
        else:
            self.layers = []

        for la in layers:
            la_type = la.__class__.__name__
            la_scope = 'layer' + str(len(self.layers))
            current_layer = {'instance': la, 'type': la_type, 'scope': la_scope}
            self.layers.append(current_layer)
            # set layer as an attribute so that parameters are automatically registered
            setattr(self, 'layer_{}'.format(len(self.layers)), la)

        # check shape of first layer and last layer
        first = self.layers[0]['instance']
        if first.in_features != len(self.descriptor):
            raise InputError(
                '"in_features" of first layer should be equal to descriptor size.')
        last = self.layers[-1]['instance']
        if last.out_features != 1:
            raise InputError('"out_features" of last layer should be 1.')

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

        self.grad = self.model.descriptor.grad
        self.dtype = self.model.descriptor.dtype

    def create(self, configs, nprocs=mp.cpu_count()):
        """Preprocess configs into fingerprints, and create data loader for them.
        """
        self.configs = configs

        # generate pickled fingerprints
        fname = self.model.descriptor.generate_train_fingerprints(configs, nprocs=nprocs)
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
            zeta = x['zeta'][0]
            energy = x['energy'][0]
            species = x['species'][0]
            natoms = len(species)
            if self.grad:
                zeta.requires_grad = True
            y = self.model(zeta)
            pred_energy = y.sum()
            if self.grad:
                dzeta_dr = x['dzeta_dr'][0]
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


class NeuralNetworkError(Exception):
    def __init__(self, msg):
        super(NeuralNetworkError, self).__init__(msg)
        self.msg = msg

    def __str__(self):
        return self.msg
