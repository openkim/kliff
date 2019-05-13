import numpy as np
import os
import multiprocessing as mp
from collections.abc import Iterable
from kliff.descriptors.descriptor import load_fingerprints
from kliff.error import InputError
from kliff.dataset import Configuration
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# pytorch buildin (use them directly)
from torch.nn.modules.linear import Linear, Bilinear
from torch.nn.modules.conv import (
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
)
from torch.nn.modules.activation import (
    Threshold,
    ReLU,
    Hardtanh,
    ReLU6,
    Sigmoid,
    Tanh,
    Softmax,
    Softmax2d,
    LogSoftmax,
    ELU,
    SELU,
    CELU,
    Hardshrink,
    LeakyReLU,
    LogSigmoid,
    Softplus,
    Softshrink,
    PReLU,
    Softsign,
    Softmin,
    Tanhshrink,
    RReLU,
    GLU,
)
from torch.nn.modules.pooling import (
    AvgPool1d,
    AvgPool2d,
    AvgPool3d,
    MaxPool1d,
    MaxPool2d,
    MaxPool3d,
    MaxUnpool1d,
    MaxUnpool2d,
    MaxUnpool3d,
    FractionalMaxPool2d,
    LPPool1d,
    LPPool2d,
    AdaptiveMaxPool1d,
    AdaptiveMaxPool2d,
    AdaptiveMaxPool3d,
    AdaptiveAvgPool1d,
    AdaptiveAvgPool2d,
    AdaptiveAvgPool3d,
)
from torch.nn.modules.batchnorm import BatchNorm1d, BatchNorm2d, BatchNorm3d
from torch.nn.modules.normalization import (
    LocalResponseNorm,
    CrossMapLRN2d,
    LayerNorm,
    GroupNorm,
)
from torch.nn.modules.rnn import (
    RNNBase,
    RNN,
    LSTM,
    GRU,
    RNNCellBase,
    RNNCell,
    LSTMCell,
    GRUCell,
)


@torch._jit_internal.weak_module
class Dropout(torch.nn.modules.dropout._DropoutNd):
    """A Dropout layer that zeros the same element of descriptor values for all
    atoms.

    Note `torch.nn.Dropout` dropout each component independently.

    Parameters
    ----------
    p: float
        probability of an element to be zeroed. Default: 0.5

    inplace: bool
        If set to `True`, will do this operation in-place. Default: `False`

    Shapes
    ------
        Input: [N, D] or [1, N, D]
        Outut: [N, D] or [1, N, D] (same as Input)
        The first dimension 1 is beause the dataloader provides only sample each
        iteration.
    """

    @torch._jit_internal.weak_script_method
    def forward(self, input):
        dim = input.dim()
        shape = input.shape
        if dim == 2:
            shape_4D = (1, *shape, 1)
        elif dim == 3:
            if shape[0] != 1:
                raise Exception('Shape[0] needs to be 1 for a 3D tensor.')
            shape_4D = (*shape, 1)

        else:
            raise Exception(
                'Input need to be 2D or 3D tensor, but got a ' '{}D tensor.'.format(dim)
            )
        x = torch.reshape(input, shape_4D)
        x = torch.transpose(x, 1, 2)
        y = torch.nn.functional.dropout2d(x, self.p, self.training, self.inplace)
        y = torch.transpose(y, 1, 2)
        y = torch.reshape(y, shape)
        return y


class FingerprintsDataset(Dataset):
    """Atomic environment fingerprints dataset.

    Parameters
    ----------
    fname: string
        Name of the fingerprints file.

    transform: callable (optional):
        Optional transform to be applied on a sample.
    """

    def __init__(self, fname, transform=None):
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

    The dataset loader will load an element from the next batch if a batch is
    fully iterarated. This, in effect, looks like concatenating the dataset the
    number of epochs times.

    Parameters
    ----------
    num_epochs: int
        Number of epochs to iterate through the dataset.
    """

    def __init__(self, num_epochs=1, *args, **kwargs):
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
    """Neural Network model.

    A feed-forward neural network model built using PyTorch.

    Parameters
    ----------
    descriptor: object
        A descriptor that transforms atomic environment information to the
        fingerprints, which are used as the input for the neural network.

    seed: int (optional)
        Global seed for random numbers.
    """

    def __init__(self, descriptor, seed=35):
        super(NeuralNetwork, self).__init__()
        self.descriptor = descriptor
        self.seed = seed

        dtype = self.descriptor.get_dtype()
        if dtype == np.float32:
            self.dtype = torch.float32
        elif dtype == np.float64:
            self.dtype = torch.float64
        else:
            raise NeuralNetworkError('Not support dtype "{}".'.format(dtype))

        torch.manual_seed(seed)
        self.layers = None
        self.save_prefix = None
        self.save_start = None
        self.save_frequency = None

    def add_layers(self, *layers):
        """Add layers to the sequential model.

        Parameters
        ----------
        layers: torch.nn layers
            ``torch.nn`` layers that are used to build a sequential model.
            Available ones including: torch.nn.Linear, torch.nn.Dropout, and
            torch.nn.Sigmoid among others. See
            https://pytorch.org/docs/stable/nn.html
            for a full list of torch.nn layers.
        """
        if self.layers is not None:
            raise NeuralNetworkError(
                '"add_layers" called multiple times. ' 'It should be called only once.'
            )
        else:
            self.layers = []

        for la in layers:
            self.layers.append(la)
            # set it as attr so that parameters are automatically registered
            setattr(self, 'layer_{}'.format(len(self.layers)), la)

        # check shape of first layer and last layer
        first = self.layers[0]
        if first.in_features != len(self.descriptor):
            raise InputError(
                '"in_features" of first layer should be equal to ' 'descriptor size.'
            )
        last = self.layers[-1]
        if last.out_features != 1:
            raise InputError('"out_features" of last layer should be 1.')

        # cast types
        self.type(self.dtype)

    def forward(self, x):
        for j, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def set_save_metadata(self, prefix, start, frequency):
        """Set metadata that controls how the model are saved during training.

        If this function is called before minimization starts, the model will be
        saved to the directory specified by ``prefix`` every ``frequency``
        epochs, beginning at the ``start`` epoch.

        Parameters
        ----------
        prefix: str
            Directory where the models are saved.
            Model will be named as `{prefix}/model_epoch{ep}.pt`, where `ep` is
            the epoch number.

        start: int
            Eopch number at which begins to save the model.

        frequency: int
            Save the model every ``frequency`` epochs.
        """
        self.save_prefix = str(prefix)
        self.save_start = int(start)
        self.save_frequency = int(frequency)

    def save(self, path):
        """Save a model to disk.

        Parameters
        ----------
        path: str
            Path where to store the model.
        """
        dirname = os.path.dirname(path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        torch.save(self.state_dict(), path)

    def load(self, path, mode):
        """Load a model on disk into memory.

        Parameters
        ----------
        path: str
            Path where the model is stored.

        mode: str
            Purpose of the loaded model. Should be either ``train`` or ``eval``.
        """

        self.load_state_dict(torch.load(path))
        if mode == 'train':
            self.train()
        elif mode == 'eval':
            self.eval()
        else:
            raise NeuralNetworkError(
                'Uncongnized mode "{}" in model.load().'.format(mode)
            )

    def write_kim_model(self, path=None):
        # supported
        param_layer = ['Linear']
        activ_layer = ['Sigmoid', 'Tanh', 'ReLU', 'ELU']
        dropout_layer = ['Dropout']
        layer_groups = self._group_layers(param_layer, activ_layer, dropout_layer)

        weights, biases = self._get_weights_and_biases(layer_groups, param_layer)
        activations = self._get_activations(layer_groups, activ_layer)
        drop_ratios = self._get_drop_ratios(layer_groups, dropout_layer)

        descriptor = self.descriptor
        dtype = self.dtype

        if path is None:
            path = os.path.join(os.getcwd(), 'NeuralNetwork__MO_000000111111_000')
        if path and not os.path.exists(path):
            os.makedirs(path)

        # TODO write CMakeList and probably pull the Eigen library
        with open(os.path.join(path, 'CMakeLists.txt'), 'w') as fout:
            fout.write('to be filled')

        # write parameter file
        fname = 'kliff_trained.params'
        with open(os.path.join(path, fname), 'w') as fout:
            fout.write('#' + '=' * 80 + '\n')
            fout.write(
                '# KIM ANN potential parameters, generated by `kliff` '
                'fitting program.\n'
            )
            fout.write('#' + '=' * 80 + '\n\n')

            # cutoff
            cutname, rcut = descriptor.get_cutoff()
            maxrcut = max(rcut.values())

            fout.write('# cutoff    rcut\n')
            if dtype == torch.float64:
                fout.write('{}  {:.15g}\n\n'.format(cutname, maxrcut))
            else:
                fout.write('{}  {:.7g}\n\n'.format(cutname, maxrcut))

            # symmetry functions
            # header
            fout.write('#' + '=' * 80 + '\n')
            fout.write('# symmetry functions\n')
            fout.write('#' + '=' * 80 + '\n\n')

            desc = descriptor.get_hyperparams()
            num_desc = len(desc)
            fout.write('{}    #number of symmetry funtion types\n\n'.format(num_desc))

            # descriptor values
            fout.write('# sym_function    rows    cols\n')
            for name, values in desc.items():
                if name == 'g1':
                    fout.write('g1\n\n')
                else:
                    rows = len(values)
                    cols = len(values[0])
                    fout.write('{}    {}    {}\n'.format(name, rows, cols))
                    if name == 'g2':
                        for val in values:
                            if dtype == torch.float64:
                                fout.write('{:.15g} {:.15g}'.format(val[0], val[1]))
                            else:
                                fout.write('{:.7g} {:.7g}'.format(val[0], val[1]))
                            fout.write('    # eta  Rs\n')
                        fout.write('\n')
                    elif name == 'g3':
                        for val in values:
                            if dtype == torch.float64:
                                fout.write('{:.15g}'.format(val[0]))
                            else:
                                fout.write('{:.7g}'.format(val[0]))
                            fout.write('    # kappa\n')
                        fout.write('\n')
                    elif name == 'g4':
                        for val in values:
                            zeta = val[0]
                            lam = val[1]
                            eta = val[2]
                            if dtype == torch.float64:
                                fout.write(
                                    '{:.15g} {:.15g} {:.15g}'.format(zeta, lam, eta)
                                )
                            else:
                                fout.write('{:.7g} {:.7g} {:.7g}'.format(zeta, lam, eta))
                            fout.write('    # zeta  lambda  eta\n')
                        fout.write('\n')
                    elif name == 'g5':
                        for val in values:
                            zeta = val[0]
                            lam = val[1]
                            eta = val[2]
                            if dtype == torch.float64:
                                fout.write(
                                    '{:.15g} {:.15g} {:.15g}'.format(zeta, lam, eta)
                                )
                            else:
                                fout.write('{:.7g} {:.7g} {:.7g}'.format(zeta, lam, eta))
                            fout.write('    # zeta  lambda  eta\n')
                        fout.write('\n')

            # data centering and normalization
            # header
            fout.write('#' + '=' * 80 + '\n')
            fout.write('# Preprocessing data to center and normalize\n')
            fout.write('#' + '=' * 80 + '\n')

            # mean and stdev
            mean = descriptor.get_mean()
            stdev = descriptor.get_stdev()
            if mean is None and stdev is None:
                fout.write('center_and_normalize  False\n')
            else:
                fout.write('center_and_normalize  True\n\n')

                fout.write('# mean\n')
                for i in mean:
                    if dtype == torch.float64:
                        fout.write('{:23.15e}\n'.format(i))
                    else:
                        fout.write('{:15.7e}\n'.format(i))
                fout.write('\n# standard deviation\n')
                for i in stdev:
                    if dtype == torch.float64:
                        fout.write('{:23.15e}\n'.format(i))
                    else:
                        fout.write('{:15.7e}\n'.format(i))
                fout.write('\n')

            # ann structure and parameters
            # header
            fout.write('#' + '=' * 80 + '\n')
            fout.write('# ANN structure and parameters\n')
            fout.write('#\n')
            fout.write(
                '# Note that the ANN assumes each row of the input "X" '
                'is an observation, i.e.\n'
            )
            fout.write('# the layer is implemented as\n')
            fout.write('# Y = activation(XW + b).\n')
            fout.write(
                '# You need to transpose your weight matrix if each '
                'column of "X" is an observation.\n'
            )
            fout.write('#' + '=' * 80 + '\n\n')

            # number of layers
            num_layers = len(weights)
            fout.write(
                '{}    # number of layers (excluding input layer, '
                'including output layer)\n'.format(num_layers)
            )

            # size of layers
            for b in biases:
                fout.write('{}  '.format(len(b)))
            fout.write('  # size of each layer (last must be 1)\n')

            # activation function
            # TODO enable writing different activations for each layer
            activation = activations[0]
            fout.write('{}    # activation function\n'.format(activation))

            # keep probability
            for i in drop_ratios:
                fout.write('{:.15g}  '.format(1.0 - i))
            fout.write('  # keep probability of input for each layer\n\n')

            # weights and biases
            for i, (w, b) in enumerate(zip(weights, biases)):

                # weight
                rows, cols = w.shape
                if i != num_layers - 1:
                    fout.write(
                        '# weight of hidden layer {} (shape({}, {}))\n'.format(
                            i + 1, rows, cols
                        )
                    )
                else:
                    fout.write(
                        '# weight of output layer (shape({}, {}))\n'.format(rows, cols)
                    )
                for line in w:
                    for item in line:
                        if dtype == torch.float64:
                            fout.write('{:23.15e}'.format(item))
                        else:
                            fout.write('{:15.7e}'.format(item))
                    fout.write('\n')

                # bias
                if i != num_layers - 1:
                    fout.write(
                        '# bias of hidden layer {} (shape({}, {}))\n'.format(
                            i + 1, rows, cols
                        )
                    )
                else:
                    fout.write(
                        '# bias of output layer (shape({}, {}))\n'.format(rows, cols)
                    )
                for item in b:
                    if dtype == torch.float64:
                        fout.write('{:23.15e}'.format(item))
                    else:
                        fout.write('{:15.7e}'.format(item))
                fout.write('\n\n')

    def _group_layers(self, param_layer, activ_layer, dropout_layer):
        """Divide all the layers into groups.

        The first group is either an empty list or a `Dropout` layer for the
        input layer. The last group typically contains only a `Linear` layer.
        For other groups, each group contains two, or three layers. `Linear`
        layer and an activation layer are mandatory, and a third `Dropout` layer
        is optional.

        Return
        ------
        groups: list of list of layers
        """

        groups = []
        new_group = []

        supported = param_layer + activ_layer + dropout_layer
        for i, layer in enumerate(self.layers):
            name = layer.__class__.__name__
            if name not in supported:
                raise NeuralNetworkError(
                    'Layer "{}" not supported by KIM model. Cannot proceed '
                    'to write.'.format(name)
                )
            if name in activ_layer:
                if i == 0:
                    raise NeuralNetworkError(
                        'First layer cannot be a "{}" layer'.format(name)
                    )
                if self.layers[i - 1].__class__.__name__ not in param_layer:
                    raise NeuralNetworkError(
                        'Cannot convert to KIM model. a "{}" layer must follow '
                        'a "Linear" layer.'.format(name)
                    )
            if name[:7] in dropout_layer:
                if self.layers[i - 1].__class__.__name__ not in activ_layer:
                    raise NeuralNetworkError(
                        'Cannot convert to KIM model. a "{}" layer must follow '
                        'an activation layer.'.format(name)
                    )
            if name in param_layer:
                groups.append(new_group)
                new_group = []
            new_group.append(layer)
        groups.append(new_group)

        return groups

    @staticmethod
    def _get_weights_and_biases(groups, supported):
        """Get weights and biases of all layers that have weights and biases.
        """
        weights = []
        biases = []
        for i, g in enumerate(groups):
            if i != 0:
                layer = g[0]
                name = layer.__class__.__name__
                if name in supported:
                    weight = layer.weight
                    bias = layer.bias
                    weights.append(weight)
                    biases.append(bias)
        return weights, biases

    @staticmethod
    def _get_activations(groups, supported):
        """Get the activation of all layers."""
        activations = []
        for i, g in enumerate(groups):
            if i != 0 and i != (len(groups) - 1):
                layer = g[1]
                name = layer.__class__.__name__
                if name in supported:
                    activations.append(name.lower())
        return activations

    @staticmethod
    def _get_drop_ratios(groups, supported):
        """Get the dropout ratio of all layers."""
        drop_ratios = []
        for i, g in enumerate(groups):
            if i == 0:
                if len(g) != 0:
                    layer = g[0]
                    name = layer.__class__.__name__
                    if name in supported:
                        drop_ratios.append(layer.p)
                else:
                    drop_ratios.append(0.0)
            elif i == len(groups) - 1:
                pass
            else:
                if len(g) == 3:
                    layer = g[2]
                    name = layer.__class__.__name__
                    if name in supported:
                        drop_ratios.append(layer.p)
                else:
                    drop_ratios.append(0.0)

        return drop_ratios


class PytorchANNCalculator(object):
    """ A neural network calculator.

    Parameters
    ----------
    model: obj
        Instance of :class:`~kliff.neuralnetwork.NeuralNetwork`.

    Attributes
    ----------
    attr1: list
        This is an example attribute.
    """

    # TODO should be moved to Model
    implemented_property = ['energy', 'forces']

    def __init__(self, model):

        self.model = model
        self.dtype = self.model.descriptor.dtype
        self.train_fingerprints_path = None

        self.use_energy = None
        self.use_forces = None

        self.results = dict([(i, None) for i in self.implemented_property])

    def create(
        self,
        configs,
        use_energy=True,
        use_forces=True,
        use_stress=False,
        reuse=False,
        nprocs=mp.cpu_count(),
    ):
        """Preprocess configs into fingerprints.

        Parameters
        ----------

        configs: list of Configuration object

        use_energy: bool (optional)
            Whether to require the calculator to compute energy.

        use_forces: bool (optional)
            Whether to require the calculator to compute forces.

        use_stress: bool (optional)
            Whether to require the calculator to compute stress.

        nprocs: int (optional)
            Number if processors.

        """
        if use_stress:
            raise NotImplementedError('"stress" is not supported by NN calculator.')

        self.configs = configs
        self.use_energy = use_energy
        self.use_forces = use_forces

        if isinstance(configs, Configuration):
            configs = [configs]

        # generate pickled fingerprints
        print('Start generating fingerprints')
        fname = self.model.descriptor.generate_train_fingerprints(
            configs, grad=use_forces, reuse=reuse, nprocs=nprocs
        )
        print('Finish generating fingerprints')
        self.train_fingerprints_path = fname

    def get_train_fingerprints_path(self):
        """Return the path to the training set fingerprints: `train.pkl`."""
        return self.train_fingerprints_path

    def compute(self, x):

        grad = self.use_forces
        zeta = x['zeta'][0]

        if grad:
            zeta.requires_grad = True
        y = self.model(zeta)
        pred_energy = y.sum()
        if grad:
            dzeta_dr = x['dzeta_dr'][0]
            forces = self.compute_forces(pred_energy, zeta, dzeta_dr)
            zeta.requires_grad = False
        else:
            forces = None

        return {'energy': pred_energy, 'forces': forces}

    @staticmethod
    def compute_forces(energy, zeta, dzeta_dr):
        denergy_dzeta = torch.autograd.grad(energy, zeta, create_graph=True)[0]
        forces = -torch.tensordot(denergy_dzeta, dzeta_dr, dims=([0, 1], [0, 1]))
        return forces


def cost_single_config(pred_energy, energy=None, forces=None):
    cost = (pred_energy - energy) ** 2
    return cost


class NeuralNetworkError(Exception):
    def __init__(self, msg):
        super(NeuralNetworkError, self).__init__(msg)
        self.msg = msg

    def __expr__(self):
        return self.msg
