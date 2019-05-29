import os
import numpy as np
import multiprocessing as mp
import kliff
from ..dataset.dataset import Configuration
from ..dataset.dataset_torch import FingerprintsDataset, FingerprintsDataLoader
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallelCPU as DDPCPU


logger = kliff.logger.get_logger(__name__)


class ModelTorch(nn.Module):
    """Base class for machine learning models."""

    def __init__(self, descriptor, seed=35):
        super(ModelTorch, self).__init__()

        self.seed = seed
        torch.manual_seed(seed)

        self.descriptor = descriptor
        dtype = self.descriptor.get_dtype()
        if dtype == np.float32:
            self.dtype = torch.float32
        elif dtype == np.float64:
            self.dtype = torch.float64
        else:
            raise ModelTorchError('Not support dtype "{}".'.format(dtype))

        self.save_prefix = None
        self.save_start = None
        self.save_frequency = None

    def forward(self, x):
        raise ModelTorchError('"forward" not implemented.')

    def fit(self, path):
        raise ModelTorchError(
            '"fit" not supported by this model. Minimize a loss function to train the '
            'model instead.'
        )

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
            Epoch number at which begins to save the model.

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
            raise ModelTorchError('Unrecognized mode "{}" in model.load().'.format(mode))

    def write_kim_model(self, path=None):
        raise ModelTorchError('"write_kim_model" not implemented.')


class LinearRegression(ModelTorch):
    """Linear regression model.
    """

    def __init__(self, descriptor, seed=35):
        super(LinearRegression, self).__init__(descriptor, seed)

        desc_size = self.descriptor.get_size()
        self.layer = nn.Linear(desc_size, 1)

    def forward(self, x):
        return self.layer(x)

    def fit(self, path):
        """Fit the model using analytic solution."""
        fp = FingerprintsDataset(path)
        data_loader = FingerprintsDataLoader(dataset=fp, num_epochs=1)
        X, y = self.prepare_data(data_loader)
        A = torch.inverse(torch.mm(X.t(), X))
        beta = torch.mv(torch.mm(A, X.t()), y)

        self.set_params(beta)

        msg = 'fit model "{}" finished.'.format(self.__class__.__name__)
        logger.info(msg)
        print(msg)

    def set_params(self, beta):
        """Set linear linear weight and bias.

        Parameters
        ----------
        beta: Tensor
            First component is bias and the remaining components is weight.
        """
        self.layer.weight = torch.nn.Parameter(beta[1:])
        self.layer.bias = torch.nn.Parameter(beta[0:1])

    def prepare_data(self, data_loader, use_energy=True, use_forces=False):
        def get_next():
            inp = data_loader.next_element()

            if use_energy:
                zeta = inp['zeta'][0]  # 2D tensor
                intercept = torch.ones(zeta.size()[0], 1)
                zeta = torch.cat((intercept, zeta), dim=1)

                # sum to get energy of the configuration; we can do this because the model
                # is linear
                zeta = torch.sum(zeta, 0, keepdim=True)  # 2D tensor
                e = torch.tensor([inp['energy'][0]])  # 1D tensor

            if use_forces:
                dzeta = inp['dzeta_dr'][0]  # 3D tensor (atom, desc, coords)
                # torch.zeros because derivative of intercept is 0
                intercept = torch.zeros(dzeta.size()[0], dzeta.size()[2])
                dzeta = torch.cat((intercept, dzeta), dim=1)

                dzeta = torch.sum(dzeta, 0)  # 2D tensor
                f = inp['forces'][0]  # 1D tensor

            if use_energy and use_forces:
                x = torch.cat((zeta, torch.transpose(dzeta)))
                y = torch.cat((e, f))
            elif use_energy:
                x = zeta
                y = e
            else:
                x = torch.transpose(dzeta)
                y = f

            return x, y

        x_, y_ = get_next()
        X = x_
        y = y_

        while True:
            try:
                x_, y_ = get_next()
                X = torch.cat((X, x_))
                y = torch.cat((y, y_))
            except StopIteration:
                break

        return X, y


class CalculatorTorch:
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
        """Process configs into fingerprints.

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

    def fit(self):
        path = self.get_train_fingerprints_path()
        self.model.fit(path)


class CalculatorTorchDDPCPU(CalculatorTorch):
    def compute(self, x):
        grad = self.use_forces
        zeta = x['zeta'][0]

        if grad:
            zeta.requires_grad = True

        model = DDPCPU(self.model)
        y = model(zeta)
        pred_energy = y.sum()
        if grad:
            dzeta_dr = x['dzeta_dr'][0]
            forces = self.compute_forces(pred_energy, zeta, dzeta_dr)
            zeta.requires_grad = False
        else:
            forces = None

        return {'energy': pred_energy, 'forces': forces}


class ModelTorchError(Exception):
    def __init__(self, msg):
        super(ModelTorchError, self).__init__(msg)
        self.msg = msg

    def __expr__(self):
        return self.msg
