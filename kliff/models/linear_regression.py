import torch
import torch.nn as nn
import kliff
from ..dataset.dataset_torch import FingerprintsDataset, FingerprintsDataLoader
from .model_torch import ModelTorch

logger = kliff.logger.get_logger(__name__)


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
