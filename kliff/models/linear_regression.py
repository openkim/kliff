import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..dataset.dataset_torch import FingerprintsDataset, fingerprints_collate_fn
from .model_torch import ModelTorch

logger = logging.getLogger(__name__)


class LinearRegression(ModelTorch):
    r"""Linear regression model."""

    def __init__(self, descriptor, seed=35):
        super(LinearRegression, self).__init__(descriptor, seed)

        desc_size = self.descriptor.get_size()
        self.layer = nn.Linear(desc_size, 1)

    def forward(self, x):
        return self.layer(x)

    def fit(self, path):
        r"""Fit the model using analytic solution."""
        fp = FingerprintsDataset(path)

        loader = DataLoader(
            dataset=fp, batch_size=1, collate_fn=fingerprints_collate_fn
        )

        X, y = self.prepare_data(loader)
        A = torch.inverse(torch.mm(X.t(), X))
        beta = torch.mv(torch.mm(A, X.t()), y)

        self.set_params(beta)

        msg = 'fit model "{}" finished.'.format(self.__class__.__name__)
        logger.info(msg)
        print(msg)

    def set_params(self, beta):
        r"""Set linear weight and bias.

        Parameters
        ----------
        beta: Tensor
            First component is bias and the remaining components is weight.
        """
        self.layer.weight = torch.nn.Parameter(beta[1:])
        self.layer.bias = torch.nn.Parameter(beta[0:1])

    def prepare_data(self, loader, use_energy=True, use_forces=False):
        X = []
        y = []
        for batch in loader:
            sample = batch[0]
            if use_energy:
                zeta = sample["zeta"]
                intercept = torch.ones(zeta.size()[0], 1)
                zeta = torch.cat((intercept, zeta), dim=1)

                # sum to get energy of the configuration; we can do this because the model
                # is linear
                zeta = torch.sum(zeta, 0, keepdim=True)  # 2D tensor
                e = torch.tensor([sample["energy"]])  # 1D tensor

            if use_forces:
                dzeta = sample["dzeta_dr"]  # 3D tensor (atom, desc, coords)
                # torch.zeros because derivative of intercept is 0
                intercept = torch.zeros(dzeta.size()[0], dzeta.size()[2])
                dzeta = torch.cat((intercept, dzeta), dim=1)

                dzeta = torch.sum(dzeta, 0)  # 2D tensor
                f = sample["forces"][0]  # 1D tensor

            if use_energy and use_forces:
                x_ = torch.cat((zeta, torch.transpose(dzeta)))
                y_ = torch.cat((e, f))
            elif use_energy:
                x_ = zeta
                y_ = e
            elif use_forces:
                x_ = torch.transpose(dzeta)
                y_ = f
            else:
                raise LinearRegressionError(
                    'Both "use_energy" and "use_forces" are "False".'
                )

            X.append(x_)
            y.append(y_)

        X = torch.cat(X)
        y = torch.cat(y)

        return X, y


class LinearRegressionError(Exception):
    def __init__(self, msg):
        super(LinearRegressionError, self).__init__(msg)
        self.msg = msg

    def __expr__(self):
        return self.msg
