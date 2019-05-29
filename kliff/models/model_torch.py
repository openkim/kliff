import os
import numpy as np
import torch
import torch.nn as nn
import kliff

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


class ModelTorchError(Exception):
    def __init__(self, msg):
        super(ModelTorchError, self).__init__(msg)
        self.msg = msg

    def __expr__(self):
        return self.msg
