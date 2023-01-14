from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

# from kliff.legacy.descriptors.descriptor import Descriptor
from kliff.utils import create_directory, seed_all, to_path


class ModelTorch(nn.Module):
    """
    Base class for machine learning models.

    Typically, a user will not directly use this.

    Args:
        descriptor: atomic environment descriptor for computing configuration
            fingerprints. See :meth:`~kliff.descriptors.SymmetryFunction` and
            :meth:`~kliff.descriptors.Bispectrum`.
        seed: random seed.
    """

    def __init__(self, descriptor, seed: int = 35):
        super(ModelTorch, self).__init__()

        self._descriptor = descriptor
        self.seed = seed
        seed_all(seed)

        dtype = self._descriptor.get_dtype()

        if dtype == np.float32:
            self._dtype = torch.float32
        elif dtype == np.float64:
            self._dtype = torch.float64
        else:
            raise ModelTorchError(f"Not support dtype {dtype}.")

        self._save_prefix = Path.cwd() / "kliff_saved_model"
        self._save_start = 1
        self._save_frequency = 10

    def forward(self, x: Any):
        """
        Use the model to perform computation.

        Args:
            x: input to the model
        """
        raise NotImplementedError("`forward` not implemented.")

    def write_kim_model(self, path: Path = None):
        """
        Write the model out as a KIM-API compatible one.

        Args:
            path: path to write the model
        """
        raise NotImplementedError("`write_kim_model` not implemented.")

    def fit(self, path: Path):
        """
        Fit the model using analytic solution.

        Args:
            path: path to the fingerprints generated by the descriptor.
        """
        raise ModelTorchError(
            "Analytic fitting not supported for this model. Minimize a loss function "
            "to train the model instead."
        )

    def save(self, filename: Path):
        """
        Save a model to disk.

        Args:
            filename: Path to store the model.
        """
        state_dict = {
            "model_state_dict": self.state_dict(),
            "descriptor_state_dict": self.descriptor.state_dict(),
        }

        filename = to_path(filename)
        create_directory(filename)

        torch.save(state_dict, str(filename))

    def load(self, filename: Path, mode: str = "train"):
        """
        Load a save model.

        Args:
            filename: Path where the model is stored, e.g. kliff_model.pkl
            mode: Purpose of the loaded model. Should be either `train` or `eval`.
        """
        filename = to_path(filename)
        state_dict = torch.load(str(filename))

        # load model state dict
        self.load_state_dict(state_dict["model_state_dict"])

        if mode == "train":
            self.train()
        elif mode == "eval":
            self.eval()
        else:
            raise ModelTorchError(f"Unrecognized mode `{mode}`.")

        # load descriptor state dict
        self.descriptor.load_state_dict(state_dict["descriptor_state_dict"])

        logger.info(f"Model loaded from `{filename}`")

    def set_save_metadata(self, prefix: Path, start: int, frequency: int = 1):
        """
        Set metadata that controls how the model are saved during training.

        If this function is called before minimization starts, the model will be
        saved to the directory specified by `prefix` every `frequency` epochs,
        beginning at the `start` epoch.

        Args:
            prefix: Path to the directory where the models are saved.
                Model will be named as `{prefix}/model_epoch{ep}.pt`, where `ep` is
                the epoch number.
            start: Epoch number at which begins to save the model.
            frequency: Save the model every `frequency` epochs.
        """
        self._save_prefix = to_path(prefix)
        self._save_start = int(start)
        self._save_frequency = int(frequency)

    @property
    def descriptor(self):
        return self._descriptor

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return self._dtype

    @property
    def save_prefix(self):
        return self._save_prefix

    @property
    def save_start(self):
        return self._save_start

    @property
    def save_frequency(self):
        return self._save_frequency


class ModelTorchError(Exception):
    def __init__(self, msg):
        super(ModelTorchError, self).__init__(msg)
        self.msg = msg
