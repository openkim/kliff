__version__ = "0.4.1"

import warnings

from kliff.log import set_logger
from kliff.utils import torch_available

set_logger(level="INFO", stderr=True)

if not torch_available():
    warnings.warn(
        "'PyTorch' not found. All kliff machine learning modules (e.g. NeuralNetwork) "
        "are not imported. Ignore this if you want to use kliff to train "
        "physics-based models."
    )
