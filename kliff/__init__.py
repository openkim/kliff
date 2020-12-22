import warnings

from kliff.log import set_up_logger
from kliff.utils import torch_available

__version__ = "0.1.7"

set_up_logger()

if not torch_available():
    warnings.warn(
        "'PyTorch' not found. All kliff machine learning modules (e.g. NeuralNetwork) "
        "are not imported. Ignore this if you want to use kliff to train "
        "physics-based models."
    )
