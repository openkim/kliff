__version__ = "0.2.2"

import sys
import warnings

import loguru

from kliff.utils import torch_available

# set loguru logger
loguru.logger.remove()  # remove default logger, which log to stderr with level DEBUG
loguru.logger.add(sys.stderr, level="INFO")
loguru.logger.add("kliff.log", level="INFO")

if not torch_available():
    warnings.warn(
        "'PyTorch' not found. All kliff machine learning modules (e.g. NeuralNetwork) "
        "are not imported. Ignore this if you want to use kliff to train "
        "physics-based models."
    )
