from ..utils import torch_available
from .kim import KIM
from .lennard_jones import LennardJones
from .model import ComputeArguments, Model

__all__ = ["ComputeArguments", "Model", "LennardJones", "KIM"]

if torch_available():
    from .linear_regression import LinearRegression
    from .model_torch import ModelTorch
    from .neural_network import NeuralNetwork

    __all__.extend(["ModelTorch", "NeuralNetwork", "LinearRegression"])
