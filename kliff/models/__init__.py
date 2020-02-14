from ..utils import check_torch_available
from .model import ComputeArguments
from .model import Model
from .lennard_jones import LennardJones
from .kim import KIM

__all__ = ["ComputeArguments", "Model", "LennardJones", "KIM"]

if check_torch_available():
    from .model_torch import ModelTorch
    from .neural_network import NeuralNetwork
    from .linear_regression import LinearRegression

    __all__.extend(["ModelTorch", "NeuralNetwork", "LinearRegression"])
