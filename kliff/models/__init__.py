from kliff.models.kim import KIMModel
from kliff.models.lennard_jones import LennardJones
from kliff.models.model import ComputeArguments, Model
from kliff.models.parameter import OptimizingParameters, Parameter
from kliff.utils import torch_available

__all__ = [
    "Parameter",
    "OptimizingParameters",
    "ComputeArguments",
    "Model",
    "LennardJones",
    "KIMModel",
]

if torch_available():
    from kliff.models.linear_regression import LinearRegression
    from kliff.models.model_torch import ModelTorch
    from kliff.models.neural_network import NeuralNetwork

    __all__.extend(["ModelTorch", "NeuralNetwork", "LinearRegression"])
