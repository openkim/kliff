from kliff.utils import torch_available

from .calculator import Calculator

__all__ = ["Calculator"]


if torch_available():
    from .calculator_torch import CalculatorTorch

    __all__.extend(["CalculatorTorch"])
