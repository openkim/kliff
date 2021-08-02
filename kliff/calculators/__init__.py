from kliff.calculators.calculator import Calculator
from kliff.utils import torch_available

__all__ = ["Calculator"]


if torch_available():
    from .calculator_torch import CalculatorTorch

    __all__.extend(["CalculatorTorch"])
