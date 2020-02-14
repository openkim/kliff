from ..utils import check_torch_available
from .calculator import Calculator

__all__ = ["Calculator"]


if check_torch_available():
    from .calculator_torch import CalculatorTorch, CalculatorTorchDDPCPU

    __all__.extend(["CalculatorTorch", "CalculatorTorchDDPCPU"])
