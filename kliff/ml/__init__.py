from loguru import logger

from . import graphs
# from . import descriptors
from .descriptors import Descriptor
from .opt import OptimizerScipy, OptimizerTorch

try:
    from .training_wheels import TrainingWheels, TorchGraphFunction, TorchDescriptorFunction, TorchLegacyDescriptorFunction
    __all__ = ["graphs", "Descriptor", "OptimizerScipy", "OptimizerTorch", "TrainingWheels", "TorchGraphFunction",
           "TorchDescriptorFunction"]
except ImportError:
    logger.warning("Torch is not installed. TrainingWheels will not be available.")
    __all__ = ["graphs", "Descriptor", "OptimizerScipy"]
    pass

