from kliff.utils import torch_available

if torch_available():
    from .generate_graph import KIMTorchGraphGenerator, KIMTorchGraph
    __all__ = ['KIMTorchGraphGenerator', 'KIMTorchGraph']
else:
    from .generate_graph import KIMTorchGraphGenerator
    __all__ = ['KIMTorchGraphGenerator']