from kliff.utils import torch_geometric_available

if torch_geometric_available():
    from .generate_graph import KLIFFTorchGraph, KLIFFTorchGraphGenerator

    __all__ = ["KLIFFTorchGraphGenerator", "KLIFFTorchGraph"]
else:
    from .generate_graph import KLIFFTorchGraphGenerator

    __all__ = ["KLIFFTorchGraphGenerator"]
