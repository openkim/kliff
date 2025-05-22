from pathlib import Path
from typing import Any, Dict, List

from loguru import logger
from monty.dev import requires

from kliff.dataset import Configuration
from kliff.transforms.configuration_transforms import ConfigurationTransform
from kliff.transforms.configuration_transforms.graphs import graph_module
from kliff.utils import torch_available, torch_geometric_available

if torch_available():
    import torch

if torch_geometric_available():
    from torch_geometric.data import Data
else:
    Data = object


class PyGGraph(Data):
    """
    A Pytorch Geometric compatible graph representation of a configuration. When loaded
    into a class:`torch_geometric.data.DataLoader` the graphs of type PyGGraph
    will be automatically collated and batched.
    """

    @requires(
        torch_geometric_available(),
        "Pytorch Geometric is not available. It is required for PyGGraph.",
    )
    def __init__(self):
        super().__init__()
        self.num_nodes = (
            None  # Simplify sizes and frees up pos key word, coords is cleaner
        )
        self.energy = None
        self.forces = None
        self.n_layers = None
        self.coords = None
        self.images = None
        self.species = None
        self.z = None
        self.cell = None
        self.contributions = None
        self.idx = None
        self.shifts = None

    def __inc__(self, key: str, value: torch.Tensor, *args, **kwargs):
        if "index" in key or "face" in key:
            return self.num_nodes
        elif "contributions" in key:
            return 2
        elif "images" in key:
            return torch.max(value) + 1
        else:
            return 0

    def __cat_dim__(self, key: str, value: torch.Tensor, *args, **kwargs):
        if "index" in key or "face" in key:
            return 1
        else:
            return 0

    @classmethod
    def from_dict(cls, mapping: Dict[str, Any]):
        """
        Create a PyGGraph object from a dictionary.

        Args:
            mapping: Dictionary containing the graph data.

        Returns:
            PyGGraph object.
        """
        graph = cls()
        for key, value in mapping.items():
            setattr(graph, key, torch.as_tensor(value))
        return graph

    def to_dict(self):
        """
        Convert the PyGGraph object to a dictionary.

        Returns:
            Dictionary containing the graph data.
        """
        graph_dict = {}
        for key, value in self.__dict__["_store"].items():
            graph_dict[key] = value.detach().cpu().numpy()
        return graph_dict


class RadialGraph(ConfigurationTransform):
    """
    Generate a graph representation of a configuration. This generator will also save the
    required parameters for porting the model over to KIM-API using TorchMLModelDriver.
    The configuration file saved here will generate identical graphs at KIM-API runtime.
    For porting the graph representation you also need to provide the TorchScript model file name.

    Args:
        species (list): List of species.
        cutoff (float): Cutoff distance.
        n_layers (int): Number of convolution layers.
        copy_to_config (bool): If True, the graph will be copied to
            the Configuration object's fingerprint attribute.
        mic (bool): If True, module will return conventional MIC graphs, as opposed to
            the parallel staged graphs.
    """

    def __init__(
        self,
        species: List[str],
        cutoff: float,
        n_layers: int = 1,
        copy_to_config: bool = False,
        mic: bool = False,
    ):
        super().__init__(copy_to_config=copy_to_config)
        self.species = species
        self.cutoff = cutoff
        self.n_layers = n_layers
        self.infl_dist = n_layers * cutoff
        self.mic = mic
        self._tg = graph_module

    def forward(self, configuration: Configuration) -> PyGGraph:
        """
        Generate a graph representation of a configuration.

        Args:
            configuration: Instance of ~:class:`kliff.dataset.Configuration`. For which the
                graph representation is to be generated.

        Returns:
            C++ custom graph object or Pytorch Geometric Data object.
        """
        if self.mic:
            graph = graph_module.get_mic_graph(
                self.cutoff,
                configuration.species,
                configuration.coords,
                configuration.cell,
                configuration.PBC,
            )
        else:
            graph = graph_module.get_staged_graph(
                self.n_layers,
                self.cutoff,
                configuration.species,
                configuration.coords,
                configuration.cell,
                configuration.PBC,
            )
        graph.energy = configuration.energy
        graph.forces = configuration.forces
        graph.idx = configuration.metadata.get("index", -1)
        return self.to_py_graph(graph)

    @staticmethod
    def to_py_graph(graph: graph_module.GraphData) -> PyGGraph:
        """
        Convert a C++ graph object to a KLIFF Geometric Graph Data object, ``GraphData``.

        Args:
            graph: C++ graph data object.

        Returns:
            PyGGraph object.
        """
        pyg_graph = PyGGraph()
        pyg_graph.energy = torch.as_tensor(graph.energy)
        pyg_graph.forces = torch.as_tensor(graph.forces)
        pyg_graph.n_layers = torch.as_tensor(graph.n_layers)
        pyg_graph.coords = torch.as_tensor(graph.coords)
        pyg_graph.images = torch.as_tensor(graph.images)
        pyg_graph.species = torch.as_tensor(graph.species)
        pyg_graph.z = torch.as_tensor(graph.z)
        pyg_graph.cell = torch.as_tensor(graph.cell)
        pyg_graph.contributions = torch.as_tensor(graph.contributions)
        pyg_graph.num_nodes = torch.as_tensor(graph.n_nodes)
        pyg_graph.idx = torch.as_tensor(graph.idx)
        pyg_graph.shifts = torch.as_tensor(graph.shifts)
        for i in range(graph.n_layers):
            pyg_graph.__setattr__(
                f"edge_index{i}", torch.as_tensor(graph.edge_index[i])
            )
        # pyg_graph.coords.requires_grad_(True)
        return pyg_graph

    def __call__(
        self, configuration: Configuration, return_extended_state=False
    ) -> PyGGraph:
        graph = self.forward(configuration)
        if return_extended_state:
            graph = graph.to_dict()
        if self.copy_to_config:
            configuration.fingerprint = graph
        return graph

    def export_kim_model(self, path: Path, model: str):
        """
        Save the transform toa text file for reuse. This is currently used  to load the
        model into KIM-API for pre-processing. The model name is also required to correctly
        load the model into KIM-API.

        Args:
            path: Path to save the parameter file.
            model: name of model to save.
        """
        with open(f"{path}/kliff_graph.param", "w") as f:
            n_elements = len(self.species)
            f.write(f"# Number of species\n")
            f.write(f"{n_elements}\n")
            f.write(f"{' '.join(self.species)}\n\n")

            f.write("# Preprocessing kind\n")
            f.write("Graph\n\n")

            f.write("# Cutoff and n_conv layers\n")
            f.write(f"{self.cutoff}\n{self.n_layers}\n\n")

            f.write("# Model\n")
            f.write(f"{model}\n\n")

            f.write("# Returns Forces\n")
            f.write("False\n")

            f.write("# Number of inputs\n")
            f.write(f"{3 + self.n_layers}\n\n")

            f.write("# Any descriptors?\n")
            f.write("None\n")
