from loguru import logger
from kliff.utils import torch_available

if torch_available():
    import torch

import kliff.transforms.configuration_transforms.graph_module as graph_module

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from kliff.dataset import Configuration

try:
    from torch_geometric.data import Data
    pyg_available = True

    class KLIFFTorchGraph(Data):
        """
        A Pytorch Geometric compatible graph representation of a configuration. When loaded into a
        class:`torch_geometric.data.DataLoader` the graphs of type KIMTorchGraph will be automatically collated and
         batched.
        """
        def __init__(self):
            super(KLIFFTorchGraph, self).__init__()
            self.num_nodes = None  # Simplify sizes and frees up pos key word, coords is cleaner
            self.energy = None
            self.forces = None
            self.n_layers = None
            self.coords = None
            self.images = None
            self.species = None
            self.z = None
            self.contributions = None

        def __inc__(self, key, value, *args, **kwargs):
            if 'index' in key or 'face' in key:
                return self.num_nodes
            elif 'contributions' in key:
                return 2
            elif 'images' in key:
                return torch.max(value) + 1
            else:
                return 0

        def __cat_dim__(self, key, value, *args, **kwargs):
            if 'index' in key or 'face' in key:
                return 1
            else:
                return 0
except ImportError:
    pyg_available = False
    logger.warning("Torch geometric is not available")


class KLIFFTorchGraphGenerator:
    """
    Generate a graph representation of a configuration. This generator will also save the required parameters
    for porting the model over to KIM-API using TorchMLModelDriver. The configuration file saved here will generate
    identical graphs at KIM-API runtime. For porting the graph representation you also need to provide the
    TorchScript model file name.
    Args:
        species (list): List of species.
        cutoff (float): Cutoff distance.
        n_layers (int): Number of convolution layers.
        as_torch_geometric_data (bool): If True, the graph will be returned as a Pytorch Geometric Data object.
    """
    def __init__(self, species, cutoff, n_layers, as_torch_geometric_data=False):
        self.species = species
        self.cutoff = cutoff
        self.n_layers = n_layers
        self.infl_dist = n_layers * cutoff
        self._tg = graph_module
        if as_torch_geometric_data and not pyg_available:
            raise ImportError("Torch geometric is not available")
        self.as_torch_geometric_data = as_torch_geometric_data

    def generate_graph(self, configuration: Configuration):
        """
        Generate a graph representation of a configuration.
        :param configuration:
        :return: C++ custom graph object or Pytorch Geometric Data object.
        """
        graph = graph_module.get_complete_graph(
            self.n_layers,
            self.cutoff,
            configuration.species,
            configuration.coords,
            configuration.cell,
            configuration.PBC
        )

        graph.energy = configuration.energy
        graph.forces = configuration.forces
        if torch_available:
            if self.as_torch_geometric_data:
                return self._to_py_graph(graph)
        return graph

    def __call__(self, configuration: Configuration):
        """
        Function wrapper over `generate_graph` for convenience and consistency.
        :param configuration:
        :return: Custom graph object or Pytorch Geometric Data object.
        """
        return self.generate_graph(configuration)

    @staticmethod
    def _to_py_graph(graph):
        """
        Convert a C++ graph object to a Pytorch Geometric Data object.
        :param graph: C++ graph object.
        :return: KIMTorchGraph object.
        """
        torch_geom_graph = KLIFFTorchGraph()
        torch_geom_graph.energy = torch.as_tensor(graph.energy)
        torch_geom_graph.forces = torch.as_tensor(graph.forces)
        torch_geom_graph.n_layers = torch.as_tensor(graph.n_layers)
        torch_geom_graph.coords = torch.as_tensor(graph.coords)
        torch_geom_graph.images = torch.as_tensor(graph.images)
        torch_geom_graph.species = torch.as_tensor(graph.species)
        torch_geom_graph.z = torch.as_tensor(graph.z)
        torch_geom_graph.contributions = torch.as_tensor(graph.contributions)
        torch_geom_graph.num_nodes = torch.as_tensor(graph.n_nodes)
        for i in range(graph.n_layers):
            torch_geom_graph.__setattr__(f"edge_index{i}", torch.as_tensor(graph.edge_index[i]))
        torch_geom_graph.coords.requires_grad_(True)
        return torch_geom_graph

    def collate_fn(self, config_list):
        """
        Collate function for use with a Pytorch DataLoader.
        :param config_list:
        :return: list of graphs.
        """
        graph_list = []
        for conf in config_list:
            graph = self.generate_graph(conf)
            graph_list.append(graph)
        return graph_list

    def collate_fn_single_conf(self, config_list):
        graph = self.generate_graph(config_list[0])
        return graph

    def save_for_kim_model(self, path: str, model: str):
        """
        Save a KIM model parameter file. This is used to load the model into KIM-API.
        :param path: Path to save the parameter file.
        :param model: name of model to save.
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
