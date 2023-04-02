from loguru import logger

try:
    import torch
    torch_available = True
except ImportError:
    logger.warning("Torch is not installed. TrainingWheels will not be available.")
    torch_available = False

import kliff.ml.graphs.tg as tg
from kliff.dataset import Configuration

try:
    from torch_geometric.data import Data
    pyg_available = True
    class KIMTorchGraph(Data):
        def __init__(self):
            super(KIMTorchGraph, self).__init__()
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


class KIMTorchGraphGenerator:

    def __init__(self, species, cutoff, n_layers, as_torch_geometric_data=False):
        self.species = species
        self.cutoff = cutoff
        self.n_layers = n_layers
        self.infl_dist = n_layers * cutoff
        self._tg = tg
        if as_torch_geometric_data and not pyg_available:
            raise ImportError("Torch geometric is not available")
        self.as_torch_geometric_data = as_torch_geometric_data

    def generate_graph(self, configuration: Configuration):
        graph = tg.get_complete_graph(
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

    @staticmethod
    def _to_py_graph(graph):
        torch_geom_graph = KIMTorchGraph()
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
        graph_list = []
        for conf in config_list:
            graph = self.generate_graph(conf)
            graph_list.append(graph)
        return graph_list

    def collate_fn_single_conf(self, config_list):
        graph = self.generate_graph(config_list[0])
        return graph

    def save_kim_model(self, path: str, model: str):
        with open(f"{path}/kim_model.param", "w") as f:
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
