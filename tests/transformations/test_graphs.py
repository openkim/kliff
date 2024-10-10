import numpy as np
import pytest
from ase import Atoms
from torch import tensor

from kliff.dataset import Configuration
from kliff.transforms.configuration_transforms.graphs import RadialGraph


def test_implicit_copying(test_data_dir):
    config = Configuration.from_file(
        test_data_dir / "configs" / "Si_4" / "Si_T300_step_0.xyz"
    )

    graph_generator = RadialGraph(
        species=["Si"], cutoff=4.5, n_layers=2, copy_to_config=True
    )
    graph = graph_generator(config)

    assert graph is config.fingerprint


def test_graph_generation(test_data_dir):
    config = Configuration.from_file(test_data_dir / "configs" / "Si.xyz")
    graph_generator = RadialGraph(species=["Si"], cutoff=3.7, n_layers=2)
    graph = graph_generator(config)
    assert np.allclose(
        graph.edge_index0.numpy(),
        np.array(
            [
                [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 4, 4, 4, 4, 5, 6, 6, 6, 7],
                [1, 4, 0, 2, 4, 6, 1, 3, 4, 6, 2, 0, 1, 2, 5, 4, 1, 2, 7, 6],
            ]
        ),
    )
    assert np.allclose(graph.edge_index1.numpy(), graph.edge_index0.numpy())
    assert graph.n_layers == 2
    assert np.allclose(graph.species.numpy(), np.array([0, 0, 0, 0, 0, 0, 0, 0]))
    assert np.allclose(graph.z.numpy(), np.array([14, 14, 14, 14, 14, 14, 14, 14]))
