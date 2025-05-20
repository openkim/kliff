import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
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


def test_staged_graph_generation(test_data_dir):
    config = Configuration.from_file(test_data_dir / "configs" / "Si.xyz")
    graph_generator = RadialGraph(species=["Si"], cutoff=3.7, n_layers=2)
    graph = graph_generator(config)
    # TODO: This is not a good test, as it will not work once we parallelize the graph
    # generator, better will be something like mic below.
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


def test_mic_graph_generation(test_data_dir):
    cutoff = 3.7
    atoms = bulk("Si")
    calc = SinglePointCalculator(
        atoms, energy=0.0, forces=np.zeros_like(atoms.get_positions())
    )
    atoms.calc = calc
    config = Configuration.from_ase_atoms(atoms)
    graph_generator = RadialGraph(species=["Si"], cutoff=cutoff, mic=True)
    graph = graph_generator(config)
    n_atoms = atoms.get_global_number_of_atoms()

    assert graph.edge_index0.max().item() < n_atoms  # largest index < n_atom for mic
    assert np.allclose(graph.species.numpy(), np.array([0, 0]))
    assert np.allclose(graph.z.numpy(), np.array([14, 14]))

    vectors = (
        graph.coords[graph.edge_index0[1]]
        + graph.shifts
        - graph.coords[graph.edge_index0[0]]
    )
    distances = vectors.norm(dim=-1)
    assert np.all(
        distances.numpy() < cutoff
    )  # all corrected vectors are of correct length
