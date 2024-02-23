import numpy as np
from ase.data import atomic_numbers

from kliff.dataset import Configuration
from kliff.transforms.configuration_transforms import ConfigurationTransform


class GlobalCoulombMatrix(ConfigurationTransform):
    """
    Coulomb matrix representation of the configuration.
    """

    def __init__(self, max_atoms: int = 5, copy_to_config: bool = False):
        super().__init__(copy_to_config)
        self.max_atoms = max_atoms

    def forward(self, configuration: Configuration):
        """
        Generate the Coulomb matrix for the configuration.

        Args:
            configuration: Instance of ~:class:`kliff.dataset.Configuration`. For which the
                Coulomb matrix is to be generated.

        Returns:
            Coulomb matrix of the configuration.
        """
        coords = configuration.coords
        n_atoms = configuration.get_num_atoms()
        coulomb_mat = np.zeros((self.max_atoms, self.max_atoms))
        species = [atomic_numbers[elem] for elem in configuration.species]
        for i in range(n_atoms):
            for j in range(i + 1):
                if i == j:
                    coulomb_mat[i, j] = 0.5 * (species[i] ** 2.4)
                else:
                    r = np.linalg.norm(coords[i] - coords[j])
                    coulomb_mat[i, j] = species[i] * species[j] / r
                    coulomb_mat[j, i] = coulomb_mat[i, j]
        return coulomb_mat

    def backward(self, fingerprint, configuration):
        """
        Inverse mapping of the transform. This is not implemented for any of the transforms,
        but is there for future use.
        """
        NotImplementedError(
            "Any of the implemented transforms do not support inverse mapping.\n"
            "For computing jacobian-vector product use `backward` function."
        )

    def collate_fn(self, config_list):
        """
        Collate function for the Coulomb matrix transform.
        """
        return [self.forward(config) for config in config_list]


def test_configuration_transform(test_data_dir):
    config = Configuration.from_file(test_data_dir / "configs/Si.xyz")
    transform = GlobalCoulombMatrix(max_atoms=8)
    fingerprint = transform(config)
    assert fingerprint.shape == (8, 8)
    assert np.allclose(fingerprint, fingerprint.T)
    assert np.allclose(
        fingerprint, np.load(test_data_dir / "precomputed_numpy_arrays/cm_si.npy")
    )
    assert config.fingerprint is None
    dataset = [config, config]
    fingerprints = transform.collate_fn(dataset)
    assert len(fingerprints) == 2
    assert fingerprints[0].shape == (8, 8)
    assert fingerprints[1].shape == (8, 8)
    assert np.array(fingerprints).shape == (2, 8, 8)
