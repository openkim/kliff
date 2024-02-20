import numpy as np

from kliff.dataset import Configuration, Dataset
from kliff.transforms.property_transforms import NormalizedPropertyTransform


def test_normalized_property_transforms(test_data_dir):
    # Test NormalizedPropertyTransform
    dataset = Dataset.from_ase(test_data_dir / "configs/Si_4.xyz", energy_key="Energy")
    pt = NormalizedPropertyTransform()

    assert pt._get_property_values(dataset).tolist() == [123.45, 0.0, 0.0, 100.0]
    pt.transform(dataset)
    assert pt.mean == 55.8625
    assert np.abs(pt.std - 56.474389936943986) < 10**-10
    transformed_energy = np.array(
        [
            1.1967814096879006,
            -0.9891651784529732,
            -0.9891651784529732,
            0.7815489472180464,
        ]
    )
    assert np.allclose(pt._get_property_values(dataset), transformed_energy)

    pt.inverse(dataset)
    assert np.allclose(pt._get_property_values(dataset), [123.45, 0.0, 0.0, 100.0])
