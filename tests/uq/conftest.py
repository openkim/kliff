from pathlib import Path

import pytest

from kliff.dataset import Dataset
from kliff.models import KIMModel


@pytest.fixture(scope="session")
def uq_test_dir():
    # Directory of uq test files
    return Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def uq_test_data_dir():
    # Directory of uq test data
    return Path(__file__).resolve().parents[1] / "test_data/configs/Si_4"


@pytest.fixture(scope="session")
def uq_test_configs(uq_test_data_dir):
    # Load test configs
    data = Dataset.from_path(uq_test_data_dir)
    return data.get_configs()


@pytest.fixture(scope="session")
def uq_kim_model():
    # Load a KIM model
    modelname = "SW_StillingerWeber_1985_Si__MO_405512056662_006"
    model = KIMModel(modelname)
    model.set_opt_params(A=[["default"]])
    return model


@pytest.fixture(scope="session")
def uq_nn_orig_state_filename(uq_test_dir):
    """Return the original state filename for the NN model."""
    return uq_test_dir / "orig_model.pkl"
