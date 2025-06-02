from pathlib import Path

import numpy as np
import pytest
from ase import Atoms, build, io

from kliff.dataset import Configuration, Dataset


def test_lmdb_read_write():
    """Check data export and import to lmdb dataset"""
    config_list = [Configuration.bulk(name="Si", a=r) for r in [5.0, 5.5, 6.0]]
    ds = Dataset(config_list)
    ds.to_lmdb("ds.lmdb")
    assert Path("ds.lmdb").exists()

    ds_read = Dataset.from_lmdb(Path("ds.lmdb"))

    for c1, c2 in zip(ds, ds_read):
        assert np.allclose(c1.coords, c2.coords)
        assert np.allclose(c1.cell, c2.cell)
        assert c1.species == c2.species


def test_hf_read():
    """Check data export and import to HF dataset"""

    ds = Dataset.from_huggingface(
        "colabfit/carbon_X", "train", energy_key=None, forces_key=None
    )
    # convert it to ase list and check if formula is correct, easiest test.
    assert ds.to_ase_list()[0].symbols.formula.format() == "C6"
