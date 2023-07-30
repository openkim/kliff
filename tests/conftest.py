import os
import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def test_data_dir():
    return Path(__file__).resolve().parent / "test_data"


@pytest.fixture(scope="session")
def clean_dir(debug_mode):
    """
    Create a new temp directory.

    When passed as an argument of a test function, the test will automatically be run
    in this directory.
    """
    old_cwd = Path.cwd()
    working_dir = tempfile.mkdtemp()
    os.chdir(working_dir)
    try:
        yield
    finally:
        if debug_mode:
            print(f"Tests ran in {working_dir}")
        else:
            os.chdir(old_cwd)
            shutil.rmtree(working_dir)


@pytest.fixture()
def tmp_dir(debug_mode):
    """Same as clean_dir() but is fresh for every test."""

    old_cwd = Path.cwd()
    working_dir = tempfile.mkdtemp()
    os.chdir(working_dir)
    try:
        yield
    finally:
        if debug_mode:
            print(f"Tests ran in {working_dir}")
        else:
            os.chdir(old_cwd)
            shutil.rmtree(working_dir)


@pytest.fixture(scope="session")
def debug_mode():
    return False
