import os


def test_basic():
    cmd = 'python running_fit.py'
    os.system(cmd)
    assert 1 == 2
