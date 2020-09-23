import numpy as np
from src import disturbances


def test_thermal():
    x = np.array((0, 0))
    y = disturbances.thermal(x)
    assert y[0][0] == 0
    assert y[0][1] == 0


def test_gaussian():
    x = np.array((0, 0))
    y = disturbances.gaussians(x)
    assert y[0][0] == 0
    assert y[0][1] == 0