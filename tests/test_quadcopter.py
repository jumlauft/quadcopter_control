from src import quadcopter
import numpy as np


def test_position():
    pos0 = np.array((0, 0, 0))
    yaw0 = 0
    quad = quadcopter.Quadcopter(pos0, yaw0)
    assert (quad.get_position() == pos0).all()


def test_simulation():
    pos0 = np.array((0, 0, 0))
    yaw0 = 0
    quad = quadcopter.Quadcopter(pos0, yaw0)
    disturbance = lambda x : np.zeros(3)
    ctrl = lambda x : np.array([quad.GRAVITY_CONST*quad.MASS,0,0,0])
    quad.simulate(ctrl,disturbance)
    assert np.linalg.norm(quad.get_position() - pos0) < 5e-7
