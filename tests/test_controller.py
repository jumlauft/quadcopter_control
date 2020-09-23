import numpy as np
from src import controller

def test_model_free():
    state = np.array([0,0,0,0,0,0,1,0,0,0,0,0,0])
    desired_state = {'pos': np.array([0,0,0]),
                     'vel': np.array([0,0,0]),
                     'acc': np.array([0,0,0]),
                     'yaw': np.array([0]),
                     'yawdot': np.array([0])}
    ctrl = controller.Controller(None, desired_state)
    assert (ctrl.run_ctrl(state) == np.array([1.7658,0,0,0]) ).all()


