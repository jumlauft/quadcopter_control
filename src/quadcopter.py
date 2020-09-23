import numpy as np
from scipy.integrate import solve_ivp
import itertools
from pyquaternion import Quaternion
from scipy.optimize import minimize, LinearConstraint


class Quadcopter:
    MASS = 0.18
    GRAVITY_CONST = 9.81
    DT = 0.005
    ARM_LENGTH = 0.086  # meter
    HEIGHT = 0.05
    INERTIA = [[0.00025, 0, 0], [0, 0.00025, 0], [0, 0, 0.0003738]]
    MIN_THROTTLE = 0
    MAX_THROTTLE = 4.0 * MASS * GRAVITY_CONST
    R_C = 15 / 611
    GRAVITY_ACC = [0, 0, -GRAVITY_CONST]
    #  [ F  ]          [ F1 ]
    #  | M1 |  = M2R * | F2 |
    #  | M2 |          | F3 |
    #  [ M3 ]          [ F4 ]
    M2R = [[1, 1, 1, 1],
            [0, ARM_LENGTH, 0, -ARM_LENGTH],
            [-ARM_LENGTH, 0, ARM_LENGTH, 0],
            [R_C, -R_C, R_C, -R_C]]

    def __init__(self, pos0, yaw0):
        """

        Args:
            pos0: initial x,y,z position
            yaw0: initial yaw angle

            ipos (list): indices for position variables x,y,z
            ivel (list): indices for velocity variables xdot, ydot, zdot
            irot (list): indices for rotational variables q1,q2,q3,q4
            iome (list): indices for rotational velocity variables
            imom (list): indices for moments
            ifor (list): indices for forces
            istate (list): indices for states
            dstate (int): dimensionality of state
            daction (int): dimensionality of actions (moments and forces)
            state: x,y,z, xdot, ydot, zdot, quatw, quatx, quaty, quatz, p, q, r
            iomo:
        Attributes:
        """
        # state =[x,y,z, xdot, ydot, zdot, quatw, quatx, quaty, quatz, p, q, r]
        self.ipos = [0, 1, 2]
        self.ivel = [3, 4, 5]
        self.irot = [6, 7, 8, 9]
        self.iome = [10, 11, 12]
        self.imom = [1, 2, 3]
        self.ifor = [0]
        self.istate = self.ipos + self.ivel + self.irot + self.iome
        self.dstate = len(self.istate)
        self.daction = len(self.imom + self.ifor)

        self.state = np.zeros(self.dstate)
        self.state[self.irot] = Quaternion(axis=(0, 0, 1),
                                           radians=yaw0).normalised.elements
        self.state[self.ipos] = np.array(pos0)

        self.iomo = np.stack(list(itertools.product(self.iome, self.imom)))

        self.R2M = np.linalg.inv(np.array(self.M2R))

    def _dyn(self, t, state, u, w):
        """ dynamics of quadcopter

        dstate/dt = dyn(t,state,u,w)

        Args:
            t: time
            state: x,y,z, xdot, ydot, zdot, quatw, quatx, quaty, quatz, p, q, r
            u: actions
            w: disturbances

        Returns:
            time derivative of the state dstate/dt
        """
        inertia = np.array(self.INERTIA)
        rot = Quaternion(state[self.irot]).normalised
        omega = state[self.iome]
        state_dot = np.zeros(self.dstate)

        # translatory velocity
        state_dot[self.ipos] = state[self.ivel]

        # translatory acceleration
        state_dot[self.ivel] = np.array(self.GRAVITY_ACC) + w / self.MASS + \
                               rot.inverse.rotate(
                                   [0, 0, u[self.ifor] / self.MASS])

        # rotatory velocity
        state_dot[self.irot] = (
                0.5 * Quaternion(vector=-omega) * rot).normalised.elements

        # rotatory acceleration
        state_dot[self.iome] = np.linalg.solve(inertia, u[self.imom]
                                               - np.cross(omega, inertia.dot(
            omega)))
        return state_dot

    def simulate(self, ctrl, disturbance):
        """ simulates the quadcopter by one time step

        Args:
            ctrl: control actions
            disturbance: disturbance function

        Returns:
            state at the next time step and the observed disturbance
        """
        u = self._impose_actuator_limits(ctrl(self.state))
        w = disturbance(self.state[self.ipos])
        ivp_sol = solve_ivp(self._dyn, [0, self.DT], self.state, args=(u, w))
        self.state = ivp_sol.y[:, -1]
        return self.state, w

    def _impose_actuator_limits(self, ubar):
        """ Enforces the actuator limits on the actions

        Args:
            ubar: actions given by the controller

        Returns:
            projection of ubar on the feasible action set
        """
        limits = LinearConstraint(self.R2M, self.MIN_THROTTLE,
                                  self.MAX_THROTTLE / 4)

        def obj_fun(u):
            return np.linalg.norm(ubar - u)

        solmin = minimize(obj_fun, np.zeros_like(ubar), method='SLSQP',
                          constraints=limits)
        return solmin.x

    def get_position(self):
        """ extracts position of quadcopter state

        Returns:
            x,y,z position of quadcopter
        """
        return self.state[self.ipos]
