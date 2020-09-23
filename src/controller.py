import numpy as np
from pyquaternion import Quaternion


def quaternion2euler(q):
    """ converts quaternion rotation into euler angles

    Args:
        q: quaterion rotation 4 dimensional

    Returns:
        Euler angles as numpy array
    """
    #
    from math import atan2, asin
    phi = atan2(2 * (q[0] * q[1] + q[2] * q[3]),
                1 - 2 * (q[1] ** 2 + q[2] ** 2))
    theta = asin(2 * (q[0] * q[2] - q[3] * q[1]))
    psi = atan2(2 * (q[0] * q[3] + q[1] * q[2]),
                1 - 2 * (q[2] ** 2 + q[3] ** 2))
    return np.array([phi, theta, psi])


class Controller:
    MASS = 0.18
    GRAVITY_CONST = 9.81
    GAIN_BOOST = 2
    KP_TRA = [20, 20, 200]
    KD_TRA = [20, 20, 100]
    KP_ROT = [130, 130, 60]
    KD_ROT = [3, 3, 5]

    def __init__(self, dmodel, desired_state):
        """ A PD controller for a quadcopter with disturbance compensation

        Args:
            dmodel: disturbance model
            desired_state: desired state

        Attributes:
            dmodel: disturbance model
            desired_state: desired state
            base_gains (dict): proportional (P) and differential (D) gains
            ipos (list): indices for position variables x,y,z
            ivel (list): indices for velocity variables xdot, ydot, zdot
            irot (list): indices for rotational variables q1,q2,q3,q4
            iome (list): indices for rotational velocity variables
            imom (list): indices for moments
            ifor (list): indices for forces
            daction (int): dimensionality of actions (moments and forces)


        """
        self.dmodel = dmodel
        self.des_state = desired_state
        self.base_gains = dict()
        self.base_gains['Kp_tra'] = np.diag(self.KP_TRA)  # 50, 50, 500
        self.base_gains['Kd_tra'] = np.diag(self.KD_TRA)  # 30, 30, 200
        self.base_gains['Kp_rot'] = np.diag(self.KP_ROT)  # 160, 160, 80
        self.base_gains['Kd_rot'] = np.diag(self.KD_ROT)  # 3, 3, 5,
        self.ipos = [0, 1, 2]
        self.ivel = [3, 4, 5]
        self.irot = [6, 7, 8, 9]
        self.iome = [10, 11, 12]
        self.imom = [1, 2, 3]
        self.ifor = [0]
        self.daction = len(self.imom + self.ifor)

    def _get_gains(self, alea_unc):
        """ Computes dynamically the gains of the PD controller

        The gain in z direction is scaled based on the aleatoric uncertainty
        and GAIN_BOOST

        Args:
            alea_unc (float): aleatoric uncertainty

        Returns:
            gains (dict): rotational and translatory PD gains
        """
        gain_factor = 1 + self.GAIN_BOOST * float(alea_unc)
        gains = dict()
        gains['Kp_tra'] = self.base_gains['Kp_tra'] * np.diag(
            [1, 1, gain_factor])
        gains['Kd_tra'] = self.base_gains['Kd_tra'] * np.diag(
            [1, 1, gain_factor])
        gains['Kp_rot'] = self.base_gains['Kp_rot']
        gains['Kd_rot'] = self.base_gains['Kd_rot']

        return gains

    def get_last_epi(self):
        """ get most recent evaluation of epistemic uncertainty

        Returns:
            epistemic uncertainty
        """
        return self.last_epi

    def run_ctrl(self, state):
        """ executes the controller and computes commands

        Args:
            state: current state of the quadcopter

        Returns:
            the actions i.e. moments and forces for the quadcopter
        """
        action = np.zeros(self.daction)
        if self.dmodel is not None:
            dist_mean, alea_unc, epi = self.dmodel.predict(state[0:2])
            self.last_epi = epi
            predicted_disturbance = (1 - epi) * dist_mean
        else:
            predicted_disturbance = 0
            alea_unc = 0

        gains = self._get_gains(alea_unc)
        dstate = self.des_state
        comm_acc = dstate['acc'] \
                   + gains['Kd_tra'].dot(dstate['vel'] - state[self.ivel]) \
                   + gains['Kp_tra'].dot(dstate['pos'] - state[self.ipos])

        action[self.ifor] = self.MASS * (
                self.GRAVITY_CONST + comm_acc[2]) - predicted_disturbance

        des_rot = Quaternion(axis=(0, 0, 1),
                             angle=np.pi / 2 - dstate['yaw']).rotate(
            comm_acc) / self.GRAVITY_CONST
        des_rot[2] = dstate['yaw']

        des_omega = np.array([0, 0, dstate['yawdot']])

        rot = - quaternion2euler(Quaternion(state[self.irot]))
        action[self.imom] = gains['Kp_rot'].dot(des_rot - rot) + \
                            gains['Kd_rot'].dot(des_omega - state[self.iome])
        if np.any(np.isnan(action)):
            print('is NAN')
        return action

    def set_desired_state(self, desired_state):
        self.des_state = desired_state
