"""
author: Peter Huang
email: hbd730@gmail.com
license: BSD
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np


def get_poly_cc(n, k, t):
    """ computes coefficients for polynomial at given time

    Args:
        n: order of polynomial
        k: order of derivative
        t: time
    """
    assert (n > 0 and k >= 0), "order and derivative must be positive."

    cc = np.ones(n)
    d = np.linspace(0, n - 1, n)

    for i in range(n):
        for j in range(k):
            cc[i] = cc[i] * d[i]
            d[i] = d[i] - 1
            if d[i] == -1:
                d[i] = 0

    for i, c in enumerate(cc):
        cc[i] = c * np.power(t, d[i])

    return cc


def mst(waypoints):
    """ This function takes a list of desired waypoint [x0, x1, x2...xN]
     and returns a [8N,1] coefficients matrix for the N+1 waypoints.

    1.The Problem
    Generate a full trajectory across N+1 waypoint made of N polynomial
    line segment.Each segment is defined as 7 order polynomial defined as
    follow:
    Pi=ai_0+ ai1*t+ ai2*t^2+ ai3*t^3+ ai4*t^4+ ai5*t^5 + ai6*t^6 + ai7*t^7

    Each polynomial has 8 unknown coefficients, thus we will have 8*N
    unknown to solve in total, so we need to come up with 8*N constraints.

    2.The constraints
    In general, the constraints is a set of condition which define the
    initial and final state, continuity between each piecewise function.
    includes specifying continuity in higher derivatives of the trajectory
    at the intermediate waypoints.

    3.Matrix Design
    Since we have 8*N unknown coefficients to solve, and if we are given 8*N
    equations(constraints), then the problem becomes solving a linear equation.

    a * coeff = b

    Let's look at b matrix first, b matrix is simple because it is just some
    constants on the right hand side of the equation. There are 8xN constraints,
    so b matrix will be [8N, 1].

    Now, how do we determine the dimension of coeff matrix? coeff is the final
    output matrix consists of 8*N elements. Since b matrix is only one column,
    thus coeff matrix must be [8N, 1].

    coeff.transpose = [a10 a11..a17...aN0 aN1..aN7]

    a matrix is tricky, we then can think of a matrix as a coeffient-coeffient
    matrix. We are no longer looking at a particular polynomial Pi, but rather
    P1, P2...PN as a whole. Since now our coeff matrix is [8N, 1],
    and b is [8N, 8N], thus a matrix must have the form [8N, 8N].

    a = [A10 A12 ... A17 ... AN0 AN1 ...AN7
         ...
        ]

    Each element in a row represents the coefficient of coeffient aij under
    a certain constraint, where aij is the jth coeffient of Pi
    with i = 1...N, j = 0...7.
    """

    n = len(waypoints) - 1

    # initialize a, and b matrix
    a = np.zeros((8 * n, 8 * n))
    b = np.zeros((8 * n, 1))

    # populate b matrix.
    for i in range(n):
        b[i] = waypoints[i]
        b[i + n] = waypoints[i + 1]

    # Constraint 1
    for i in range(n):
        a[i][8 * i:8 * (i + 1)] = get_poly_cc(8, 0, 0)

    # Constraint 2
    for i in range(n):
        a[i + n][8 * i:8 * (i + 1)] = get_poly_cc(8, 0, 1)

    # Constraint 3
    for i in range(n):
        a[i + 2 * n][8 * i:8 * (i + 1)] = get_poly_cc(8, 1, 0)

    # Constraint 4
    for i in range(n):
        a[i + 3 * n][8 * i:8 * (i + 1)] = get_poly_cc(8, 1, 1)

    # Constraint 5
    for i in range(n):
        a[i + 4 * n][8 * i:8 * (i + 1)] = get_poly_cc(8, 2, 0)

    # Constraint 6
    for i in range(n):
        a[i + 5 * n][8 * i:8 * (i + 1)] = get_poly_cc(8, 2, 1)

    # Constraint 7
    for i in range(n):
        a[i + 6 * n][8 * i:8 * (i + 1)] = get_poly_cc(8, 3, 0)

    # Constraint 8
    for i in range(n):
        a[i + 7 * n][8 * i:8 * (i + 1)] = get_poly_cc(8, 3, 1)

    # solve for the coefficients
    coeff = np.linalg.solve(a, b)
    return coeff


class TrajectoryGenerator:
    def __init__(self, waypoints, t_total):
        self.waypoints = waypoints.tolist()
        self.t_total = t_total
        self._trajcoeff = self.get_mst_coefficients()

    def get_mst_coefficients(self):
        # generate MST coefficients for each segment, coeff is  1D array [64,]
        wp = np.array(self.waypoints)
        coeff_x = mst(wp[:, 0]).transpose()[0]
        coeff_y = mst(wp[:, 1]).transpose()[0]
        coeff_z = mst(wp[:, 2]).transpose()[0]
        return coeff_x, coeff_y, coeff_z

    def get_desired_state(self, t):
        """ The function takes known number of waypoints and time, then
        generates a minimum snap trajectory which goes through each waypoint.
        The output is the desired state associated with the next waypont for
        the time t.
        waypoints is [N,3] matrix, waypoints = [[x0,y0,z0]...[xn,yn,zn]].
        v is velocity in m/s
        """
        wp = np.array(self.waypoints)
        coeff_x, coeff_y, coeff_z = self._trajcoeff
        yaw = 0
        yawdot = 0.0
        acc = np.zeros(3)
        vel = np.zeros(3)

        # distance vector array, represents each segment's distance
        # distance = waypoints[0:-1] - waypoints[1:]
        n = len(wp)
        # t_seg is now each segment's travel time
        t_seg = np.ones(n - 1) * self.t_total / (n - 1)

        # accumulated time
        s = np.zeros(len(t_seg) + 1)
        s[1:] = np.cumsum(t_seg)

        # find which segment current t belongs to
        t_index = np.where(t >= s)[0][-1]

        # prepare the next desired state
        if t == 0:
            pos = wp[0]
            # t0 = get_poly_cc(8, 1, 0)
            # current_heading = np.array(
            #   [coeff_x[0:8].dot(t0), coeff_y[0:8].dot(t0)]) * (1.0 / t_seg[0])
        # stay hover at the last waypoint position
        elif t >= s[-1]:
            pos = wp[-1]
        else:
            # scaled time
            scale = (t - s[t_index]) / t_seg[t_index]
            start = 8 * t_index
            end = 8 * (t_index + 1)

            t0 = get_poly_cc(8, 0, scale)
            pos = np.array(
                [coeff_x[start:end].dot(t0), coeff_y[start:end].dot(t0),
                 coeff_z[start:end].dot(t0)])

            t1 = get_poly_cc(8, 1, scale)
            # chain rule applied
            vel = np.array(
                [coeff_x[start:end].dot(t1), coeff_y[start:end].dot(t1),
                 coeff_z[start:end].dot(t1)]) * (
                          1.0 / t_seg[t_index])

            t2 = get_poly_cc(8, 2, scale)
            # chain rule applied
            acc = np.array(
                [coeff_x[start:end].dot(t2), coeff_y[start:end].dot(t2),
                 coeff_z[start:end].dot(t2)]) * (
                          1.0 / t_seg[t_index] ** 2)

        return {'pos': pos, 'vel': vel, 'acc': acc, 'yaw': yaw,
                'yawdot': yawdot}
