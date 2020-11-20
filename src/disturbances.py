import numpy as np
from scipy.interpolate import RectBivariateSpline
try:
    dist_data_z = np.loadtxt("../data/thermal_data_z.txt", dtype=np.float32)
except:
    dist_data_z = np.loadtxt("./data/thermal_data_z.txt", dtype=np.float32)


dist_data_vec_y = np.linspace(-0.05, 0.15, dist_data_z.shape[1])
dist_data_vec_x = np.linspace(-0.05, 0.15, dist_data_z.shape[0])
interp_dist = RectBivariateSpline(dist_data_vec_x, dist_data_vec_y,
                                  2 * dist_data_z, s=50)


def thermal(x):
    """ thermal disturbance model

    Data taken from https://thermal.kk7.ch/

    Args:
        x: current input nparray [n,2]

    Returns:
        disturbance as nparray [n,3]
    """
    if x.ndim == 1:
        x = x[0:2].reshape(-1, 2)
    n = x.shape[0]
    w = interp_dist(x[:, 0], x[:, 1], grid=False) * (1 + np.random.rand(n)) / 2
    return np.concatenate((np.zeros((n, 2)), w.reshape(n, 1)), axis=1)


def gaussians(x):
    """ Synthetic disturbance model build of Gaussians

    Args:
        x: current input nparray [n,2]

    Returns:
        disturbance as nparray [n,3]
    """
    from scipy.stats import multivariate_normal
    if x.ndim == 1:
        x = x[0:2].reshape(-1, 2)
    n = x.shape[0]
    sigma = np.array([[0.001, 0], [0, 0.001]])
    c = np.sqrt(((2 * np.pi) ** 2) * np.linalg.det(sigma))
    mv11 = multivariate_normal(mean=[0.1, 0.1], cov=sigma).pdf(x[:, 0:2]) * c
    mv00 = multivariate_normal(mean=[0, 0], cov=sigma).pdf(x[:, 0:2]) * c
    mv10 = multivariate_normal(mean=[0.1, 0], cov=sigma).pdf(x[:, 0:2]) * c
    w = 0.6 * mv00 * (np.random.rand(n) - 2.5) \
        + 0.8 * mv11 * (np.random.rand(n) - 2.5) \
        + 0.3 * mv10 * (np.random.rand(n) - 3.5)
    return np.concatenate((np.zeros((n, 2)), w.reshape(n, 1)), axis=1)


def gaussiansL(x):
    """ Synthetic disturbance model build of Gaussians

    Args:
        x: current input nparray [n,2]

    Returns:
        disturbance as nparray [n,3]
    """
    from scipy.stats import multivariate_normal
    if x.ndim == 1:
        x = x[0:2].reshape(-1, 2)
    n = x.shape[0]
    sigma = np.array([[0.01, 0], [0, 0.01]])
    c = np.sqrt(((2 * np.pi) ** 2) * np.linalg.det(sigma))
    mv11 = multivariate_normal(mean=[1, 1], cov=sigma).pdf(x[:, 0:2]) * c
    mv00 = multivariate_normal(mean=[0, 0], cov=sigma).pdf(x[:, 0:2]) * c
    mv10 = multivariate_normal(mean=[1, 0], cov=sigma).pdf(x[:, 0:2]) * c
    w = 0.6 * mv00 * (np.random.rand(n) - 2.5) \
        + 0.8 * mv11 * (np.random.rand(n) - 2.5) \
        + 0.3 * mv10 * (np.random.rand(n) - 3.5)
    return np.concatenate((np.zeros((n, 2)), w.reshape(n, 1)), axis=1)
