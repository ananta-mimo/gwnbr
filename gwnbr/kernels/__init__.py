"""
gwnbr.kernels
-------------
Spatial kernel weighting functions for GWR-based models.

Supported kernels
-----------------
gaussian    : Fixed-bandwidth Gaussian kernel (default for GWNBR).
              w_ij = exp(-0.5 * (d_ij / h)^2)
              Matches 'method=fixed' in the SAS macro.

bisquare    : Adaptive bisquare (bi-square) kernel.
              w_ij = (1 - (d_ij / h)^2)^2  if d_ij <= h, else 0.
              Matches 'method=adaptiven' in the SAS macro.

adaptive_nn : Adaptive nearest-neighbour (k-NN bisquare).
              Bandwidth h = number of neighbours k.
              Matches 'method=adaptive1' in the SAS macro.

References
----------
Fotheringham, Brunsdon & Charlton (2002). Geographically Weighted
    Regression. Wiley.
Silva & Rodrigues (2014). Statistics and Computing, 24, 769-783.
"""

import numpy as np


def gaussian_kernel(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    """
    Fixed-bandwidth Gaussian kernel.

    Parameters
    ----------
    distances : np.ndarray, shape (n,)
        Distances from focal point i to all n observations.
    bandwidth : float
        Fixed bandwidth h (same units as distances).

    Returns
    -------
    weights : np.ndarray, shape (n,)
        Kernel weights in [0, 1].
    """
    return np.exp(-0.5 * (distances / bandwidth) ** 2)


def bisquare_kernel(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    """
    Adaptive bisquare kernel.

    w_ij = (1 - (d/h)^2)^2  for d <= h
    w_ij = 0                 for d >  h

    Parameters
    ----------
    distances : np.ndarray, shape (n,)
    bandwidth : float  Maximum distance threshold.

    Returns
    -------
    weights : np.ndarray, shape (n,)
    """
    ratio = distances / bandwidth
    weights = np.where(ratio <= 1.0, (1.0 - ratio ** 2) ** 2, 0.0)
    return weights


def adaptive_nn_kernel(distances: np.ndarray, k: int) -> np.ndarray:
    """
    Adaptive nearest-neighbour bisquare kernel.

    The bandwidth adapts per focal point: h_i = distance to the
    k-th nearest neighbour.  Points within k neighbours receive
    a bisquare weight; all others receive 0.

    Parameters
    ----------
    distances : np.ndarray, shape (n,)
        Distances from focal point i to all other observations.
    k : int
        Number of nearest neighbours to include.

    Returns
    -------
    weights : np.ndarray, shape (n,)
    """
    sorted_d = np.sort(distances)
    k_clamped = min(k, len(sorted_d) - 1)
    h_i = sorted_d[k_clamped]       # bandwidth = distance to k-th neighbour
    if h_i < 1e-10:
        h_i = 1e-10
    return bisquare_kernel(distances, h_i)


def get_kernel(kernel: str):
    """
    Retrieve kernel function by name.

    Parameters
    ----------
    kernel : str
        One of {'gaussian', 'bisquare', 'adaptive_nn'}.

    Returns
    -------
    callable  Kernel function.

    Raises
    ------
    ValueError  If kernel name is not recognised.
    """
    _map = {
        "gaussian":    gaussian_kernel,
        "bisquare":    bisquare_kernel,
        "adaptive_nn": adaptive_nn_kernel,
    }
    if kernel not in _map:
        raise ValueError(
            f"Unknown kernel '{kernel}'. "
            f"Choose from: {list(_map.keys())}"
        )
    return _map[kernel]


def compute_weights(distances: np.ndarray,
                    bandwidth: float,
                    kernel: str = "gaussian") -> np.ndarray:
    """
    Compute kernel weights for a focal location.

    Convenience wrapper that dispatches to the correct kernel.

    Parameters
    ----------
    distances : np.ndarray, shape (n,)
        Distances from focal point i to all observations.
    bandwidth : float
        Bandwidth value (km for Haversine, same units for Euclidean,
        or integer k for adaptive_nn).
    kernel : str
        Kernel type: 'gaussian', 'bisquare', or 'adaptive_nn'.

    Returns
    -------
    weights : np.ndarray, shape (n,)
    """
    fn = get_kernel(kernel)
    if kernel == "adaptive_nn":
        return fn(distances, int(bandwidth))
    return fn(distances, bandwidth)
