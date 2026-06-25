"""
gwnbr.utils.nr_solver
---------------------
Newton-Raphson solver for the local overdispersion parameter (alpha)
of the Negative Binomial distribution.

Translated from the NR loop in Silva & Rodrigues (2014) SAS macro.

The NB-2 parameterization uses:
    Var(Y) = mu + alpha * mu^2

where alpha = 1 / theta (theta = r = size parameter).

The local log-likelihood being maximized is Equation (11) of
Silva & Rodrigues (2014), weighted by spatial kernel weights w_i.
"""

import numpy as np
from scipy.special import digamma, polygamma


# polygamma(1, x) = trigamma(x)
trigamma = lambda x: polygamma(1, x)


def _score(theta: float, y: np.ndarray, mu: np.ndarray,
           weights: np.ndarray) -> float:
    """
    First derivative of weighted local log-likelihood w.r.t. theta.

    Equation (13) from Silva & Rodrigues (2014).

    Parameters
    ----------
    theta : float
        Current theta = 1/alpha (size parameter, must be > 0).
    y     : np.ndarray, shape (n,)  Observed counts.
    mu    : np.ndarray, shape (n,)  Current fitted means.
    weights : np.ndarray, shape (n,) Spatial kernel weights.

    Returns
    -------
    float  Score (first derivative).
    """
    term = (digamma(theta + y)
            - digamma(theta)
            + np.log(theta)
            + 1.0
            - np.log(theta + mu)
            - (theta + y) / (theta + mu))
    return float(np.sum(term * weights))


def _hessian(theta: float, y: np.ndarray, mu: np.ndarray,
             weights: np.ndarray) -> float:
    """
    Second derivative of weighted local log-likelihood w.r.t. theta.

    Equation (14) from Silva & Rodrigues (2014).

    Parameters
    ----------
    theta, y, mu, weights : as in _score().

    Returns
    -------
    float  Hessian (second derivative).
    """
    term = (trigamma(theta + y)
            - trigamma(theta)
            + 1.0 / theta
            - 2.0 / (theta + mu)
            + (y + theta) / ((theta + mu) ** 2))
    h = float(np.sum(term * weights))

    # Numerical stability guards (from SAS macro)
    if abs(h) < 1e-23:
        h = np.sign(h) * 1e-23 if h != 0 else 1e-23
    return h


def fit_alpha_nr(y: np.ndarray,
                 mu: np.ndarray,
                 weights: np.ndarray,
                 theta_init: float = 1.0,
                 max_iter: int = 100,
                 tol: float = 1e-3,
                 alpha_max: float = 1e8) -> tuple:
    """
    Estimate the local dispersion parameter via Newton-Raphson.

    Returns alpha = 1/theta and its standard error via the delta method
    (Equation 15, Silva & Rodrigues 2014):
        se(alpha) = se(theta) / theta^2
        se(theta) = sqrt(1 / |hessian|)

    Parameters
    ----------
    y          : np.ndarray  Observed counts.
    mu         : np.ndarray  Current fitted means.
    weights    : np.ndarray  Spatial kernel weights for focal tract i.
    theta_init : float       Starting value for theta (default 1.0).
    max_iter   : int         Maximum NR iterations.
    tol        : float       Convergence tolerance on delta_theta.
    alpha_max  : float       Cap on alpha to handle near-Poisson tracts.

    Returns
    -------
    alpha  : float  Estimated overdispersion  (alpha = 1/theta).
    se_alpha : float  Standard error of alpha.
    converged : bool
    """
    theta = max(theta_init, 1e-10)
    converged = False
    count = 0   # track consecutive negative-theta resets

    for _iter in range(max_iter):
        theta = max(theta, 1e-10)

        g = _score(theta, y, mu, weights)
        h = _hessian(theta, y, mu, weights)

        theta_old = theta
        theta = theta - g / h

        # Handle non-positive theta
        if theta <= 0:
            count += 1
            if count == 1:
                theta = 1e-6
            elif count == 2:
                theta = 1e-4
            else:
                theta = max(1.0 / alpha_max, 1e-10)

        delta = abs(theta - theta_old)
        if theta < 1e-3:
            delta *= 100          # scale up for tiny theta (SAS trick)

        if delta < tol:
            converged = True
            break

    theta = max(theta, 1e-10)
    alpha = 1.0 / theta

    # Standard error via delta method
    h_final = _hessian(theta, y, mu, weights)
    se_theta = np.sqrt(1.0 / abs(h_final)) if abs(h_final) > 1e-20 else np.nan
    se_alpha = se_theta / (theta ** 2)

    return alpha, se_alpha, converged
