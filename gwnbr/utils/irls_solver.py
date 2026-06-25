"""
gwnbr.utils.irls_solver
------------------------
Iteratively Reweighted Least Squares (IRLS) solver for local
beta coefficients of the GWNBR / GWNBRg model.

Translates the IRLS inner loop from the SAS macro of
Silva & Rodrigues (2014), specifically the block:

    /* computing beta */
    do while (abs(ddev) > 0.000001);
        Ai = ...
        zj = ...
        bi = inv(X'*(wi#Ai#X)) * X'*(wi#Ai#zj)
        ...
    end;

The working weight matrix A_i (Equation 8 in the paper) uses the
observed Fisher Information formulation for the NB-2 distribution.
"""

import numpy as np
from scipy.special import gammaln


def _working_weights(mu: np.ndarray, alpha: float) -> np.ndarray:
    """
    Compute the IRLS working weight vector A_i.

    This is the diagonal of the GLM weight matrix (Equation 8,
    Silva & Rodrigues 2014) for NB-2:

        A_i = mu/(1 + alpha*mu)
              + (y - mu) * alpha*mu / (1 + alpha*mu)^2

    For alpha = 0 (Poisson), this reduces to A_i = mu.

    Parameters
    ----------
    mu    : np.ndarray, shape (n,)  Current fitted means.
    alpha : float                   Overdispersion parameter.

    Returns
    -------
    A : np.ndarray, shape (n,)
    """
    if alpha < 1e-10:   # Poisson limit
        return mu.copy()

    denom1 = 1.0 + alpha * mu
    denom2 = denom1 ** 2
    A = mu / denom1 + (alpha * mu) / denom2   # simplified from SAS
    A = np.where(A <= 0, 1e-5, A)             # stability clamp
    return A


def _nb_deviance(y: np.ndarray, mu: np.ndarray,
                 alpha: float) -> float:
    """
    NB-2 deviance (Equation 11 simplified, Silva & Rodrigues 2014).

    For alpha = 0, reduces to Poisson deviance.
    """
    tt = np.where(mu > 0, y / mu, 1e-10)
    tt = np.where(tt == 0, 1e-10, tt)

    if alpha < 1e-10:
        return float(2.0 * np.sum(y * np.log(tt) - (y - mu)))

    inv_alpha = 1.0 / alpha
    ratio1 = 1.0 + alpha * y
    ratio2 = 1.0 + alpha * mu
    ratio1 = np.where(ratio1 <= 0, 1e-10, ratio1)
    ratio2 = np.where(ratio2 <= 0, 1e-10, ratio2)
    return float(2.0 * np.sum(y * np.log(tt)
                              - (y + inv_alpha) * np.log(ratio1 / ratio2)))


def irls(X: np.ndarray,
         y: np.ndarray,
         spatial_weights: np.ndarray,
         offset: np.ndarray,
         alpha: float,
         beta_init: np.ndarray = None,
         max_iter: int = 100,
         tol: float = 1e-6) -> tuple:
    """
    Run one IRLS pass to estimate local beta coefficients.

    Solves the locally weighted NB GLM for focal location i:

        beta_i = (X' W_s A X)^{-1}  X' W_s A z

    where:
        W_s = diag(spatial_weights)   spatial kernel weights
        A   = diag(working weights)   IRLS GLM weights
        z   = adjusted dependent variable (working response)

    Parameters
    ----------
    X               : np.ndarray, shape (n, p)
                      Design matrix (with intercept column).
    y               : np.ndarray, shape (n,)
                      Observed counts.
    spatial_weights : np.ndarray, shape (n,)
                      Kernel weights for focal location i.
    offset          : np.ndarray, shape (n,)
                      Log-offset (e.g. log population). Zeros if unused.
    alpha           : float
                      Fixed overdispersion parameter for this iteration.
    beta_init       : np.ndarray or None
                      Starting beta. If None, uses log(mean(y)).
    max_iter        : int   Maximum IRLS iterations.
    tol             : float Convergence tolerance on deviance change.

    Returns
    -------
    beta    : np.ndarray, shape (p,)   Estimated coefficients.
    mu      : np.ndarray, shape (n,)   Fitted means.
    cov_mat : np.ndarray, shape (p, p) Covariance matrix of beta.
    converged : bool
    """
    n, p = X.shape

    # --- Initialise ---
    if beta_init is not None:
        beta = beta_init.copy()
    else:
        mu_init = np.full(n, max(np.mean(y), 0.5))
        beta = np.zeros(p)
        beta[0] = np.log(np.mean(mu_init))

    eta = X @ beta + offset
    eta = np.clip(eta, -100, 100)
    mu = np.exp(eta)

    dev_old = _nb_deviance(y, mu, alpha)
    converged = False

    for _iter in range(max_iter):
        A = _working_weights(mu, alpha)

        # Working response z_j (adjusted dependent variable)
        z = eta + (y - mu) / (A * (1.0 + alpha * mu)) - offset

        # Weighted design: W_combined = spatial_weights * A
        W = spatial_weights * A           # element-wise, shape (n,)

        # Normal equations: (X' W X) beta = X' W z
        XtWX = (X * W[:, None]).T @ X    # (p, p)
        XtWz = (X * W[:, None]).T @ z    # (p,)

        if abs(np.linalg.det(XtWX)) < 1e-20:
            # Singular matrix — keep previous beta
            break

        beta = np.linalg.solve(XtWX, XtWz)

        eta = X @ beta + offset
        eta = np.clip(eta, -100, 100)
        mu = np.exp(eta)

        dev_new = _nb_deviance(y, mu, alpha)
        d_dev = dev_new - dev_old

        if abs(d_dev) < tol:
            converged = True
            break
        dev_old = dev_new

    # Covariance matrix: (X' W A X)^{-1}
    A_final = _working_weights(mu, alpha)
    W_final = spatial_weights * A_final
    XtWX_final = (X * W_final[:, None]).T @ X
    if abs(np.linalg.det(XtWX_final)) < 1e-20:
        cov_mat = np.full((p, p), np.nan)
    else:
        cov_mat = np.linalg.inv(XtWX_final)

    return beta, mu, cov_mat, converged
