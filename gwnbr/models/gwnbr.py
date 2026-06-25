"""
gwnbr.models.gwnbr
-------------------
Full Geographically Weighted Negative Binomial Regression (GWNBR).

In GWNBR, both the regression coefficients beta_i AND the
overdispersion parameter alpha_i are estimated locally for each
focal location, via alternating NR and IRLS iterations.

Bandwidth selection uses cross-validation (CV) because the effective
number of parameters contributed by the alpha surface (k_2) is not
analytically tractable. AICc is therefore not directly applicable.

Reference
---------
Silva & Rodrigues (2014). Statistics and Computing, 24, 769-783.
"""

from __future__ import annotations
import numpy as np
from scipy import stats
from joblib import Parallel, delayed

from gwnbr.models.base import GWRBase, _nb_log_likelihood, _nb_deviance
from gwnbr.models.gwnbrg import _fit_global_nb
from gwnbr.kernels import compute_weights
from gwnbr.utils.distance import pairwise_distances
from gwnbr.utils.nr_solver import fit_alpha_nr
from gwnbr.utils.irls_solver import irls, _working_weights


# -----------------------------------------------------------------------
# Single-tract fit helper (local alpha version)
# -----------------------------------------------------------------------

def _fit_single_tract_local(i: int,
                             X: np.ndarray,
                             y: np.ndarray,
                             distances_i: np.ndarray,
                             offset: np.ndarray,
                             alpha_init: float,
                             bandwidth: float,
                             kernel: str,
                             beta_init: np.ndarray,
                             max_outer: int,
                             max_irls: int,
                             tol_outer: float,
                             tol_irls: float) -> dict:
    """
    Estimate local beta_i and alpha_i for focal tract i (full GWNBR).
    Alternates NR (alpha) and IRLS (beta) until convergence.
    """
    w_i = compute_weights(distances_i, bandwidth, kernel)

    alpha = max(alpha_init, 1e-10)
    beta  = beta_init.copy() if beta_init is not None else None
    mu    = np.full(len(y), max(np.mean(y), 0.5))
    se_alpha = np.nan

    for _outer in range(max_outer):
        alpha_old = alpha

        # --- IRLS: update beta with current alpha ---
        beta, mu, cov_i, _ = irls(
            X, y, w_i, offset,
            alpha=alpha,
            beta_init=beta,
            max_iter=max_irls,
            tol=tol_irls
        )

        # --- NR: update alpha with current mu ---
        alpha_new, se_alpha, _ = fit_alpha_nr(
            y, mu, w_i,
            theta_init=max(1.0 / alpha, 1e-10),
            max_iter=100, tol=1e-3
        )
        alpha = max(alpha_new, 1e-10)

        d_alpha = abs(alpha - alpha_old)
        if alpha < 1e-3:
            d_alpha *= 100
        if d_alpha < tol_outer:
            break

    # Hat matrix row for focal tract i
    A_i = _working_weights(mu, alpha)
    W_combined = w_i * A_i
    XtWX = (X * W_combined[:, None]).T @ X
    if abs(np.linalg.det(XtWX)) < 1e-20:
        hat_row = np.zeros(len(y))
    else:
        hat_row = X[i] @ np.linalg.inv(XtWX) @ (X * W_combined[:, None]).T

    se_beta = np.sqrt(np.maximum(np.diag(cov_i), 0.0))

    return {
        "i"        : i,
        "beta"     : beta,
        "se_beta"  : se_beta,
        "alpha"    : alpha,
        "se_alpha" : se_alpha,
        "hat_row"  : hat_row,
        "y_hat_i"  : mu[i],
    }


# -----------------------------------------------------------------------
# GWNBR class
# -----------------------------------------------------------------------

class GWNBR(GWRBase):
    """
    Full Geographically Weighted Negative Binomial Regression (GWNBR).

    Both beta_i and alpha_i are estimated locally for each focal
    location via alternating NR and IRLS.

    Parameters
    ----------
    coords         : array-like, shape (n, 2)  [lon/x, lat/y].
    y              : array-like, shape (n,)    Count response variable.
    X              : array-like, shape (n, k)  Predictor matrix (no intercept).
    offset         : array-like, shape (n,) or None.
    variable_names : list of str, optional.

    Notes
    -----
    - Use GWNBRg when you want AICc-based bandwidth selection.
    - Use GWNBR when full local overdispersion surfaces are needed.
    - For bandwidth selection with GWNBR, use CV (see BandwidthSelector).

    Example
    -------
    >>> model = GWNBR(coords=coords, y=y, X=X,
    ...               offset=np.log(population))
    >>> model.fit(bandwidth=50.0, kernel='gaussian')
    >>> print(model.summary())
    """

    def __init__(self, coords, y, X, offset=None, variable_names=None):
        super().__init__(coords, y, X, offset, variable_names)

    def fit(self,
            bandwidth: float,
            kernel: str = "gaussian",
            n_jobs: int = -1,
            max_outer: int = 50,
            max_irls: int = 100,
            tol_outer: float = 1e-5,
            tol_irls: float = 1e-6,
            verbose: bool = True) -> "GWNBR":
        """
        Fit the full GWNBR model.

        Parameters
        ----------
        bandwidth  : float  Spatial bandwidth.
        kernel     : str    'gaussian', 'bisquare', or 'adaptive_nn'.
        n_jobs     : int    Parallel jobs. -1 = all CPUs.
        max_outer  : int    Max alternating NR/IRLS iterations per tract.
        max_irls   : int    Max IRLS iterations per outer step.
        tol_outer  : float  Outer convergence tolerance (on delta alpha).
        tol_irls   : float  IRLS convergence tolerance.
        verbose    : bool   Print progress.

        Returns
        -------
        self
        """
        self.bandwidth = bandwidth
        self._kernel = kernel

        if verbose:
            print(f"[GWNBR] Computing {self.n}×{self.n} distance matrix...")

        D = pairwise_distances(self.coords)

        if verbose:
            print("[GWNBR] Estimating global NB for initialisation...")

        alpha_g, beta_g, _ = _fit_global_nb(self.X, self.y, self.offset)

        if verbose:
            print(f"[GWNBR] Global alpha init = {alpha_g:.6f}")
            print(f"[GWNBR] Fitting {self.n} local NR+IRLS loops "
                  f"(bandwidth={bandwidth}, kernel='{kernel}') ...")

        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_fit_single_tract_local)(
                i, self.X, self.y, D[i], self.offset,
                alpha_g, bandwidth, kernel, beta_g,
                max_outer, max_irls, tol_outer, tol_irls
            )
            for i in range(self.n)
        )

        # --- Collect ---
        self.betas      = np.zeros((self.n, self.p))
        self.se_betas   = np.zeros((self.n, self.p))
        self.alphas     = np.zeros(self.n)
        self.se_alphas  = np.zeros(self.n)
        self.y_hat      = np.zeros(self.n)
        self.hat_matrix = np.zeros((self.n, self.n))

        for r in results:
            i = r["i"]
            self.betas[i]     = r["beta"]
            self.se_betas[i]  = r["se_beta"]
            self.alphas[i]    = r["alpha"]
            self.se_alphas[i] = r["se_alpha"] if not np.isnan(r["se_alpha"]) else 0.0
            self.y_hat[i]     = r["y_hat_i"]
            self.hat_matrix[i] = r["hat_row"]

        self.t_stats  = self.betas / np.where(self.se_betas < 1e-20,
                                               1e-20, self.se_betas)
        self.p_values = 2.0 * (1.0 - stats.norm.cdf(np.abs(self.t_stats)))

        # Effective parameters = trace(S) + 1
        # Note: k_2 is technically unknown for GWNBR; using +1 as approximation
        self.n_params = float(np.trace(self.hat_matrix)) + 1.0

        self.deviance = _nb_deviance(self.y, self.y_hat, self.alphas)
        self.log_likelihood = _nb_log_likelihood(
            self.y, self.y_hat, self.alphas)

        self._compute_diagnostics()
        self._fitted = True

        if verbose:
            print(f"[GWNBR] Done.  AICc={self.AICc:.2f}  "
                  f"Pseudo-R2={self.pct_deviance:.4f}  "
                  f"Mean alpha={np.mean(self.alphas):.4f}")

        return self

    def significant_betas(self, alpha_level: float = 0.05) -> np.ndarray:
        """
        Multiple-testing corrected significance mask (n, p).
        Silva & Fotheringham (2015) correction.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        adj_alpha = alpha_level * (self.p / self.n_params)
        return self.p_values < adj_alpha
