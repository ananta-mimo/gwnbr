"""
gwnbr.bandwidth
----------------
Bandwidth selection for GW count regression models.

Implements the Golden Section Search algorithm from the SAS %golden
macro of Silva & Rodrigues (2014), translated to Python.

Supports three selection criteria:
- 'aicc' : Corrected AIC (recommended for GWNBRg)
- 'cv'   : Cross-validation sum of squared prediction errors
- 'aic'  : Standard AIC

And three bandwidth types:
- 'fixed'       : Fixed distance bandwidth (km or map units)
- 'adaptive_nn' : Adaptive k-nearest-neighbour bandwidth (integer k)
- 'bisquare'    : Fixed bisquare bandwidth

References
----------
Silva & Rodrigues (2014). Statistics and Computing, 24, 769-783.
Fotheringham, Brunsdon & Charlton (2002). GWR. Wiley.
"""

from __future__ import annotations
import numpy as np
from typing import Callable


class BandwidthSelector:
    """
    Optimal bandwidth search for GWNBRg / GWNBR / GWPR models.

    Uses the Golden Section Search method (translated from SAS %golden),
    which finds the bandwidth minimising a chosen criterion.

    Parameters
    ----------
    model_class  : class   One of GWNBRg, GWNBR, or GWPR.
    coords       : np.ndarray, shape (n, 2)
    y            : np.ndarray, shape (n,)
    X            : np.ndarray, shape (n, p)
    offset       : np.ndarray or None
    variable_names : list or None
    kernel       : str   'gaussian', 'bisquare', or 'adaptive_nn'.
    criterion    : str   'aicc' (default for GWNBRg), 'cv', or 'aic'.
    bw_min       : float or None  Lower bound for search. Auto if None.
    bw_max       : float or None  Upper bound for search. Auto if None.
    n_jobs       : int   Parallel jobs for model fitting.
    verbose      : bool

    Example
    -------
    >>> selector = BandwidthSelector(
    ...     GWNBRg, coords, y, X, offset=np.log(pop),
    ...     kernel='gaussian', criterion='aicc'
    ... )
    >>> optimal_bw = selector.search()
    >>> model = GWNBRg(coords, y, X, offset=np.log(pop))
    >>> model.fit(bandwidth=optimal_bw)
    """

    def __init__(self,
                 model_class,
                 coords: np.ndarray,
                 y: np.ndarray,
                 X: np.ndarray,
                 offset: np.ndarray = None,
                 variable_names: list = None,
                 kernel: str = "gaussian",
                 criterion: str = "aicc",
                 bw_min: float = None,
                 bw_max: float = None,
                 n_jobs: int = -1,
                 verbose: bool = True):

        self.model_class = model_class
        self.coords = np.asarray(coords, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.X = np.asarray(X, dtype=float)
        self.offset = offset
        self.variable_names = variable_names
        self.kernel = kernel
        self.criterion = criterion.lower()
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Auto-set bounds
        from gwnbr.utils.distance import pairwise_distances
        D = pairwise_distances(self.coords)
        self._max_dist = float(np.max(D))
        n = len(y)

        if kernel == "adaptive_nn":
            self._bw_min = float(bw_min) if bw_min is not None else 5.0
            self._bw_max = float(bw_max) if bw_max is not None else float(n)
            self._tol = 0.9
            self._integer_bw = True
        else:
            self._bw_min = float(bw_min) if bw_min is not None else 0.0
            self._bw_max = float(bw_max) if bw_max is not None else self._max_dist
            self._tol = 0.1
            self._integer_bw = False

        self._history = []   # list of (bandwidth, criterion_value)
        self.optimal_bandwidth = None

    def _evaluate(self, bandwidth: float) -> float:
        """Fit model at given bandwidth and return criterion value."""
        if self._integer_bw:
            bandwidth = round(bandwidth)

        model = self.model_class(
            self.coords, self.y, self.X,
            offset=self.offset,
            variable_names=self.variable_names
        )
        model.fit(bandwidth=bandwidth, kernel=self.kernel,
                  n_jobs=self.n_jobs, verbose=False)

        if self.criterion == "aicc":
            val = model.AICc
        elif self.criterion == "aic":
            val = model.AIC
        elif self.criterion == "cv":
            val = float(np.sum((self.y - model.y_hat) ** 2))
        else:
            raise ValueError(f"Unknown criterion '{self.criterion}'. "
                             "Use 'aicc', 'aic', or 'cv'.")

        self._history.append((bandwidth, val))
        if self.verbose:
            print(f"  bw={bandwidth:.4f}  {self.criterion.upper()}={val:.4f}")
        return val

    def search(self) -> float:
        """
        Run the Golden Section Search and return the optimal bandwidth.

        The search is translated directly from the %golden SAS macro,
        including the integer-snapping step for adaptive_nn kernels.

        Returns
        -------
        float  Optimal bandwidth.
        """
        GOLDEN_RATIO = 0.61803399
        C = 1.0 - GOLDEN_RATIO

        h0 = self._bw_min
        h3 = self._bw_max
        h1 = h0 + C * (h3 - h0)
        h2 = h0 + GOLDEN_RATIO * (h3 - h0)

        if self.verbose:
            print(f"[BandwidthSelector] Golden Section Search")
            print(f"  kernel={self.kernel}  criterion={self.criterion}")
            print(f"  search range: [{h0:.2f}, {h3:.2f}]")
            print(f"  initial bracket: h1={h1:.4f}  h2={h2:.4f}")

        res1 = self._evaluate(h1)
        res2 = self._evaluate(h2)

        n_iter = 0
        while abs(h3 - h0) > self._tol * 2:
            n_iter += 1
            if res2 < res1:
                h0 = h1
                h1 = h2
                h2 = C * h1 + GOLDEN_RATIO * h3
                res1 = res2
                res2 = self._evaluate(h2)
            else:
                h3 = h2
                h2 = h1
                h1 = C * h2 + GOLDEN_RATIO * h0
                res2 = res1
                res1 = self._evaluate(h1)

            if n_iter > 200:
                if self.verbose:
                    print("  [Warning] Max iterations reached.")
                break

        # Final bandwidth
        if self._integer_bw:
            xmin = (h3 + h0) / 2.0
            h_lo = int(np.floor(xmin))
            h_hi = int(np.ceil(xmin))
            v_lo = self._evaluate(float(h_lo))
            v_hi = self._evaluate(float(h_hi))
            optimal = h_lo if v_lo <= v_hi else h_hi
        else:
            optimal = (h3 + h0) / 2.0

        self.optimal_bandwidth = optimal

        if self.verbose:
            print(f"\n[BandwidthSelector] Optimal bandwidth: {optimal}")
            print(f"  {self.criterion.upper()} = "
                  f"{self._evaluate(optimal):.4f}")

        return optimal

    def history_dataframe(self):
        """Return search history as a pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self._history,
                            columns=["bandwidth", self.criterion])
