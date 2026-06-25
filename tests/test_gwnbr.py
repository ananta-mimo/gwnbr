"""
tests/test_gwnbr.py
--------------------
Unit and integration tests for the gwnbr package.

Run with:  pytest tests/ -v
"""

import numpy as np
import pytest
from gwnbr.models import GWNBRg, GWNBR, GWPR
from gwnbr.kernels import compute_weights, gaussian_kernel, bisquare_kernel
from gwnbr.utils.distance import haversine_distances, euclidean_distances
from gwnbr.utils.nr_solver import fit_alpha_nr
from gwnbr.utils.irls_solver import irls


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def small_nb_data():
    """
    Small synthetic dataset: 60 tracts, 2 predictors.
    Returns coords, y, X, offset.
    """
    rng = np.random.default_rng(42)
    n = 60

    # Lat/lon grid (roughly Maryland-sized)
    lon = rng.uniform(-79.5, -75.0, n)
    lat = rng.uniform(37.9, 39.7, n)
    coords = np.column_stack([lon, lat])

    # Predictors
    income     = rng.standard_normal(n)
    unemploy   = rng.standard_normal(n)
    X = np.column_stack([income, unemploy])

    # Population (offset)
    population = rng.integers(1000, 50000, n).astype(float)
    offset = np.log(population)

    # True parameters (vary spatially for realism)
    beta0 = -2.5 + 0.3 * lat
    beta1 = -0.3 + 0.05 * lon
    beta2 = 0.2

    mu_true = np.exp(beta0 + beta1 * income + beta2 * unemploy + offset)
    alpha_true = 0.5

    # Draw NB counts
    p_nb = 1.0 / (1.0 + alpha_true * mu_true)
    r_nb = 1.0 / alpha_true
    y = rng.negative_binomial(r_nb, p_nb).astype(float)

    return coords, y, X, offset


# -----------------------------------------------------------------------
# Distance tests
# -----------------------------------------------------------------------

class TestDistance:
    def test_haversine_symmetry(self, small_nb_data):
        coords, *_ = small_nb_data
        D = haversine_distances(coords)
        assert D.shape == (60, 60)
        np.testing.assert_allclose(D, D.T, atol=1e-10)

    def test_haversine_diagonal_zero(self, small_nb_data):
        coords, *_ = small_nb_data
        D = haversine_distances(coords)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-10)

    def test_euclidean_positive(self):
        coords = np.array([[0, 0], [3, 4], [6, 8]], dtype=float)
        D = euclidean_distances(coords)
        assert D[0, 1] == pytest.approx(5.0, rel=1e-5)

    def test_haversine_known_distance(self):
        """DC to Baltimore is roughly 60 km."""
        dc   = np.array([[-77.036, 38.907]])
        balt = np.array([[-76.612, 39.290]])
        coords = np.vstack([dc, balt])
        D = haversine_distances(coords)
        assert 50 < D[0, 1] < 70


# -----------------------------------------------------------------------
# Kernel tests
# -----------------------------------------------------------------------

class TestKernels:
    def test_gaussian_at_zero(self):
        w = gaussian_kernel(np.array([0.0]), bandwidth=10.0)
        assert w[0] == pytest.approx(1.0)

    def test_gaussian_decreasing(self):
        d = np.array([0, 5, 10, 20])
        w = gaussian_kernel(d, bandwidth=10.0)
        assert np.all(np.diff(w) < 0)

    def test_bisquare_cutoff(self):
        d = np.array([0.0, 5.0, 10.0, 10.01])
        w = bisquare_kernel(d, bandwidth=10.0)
        assert w[-1] == 0.0
        assert w[0] == pytest.approx(1.0)

    def test_compute_weights_dispatch(self):
        d = np.array([0.0, 5.0, 20.0])
        w = compute_weights(d, bandwidth=15.0, kernel="gaussian")
        assert len(w) == 3
        assert w[0] > w[1] > w[2]


# -----------------------------------------------------------------------
# NR solver tests
# -----------------------------------------------------------------------

class TestNRSolver:
    def test_alpha_positive(self, small_nb_data):
        _, y, _, _ = small_nb_data
        mu = np.full(len(y), np.mean(y))
        w  = np.ones(len(y))
        alpha, se, conv = fit_alpha_nr(y, mu, w)
        assert alpha > 0

    def test_alpha_converges(self, small_nb_data):
        _, y, _, _ = small_nb_data
        mu = np.full(len(y), np.mean(y))
        w  = np.ones(len(y))
        _, _, conv = fit_alpha_nr(y, mu, w)
        assert conv


# -----------------------------------------------------------------------
# IRLS solver tests
# -----------------------------------------------------------------------

class TestIRLS:
    def test_irls_shapes(self, small_nb_data):
        coords, y, X_raw, offset = small_nb_data
        n = len(y)
        X = np.hstack([np.ones((n, 1)), X_raw])
        w = np.ones(n)
        beta, mu, cov, conv = irls(X, y, w, offset, alpha=0.5)
        assert beta.shape == (3,)
        assert mu.shape  == (n,)
        assert cov.shape == (3, 3)

    def test_irls_mu_positive(self, small_nb_data):
        coords, y, X_raw, offset = small_nb_data
        n = len(y)
        X = np.hstack([np.ones((n, 1)), X_raw])
        w = np.ones(n)
        _, mu, _, _ = irls(X, y, w, offset, alpha=0.5)
        assert np.all(mu > 0)


# -----------------------------------------------------------------------
# GWPR tests
# -----------------------------------------------------------------------

class TestGWPR:
    def test_fit_completes(self, small_nb_data):
        coords, y, X, offset = small_nb_data
        model = GWPR(coords, y, X, offset=offset,
                     variable_names=["income", "unemploy"])
        model.fit(bandwidth=200.0, kernel="gaussian",
                  n_jobs=1, verbose=False)
        assert model._fitted

    def test_output_shapes(self, small_nb_data):
        coords, y, X, offset = small_nb_data
        n = len(y)
        model = GWPR(coords, y, X, offset=offset)
        model.fit(bandwidth=200.0, n_jobs=1, verbose=False)
        assert model.betas.shape   == (n, 3)
        assert model.y_hat.shape   == (n,)
        assert model.t_stats.shape == (n, 3)

    def test_diagnostics_finite(self, small_nb_data):
        coords, y, X, offset = small_nb_data
        model = GWPR(coords, y, X, offset=offset)
        model.fit(bandwidth=200.0, n_jobs=1, verbose=False)
        assert np.isfinite(model.AICc)
        assert np.isfinite(model.pct_deviance)
        assert 0 <= model.pct_deviance <= 1


# -----------------------------------------------------------------------
# GWNBRg tests
# -----------------------------------------------------------------------

class TestGWNBRg:
    def test_fit_completes(self, small_nb_data):
        coords, y, X, offset = small_nb_data
        model = GWNBRg(coords, y, X, offset=offset,
                       variable_names=["income", "unemploy"])
        model.fit(bandwidth=200.0, n_jobs=1, verbose=False)
        assert model._fitted

    def test_alpha_positive(self, small_nb_data):
        coords, y, X, offset = small_nb_data
        model = GWNBRg(coords, y, X, offset=offset)
        model.fit(bandwidth=200.0, n_jobs=1, verbose=False)
        assert model.alpha_global > 0

    def test_better_than_gwpr(self, small_nb_data):
        """GWNBRg should have lower AICc than GWPR on overdispersed data."""
        coords, y, X, offset = small_nb_data
        gwnbrg = GWNBRg(coords, y, X, offset=offset)
        gwnbrg.fit(bandwidth=200.0, n_jobs=1, verbose=False)
        gwpr   = GWPR(coords, y, X, offset=offset)
        gwpr.fit(bandwidth=200.0, n_jobs=1, verbose=False)
        assert gwnbrg.AICc < gwpr.AICc

    def test_to_dataframe(self, small_nb_data):
        coords, y, X, offset = small_nb_data
        model = GWNBRg(coords, y, X, offset=offset,
                       variable_names=["income", "unemploy"])
        model.fit(bandwidth=200.0, n_jobs=1, verbose=False)
        df = model.to_dataframe()
        assert len(df) == len(y)
        assert "beta_income" in df.columns

    def test_summary_string(self, small_nb_data):
        coords, y, X, offset = small_nb_data
        model = GWNBRg(coords, y, X, offset=offset)
        model.fit(bandwidth=200.0, n_jobs=1, verbose=False)
        s = model.summary()
        assert "GWNBRg" in s
        assert "AICc" in s

    def test_significant_betas_shape(self, small_nb_data):
        coords, y, X, offset = small_nb_data
        model = GWNBRg(coords, y, X, offset=offset)
        model.fit(bandwidth=200.0, n_jobs=1, verbose=False)
        sig = model.significant_betas()
        assert sig.shape == (len(y), 3)
        assert sig.dtype == bool


# -----------------------------------------------------------------------
# Full GWNBR tests
# -----------------------------------------------------------------------

class TestGWNBR:
    def test_fit_completes(self, small_nb_data):
        coords, y, X, offset = small_nb_data
        model = GWNBR(coords, y, X, offset=offset,
                      variable_names=["income", "unemploy"])
        model.fit(bandwidth=200.0, n_jobs=1, verbose=False)
        assert model._fitted

    def test_local_alphas_positive(self, small_nb_data):
        coords, y, X, offset = small_nb_data
        model = GWNBR(coords, y, X, offset=offset)
        model.fit(bandwidth=200.0, n_jobs=1, verbose=False)
        assert np.all(model.alphas > 0)

    def test_diagnostics_finite(self, small_nb_data):
        coords, y, X, offset = small_nb_data
        model = GWNBR(coords, y, X, offset=offset)
        model.fit(bandwidth=200.0, n_jobs=1, verbose=False)
        assert np.isfinite(model.AICc)
        assert np.isfinite(model.pct_deviance)
