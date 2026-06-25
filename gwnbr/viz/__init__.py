"""
gwnbr.viz
----------
Visualisation utilities for GWNBR model results.

Functions
---------
plot_coefficient_map   : Choropleth of local beta coefficients.
plot_alpha_map         : Choropleth of local alpha (GWNBR only).
plot_significance_map  : Map of locally significant tracts.
plot_bandwidth_search  : Plot Golden Section Search history.
plot_residuals         : Residual diagnostics.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

try:
    import geopandas as gpd
    _GPD_AVAILABLE = True
except ImportError:
    _GPD_AVAILABLE = False


def _check_mpl():
    if not _MPL_AVAILABLE:
        raise ImportError("matplotlib is required for plotting. "
                          "Install with: pip install matplotlib")


def plot_coefficient_map(model,
                         variable: str,
                         gdf=None,
                         figsize: tuple = (10, 8),
                         cmap: str = "RdBu_r",
                         title: str = None,
                         ax=None):
    """
    Choropleth map of local beta coefficients for one variable.

    If a GeoDataFrame is supplied, coefficients are joined to it
    for a proper polygon map. Otherwise, a scatter plot by coordinate
    is produced.

    Parameters
    ----------
    model    : fitted GWNBRg / GWNBR / GWPR object.
    variable : str   Variable name (must match model.var_names).
    gdf      : GeoDataFrame or None.
                If supplied, must have same row order as model data.
    figsize  : tuple
    cmap     : str   Matplotlib colormap.
    title    : str or None.
    ax       : matplotlib Axes or None.

    Returns
    -------
    fig, ax
    """
    _check_mpl()
    if not model._fitted:
        raise RuntimeError("Model must be fitted before plotting.")

    if variable not in model.var_names:
        raise ValueError(f"'{variable}' not in model.var_names: "
                         f"{model.var_names}")

    j = model.var_names.index(variable)
    coefs = model.betas[:, j]
    title = title or f"Local β — {variable}"

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    # Centre colormap at 0
    vabs = max(abs(np.nanmin(coefs)), abs(np.nanmax(coefs)))
    norm = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)

    if gdf is not None and _GPD_AVAILABLE:
        gdf = gdf.copy()
        gdf["_coef"] = coefs
        gdf.plot(column="_coef", cmap=cmap, norm=norm, ax=ax,
                 legend=True, legend_kwds={"label": "Coefficient"})
    else:
        sc = ax.scatter(model.coords[:, 0], model.coords[:, 1],
                        c=coefs, cmap=cmap, norm=norm, s=15)
        plt.colorbar(sc, ax=ax, label="Coefficient")

    ax.set_title(title)
    ax.set_xlabel("Longitude / X")
    ax.set_ylabel("Latitude / Y")
    fig.tight_layout()
    return fig, ax


def plot_significance_map(model,
                          variable: str,
                          alpha_level: float = 0.05,
                          gdf=None,
                          figsize: tuple = (10, 8),
                          title: str = None,
                          ax=None):
    """
    Map locally significant tracts for one variable.

    Uses the multiple-testing corrected significance from
    Silva & Fotheringham (2015).

    Parameters
    ----------
    model       : fitted model.
    variable    : str
    alpha_level : float  Nominal significance level.
    gdf, figsize, title, ax : as in plot_coefficient_map.
    """
    _check_mpl()
    if not model._fitted:
        raise RuntimeError("Model must be fitted before plotting.")

    j = model.var_names.index(variable)
    sig_mask = model.significant_betas(alpha_level)[:, j]
    coefs = model.betas[:, j]
    title = title or f"Local significance — {variable} (α={alpha_level})"

    colors = np.where(sig_mask,
                      np.where(coefs > 0, "#d7191c", "#2c7bb6"),
                      "#aaaaaa")

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    if gdf is not None and _GPD_AVAILABLE:
        gdf = gdf.copy()
        gdf["_color"] = colors
        gdf["_color"].fillna("#aaaaaa", inplace=True)
        gdf.plot(color=gdf["_color"], ax=ax)
    else:
        for color, label in [("#d7191c", "Sig. positive"),
                              ("#2c7bb6", "Sig. negative"),
                              ("#aaaaaa", "Not significant")]:
            mask = colors == color
            ax.scatter(model.coords[mask, 0], model.coords[mask, 1],
                       c=color, label=label, s=15)
        ax.legend()

    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_alpha_map(model,
                   gdf=None,
                   figsize: tuple = (10, 8),
                   cmap: str = "YlOrRd",
                   ax=None):
    """
    Map local overdispersion (alpha) — only meaningful for full GWNBR.
    """
    _check_mpl()
    if not model._fitted:
        raise RuntimeError("Model must be fitted before plotting.")

    alphas = np.atleast_1d(model.alphas)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    if gdf is not None and _GPD_AVAILABLE:
        gdf = gdf.copy()
        gdf["_alpha"] = alphas
        gdf.plot(column="_alpha", cmap=cmap, ax=ax, legend=True,
                 legend_kwds={"label": "Alpha (overdispersion)"})
    else:
        sc = ax.scatter(model.coords[:, 0], model.coords[:, 1],
                        c=alphas, cmap=cmap, s=15)
        plt.colorbar(sc, ax=ax, label="Alpha (overdispersion)")

    ax.set_title("Local Overdispersion (α)")
    fig.tight_layout()
    return fig, ax


def plot_bandwidth_search(selector, figsize: tuple = (8, 5), ax=None):
    """
    Plot the bandwidth search history from BandwidthSelector.

    Parameters
    ----------
    selector : fitted BandwidthSelector object.
    """
    _check_mpl()
    df = selector.history_dataframe()
    df = df.drop_duplicates("bandwidth").sort_values("bandwidth")

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(df["bandwidth"], df[selector.criterion], "o-", markersize=4)
    if selector.optimal_bandwidth is not None:
        ax.axvline(selector.optimal_bandwidth, color="red",
                   linestyle="--", label=f"Optimal: {selector.optimal_bandwidth:.2f}")
        ax.legend()
    ax.set_xlabel("Bandwidth")
    ax.set_ylabel(selector.criterion.upper())
    ax.set_title(f"Bandwidth Selection — {selector.criterion.upper()}")
    fig.tight_layout()
    return fig, ax


def plot_residuals(model, figsize: tuple = (12, 4), axes=None):
    """
    Three-panel residual diagnostic plot.

    Panels: (1) Residuals vs fitted, (2) histogram, (3) spatial scatter.
    """
    _check_mpl()
    if not model._fitted:
        raise RuntimeError("Fit model before plotting residuals.")

    raw_resid = model.y - model.y_hat

    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
    else:
        fig = axes[0].figure

    # 1. Residuals vs fitted
    axes[0].scatter(model.y_hat, raw_resid, s=8, alpha=0.5)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Fitted values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Fitted")

    # 2. Histogram of residuals
    axes[1].hist(raw_resid, bins=30, edgecolor="white")
    axes[1].set_xlabel("Residual")
    axes[1].set_title("Residual Distribution")

    # 3. Spatial residuals
    sc = axes[2].scatter(model.coords[:, 0], model.coords[:, 1],
                         c=raw_resid, cmap="RdBu_r", s=10)
    plt.colorbar(sc, ax=axes[2], label="Residual")
    axes[2].set_title("Spatial Residuals")

    fig.tight_layout()
    return fig, axes
