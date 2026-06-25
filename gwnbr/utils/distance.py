"""
gwnbr.utils.distance
--------------------
Pairwise distance calculations for geographic coordinates.

Supports:
- Haversine (great-circle) distance for lat/lon coordinates
- Euclidean distance for projected coordinates

These match the distance logic in the original SAS macro by
Silva & Rodrigues (2014), which uses arco * 6371 for lat/lon
and Euclidean otherwise (detected by |coord| < 180).
"""

import numpy as np


EARTH_RADIUS_KM = 6371.0


def _is_latlon(coords: np.ndarray) -> bool:
    """
    Detect whether coordinates are lat/lon degrees.
    Mirrors the SAS logic: if abs(coord[,1]) < 180 → lat/lon.

    Parameters
    ----------
    coords : np.ndarray, shape (n, 2)
        Array of [longitude, latitude] or projected [x, y].

    Returns
    -------
    bool
    """
    return bool(np.all(np.abs(coords[:, 0]) < 180))


def haversine_distances(coords: np.ndarray) -> np.ndarray:
    """
    Compute full pairwise great-circle distances (km) using the
    Haversine formula.

    Parameters
    ----------
    coords : np.ndarray, shape (n, 2)
        Columns are [longitude_deg, latitude_deg].

    Returns
    -------
    D : np.ndarray, shape (n, n)
        Symmetric matrix of pairwise distances in kilometres.
    """
    lon = np.radians(coords[:, 0])
    lat = np.radians(coords[:, 1])

    # Broadcast pairwise differences
    dlon = lon[:, None] - lon[None, :]   # (n, n)
    dlat = lat[:, None] - lat[None, :]

    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat[:, None]) * np.cos(lat[None, :])
         * np.sin(dlon / 2.0) ** 2)

    # Clamp for numerical safety before arcsin
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


def euclidean_distances(coords: np.ndarray) -> np.ndarray:
    """
    Compute full pairwise Euclidean distances.

    Parameters
    ----------
    coords : np.ndarray, shape (n, 2)
        Projected coordinates [x, y].

    Returns
    -------
    D : np.ndarray, shape (n, n)
    """
    diff = coords[:, None, :] - coords[None, :, :]   # (n, n, 2)
    return np.sqrt(np.sum(diff ** 2, axis=2))


def pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """
    Auto-detect coordinate type and return pairwise distance matrix.

    Uses Haversine for lat/lon (|lon| < 180) and Euclidean otherwise,
    matching the original SAS macro detection logic.

    Parameters
    ----------
    coords : np.ndarray, shape (n, 2)
        [longitude/x, latitude/y].

    Returns
    -------
    D : np.ndarray, shape (n, n)
        Distance matrix in km (lat/lon) or input units (projected).
    """
    if _is_latlon(coords):
        return haversine_distances(coords)
    return euclidean_distances(coords)
