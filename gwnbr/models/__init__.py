"""
gwnbr.models
------------
Model classes for GW count regression.

GWNBRg : GWNBR with globally estimated overdispersion (recommended start).
GWNBR  : Full GWNBR with locally estimated overdispersion.
GWPR   : Geographically Weighted Poisson Regression (baseline).
"""

from gwnbr.models.gwnbrg import GWNBRg
from gwnbr.models.gwnbr import GWNBR
from gwnbr.models.gwpr import GWPR

__all__ = ["GWNBRg", "GWNBR", "GWPR"]
