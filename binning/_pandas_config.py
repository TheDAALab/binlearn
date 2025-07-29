"""Pandas configuration for binning framework.

This module attempts to import pandas and sets a flag indicating its availability.
"""

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:  # pragma: no cover
    PANDAS_AVAILABLE = False
    pd = None
