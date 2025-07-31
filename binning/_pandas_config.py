"""Pandas configuration for binning framework.

This module attempts to import pandas and sets a flag indicating its availability.
"""

try:
    # pylint: disable=import-error,unused-import
    import pandas as pd  # pragma: no cover

    PANDAS_AVAILABLE = True  # pragma: no cover
except ImportError:  # pragma: no cover
    # pylint: disable=invalid-name
    pd = None  # pragma: no cover
    PANDAS_AVAILABLE = False  # pragma: no cover
