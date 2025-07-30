"""Pandas configuration for binning framework.

This module attempts to import pandas and sets a flag indicating its availability.
"""

try:
    import pandas as pd  # pylint: disable=import-error,unused-import  # noqa
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None  # pylint: disable=invalid-name
    PANDAS_AVAILABLE = False
