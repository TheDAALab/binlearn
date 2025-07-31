"""Polars configuration for binning framework.

This module attempts to import polars and sets a flag indicating its availability.
"""

try:
    # pylint: disable=import-error,unused-import
    import polars as pl  # pragma: no cover

    POLARS_AVAILABLE = True  # pragma: no cover
except ImportError:  # pragma: no cover
    # pylint: disable=invalid-name
    pl = None  # pragma: no cover
    POLARS_AVAILABLE = False  # pragma: no cover
