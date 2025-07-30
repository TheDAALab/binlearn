"""Polars configuration for binning framework.

This module attempts to import polars and sets a flag indicating its availability.
"""

try:
    import polars as pl  # pylint: disable=import-error,unused-import  # noqa

    POLARS_AVAILABLE = True
except ImportError:  # pragma: no cover
    pl = None  # pylint: disable=invalid-name  # pragma: no cover
    POLARS_AVAILABLE = False  # pragma: no cover
