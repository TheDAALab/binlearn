"""Polars configuration for binning framework.

This module attempts to import polars and sets a flag indicating its availability.
"""

from typing import Any

# Initialize variables
pl: Any | None = None
POLARS_AVAILABLE = False

try:
    # pylint: disable=import-error,unused-import
    import polars  # pragma: no cover

    pl = polars  # pragma: no cover

    POLARS_AVAILABLE = True  # pragma: no cover
except ImportError:  # pragma: no cover
    # pl remains None as initialized above
    pass  # pragma: no cover
