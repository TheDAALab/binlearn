"""Constants for binning framework.

This module defines special integer values used to represent missing and out-of-range bin indices.
"""

MISSING_VALUE = -2
"""Integer value representing a missing bin index (e.g., for NaN input)."""

BELOW_RANGE = -3
"""Integer value representing an input below the lowest bin edge."""

ABOVE_RANGE = -4
"""Integer value representing an input above the highest bin edge."""
