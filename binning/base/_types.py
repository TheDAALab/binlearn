"""Type definitions for binning classes.

This module provides type aliases for bin specifications and representatives
used throughout the binning framework.
"""

from collections.abc import Hashable
from typing import Any, Dict, List

BinEdges = List[float]
"""A list of bin edge values for interval binning."""

IntervalBinSpec = Dict[Hashable, BinEdges]
"""A mapping from column identifier to a list of bin edges."""

FlexibleBinSpec = Dict[Hashable, List[Dict[str, Any]]]
"""A mapping from column identifier to a list of flexible bin definitions.
Each bin definition is a dictionary, e.g. {"singleton": value} or {"interval": (a, b)}.
"""

BinRepsType = Dict[Hashable, List[float]]
"""A mapping from column identifier to a list of bin representative values."""

IntervalBinSpec = Dict[Hashable, List[float]]
FlexibleBinSpec = Dict[Hashable, List[dict]]
BinEdges = List[float]
