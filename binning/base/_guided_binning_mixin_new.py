"""
Legacy mixin for guided binning methods.

Note: This functionality has been moved to SupervisedBinningBase. 
This mixin is kept for backward compatibility but is no longer recommended.
Use SupervisedBinningBase for new supervised binning methods.
"""

from typing import Any, Dict, List, Tuple, Optional, Union
import numpy as np
from ..errors import InvalidDataError, DataQualityWarning


class GuidedBinningMixin:
    """
    Legacy mixin for guided binning methods.
    
    DEPRECATED: Use SupervisedBinningBase instead for new implementations.
    This class is maintained only for backward compatibility.
    """
    
    def __init__(self):
        import warnings
        warnings.warn(
            "GuidedBinningMixin is deprecated. Use SupervisedBinningBase instead.",
            DeprecationWarning,
            stacklevel=2
        )
