"""
This module brings together all tools.
"""

# Configuration management
from .config import get_config, set_config, load_config, reset_config

# Error handling
from .errors import (
    BinningError, InvalidDataError, ConfigurationError, 
    FittingError, TransformationError, ValidationError
)

# Sklearn utilities
from .sklearn_utils import BinningFeatureSelector, BinningPipeline

# Base classes and utilities
from .base import (
    # Constants
    MISSING_VALUE, ABOVE_RANGE, BELOW_RANGE,
    # Types  
    BinEdges, IntervalBinSpec, FlexibleBinSpec, BinRepsType,
    # Base classes
    GeneralBinningBase, IntervalBinningBase, FlexibleBinningBase, SupervisedBinningBase,
    # Utility functions
    prepare_array, return_like_input, prepare_input_with_columns,
    ensure_bin_dict, validate_bins, default_representatives, create_bin_masks
)

# Concrete binning methods
from .methods import EqualWidthBinning, OneHotBinning, SupervisedBinning

# Optional pandas/polars configurations (if available)
try:
    from ._pandas_config import *
except ImportError:
    pass

try:
    from ._polars_config import *  
except ImportError:
    pass

__all__ = [
    # Configuration
    'get_config', 'set_config', 'load_config', 'reset_config',
    
    # Errors
    'BinningError', 'InvalidDataError', 'ConfigurationError', 
    'FittingError', 'TransformationError', 'ValidationError',
    
    # Sklearn utilities
    'BinningFeatureSelector', 'BinningPipeline',
    
    # Constants
    'MISSING_VALUE', 'ABOVE_RANGE', 'BELOW_RANGE',
    
    # Types
    'BinEdges', 'IntervalBinSpec', 'FlexibleBinSpec', 'BinRepsType',
    
    # Base classes
    'GeneralBinningBase', 'IntervalBinningBase', 'FlexibleBinningBase', 'SupervisedBinningBase',
    
    # Utility functions
    'prepare_array', 'return_like_input', 'prepare_input_with_columns',
    'ensure_bin_dict', 'validate_bins', 'default_representatives', 'create_bin_masks',
    
    # Concrete methods
    'EqualWidthBinning', 'OneHotBinning', 'SupervisedBinning'
]
