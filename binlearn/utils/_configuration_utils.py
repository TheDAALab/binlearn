"""
Configuration utilities for binning methods.

This module provides standardized configuration handling patterns used across
multiple binning implementations to reduce code duplication and ensure consistency.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from binlearn.config import apply_config_defaults


def standardize_init_pattern(
    self: Any,
    method_name: str,
    user_params: dict[str, Any],
    extra_validations: dict[str, Callable[..., Any]] | None = None,
) -> None:
    """Standardized initialization pattern for binning methods.

    This function encapsulates the common initialization pattern used across
    all binning methods: applying configuration defaults, setting parameters
    as attributes, and running validations.

    Args:
        self: The binning method instance being initialized
        method_name: Name of the method for configuration lookup
        user_params: Dictionary of user-provided parameters
        extra_validations: Optional dictionary of {param_name: validator_func}
            for custom parameter validation
    """
    # Apply configuration defaults
    params = apply_config_defaults(method_name, user_params)

    # Set all parameters as instance attributes
    for key, value in params.items():
        setattr(self, key, value)

    # Run custom validations if provided
    if extra_validations:
        for param_name, validator in extra_validations.items():
            if hasattr(self, param_name):
                validator(getattr(self, param_name))


def create_param_dict_for_config(
    n_bins: Any = None, random_state: Any = None, **kwargs: Any
) -> dict[str, Any]:
    """Create parameter dictionary for configuration, filtering None values.

    Args:
        n_bins: Number of bins parameter
        random_state: Random state parameter
        **kwargs: Additional parameters

    Returns:
        Dictionary with non-None parameters
    """
    params = {"n_bins": n_bins, "random_state": random_state}
    params.update(kwargs)

    # Filter out None values
    return {k: v for k, v in params.items() if v is not None}


def get_effective_n_bins(n_bins: int | str, data_size: int) -> int:
    """Get effective number of bins based on parameter and data size.

    Args:
        n_bins: Number of bins (integer) or method string ('auto', 'sqrt', etc.)
        data_size: Size of the data to be binned

    Returns:
        Effective integer number of bins
    """
    if isinstance(n_bins, int):
        return n_bins

    # Handle string methods
    if n_bins == "auto":
        return min(50, max(2, int(data_size**0.5)))
    elif n_bins == "sqrt":
        return max(2, int(data_size**0.5))
    elif n_bins == "log":
        return max(2, int(np.log2(data_size)))
    else:
        # Default fallback
        return 10


def prepare_sklearn_estimator_params(
    user_params: dict[str, Any], sklearn_param_mapping: dict[str, str]
) -> dict[str, Any]:
    """Prepare parameters for sklearn estimators with parameter mapping.

    Args:
        user_params: User-provided parameters
        sklearn_param_mapping: Mapping from our parameter names to sklearn names
            e.g., {'n_bins': 'n_clusters', 'random_state': 'random_state'}

    Returns:
        Dictionary with parameters mapped to sklearn parameter names
    """
    sklearn_params = {}
    for our_name, sklearn_name in sklearn_param_mapping.items():
        if our_name in user_params:
            sklearn_params[sklearn_name] = user_params[our_name]
    return sklearn_params


def handle_common_warnings(method_name: str, **conditions: Any) -> None:
    """Handle common warnings based on conditions.

    Args:
        method_name: Name of the binning method for warning messages
        **conditions: Boolean conditions that trigger warnings
            e.g., small_data=True, few_bins=True
    """
    import warnings

    if conditions.get("small_data"):
        warnings.warn(
            f"{method_name} may not work well with very small datasets. "
            "Consider using a simpler binning method.",
            category=UserWarning,
            stacklevel=3,
        )

    if conditions.get("few_bins"):
        warnings.warn(
            f"{method_name} with very few bins may not capture data patterns well. "
            "Consider increasing the number of bins.",
            category=UserWarning,
            stacklevel=3,
        )

    if conditions.get("many_bins"):
        warnings.warn(
            f"{method_name} with many bins may overfit to the data. "
            "Consider reducing the number of bins.",
            category=UserWarning,
            stacklevel=3,
        )
