"""
Enhanced error handling for the binning framework.
"""

import warnings
from typing import Any, cast

import numpy as np

from binlearn.config import get_config


class BinningError(Exception):
    """Base exception for all binning-related errors."""

    def __init__(self, message: str, suggestions: list[str] | None = None):
        super().__init__(message)
        self.suggestions = suggestions or []

    def __str__(self) -> str:
        msg = super().__str__()
        if self.suggestions:
            suggestions_text = "\n".join(f"  - {s}" for s in self.suggestions)
            msg += f"\n\nSuggestions:\n{suggestions_text}"
        return msg


class InvalidDataError(BinningError):
    """Raised when input data is invalid or incompatible."""


class ConfigurationError(BinningError):
    """Raised when configuration parameters are invalid."""


class FittingError(BinningError):
    """Raised when fitting process fails."""


class TransformationError(BinningError):
    """Raised when transformation fails."""


class ValidationError(BinningError):
    """Raised when validation fails."""


class BinningWarning(UserWarning):
    """Base warning for binning operations."""


class DataQualityWarning(BinningWarning):
    """Warning about data quality issues."""


class PerformanceWarning(BinningWarning):
    """Warning about potential performance issues."""


def validate_tree_params(task_type: str, tree_params: dict[str, Any]) -> dict[str, Any]:
    """Validate tree parameters for SupervisedBinning."""
    _ = task_type

    if not tree_params:
        return {}

    valid_params = {
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "random_state",
        "max_leaf_nodes",
        "min_impurity_decrease",
        "class_weight",
        "ccp_alpha",
        "criterion",
    }

    invalid_params = set(tree_params.keys()) - valid_params
    if invalid_params:
        raise ConfigurationError(
            f"Invalid tree parameters: {invalid_params}",
            suggestions=[
                f"Valid parameters are: {sorted(valid_params)}",
                "Check scikit-learn documentation for DecisionTree parameters",
            ],
        )

    # Validate specific parameter values
    if "max_depth" in tree_params:
        max_depth = tree_params["max_depth"]
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 1):
            raise ConfigurationError(
                f"max_depth must be a positive integer or None, got {max_depth}",
                suggestions=["Use positive integers like 3, 5, 10, or None for unlimited depth"],
            )

    if "min_samples_split" in tree_params:
        min_split = tree_params["min_samples_split"]
        if not isinstance(min_split, int) or min_split < 2:
            raise ConfigurationError(
                f"min_samples_split must be an integer >= 2, got {min_split}",
                suggestions=["Use values like 2, 5, 10 depending on your dataset size"],
            )

    if "min_samples_leaf" in tree_params:
        min_leaf = tree_params["min_samples_leaf"]
        if not isinstance(min_leaf, int) or min_leaf < 1:
            raise ConfigurationError(
                f"min_samples_leaf must be a positive integer, got {min_leaf}",
                suggestions=["Use values like 1, 3, 5 depending on your dataset size"],
            )

    return tree_params


def suggest_alternatives(method_name: str) -> list[str]:
    """Suggest alternative method names for common misspellings."""
    alternatives = {
        "supervised": ["tree", "decision_tree"],
        "equal_width": ["uniform", "equidistant"],
        "singleton": ["categorical", "nominal"],
        "quantile": ["percentile"],
    }

    suggestions = []
    for correct, aliases in alternatives.items():
        if method_name.lower() in aliases or method_name.lower() == correct:
            suggestions.extend([correct] + aliases)

    return list(set(suggestions))
