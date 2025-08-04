"""
Enhanced error handling for the binning framework.
"""

import warnings
from typing import Any, Dict, List, Optional, cast

import numpy as np

from binlearn.config import get_config


class BinningError(Exception):
    """Base exception for all binning-related errors."""

    def __init__(self, message: str, suggestions: Optional[List[str]] = None):
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


class ValidationMixin:
    """Mixin class providing enhanced validation capabilities."""

    @staticmethod
    def validate_array_like(
        data: Any, name: str = "data", allow_none: bool = False
    ) -> Optional[np.ndarray]:
        """Validate and convert array-like input."""
        if data is None and allow_none:
            return None

        if data is None:
            raise InvalidDataError(
                f"{name} cannot be None",
                suggestions=[
                    f"Provide a valid array-like object for {name}",
                    "Check if your data loading was successful",
                ],
            )

        try:
            arr = np.asarray(data)
        except Exception as e:
            raise InvalidDataError(
                f"Could not convert {name} to array: {str(e)}",
                suggestions=[
                    "Ensure input is array-like (list, numpy array, pandas DataFrame/Series)",
                    "Check for any invalid values in your data",
                    "Consider converting data types explicitly",
                ],
            ) from e

        # Check if array is empty - let specific methods handle this with their own error messages
        # if array.size == 0:
        #     raise ValueError(f"{name} is empty")

        return arr  # type: ignore[no-any-return]

    @staticmethod
    def validate_column_specification(columns: Any, data_shape: tuple) -> List[Any]:
        """Validate column specifications."""
        if columns is None:
            return list(range(data_shape[1]))

        # Convert single column to list
        if not isinstance(columns, (list, tuple, np.ndarray)):
            columns = [columns]

        # Validate each column
        validated_columns: List[Any] = []
        for col in columns:
            if isinstance(col, str):
                validated_columns.append(col)
            elif isinstance(col, int):
                if col < 0 or col >= data_shape[1]:
                    raise InvalidDataError(
                        f"Column index {col} is out of range for data with {data_shape[1]} columns",
                        suggestions=[
                            f"Use column indices between 0 and {data_shape[1] - 1}",
                            "Check if your data has the expected number of columns",
                        ],
                    )
                validated_columns.append(col)
            else:
                raise InvalidDataError(
                    f"Invalid column specification: {col} (type: {type(col)})",
                    suggestions=[
                        "Use string column names or integer indices",
                        "Ensure column specifications match your data format",
                    ],
                )

        return validated_columns

    @staticmethod
    def validate_guidance_columns(
        guidance_cols: Any, binning_cols: List[Any], data_shape: tuple
    ) -> List[Any]:
        """Validate guidance column specifications."""
        if guidance_cols is None:
            return []

        # Convert to list if needed
        if not isinstance(guidance_cols, (list, tuple)):
            guidance_cols = [guidance_cols]

        validated_guidance = ValidationMixin.validate_column_specification(
            guidance_cols, data_shape
        )

        # Check for overlap with binning columns
        overlap = set(validated_guidance) & set(binning_cols)
        if overlap:
            raise InvalidDataError(
                f"Guidance columns cannot overlap with binning columns: {overlap}",
                suggestions=[
                    "Use separate columns for guidance and binning",
                    "Consider creating a copy of the target column if needed",
                ],
            )

        return validated_guidance

    @staticmethod
    def check_data_quality(data: np.ndarray, name: str = "data") -> None:
        """Check data quality and issue warnings if needed."""

        config = get_config()

        if not config.show_warnings:
            return

        # Check for missing values - handle different dtypes
        # For numeric data, use np.isnan
        if np.issubdtype(data.dtype, np.number):
            missing_mask = np.isnan(data)
        else:
            # For object/string data, check for None and 'nan' strings
            missing_mask = np.array(
                [
                    x is None or (isinstance(x, str) and x.lower() in ["nan", "na", "null", ""])
                    for x in cast(Any, data.flat)
                ]
            ).reshape(data.shape)

        if missing_mask.any():
            missing_pct = missing_mask.mean() * 100
            if missing_pct > 50:
                warnings.warn(
                    f"{name} contains {missing_pct:.1f}% missing values. "
                    "This may significantly impact binning quality.",
                    DataQualityWarning,
                    stacklevel=2,
                )

        # Check for infinite values only for numeric types
        try:
            if np.issubdtype(data.dtype, np.number):
                if np.isinf(data).any():
                    warnings.warn(
                        f"{name} contains infinite values. "
                        "Consider clipping or removing these values.",
                        DataQualityWarning,
                        stacklevel=2,
                    )
        except (TypeError, ValueError):
            # Skip infinite value check if data type doesn't support it
            pass

        # Check for constant columns
        if data.ndim == 2:
            for i in range(data.shape[1]):
                col_data = data[:, i]
                finite_data = col_data[np.isfinite(col_data)]
                if len(finite_data) > 1 and np.var(finite_data) == 0:
                    warnings.warn(
                        f"Column {i} in {name} appears to be constant. "
                        "This will result in a single bin.",
                        DataQualityWarning,
                        stacklevel=2,
                    )


def validate_tree_params(task_type: str, tree_params: Dict[str, Any]) -> Dict[str, Any]:
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


def suggest_alternatives(method_name: str) -> List[str]:
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
