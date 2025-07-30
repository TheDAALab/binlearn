"""
Base class for supervised binning methods with decision tree integration.

This module provides the foundational SupervisedBinningBase class for all supervised
binning transformers. It handles guidance data validation, decision tree configuration,
feature-target pair processing, and automatic task type detection.

The class supports both classification and regression tasks, with intelligent fallback
strategies for insufficient data scenarios and comprehensive data quality validation.
"""

from typing import Any, Dict, Tuple, Optional
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ..utils.types import (
    BinEdges, BinEdgesDict, ColumnId, ColumnList
)
from ._interval_binning_base import IntervalBinningBase
from ..utils.errors import (
    ConfigurationError,
    DataQualityWarning,
)


class SupervisedBinningBase(IntervalBinningBase):
    """
    Base class for supervised binning methods that use single guidance columns.

    This class provides:
    - Single guidance column validation and preprocessing
    - Decision tree template management
    - Feature-target pair validation and missing value handling
    - Insufficient data handling with fallback strategies
    - Data quality warnings for both features and targets
    """

    def __init__(
        self,
        task_type: str = "classification",
        tree_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize SupervisedBinningBase.

        Args:
            task_type: Type of supervised task ("classification" or "regression")
            tree_params: Parameters for the underlying decision tree
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)

        # Validate task type
        if task_type not in ["classification", "regression"]:
            raise ConfigurationError(
                f"task_type must be 'classification' or 'regression', got '{task_type}'"
            )

        # Store parameters exactly as received for sklearn clone compatibility
        self.task_type = task_type
        self.tree_params = tree_params

        # Note: Tree template creation is deferred to fit time to allow invalid parameters
        # during initialization (for sklearn compatibility)
        self._tree_template = None

    def _create_tree_template(self):
        """Create tree template with merged parameters."""
        if self._tree_template is not None:
            return

        # Create simple tree template with default parameters
        default_params = {
            "max_depth": 3,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "random_state": None,
        }

        # Merge user params with defaults
        merged_params = {**default_params, **(self.tree_params or {})}

        # Initialize the appropriate tree model template
        try:
            if self.task_type == "classification":
                self._tree_template = DecisionTreeClassifier(**merged_params)
            else:  # regression
                self._tree_template = DecisionTreeRegressor(**merged_params)
        except TypeError as e:
            raise ConfigurationError(
                f"Invalid tree_params: {str(e)}",
                suggestions=[
                    "Check that all tree_params are valid DecisionTree parameters",
                    ("Common parameters: max_depth, min_samples_split, "
                     "min_samples_leaf, random_state"),
                ]
            ) from e

    def validate_guidance_data(
        self, guidance_data: np.ndarray, name: str = "guidance_data"
    ) -> np.ndarray:
        """
        Validate and preprocess guidance data for supervised binning.

        SupervisedBinning expects exactly one guidance column.

        Parameters
        ----------
        guidance_data : np.ndarray
            Raw guidance/target data to validate
        name : str, default="guidance_data"
            Name for error messages

        Returns
        -------
        np.ndarray
            Validated 1D guidance data

        Raises
        ------
        ValueError
            If guidance data has wrong dimensionality or column count
        """
        # Basic validation
        guidance_validated = self.validate_array_like(guidance_data, name)
        guidance_validated = np.asarray(guidance_validated)

        # Check data quality
        self.check_data_quality(guidance_validated, name)

        # Handle dimensionality - supervised binning expects single column
        if guidance_validated.ndim == 1:
            return guidance_validated
        if guidance_validated.ndim == 2:
            if guidance_validated.shape[1] != 1:
                raise ValueError(
                    f"{name} has {guidance_validated.shape[1]} columns, "
                    f"expected exactly 1. Supervised binning requires a single guidance column. "
                    f"Please specify the correct guidance column."
                )
            # Flatten to 1D for easier processing
            return guidance_validated.ravel()
        raise ValueError(
                f"{name} has {guidance_validated.ndim} dimensions, "
                f"expected 1D or 2D array with single column"
            )

    def validate_feature_target_pair(
        self, x_col: np.ndarray, guidance_data: np.ndarray, col_id: Any = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Validate feature-target pair and create valid data mask.

        Parameters
        ----------
        x_col : np.ndarray
            Feature column data
        guidance_data : np.ndarray
            Target/guidance data (must be 1D after validation)
        col_id : Any, optional
            Column identifier for error messages

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            - x_col: Validated feature data as float array
            - guidance_data: Validated guidance data
            - valid_mask: Boolean mask for valid (non-missing) pairs
        """
        # Validate inputs
        x_col_validated = self.validate_array_like(x_col, f"feature column {col_id}")
        guidance_data_validated = self.validate_guidance_data(guidance_data)

        # Check data quality (validate_array_like should not return None here)
        assert x_col_validated is not None, "x_col_validated should not be None"
        self.check_data_quality(x_col_validated, f"feature column {col_id}")
        self.check_data_quality(guidance_data_validated, "guidance data")

        # Convert feature to float for numeric operations
        x_col = np.asarray(x_col_validated, dtype=float)

        # Create valid data mask (both feature and target must be non-missing)
        feature_finite = np.isfinite(x_col)

        if guidance_data_validated.dtype == object:
            # Handle object dtype (e.g., strings, mixed types)
            guidance_finite = np.array(
                [
                    val is not None and not (isinstance(val, float) and np.isnan(val))
                    for val in guidance_data_validated
                ]
            )
        else:
            # Numeric dtype
            guidance_finite = np.isfinite(guidance_data_validated.astype(float))

        valid_mask = feature_finite & guidance_finite

        return x_col, guidance_data_validated, valid_mask

    def extract_valid_pairs(
        self, x_col: np.ndarray, guidance_data: np.ndarray, valid_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract valid feature-target pairs using the provided mask.

        Parameters
        ----------
        x_col : np.ndarray
            Feature column data
        guidance_data : np.ndarray
            Target/guidance data
        valid_mask : np.ndarray
            Boolean mask indicating valid pairs

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Valid feature and target data
        """
        return x_col[valid_mask], guidance_data[valid_mask]

    def require_guidance_data(
        self, guidance_data: Optional[np.ndarray], method_name: str = "supervised binning"
    ) -> None:
        """
        Ensure guidance data is provided for supervised methods.

        Parameters
        ----------
        guidance_data : np.ndarray or None
            Guidance data to check
        method_name : str
            Name of the method for error messages

        Raises
        ------
        ValueError
            If guidance_data is None
        """
        if guidance_data is None:
            raise ValueError(
                f"{method_name.title()} requires guidance_data (target values) to be provided. "
                f"Please specify guidance_columns when creating the transformer."
            )

    def validate_task_type(self, task_type: str, valid_types: ColumnList) -> None:
        """
        Validate that task_type is one of the valid options.

        Parameters
        ----------
        task_type : str
            Task type to validate
        valid_types : ColumnList
            List of valid task types

        Raises
        ------
        ValueError
            If task_type is not in valid_types
        """
        if task_type not in valid_types:
            raise ValueError(
                f"task_type '{task_type}' not supported. " f"Valid options are: {valid_types}"
            )

    def handle_insufficient_data(
        self, x_col: np.ndarray, valid_mask: np.ndarray, min_samples: int, col_id: Any = None
    ) -> Optional[Tuple[BinEdges, BinEdges]]:
        """
        Handle cases with insufficient valid data for supervised binning.

        Parameters
        ----------
        x_col : np.ndarray
            Feature column data
        valid_mask : np.ndarray
            Boolean mask indicating valid pairs
        min_samples : int
            Minimum number of samples required
        col_id : Any, optional
            Column identifier for warning messages

        Returns
        -------
        Optional[Tuple[BinEdges, BinEdges]]
            Fallback bin specification (edges, representatives) or None
        """
        import warnings

        n_valid = valid_mask.sum()

        if n_valid == 0:
            # No valid data - create default range
            min_val = np.nanmin(x_col) if not np.isnan(x_col).all() else 1.0
            max_val = np.nanmax(x_col) if not np.isnan(x_col).all() else 1.0
            if min_val == max_val:
                max_val = min_val + 1.0

            if col_id is not None:
                # Create a more descriptive column reference
                col_ref = (
                    f"column {col_id}" if isinstance(col_id, (int, np.integer))
                    else f"column '{col_id}'"
                )
                warnings.warn(
                    f"Data in {col_ref} has no valid data points. "
                    f"Using default bin range [{min_val}, {max_val}]",
                    DataQualityWarning,
                )

            return [min_val, max_val], [(min_val + max_val) / 2]

        if n_valid < min_samples:
            # Insufficient data for complex binning
            valid_data = x_col[valid_mask]
            min_val = np.min(valid_data)
            max_val = np.max(valid_data)

            if min_val == max_val:
                max_val = min_val + 1.0

            if col_id is not None:
                # Create a more descriptive column reference
                if isinstance(col_id, (int, np.integer)):
                    col_ref = f"column {col_id}"
                else:
                    col_ref = f"column '{col_id}'"
                warnings.warn(
                    f"Data in {col_ref} has only {n_valid} valid samples "
                    f"(minimum {min_samples} required). Creating single bin.",
                    DataQualityWarning,
                )

            return [min_val, max_val], [(min_val + max_val) / 2]

        # Sufficient data - continue with normal processing
        return None

    def create_fallback_bins(
        self, x_col: np.ndarray, default_range: Optional[Tuple[float, float]] = None
    ) -> Tuple[BinEdges, BinEdges]:
        """
        Create fallback bins when supervised binning fails.

        Parameters
        ----------
        x_col : np.ndarray
            Feature column data
        default_range : Tuple[float, float], optional
            Default range to use. If None, infers from data.

        Returns
        -------
        Tuple[BinEdges, BinEdges]
            Fallback bin specification (edges, representatives)
        """
        if default_range is not None:
            min_val, max_val = default_range
        else:
            # Infer range from valid data
            finite_mask = np.isfinite(x_col)
            if finite_mask.any():
                min_val = np.min(x_col[finite_mask])
                max_val = np.max(x_col[finite_mask])
            else:
                min_val, max_val = 0.0, 1.0

        if min_val == max_val:
            max_val = min_val + 1.0

        return [min_val, max_val], [(min_val + max_val) / 2]
