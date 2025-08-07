"""
Clean Tree binning implementation for  architecture.

This module provides TreeBinning that inherits from SupervisedBinningBase.
Uses decision tree splits to find optimal cut points based on guidance data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ..config import apply_config_defaults, get_config
from ..utils.errors import ConfigurationError, FittingError
from ..utils.types import BinEdgesDict
from ..base._supervised_binning_base import SupervisedBinningBase


class TreeBinning(SupervisedBinningBase):
    """Tree-based supervised binning implementation using  architecture.

    Creates bins using decision tree splits guided by a target column. This method
    fits a decision tree to predict the guidance column from the features to be
    binned, then uses the tree's split thresholds to define bin boundaries.

    This implementation follows the clean  architecture with straight inheritance,
    dynamic column resolution, and parameter reconstruction capabilities.
    """

    def __init__(
        self,
        task_type: str | None = None,
        tree_params: dict[str, Any] | None = None,
        clip: bool | None = None,
        preserve_dataframe: bool | None = None,
        guidance_columns: Any = None,
        bin_edges: BinEdgesDict | None = None,
        bin_representatives: BinEdgesDict | None = None,
        class_: str | None = None,  # For reconstruction compatibility
        module_: str | None = None,  # For reconstruction compatibility
    ):
        """Initialize Tree binning."""
        # Prepare user parameters for config integration (exclude never-configurable params)
        user_params = {
            "task_type": task_type,
            "tree_params": tree_params,
            "clip": clip,
            "preserve_dataframe": preserve_dataframe,
        }
        # Remove None values to allow config defaults to take effect
        user_params = {k: v for k, v in user_params.items() if v is not None}

        # Apply configuration defaults for tree method
        resolved_params = apply_config_defaults("tree", user_params)

        # Store method-specific parameters
        self.task_type = resolved_params.get("task_type", "classification")
        self.tree_params = resolved_params.get("tree_params", None)

        # Validate task type
        if self.task_type not in ["classification", "regression"]:
            raise ConfigurationError(
                f"task_type must be 'classification' or 'regression', got '{self.task_type}'"
            )

        # Initialize tree storage attributes
        self._fitted_trees: dict[Any, Any] = {}
        self._tree_importance: dict[Any, float] = {}
        self._tree_template: DecisionTreeClassifier | DecisionTreeRegressor | None = None

        # Initialize parent with resolved parameters (never-configurable params passed as-is)
        SupervisedBinningBase.__init__(
            self,
            clip=resolved_params.get("clip"),
            preserve_dataframe=resolved_params.get("preserve_dataframe"),
            guidance_columns=guidance_columns,  # Never configurable
            bin_edges=bin_edges,  # Never configurable
            bin_representatives=bin_representatives,  # Never configurable
        )

        # Create tree template after parent initialization
        self._create_tree_template()

    def _validate_params(self) -> None:
        """Validate Tree binning parameters."""
        # Call parent validation
        SupervisedBinningBase._validate_params(self)

        # Validate task_type parameter
        if self.task_type not in ["classification", "regression"]:
            raise ConfigurationError(
                f"task_type must be 'classification' or 'regression', got '{self.task_type}'"
            )

        # Validate tree_params if provided
        if self.tree_params is not None:
            if not isinstance(self.tree_params, dict):
                raise ConfigurationError(
                    "tree_params must be a dictionary",
                    suggestions=["Example: tree_params={'max_depth': 3, 'min_samples_leaf': 5}"],
                )

    def _create_tree_template(self) -> None:
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
                    "Common parameters: max_depth, min_samples_split, min_samples_leaf, random_state",
                ],
            ) from e

    def _calculate_bins(
        self,
        x_col: np.ndarray[Any, Any],
        col_id: Any,
        guidance_data: np.ndarray[Any, Any] | None = None,
    ) -> tuple[list[float], list[float]]:
        """Calculate bins using decision tree splits for a single column.

        Fits a decision tree to predict the guidance data from the feature column,
        then extracts the tree's split thresholds to create optimal bin boundaries.

        Args:
            x_col: Preprocessed column data (from base class)
            col_id: Column identifier for error reporting
            guidance_data: Target/guidance data for supervised binning (required)

        Returns:
            Tuple of (bin_edges, bin_representatives)

        Raises:
            FittingError: If guidance_data is None or tree fitting fails
        """
        # Require guidance data for supervised binning
        if guidance_data is None:
            raise FittingError(f"Column {col_id}: guidance_data is required for tree binning")

        # Check for insufficient data
        min_samples_split = (self.tree_params or {}).get("min_samples_split", 2)
        if len(x_col) < min_samples_split:
            raise FittingError(
                f"Column {col_id}: Insufficient data points ({len(x_col)}) "
                f"for tree binning. Need at least {min_samples_split}."
            )

        # Fit decision tree
        try:
            if self._tree_template is None:
                raise FittingError("Tree template not initialized")
            tree = clone(self._tree_template)
            # Reshape x_col to 2D for sklearn compatibility
            x_col_2d = x_col.reshape(-1, 1)
            tree.fit(x_col_2d, guidance_data)
        except Exception as e:
            raise FittingError(
                f"Column {col_id}: Failed to fit decision tree: {str(e)}",
                suggestions=[
                    "Check if your target values are valid for the chosen task_type",
                    "Try adjusting tree_params (e.g., reduce max_depth)",
                    "Ensure you have enough data for the tree parameters",
                ],
            ) from e

        # Extract split points from the tree
        split_points = self._extract_split_points(tree, x_col)

        # Store tree information for later access
        self._store_tree_info(tree, col_id)

        # Create bin edges
        data_min: float = float(np.min(x_col))
        data_max: float = float(np.max(x_col))

        # Combine data bounds with split points
        all_edges = [data_min] + sorted(split_points) + [data_max]

        # Remove duplicates while preserving order
        config = get_config()
        bin_edges: list[float] = []
        for edge in all_edges:
            if not bin_edges or abs(edge - bin_edges[-1]) > config.float_tolerance:
                bin_edges.append(edge)

        # Calculate representatives (midpoints of bins)
        representatives = []
        for i in range(len(bin_edges) - 1):
            rep = (bin_edges[i] + bin_edges[i + 1]) / 2
            representatives.append(rep)

        return bin_edges, representatives

    def _extract_split_points(self, tree: Any, x_data: np.ndarray[Any, Any]) -> list[float]:
        """Extract split points from a fitted decision tree.

        Args:
            tree: Fitted decision tree model
            x_data: Training data used to fit the tree

        Returns:
            List of unique split threshold values extracted from the tree
        """
        split_points = []

        # Access the tree structure
        tree_structure = tree.tree_
        feature = tree_structure.feature
        threshold = tree_structure.threshold

        # Extract thresholds for splits on our single feature (index 0)
        for node_id in range(tree_structure.node_count):
            if feature[node_id] == 0:  # Split on our feature
                split_points.append(float(threshold[node_id]))

        return split_points

    def _store_tree_info(self, tree: Any, col_id: Any) -> None:
        """Store tree information for later access.

        Args:
            tree: Fitted decision tree model
            col_id: Column identifier
        """
        self._fitted_trees[col_id] = tree

        # Calculate and store feature importance (always 1.0 for single feature)
        self._tree_importance[col_id] = 1.0
