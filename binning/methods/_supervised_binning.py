"""
SupervisedBinning transformer - creates bins using decision tree splits guided by a target column.
"""

from typing import Any, Dict, Tuple, Optional
import numpy as np
from sklearn.base import clone

from ..utils.types import (
    BinEdges, ColumnId, GuidanceColumns
)
from ..base._supervised_binning_base import SupervisedBinningBase
from ..base._repr_mixin import ReprMixin
from ..utils.errors import InvalidDataError, ConfigurationError, FittingError, validate_tree_params


class SupervisedBinning(ReprMixin, SupervisedBinningBase):
    """
    Supervised binning transformer for single guidance/target column.

    Creates bins using decision tree splits guided by a target column.
    This method fits a decision tree to predict the guidance column from the
    features to be binned, then uses the tree's leaf boundaries to define
    bin intervals. Each path from root to leaf defines an interval bin.

    Inherits all validation and guidance logic from SupervisedBinningBase.
    """

    def __init__(
        self,
        task_type: str = "classification",
        tree_params: Optional[Dict[str, Any]] = None,
        preserve_dataframe: bool = False,
        bin_edges: Any = None,
        bin_representatives: Any = None,
        guidance_columns: Optional[GuidanceColumns] = None,
        **kwargs,
    ):
        """
        Initialize the SupervisedBinning transformer.

        Parameters
        ----------
        task_type : str, default="classification"
            Type of supervised learning task. Either "classification" or "regression".

        tree_params : dict or None, default=None
            Parameters for the decision tree. Common parameters include:
            - max_depth : int or None, default=3
            - min_samples_leaf : int, default=5
            - min_samples_split : int, default=10
            - random_state : int or None, default=None

        preserve_dataframe : bool, default=False
            If True, preserve DataFrame structure in output.

        bin_edges : dict or None, default=None
            Pre-defined bin edges.

        bin_representatives : dict or None, default=None
            Pre-defined bin representatives.

        guidance_columns : list or int or None, default=None
            Column(s) to use as guidance/target for supervised binning.
        """
        # Remove fit_jointly from kwargs if present to avoid conflicts
        kwargs.pop('fit_jointly', None)

        super().__init__(
            task_type=task_type,
            tree_params=tree_params,
            clip=kwargs.get("clip"),
            preserve_dataframe=preserve_dataframe,
            bin_edges=bin_edges,
            bin_representatives=bin_representatives,
            fit_jointly=False,  # Always use per-column fitting for supervised binning
            guidance_columns=guidance_columns,
        )

        # Store original parameters for sklearn clone compatibility (after super call)
        self.task_type = task_type
        self.tree_params = tree_params

        # Initialize tree storage attributes
        self._fitted_trees: Dict[ColumnId, Any] = {}
        self._tree_importance: Dict[ColumnId, float] = {}

        # Create tree template for cloning during fitting
        self._create_tree_template()

    def _calculate_bins(
        self, x_col: np.ndarray, col_id: Any, guidance_data: Optional[np.ndarray] = None
    ) -> Tuple[BinEdges, BinEdges]:
        """
        Calculate bins using decision tree splits for a single column.

        Args:
            x_col: Data for a single column.
            col_id: Column identifier.
            guidance_data: Target values for supervised learning (must be 1D).

        Returns:
            Tuple containing:
            - List of bin edges: [edge1, edge2, ...]
            - List of representative values: [rep1, rep2, ...]
        """
        # Ensure guidance data is provided
        self.require_guidance_data(guidance_data, "SupervisedBinning")

        # At this point guidance_data is guaranteed to be not None
        assert guidance_data is not None, (
            "guidance_data should not be None after require_guidance_data"
        )

        # Validate and preprocess feature-target pair
        x_col, guidance_data_validated, valid_mask = self.validate_feature_target_pair(
            x_col, guidance_data, col_id
        )

        # Check for insufficient data
        min_samples_split = (self.tree_params or {}).get("min_samples_split", 2)
        insufficient_result = self.handle_insufficient_data(
            x_col, valid_mask, min_samples_split, col_id
        )
        if insufficient_result is not None:
            return insufficient_result

        # Extract valid pairs for tree fitting
        x_valid, y_valid = self.extract_valid_pairs(x_col, guidance_data_validated, valid_mask)

        # Fit decision tree
        try:
            if self._tree_template is None:
                raise FittingError("Tree template not initialized")
            tree = clone(self._tree_template)
            # Reshape x_valid to 2D for sklearn compatibility
            x_valid_2d = x_valid.reshape(-1, 1)
            tree.fit(x_valid_2d, y_valid)
        except Exception as e:
            raise FittingError(
                f"Failed to fit decision tree: {str(e)}",
                suggestions=[
                    "Check if your target values are valid for the chosen task_type",
                    "Try adjusting tree_params (e.g., reduce max_depth)",
                    "Ensure you have enough data for the tree parameters",
                    "Check for data type compatibility",
                ],
            ) from e

        # Extract split points from the tree
        split_points = self._extract_split_points(tree, x_valid)

        # Store tree information for later access
        self._store_tree_info(tree, col_id)

        # Create bin edges
        data_min = np.min(x_valid)
        data_max = np.max(x_valid)

        # Combine data bounds with split points
        all_edges = [data_min] + sorted(split_points) + [data_max]
        # Remove duplicates while preserving order
        from ..config import get_config
        config = get_config()
        bin_edges = []
        for edge in all_edges:
            if not bin_edges or abs(edge - bin_edges[-1]) > config.float_tolerance:
                bin_edges.append(edge)

        # Calculate representatives (midpoints of bins)
        representatives = []
        for i in range(len(bin_edges) - 1):
            rep = (bin_edges[i] + bin_edges[i + 1]) / 2
            representatives.append(rep)

        return bin_edges, representatives

    def _extract_split_points(self, tree, _x_data: np.ndarray) -> BinEdges:
        """
        Extract split points from a fitted decision tree.

        Args:
            tree: Fitted decision tree (classifier or regressor).
            _x_data: Training data used to fit the tree (unused but kept for interface compatibility).

        Returns:
            List of split threshold values.
        """
        split_points = []

        # Access the tree structure
        tree_structure = tree.tree_
        feature = tree_structure.feature
        threshold = tree_structure.threshold

        # Extract thresholds for splits on our single feature (index 0)
        for node_id in range(tree_structure.node_count):
            if feature[node_id] == 0:  # Split on our feature
                split_points.append(threshold[node_id])

        return split_points

    def get_feature_importance(self, column_id: Optional[ColumnId] = None) -> Dict[ColumnId, float]:
        """
        Get feature importance scores from the fitted decision trees.

        Parameters
        ----------
        column_id : Any, optional
            Specific column to get importance for. If None, returns all.

        Returns
        -------
        Dict[ColumnId, float]
            Mapping from column identifier to importance score.
        """
        self._check_fitted()

        if not hasattr(self, "_tree_importance"):
            raise InvalidDataError(
                "Feature importance not available. Tree may not have been fitted properly.",
                suggestions=[
                    "Ensure the transformer has been fitted with valid data",
                    "Check that the decision tree was able to make splits",
                ],
            )

        if column_id is not None:
            if column_id not in self._tree_importance:
                raise InvalidDataError(
                    f"Column {column_id} not found in fitted trees",
                    suggestions=[
                        f"Available columns: {list(self._tree_importance.keys())}",
                        "Check column identifier spelling and type",
                    ],
                )
            return {column_id: self._tree_importance[column_id]}

        return self._tree_importance.copy()

    def get_tree_structure(self, column_id: ColumnId) -> Dict[str, Any]:
        """
        Get the structure of the decision tree for a specific column.

        Parameters
        ----------
        column_id : Any
            Column identifier to get tree structure for.

        Returns
        -------
        Dict[str, Any]
            Tree structure information including splits and thresholds.
        """
        self._check_fitted()

        if not hasattr(self, "_fitted_trees"):
            raise InvalidDataError(
                "Tree structure not available. Trees may not have been stored.",
                suggestions=[
                    "Ensure the transformer has been fitted",
                    "Check that trees were fitted successfully",
                ],
            )

        if column_id not in self._fitted_trees:
            raise InvalidDataError(
                f"No tree found for column {column_id}",
                suggestions=[
                    f"Available columns: {list(self._fitted_trees.keys())}",
                    "Check column identifier",
                ],
            )

        tree = self._fitted_trees[column_id]
        tree_structure = tree.tree_

        return {
            "n_nodes": tree_structure.node_count,
            "max_depth": tree_structure.max_depth,
            "n_leaves": tree_structure.n_leaves,
            "feature_importances": tree.feature_importances_,
            "tree_": tree_structure,
        }

    def _store_tree_info(self, tree, col_id: Any) -> None:
        """Store tree information for later access."""
        self._fitted_trees[col_id] = tree
        # For single feature trees, importance is just the first (and only) importance
        self._tree_importance[col_id] = (
            tree.feature_importances_[0] if tree.feature_importances_.size > 0 else 0.0
        )

    def _validate_params(self) -> None:
        """Validate parameters for sklearn compatibility."""
        if hasattr(super(), "_validate_params"):
            super()._validate_params()

        # Validate task_type
        if self.task_type not in ["classification", "regression"]:
            raise ConfigurationError(
                f"Invalid task_type: {self.task_type}",
                suggestions=["Use 'classification' or 'regression'"],
            )

        # Validate tree_params
        if self.tree_params is not None:
            validate_tree_params(self.task_type, self.tree_params)
