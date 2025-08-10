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

from ..base import SupervisedBinningBase
from ..config import apply_config_defaults, get_config
from ..utils import (
    BinEdgesDict,
    ConfigurationError,
    FittingError,
    create_param_dict_for_config,
    safe_sklearn_call,
)


# pylint: disable=too-many-ancestors
class TreeBinning(SupervisedBinningBase):
    """Tree-based supervised binning implementation using clean architecture.

    Creates bins using decision tree splits guided by a target column. This method
    fits a decision tree to predict the guidance column from the features to be
    binned, then uses the tree's split thresholds to define bin boundaries that
    optimize the tree's ability to separate different target values.

    The decision tree learning algorithm automatically identifies the most informative
    split points for distinguishing between different target values, making this
    approach particularly effective for supervised learning tasks. The resulting bins
    correspond to the decision tree's internal nodes, creating intervals that maximize
    the separation of target classes or minimize target variance.

    This approach is especially valuable when:
    - The relationship between features and targets is complex and non-linear
    - Domain knowledge about optimal split points is limited
    - Automatic feature discretization is needed for downstream models
    - Interpretable binning rules are desired (tree splits are easy to understand)

    The method supports both classification and regression tasks, automatically
    selecting the appropriate decision tree variant based on the task type. The
    fitted trees are stored and can be accessed for analysis of feature importance
    and split decisions.

    This implementation follows the clean binlearn architecture with straight inheritance,
    dynamic column resolution, and parameter reconstruction capabilities.

    Args:
        task_type: Type of supervised task - either 'classification' or 'regression'.
            Determines whether to use DecisionTreeClassifier or DecisionTreeRegressor.
            If None, uses configuration default.
        tree_params: Dictionary of parameters to pass to the sklearn DecisionTree.
            Common parameters include max_depth, min_samples_split, min_samples_leaf,
            random_state. If None, uses configuration default or sensible defaults.
        clip: Whether to clip values outside the fitted range to the nearest bin edge.
            If None, uses configuration default.
        preserve_dataframe: Whether to preserve pandas DataFrame structure in transform
            operations. If None, uses configuration default.
        guidance_columns: Column specification for target/guidance data used in
            supervised binning. Can be column names, indices, or callable selector.
        bin_edges: Pre-computed bin edges for reconstruction. Should not be provided
            during normal usage.
        bin_representatives: Pre-computed bin representatives for reconstruction.
            Should not be provided during normal usage.
        class_: Class name for reconstruction compatibility. Internal use only.
        module_: Module name for reconstruction compatibility. Internal use only.

    Attributes:
        task_type: Type of supervised task ('classification' or 'regression')
        tree_params: Parameters passed to the decision tree
        _fitted_trees: Dictionary storing fitted tree models per column
        _tree_importance: Dictionary storing feature importance per column
        _tree_template: Template tree used for cloning during fitting

    Example:
        >>> import numpy as np
        >>> from binlearn.methods import TreeBinning
        >>> from sklearn.datasets import make_classification
        >>>
        >>> # Create sample classification data
        >>> X, y = make_classification(n_samples=1000, n_features=1, n_redundant=0, random_state=42)
        >>>
        >>> # Initialize tree binning for classification
        >>> binner = TreeBinning(
        ...     task_type='classification',
        ...     tree_params={'max_depth': 4, 'min_samples_leaf': 50, 'random_state': 42}
        ... )
        >>>
        >>> # Fit with target data
        >>> binner.fit(X, y)
        >>> X_binned = binner.transform(X)
        >>>
        >>> # Analyze tree splits
        >>> print(f"Number of bins: {len(binner.bin_edges_[0]) - 1}")
        >>> print(f"Split points: {binner.bin_edges_[0][1:-1]}")  # Exclude data bounds
        >>>
        >>> # Access fitted tree for analysis
        >>> tree = binner._fitted_trees[0]
        >>> print(f"Tree depth: {tree.tree_.max_depth}")

    Note:
        - Requires target/guidance data for supervised learning of optimal split points
        - Automatically selects DecisionTreeClassifier or DecisionTreeRegressor based on task_type
        - Split thresholds from the tree become the bin boundaries
        - Supports all sklearn DecisionTree parameters through tree_params
        - Fitted trees are stored and accessible for further analysis
        - Each column is processed independently with its corresponding target data
        - Handles both classification and regression tasks seamlessly

    See Also:
        Chi2Binning: Statistical significance-based supervised binning
        IsotonicBinning: Monotonic relationship preserving supervised binning
        SupervisedBinningBase: Base class for supervised binning methods

    References:
        Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984).
        Classification and regression trees.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        task_type: str | None = None,
        tree_params: dict[str, Any] | None = None,
        clip: bool | None = None,
        preserve_dataframe: bool | None = None,
        guidance_columns: Any = None,
        *,
        bin_edges: BinEdgesDict | None = None,
        bin_representatives: BinEdgesDict | None = None,
        class_: (
            str | None
        ) = None,  # For reconstruction compatibility  # pylint: disable=unused-argument
        module_: (
            str | None
        ) = None,  # For reconstruction compatibility  # pylint: disable=unused-argument
    ):
        """Initialize Tree binning with decision tree parameters and task configuration.

        Sets up decision tree-based binning with specified tree parameters and task type.
        Creates a tree template that will be cloned for each column during fitting.
        Applies configuration defaults for any unspecified parameters.

        Args:
            task_type: Type of supervised learning task. Must be either:
                - 'classification': Uses DecisionTreeClassifier for discrete targets
                - 'regression': Uses DecisionTreeRegressor for continuous targets
                If None, uses configuration default (typically 'classification').
            tree_params: Dictionary of parameters to pass to the sklearn DecisionTree
                constructor. Common parameters include:
                - max_depth: Maximum depth of the tree (int or None)
                - min_samples_split: Minimum samples required to split a node (int)
                - min_samples_leaf: Minimum samples required at each leaf (int)
                - random_state: Random seed for reproducible results (int or None)
                If None, uses sensible defaults.
            clip: Whether to clip transformed values outside the fitted range to the
                nearest bin edge. If None, uses configuration default.
            preserve_dataframe: Whether to preserve pandas DataFrame structure in
                transform operations. If None, uses configuration default.
            guidance_columns: Column specification for target/guidance data. Can be
                column names, indices, or callable selector. Required for supervised
                binning during fit operations.
            bin_edges: Pre-computed bin edges dictionary for reconstruction. Internal
                use only - should not be provided during normal initialization.
            bin_representatives: Pre-computed representatives dictionary for
                reconstruction. Internal use only.
            class_: Class name string for reconstruction compatibility. Internal use only.
            module_: Module name string for reconstruction compatibility. Internal use only.

        Raises:
            ConfigurationError: If task_type is not 'classification' or 'regression',
                or if tree_params contains invalid parameters.

        Example:
            >>> # Classification with custom tree parameters
            >>> binner = TreeBinning(
            ...     task_type='classification',
            ...     tree_params={
            ...         'max_depth': 5,
            ...         'min_samples_leaf': 20,
            ...         'random_state': 42
            ...     },
            ...     guidance_columns='target_class'
            ... )
            >>>
            >>> # Regression with minimal tree constraints
            >>> binner = TreeBinning(
            ...     task_type='regression',
            ...     tree_params={'max_depth': 3, 'min_samples_split': 10},
            ...     guidance_columns=['continuous_target']
            ... )
            >>>
            >>> # Use configuration defaults
            >>> binner = TreeBinning(guidance_columns='target')

        Note:
            - Parameter validation occurs during initialization
            - Tree template is created during initialization and cloned for each column
            - Configuration defaults are applied for None parameters
            - The tree_params dictionary is validated against sklearn DecisionTree parameters
            - Guidance columns must be specified for supervised binning to work properly
            - Reconstruction parameters should not be provided during normal usage
        """
        # Use standardized initialization pattern
        user_params = create_param_dict_for_config(
            task_type=task_type,
            tree_params=tree_params,
            clip=clip,
            preserve_dataframe=preserve_dataframe,
        )

        # Apply configuration defaults
        resolved_params = apply_config_defaults("supervised", user_params)

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

        # Initialize parent with resolved parameters
        SupervisedBinningBase.__init__(
            self,
            clip=resolved_params.get("clip"),
            preserve_dataframe=resolved_params.get("preserve_dataframe"),
            guidance_columns=guidance_columns,
            bin_edges=bin_edges,
            bin_representatives=bin_representatives,
        )

        # Create tree template after parent initialization
        self._create_tree_template()

    def _validate_params(self) -> None:
        """Validate Tree binning parameters."""
        # Call parent validation
        SupervisedBinningBase._validate_params(self)

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

        # Validate and clean feature-target pairs (removes NaN/inf from target)
        x_col_clean, guidance_clean = self._validate_feature_target_pair(
            x_col, guidance_data, col_id
        )

        # Check for insufficient data after cleaning
        min_samples_split = (self.tree_params or {}).get("min_samples_split", 2)
        if len(x_col_clean) < min_samples_split:
            raise FittingError(
                f"Column {col_id}: Insufficient data points ({len(x_col_clean)}) "
                f"for tree binning after cleaning. Need at least {min_samples_split}."
            )

        # Fit decision tree
        try:
            if self._tree_template is None:
                raise FittingError("Tree template not initialized")
            tree = clone(self._tree_template)
            # Reshape x_col_clean to 2D for sklearn compatibility
            x_col_2d = x_col_clean.reshape(-1, 1)
            safe_sklearn_call(tree.fit, x_col_2d, guidance_clean)
        except (
            ValueError,
            RuntimeError,
            ImportError,
            Exception,
        ) as e:  # pylint: disable=broad-exception-caught
            raise FittingError(
                f"Column {col_id}: Failed to fit decision tree: {str(e)}",
                suggestions=[
                    "Check if your target values are valid for the chosen task_type",
                    "Try adjusting tree_params (e.g., reduce max_depth)",
                    "Ensure you have enough data for the tree parameters",
                ],
            ) from e

        # Extract split points from the tree
        split_points = self._extract_split_points(tree, x_col_clean)

        # Store tree information for later access
        self._store_tree_info(tree, col_id)

        # Create bin edges
        data_min: float = float(np.min(x_col_clean))
        data_max: float = float(np.max(x_col_clean))

        # Handle constant column case: create bins with eps margins
        config = get_config()
        if abs(data_max - data_min) <= config.float_tolerance:
            # Constant column: create edges at constant_value ± eps
            constant_value = data_min  # Same as data_max
            eps = config.float_tolerance * 10  # Use larger margin than tolerance
            bin_edges = [constant_value - eps, constant_value + eps]
        else:
            # Combine data bounds with split points
            all_edges = [data_min] + sorted(split_points) + [data_max]
            # Remove duplicates while preserving order
            bin_edges = self._filter_duplicate_edges(all_edges)

        # Calculate representatives (midpoints of bins)
        representatives = []
        for i in range(len(bin_edges) - 1):
            rep = (bin_edges[i] + bin_edges[i + 1]) / 2
            representatives.append(rep)

        return bin_edges, representatives

    def _filter_duplicate_edges(self, all_edges: list[float]) -> list[float]:
        """Filter out duplicate edges based on float tolerance.

        Args:
            all_edges: List of edge values to filter

        Returns:
            Filtered list with duplicates removed based on float_tolerance
        """
        config = get_config()
        bin_edges: list[float] = []
        for edge in all_edges:
            if not bin_edges or abs(edge - bin_edges[-1]) > config.float_tolerance:
                bin_edges.append(edge)
        return bin_edges

    def _extract_split_points(
        self, tree: Any, x_data: np.ndarray[Any, Any]
    ) -> list[float]:  # pylint: disable=unused-argument
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
