"""Supervised binning transformer.

This module implements supervised binning, where bin boundaries are determined
by training decision trees on guidance/target variables. The tree splits provide
optimal cut points that maximize the relationship between features and targets.

This approach creates bins that are most informative for the prediction task,
often leading to better downstream model performance compared to unsupervised
binning methods.

Classes:
    SupervisedBinning: Main transformer for supervised binning operations.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import clone

from ..base._repr_mixin import ReprMixin
from ..base._supervised_binning_base import SupervisedBinningBase
from ..config import get_config
from ..utils.errors import ConfigurationError, FittingError, InvalidDataError, validate_tree_params
from ..utils.types import BinEdges, ColumnId, GuidanceColumns


# pylint: disable=too-many-ancestors
class SupervisedBinning(ReprMixin, SupervisedBinningBase):
    """Supervised binning transformer for single guidance/target column.

    Creates bins using decision tree splits guided by a target column. This method
    fits a decision tree to predict the guidance column from the features to be
    binned, then uses the tree's split thresholds to define bin boundaries.

    The resulting bins are optimized for the prediction task, as they capture
    the most important feature ranges for distinguishing different target values.
    This often leads to better model performance compared to unsupervised binning.

    The transformer supports both classification and regression tasks, automatically
    selecting appropriate decision tree algorithms. It's fully sklearn-compatible
    and supports pandas/polars DataFrames.

    Attributes:
        guidance_columns (list): Columns to use for supervised guidance.
        task_type (str): Type of task ("classification" or "regression").
        tree_params (dict): Parameters for the decision tree.
        min_samples_for_split (int): Minimum samples required for tree splits.
        max_unique_values (int): Maximum unique values before switching to regression.
        fit_jointly (bool): Whether to fit parameters jointly across columns.
        preserve_dataframe (bool): Whether to preserve DataFrame format.
        bin_edges_ (dict): Computed bin edges after fitting.

    Example:
        >>> import numpy as np
        >>> from binlearn.methods import SupervisedBinning
        >>> X = np.random.rand(100, 4)  # Features + target in last column
        >>> binner = SupervisedBinning(guidance_columns=[3])
        >>> X_binned = binner.fit_transform(X)
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        task_type: str = "classification",
        tree_params: Optional[Dict[str, Any]] = None,
        preserve_dataframe: bool = False,
        bin_edges: Any = None,
        bin_representatives: Any = None,
        guidance_columns: Optional[GuidanceColumns] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the SupervisedBinning transformer.

        Creates a supervised binning transformer that uses decision tree splits
        to determine optimal bin boundaries based on guidance/target data. The
        transformer fits a decision tree to predict the guidance column from the
        features, then extracts split thresholds to create informative bins.

        Args:
            task_type: Type of supervised learning task to perform. Must be either
                "classification" or "regression". Classification uses decision tree
                classifiers and is suitable for categorical targets, while regression
                uses decision tree regressors for continuous targets. Defaults to
                "classification".
            tree_params: Parameters for the underlying decision tree model. Common
                parameters include max_depth (int, default=3), min_samples_leaf
                (int, default=5), min_samples_split (int, default=10), and
                random_state (int or None). If None, default parameters optimized
                for binning will be used. Defaults to None.
            preserve_dataframe: Whether to preserve pandas/polars DataFrame format
                in the output. If True, returns DataFrame with same structure as
                input. If False, returns numpy arrays. Defaults to False.
            bin_edges: Pre-computed bin edges for each feature. Dictionary mapping
                column identifiers to bin edge arrays. Used for loading previously
                fitted transformers. Should only be provided when recreating an
                already fitted transformer. Defaults to None.
            bin_representatives: Pre-computed bin representative values for each
                feature. Dictionary mapping column identifiers to representative
                value arrays. Used for loading previously fitted transformers.
                Defaults to None.
            guidance_columns: Column identifier(s) to use as guidance/target for
                supervised binning. Can be a single column identifier (int, str)
                or list of identifiers for multi-target scenarios. Must be provided
                during fitting if not specified here. Defaults to None.
            **kwargs: Additional keyword arguments passed to the parent class.
                Common options include clip (bool) for handling out-of-range values.

        Raises:
            ValueError: If task_type is not "classification" or "regression".
            TypeError: If tree_params contains invalid parameter types.
            ConfigurationError: If guidance_columns specification is invalid.

        Example:
            >>> import numpy as np
            >>> from binlearn.methods import SupervisedBinning
            >>> X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 15.0], [4.0, 25.0]])
            >>> y = np.array([0, 0, 1, 1])  # Target for guidance
            >>> binner = SupervisedBinning(
            ...     task_type="classification",
            ...     tree_params={"max_depth": 2, "min_samples_leaf": 2}
            ... )
            >>> binner.fit(X, guidance_data=y)
            >>> X_binned = binner.transform(X)

        Note:
            - Always uses per-column fitting (fit_jointly=False) for optimal splits
            - Stores fitted decision trees for inspection and feature importance
            - Automatically handles both numeric and categorical guidance data
            - Tree parameters are validated during initialization
        """
        # Store parameters BEFORE calling super().__init__
        # because parent class calls _validate_params() which needs these attributes
        self.task_type = task_type
        self.tree_params = tree_params

        # Remove fit_jointly from kwargs if present to avoid conflicts
        kwargs.pop("fit_jointly", None)

        super().__init__(
            task_type=task_type,
            tree_params=tree_params,
            clip=kwargs.get("clip"),
            preserve_dataframe=preserve_dataframe,
            bin_edges=bin_edges,
            bin_representatives=bin_representatives,
            fit_jointly=False,  # Always use per-column fitting for supervised binning
            guidance_columns=guidance_columns,
            **kwargs,
        )

        # Initialize tree storage attributes
        self._fitted_trees: Dict[ColumnId, Any] = {}
        self._tree_importance: Dict[ColumnId, float] = {}

        # Create tree template for cloning during fitting
        self._create_tree_template()

    # pylint: disable=too-many-locals
    def _calculate_bins(
        self, x_col: np.ndarray, col_id: Any, guidance_data: Optional[np.ndarray] = None
    ) -> Tuple[BinEdges, BinEdges]:
        """Calculate bins using decision tree splits for a single column.

        Fits a decision tree to predict the guidance data from the feature column,
        then extracts the tree's split thresholds to create optimal bin boundaries.
        This supervised approach creates bins that are most informative for the
        prediction task, often leading to better downstream model performance.

        Args:
            x_col: Input feature data for a single column as 1D numpy array.
                Must contain numeric values that can be used as decision tree
                features. Missing or infinite values are handled automatically.
            col_id: Unique identifier for the column being processed. Used for
                error reporting and storing tree information. Can be any hashable
                type (int, str, etc.).
            guidance_data: Target values for supervised learning as 1D numpy array.
                Must have the same length as x_col. Should contain numeric values
                for regression tasks or categorical labels for classification tasks.
                Cannot be None for supervised binning.

        Returns:
            Tuple containing two lists:
            - bin_edges: List of bin boundary values [edge1, edge2, ..., edgeN].
              Always includes data minimum and maximum as first and last edges.
              Split points from the decision tree are inserted between these bounds.
            - representatives: List of bin representative values [rep1, rep2, ...].
              Each representative is the midpoint of its corresponding bin interval.
              Length is always len(bin_edges) - 1.

        Raises:
            FittingError: If guidance_data is None or if decision tree fitting fails.
                Common causes include incompatible data types, insufficient data,
                or invalid tree parameters.
            InvalidDataError: If x_col and guidance_data have different lengths or
                contain only invalid values (NaN/inf).
            ConfigurationError: If tree template was not properly initialized.

        Example:
            >>> x_col = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            >>> guidance = np.array([0, 0, 1, 1, 1])  # Classification targets
            >>> binner = SupervisedBinning(task_type="classification")
            >>> edges, reps = binner._calculate_bins(x_col, "feature_1", guidance)
            >>> # Returns optimized bin edges based on tree splits

        Note:
            - Automatically handles missing values by filtering them out
            - Requires minimum samples per split as defined in tree_params
            - Stores fitted tree for later inspection and feature importance
            - Split points are sorted and deduplicated using float tolerance
            - Representatives are calculated as bin midpoints for consistency
        """
        # Ensure guidance data is provided
        self.require_guidance_data(guidance_data, "SupervisedBinning")

        # At this point guidance_data is guaranteed to be not None
        assert (
            guidance_data is not None
        ), "guidance_data should not be None after require_guidance_data"

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
        data_min: float = np.min(x_valid)
        data_max: float = np.max(x_valid)

        # Combine data bounds with split points
        all_edges = [data_min] + sorted(split_points) + [data_max]
        # Remove duplicates while preserving order

        config = get_config()
        bin_edges: List[float] = []
        for edge in all_edges:
            if not bin_edges or abs(edge - bin_edges[-1]) > config.float_tolerance:
                bin_edges.append(edge)

        # Calculate representatives (midpoints of bins)
        representatives = []
        for i in range(len(bin_edges) - 1):
            rep = (bin_edges[i] + bin_edges[i + 1]) / 2
            representatives.append(rep)

        return bin_edges, representatives

    def _extract_split_points(self, tree: Any, _x_data: np.ndarray) -> BinEdges:
        """Extract split points from a fitted decision tree.

        Traverses the decision tree structure to extract all threshold values
        where the tree splits on the single feature. These thresholds represent
        the most informative cut points for separating different classes or
        regression targets, making them optimal bin boundaries.

        Args:
            tree: Fitted decision tree model (DecisionTreeClassifier or
                DecisionTreeRegressor). Must have been fitted on single-feature
                data with shape (n_samples, 1) to ensure all splits are on
                feature index 0.
            _x_data: Training data used to fit the tree. This parameter is
                maintained for interface compatibility but is not used in the
                extraction process. The tree structure contains all necessary
                information.

        Returns:
            List of unique split threshold values extracted from the tree.
            Values are in the order they appear in the tree traversal, not
            necessarily sorted. Empty list if the tree has no splits (e.g.,
            pure node or insufficient data for splitting).

        Note:
            - Only extracts splits on feature index 0 (single-feature assumption)
            - Thresholds represent decision boundaries from the tree learning
            - Each threshold optimally separates guidance data classes/values
            - The extracted points will be sorted and deduplicated later
            - Works with both classification and regression trees

        Example:
            >>> # For a tree that splits at values 2.5 and 4.1
            >>> split_points = binner._extract_split_points(fitted_tree, X)
            >>> # Returns: [2.5, 4.1] (or in tree traversal order)
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
        """Get feature importance scores from the fitted decision trees.

        Returns the importance scores computed by the decision trees during
        fitting. These scores indicate how much each feature contributes to
        the prediction of the guidance data, providing insight into which
        features are most informative for the supervised binning process.

        Args:
            column_id: Specific column identifier to get importance score for.
                If provided, returns importance only for that column. If None,
                returns importance scores for all fitted columns. Must be a
                column that was included in the fitting process.

        Returns:
            Dictionary mapping column identifiers to their importance scores.
            Importance scores are non-negative floats that sum to 1.0 across
            all features when considering the full tree. Higher values indicate
            more important features for predicting the guidance data.

        Raises:
            RuntimeError: If the transformer has not been fitted yet. Must call
                fit() before accessing feature importance scores.
            InvalidDataError: If feature importance data is not available (e.g.,
                trees failed to fit properly) or if the specified column_id
                was not found in the fitted trees.

        Example:
            >>> binner = SupervisedBinning()
            >>> binner.fit(X, guidance_data=y)
            >>>
            >>> # Get importance for all features
            >>> all_importance = binner.get_feature_importance()
            >>> # Returns: {0: 0.8, 1: 0.2} for 2-feature dataset
            >>>
            >>> # Get importance for specific feature
            >>> feature_0_importance = binner.get_feature_importance(column_id=0)
            >>> # Returns: {0: 0.8}

        Note:
            - Importance scores reflect how well each feature separates guidance data
            - Scores are derived from the sklearn decision tree feature_importances_
            - Only available after successful fitting with valid guidance data
            - Useful for feature selection and understanding data relationships
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
        """Get the structure of the decision tree for a specific column.

        Provides detailed information about the decision tree that was fitted
        for the specified column during supervised binning. This includes tree
        statistics, structure details, and the actual tree object for advanced
        analysis or visualization purposes.

        Args:
            column_id: Column identifier for which to retrieve tree structure.
                Must be a column that was included in the fitting process and
                for which a decision tree was successfully created.

        Returns:
            Dictionary containing comprehensive tree structure information:
            - "n_nodes": Total number of nodes in the tree (int)
            - "max_depth": Maximum depth of the tree (int)
            - "n_leaves": Number of leaf nodes in the tree (int)
            - "feature_importances": Array of feature importance scores (np.ndarray)
            - "tree_": The actual sklearn tree structure object for advanced access

        Raises:
            RuntimeError: If the transformer has not been fitted yet. Must call
                fit() before accessing tree structure information.
            InvalidDataError: If tree structure data is not available (trees
                were not stored properly) or if the specified column_id was
                not found in the fitted trees.

        Example:
            >>> binner = SupervisedBinning()
            >>> binner.fit(X, guidance_data=y)
            >>> tree_info = binner.get_tree_structure(column_id=0)
            >>> print(f"Tree has {tree_info['n_nodes']} nodes")
            >>> print(f"Max depth: {tree_info['max_depth']}")
            >>> print(f"Number of leaves: {tree_info['n_leaves']}")

        Note:
            - Useful for understanding how the tree makes splitting decisions
            - The tree_ object can be used with sklearn tree visualization tools
            - Feature importances are specific to the single-feature tree
            - Tree structure reflects the complexity of the guidance data relationship
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

    def _store_tree_info(self, tree: Any, col_id: Any) -> None:
        """Store tree information for later access and analysis.

        Stores the fitted decision tree and extracts its feature importance
        score for the specified column. This information is used by other
        methods like get_feature_importance() and get_tree_structure() to
        provide insights into the fitted models.

        Args:
            tree: Fitted decision tree model (DecisionTreeClassifier or
                DecisionTreeRegressor). Must have been successfully fitted
                on single-feature data with valid guidance data.
            col_id: Column identifier for which this tree was fitted. Used
                as the key for storing tree information in internal dictionaries.
                Can be any hashable type (int, str, etc.).

        Note:
            - Called automatically during the _calculate_bins process
            - Stores both the full tree object and extracted importance score
            - For single-feature trees, importance is the first array element
            - Importance defaults to 0.0 if tree has no feature importances
            - Enables later inspection and analysis of fitted models
        """
        self._fitted_trees[col_id] = tree
        # For single feature trees, importance is just the first (and only) importance
        self._tree_importance[col_id] = (
            tree.feature_importances_[0] if tree.feature_importances_.size > 0 else 0.0
        )

    def _validate_params(self) -> None:
        """Validate parameters for sklearn compatibility and configuration correctness.

        Performs comprehensive validation of SupervisedBinning parameters to ensure
        they are compatible with sklearn conventions and internal requirements.
        This includes validating the task type, tree parameters, and calling any
        parent class validation methods.

        Raises:
            ConfigurationError: If task_type is not "classification" or "regression",
                or if tree_params contains invalid parameter specifications that
                are incompatible with the chosen task type.
            ValueError: If tree_params contains parameter values that are out of
                valid ranges or have incorrect types (e.g., negative max_depth,
                non-integer min_samples_leaf).

        Note:
            - Called automatically during initialization and parameter updates
            - Supplements parent class parameter validation when available
            - Validates task_type against allowed values
            - Uses utility function validate_tree_params for tree parameter checking
            - Ensures consistency between task_type and tree_params settings

        Example:
            >>> # These will pass validation
            >>> binner = SupervisedBinning(task_type="classification")
            >>> binner = SupervisedBinning(
            ...     task_type="regression",
            ...     tree_params={"max_depth": 5, "min_samples_leaf": 3}
            ... )
            >>>
            >>> # This will raise ConfigurationError
            >>> binner = SupervisedBinning(task_type="invalid_task")
        """
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
