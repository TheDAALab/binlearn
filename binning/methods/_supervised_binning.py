"""
SupervisedBinning transformer - creates bins using decision tree splits guided by a target column.
"""

from typing import Any, Dict, List, Tuple, Optional, Union
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.base import clone
from ..base._interval_binning_base import IntervalBinningBase


class SupervisedBinning(IntervalBinningBase):
    """
    Creates bins using decision tree splits guided by a target column.
    
    This method fits a decision tree to predict the guidance column from the 
    features to be binned, then uses the tree's leaf boundaries to define 
    bin intervals. Each path from root to leaf defines an interval bin.
    
    For example:
    - Input features: [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]
    - Guidance target: [0, 1, 1] 
    - Tree might split: feature 0 <= 1.5, then feature 1 <= 25.0
    - Resulting bins based on tree leaf regions
    
    Note: Requires scikit-learn for decision tree functionality.
    """

    def __init__(
        self,
        task_type: str = "classification",
        tree_params: Optional[Dict[str, Any]] = None,
        preserve_dataframe: bool = False,
        bin_edges: Any = None,
        bin_representatives: Any = None,
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
        """
        if task_type not in ["classification", "regression"]:
            raise ValueError(
                f"task_type must be 'classification' or 'regression', got '{task_type}'"
            )

        super().__init__(
            bin_edges=bin_edges,
            bin_representatives=bin_representatives,
            preserve_dataframe=preserve_dataframe,
            **kwargs,
        )
        
        self.task_type = task_type
        
        # Set default tree parameters
        default_tree_params = {
            "max_depth": 3,
            "min_samples_leaf": 5,
            "min_samples_split": 10,
            "random_state": None,
        }
        
        if tree_params is None:
            tree_params = {}
        
        # Store the original tree_params for sklearn compatibility
        self.tree_params = tree_params
        
        # Merge with defaults for internal use
        self._merged_tree_params = {**default_tree_params, **tree_params}
        
        # Initialize the appropriate tree model
        if task_type == "classification":
            self._tree_template = DecisionTreeClassifier(**self._merged_tree_params)
        else:  # regression
            self._tree_template = DecisionTreeRegressor(**self._merged_tree_params)

    def _calculate_bins(
        self, 
        x_col: np.ndarray, 
        col_id: Any, 
        guidance_data: Optional[np.ndarray] = None
    ) -> Tuple[List[float], List[float]]:
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
        if guidance_data is None:
            raise ValueError(
                "SupervisedBinning requires guidance_data (target values) to be provided."
            )
        
        # Ensure guidance_data has exactly one column
        guidance_data = np.asarray(guidance_data)
        if guidance_data.ndim == 1:
            # 1D array is fine
            pass
        elif guidance_data.ndim == 2:
            # 2D array must have exactly 1 column
            if guidance_data.shape[1] != 1:
                raise ValueError(
                    f"SupervisedBinning expects guidance_data to have exactly 1 column, "
                    f"got {guidance_data.shape[1]} columns. Please specify a single guidance column."
                )
            # Flatten to 1D for processing
            guidance_data = guidance_data.ravel()
        else:
            raise ValueError(
                f"SupervisedBinning expects guidance_data to be 1D or 2D with 1 column, "
                f"got {guidance_data.ndim}D array."
            )
            
        # Convert to appropriate arrays and handle missing values
        x_col = np.asarray(x_col, dtype=float)
        
        # Remove rows where either feature or target is missing
        feature_finite = np.isfinite(x_col)
        if guidance_data.dtype == object:
            target_valid = guidance_data != None
        else:
            target_valid = np.isfinite(guidance_data.astype(float))
            
        valid_mask = feature_finite & target_valid
        
        if not valid_mask.any():
            # No valid data - create a single bin covering the data range
            min_val = np.nanmin(x_col) if not np.isnan(x_col).all() else 0.0
            max_val = np.nanmax(x_col) if not np.isnan(x_col).all() else 1.0
            if min_val == max_val:
                max_val = min_val + 1.0
            return [min_val, max_val], [(min_val + max_val) / 2]
        
        x_valid = x_col[valid_mask].reshape(-1, 1)
        y_valid = guidance_data[valid_mask]
        
        # Check if we have enough samples
        min_samples_split = self._merged_tree_params.get("min_samples_split", 10)
        if len(x_valid) < min_samples_split:
            # Not enough data for meaningful splits
            min_val = np.min(x_valid)
            max_val = np.max(x_valid)
            if min_val == max_val:
                max_val = min_val + 1.0
            return [min_val, max_val], [(min_val + max_val) / 2]
        
        # Fit decision tree
        tree = clone(self._tree_template)
        tree.fit(x_valid, y_valid)
        
        # Extract split points from the tree
        split_points = self._extract_split_points(tree, x_valid)
        
        # Create bin edges
        data_min = np.min(x_valid)
        data_max = np.max(x_valid)
        
        # Combine data bounds with split points
        all_edges = [data_min] + sorted(split_points) + [data_max]
        # Remove duplicates while preserving order
        bin_edges = []
        for edge in all_edges:
            if not bin_edges or abs(edge - bin_edges[-1]) > 1e-10:
                bin_edges.append(edge)
        
        # Calculate representatives (midpoints of bins)
        representatives = []
        for i in range(len(bin_edges) - 1):
            rep = (bin_edges[i] + bin_edges[i + 1]) / 2
            representatives.append(rep)
            
        return bin_edges, representatives

    def _extract_split_points(self, tree, X: np.ndarray) -> List[float]:
        """
        Extract split points from a fitted decision tree.
        
        Args:
            tree: Fitted decision tree (classifier or regressor).
            X: Training data used to fit the tree.
            
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

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        """String representation of the estimator."""
        params = []
        
        if self.task_type != "classification":
            params.append(f"task_type='{self.task_type}'")
        
        # Show tree_params if they differ from defaults
        default_tree_params = {
            "max_depth": 3,
            "min_samples_leaf": 5,
            "min_samples_split": 10,
            "random_state": None,
        }
        
        # Only show non-default parameters that were explicitly set
        if self.tree_params:
            params.append(f"tree_params={self.tree_params}")
        
        if self.preserve_dataframe:
            params.append(f"preserve_dataframe={self.preserve_dataframe}")
        if self.bin_edges is not None:
            params.append("bin_edges=...")
        if self.bin_representatives is not None:
            params.append("bin_representatives=...")

        param_str = ", ".join(params)
        return f"SupervisedBinning({param_str})"
