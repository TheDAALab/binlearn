"""
Example of how to create new guided binning methods using GuidedBinningMixin.

This demonstrates the reusable patterns factored out from SupervisedBinning.
"""

from typing import Any, Dict, List, Tuple, Optional, Union
import numpy as np
from ..base._interval_binning_base import IntervalBinningBase
from ..base._guided_binning_mixin import GuidedBinningMixin
from ..config import get_config
from ..errors import FittingError
from ..sklearn_utils import SklearnCompatibilityMixin


class ExampleGuidedBinning(IntervalBinningBase, GuidedBinningMixin, SklearnCompatibilityMixin):
    """
    Example guided binning method demonstrating how to use GuidedBinningMixin.
    
    This example creates bins based on target correlation (a simple heuristic approach).
    """

    def __init__(
        self,
        n_bins: int = 5,
        min_samples_per_bin: int = 10,
        **kwargs,
    ):
        """
        Initialize the example guided binning transformer.

        Parameters
        ----------
        n_bins : int, default=5
            Target number of bins to create
        min_samples_per_bin : int, default=10
            Minimum samples required per bin
        """
        super().__init__(**kwargs)
        self.n_bins = n_bins
        self.min_samples_per_bin = min_samples_per_bin

    def _calculate_bins(
        self, 
        x_col: np.ndarray, 
        col_id: Any, 
        guidance_data: Optional[np.ndarray] = None
    ) -> Tuple[List[float], List[float]]:
        """
        Calculate bins using target-guided correlation approach.
        
        This demonstrates the typical pattern for guided binning methods.
        """
        # 1. Ensure guidance data is provided (from mixin)
        self.require_guidance_data(guidance_data, "ExampleGuidedBinning")

        # 2. Validate and preprocess feature-target pair (from mixin)
        x_col, guidance_data_validated, valid_mask = self.validate_feature_target_pair(
            x_col, guidance_data, col_id
        )
        
        # 3. Check for insufficient data (from mixin)
        insufficient_result = self.handle_insufficient_data(
            x_col, valid_mask, self.min_samples_per_bin, col_id
        )
        if insufficient_result is not None:
            return insufficient_result
        
        # 4. Extract valid pairs for processing (from mixin)
        x_valid, y_valid = self.extract_valid_pairs(x_col, guidance_data_validated, valid_mask)
        
        # 5. Method-specific binning logic
        try:
            # Example: Sort by feature value and compute target correlation in windows
            sorted_indices = np.argsort(x_valid.ravel())
            x_sorted = x_valid.ravel()[sorted_indices]
            y_sorted = y_valid[sorted_indices]
            
            # Create candidate split points based on target variance
            candidate_splits = []
            window_size = max(len(x_sorted) // (self.n_bins * 2), self.min_samples_per_bin)
            
            for i in range(window_size, len(x_sorted) - window_size, window_size):
                left_target_mean = np.mean(y_sorted[:i])
                right_target_mean = np.mean(y_sorted[i:])
                
                # Only create split if there's meaningful difference
                if abs(left_target_mean - right_target_mean) > 0.1:
                    candidate_splits.append(x_sorted[i])
            
            # Combine with data bounds
            data_min = np.min(x_valid)
            data_max = np.max(x_valid)
            all_edges = [data_min] + sorted(candidate_splits) + [data_max]
            
            # Remove duplicates
            config = get_config()
            bin_edges = []
            for edge in all_edges:
                if not bin_edges or abs(edge - bin_edges[-1]) > config.float_tolerance:
                    bin_edges.append(edge)
            
            # Limit to target number of bins
            if len(bin_edges) > self.n_bins + 1:
                # Keep most important splits (simplified heuristic)
                bin_edges = bin_edges[:self.n_bins + 1]
            
        except Exception as e:
            # 6. Fallback handling (from mixin)
            return self.create_fallback_bins(x_col)
        
        # 7. Calculate representatives
        representatives = []
        for i in range(len(bin_edges) - 1):
            rep = (bin_edges[i] + bin_edges[i + 1]) / 2
            representatives.append(rep)
            
        return bin_edges, representatives

    def _validate_params(self) -> None:
        """Validate parameters for sklearn compatibility."""
        super()._validate_params() if hasattr(super(), '_validate_params') else None
        
        if self.n_bins < 1:
            raise ValueError("n_bins must be positive")
        if self.min_samples_per_bin < 1:
            raise ValueError("min_samples_per_bin must be positive")


# Example usage patterns that future guided methods would follow:
"""
Common patterns when creating guided binning methods:

1. **Inherit from GuidedBinningMixin**: Provides validation and data handling
2. **Use require_guidance_data()**: Ensure guidance data is provided
3. **Use validate_feature_target_pair()**: Handle missing values, type validation
4. **Use handle_insufficient_data()**: Graceful handling of edge cases
5. **Use extract_valid_pairs()**: Get clean data for your algorithm
6. **Use create_fallback_bins()**: Fallback when your method fails

Benefits of this pattern:
- Consistent error handling across all guided methods
- Robust missing value handling
- Clear validation error messages
- Reduced code duplication
- Easier testing and maintenance
"""
