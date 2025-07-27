"""Equal-width binning transformer."""

from typing import Any, Dict, Optional, Tuple, Union, List
import numpy as np
from ..base._interval_binning_base import IntervalBinningBase
from ..base._repr_mixin import ReprMixin
from ..config import get_config
from ..errors import ValidationMixin, BinningError, InvalidDataError, ConfigurationError


class EqualWidthBinning(IntervalBinningBase, ReprMixin):
    def __repr__(self):
        defaults = dict(
            n_bins=10,
            bin_range=None,
            clip=True,
            preserve_dataframe=False,
            bin_edges=None,
            bin_representatives=None,
            fit_jointly=False,
            joint_range_method='global',
        )
        params = {
            'n_bins': self.n_bins,
            'bin_range': self.bin_range,
            'clip': self.clip,
            'preserve_dataframe': self.preserve_dataframe,
            'bin_edges': self.bin_edges,
            'bin_representatives': self.bin_representatives,
            'fit_jointly': self.fit_jointly,
            'joint_range_method': self.joint_range_method,
        }
        show = []
        for k, v in params.items():
            if v != defaults[k]:
                if k in {'bin_edges', 'bin_representatives'} and v is not None:
                    show.append(f'{k}=...')
                else:
                    show.append(f'{k}={repr(v)}')
        if not show:
            return f'{self.__class__.__name__}()'
        return f'{self.__class__.__name__}(' + ', '.join(show) + ')'
    """Classic equal-width binning transformer.

    Creates bins of equal width across the range of each feature.
    Enhanced with configuration management, error handling, and sklearn integration.
    """

    def __init__(
        self,
        n_bins: Union[int, Dict[Any, int]] = 10,
        bin_range: Optional[Union[Tuple[float, float], Dict[Any, Tuple[float, float]]]] = None,
        clip: Optional[bool] = None,
        preserve_dataframe: Optional[bool] = None,
        bin_edges: Optional[Dict[Any, List[float]]] = None,
        bin_representatives: Optional[Dict[Any, List[float]]] = None,
        fit_jointly: Optional[bool] = None,
        joint_range_method: str = "global",
        **kwargs,
    ) -> None:
        """Initialize EqualWidthBinning.

        Args:
            n_bins: Number of bins to create. Can be int for all columns or dict for per-column.
            bin_range: Range for binning. If None, uses data min/max.
            clip: Whether to clip out-of-range values to nearest bin.
            preserve_dataframe: Whether to return DataFrames when input is DataFrame.
            bin_edges: Optional pre-specified bin edges.
            bin_representatives: Optional pre-specified bin representatives.
            fit_jointly: Whether to fit joint parameters across columns.
            joint_range_method: Method for joint range calculation.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(
            clip=clip,
            preserve_dataframe=preserve_dataframe,
            bin_edges=bin_edges,
            bin_representatives=bin_representatives,
            fit_jointly=fit_jointly,
            **kwargs,
        )

        # Store equal-width specific parameters
        self.n_bins = n_bins
        self.bin_range = bin_range
        self.joint_range_method = joint_range_method
        

    def _calculate_joint_parameters(self, X: np.ndarray, columns: List[Any]) -> Dict[str, Any]:
        """Calculate joint parameters for equal-width binning."""
        joint_params = {}

        # Range calculation methods
        if self.joint_range_method == "global":
            global_min = np.nanmin(X)
            global_max = np.nanmax(X)
            joint_params["global_range"] = (global_min, global_max)

        elif self.joint_range_method == "percentile":
            global_min = np.nanpercentile(X, 5)
            global_max = np.nanpercentile(X, 95)
            joint_params["global_range"] = (global_min, global_max)

        elif self.joint_range_method == "std":
            global_mean = np.nanmean(X)
            global_std = np.nanstd(X)
            n_std = 2
            global_min = global_mean - n_std * global_std
            global_max = global_mean + n_std * global_std
            joint_params["global_range"] = (global_min, global_max)

        elif self.joint_range_method == "robust":
            q25 = np.nanpercentile(X, 25)
            q75 = np.nanpercentile(X, 75)
            iqr = q75 - q25
            global_min = q25 - 1.5 * iqr
            global_max = q75 + 1.5 * iqr
            joint_params["global_range"] = (global_min, global_max)

        # Store global n_bins if it's a single value
        if isinstance(self.n_bins, int):
            joint_params["n_bins"] = self.n_bins

        return joint_params

    def _calculate_bins_jointly(
        self, x_col: np.ndarray, col_id: Any, joint_params: Dict[str, Any]
    ) -> Tuple[List[float], List[float]]:
        """Calculate equal-width bins using joint parameters."""
        # Get n_bins (prefer per-column if available)
        if isinstance(self.n_bins, dict) and col_id in self.n_bins:
            n_bins = self.n_bins[col_id]
        else:
            n_bins = joint_params.get("n_bins", 10)

        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins}")

        # Determine range to use
        if self.bin_range is None or (
            isinstance(self.bin_range, dict) and col_id not in self.bin_range
        ):
            # Use joint range if no specific range for this column
            if "global_range" in joint_params:
                min_val, max_val = joint_params["global_range"]
            else:
                min_val, max_val = self._get_data_range(x_col, col_id)
        else:
            # Use specified range for this column
            min_val, max_val = self._get_specified_range(col_id)

        return self._create_equal_width_bins(min_val, max_val, n_bins)

    def _calculate_bins(
        self, 
        x_col: np.ndarray, 
        col_id: Any, 
        guidance_data: Optional[np.ndarray] = None
    ) -> Tuple[List[float], List[float]]:
        """Calculate equal-width bins for a column (per-column logic).
        
        Args:
            x_col: Column data to bin.
            col_id: Column identifier.
            guidance_data: Optional guidance data (not used in equal-width binning).
        """
        # Get n_bins for this column
        if isinstance(self.n_bins, dict):
            n_bins = self.n_bins.get(col_id, 10)
        else:
            n_bins = self.n_bins

        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins}")

        # Get range for this column
        if self.bin_range is not None and self._has_range_for_column(col_id):
            min_val, max_val = self._get_specified_range(col_id)
        else:
            min_val, max_val = self._get_data_range(x_col, col_id)

        return self._create_equal_width_bins(min_val, max_val, n_bins)

    def _has_range_for_column(self, col_id: Any) -> bool:
        """Check if a range is specified for a column."""
        if isinstance(self.bin_range, dict):
            return col_id in self.bin_range
        else:
            return True  # Single range applies to all columns

    def _get_specified_range(self, col_id: Any) -> Tuple[float, float]:
        """Get the specified range for a column."""
        if isinstance(self.bin_range, dict):
            if col_id in self.bin_range:
                return self.bin_range[col_id]
            else:
                raise ValueError(f"No range specified for column {col_id}")
        elif self.bin_range is not None:
            return self.bin_range
        else:
            raise ValueError(f"No range specified for column {col_id}")

    def _get_data_range(self, x_col: np.ndarray, col_id: Any) -> Tuple[float, float]:
        """Get the data range for a column."""
        try:
            min_val = np.nanmin(x_col)
            max_val = np.nanmax(x_col)
        except ValueError:
            raise ValueError(f"Cannot create bins for column {col_id}: min and max must be finite.")

        if not (np.isfinite(min_val) and np.isfinite(max_val)):
            raise ValueError(f"Cannot create bins for column {col_id}: min and max must be finite.")

        return min_val, max_val

    def _create_equal_width_bins(
        self, min_val: float, max_val: float, n_bins: int
    ) -> Tuple[List[float], List[float]]:
        """Create equal-width bins given range and number of bins."""
        if min_val == max_val:
            # Handle constant data
            epsilon = 1e-8
            min_val -= epsilon
            max_val += epsilon

        # Create equal-width bin edges
        edges = np.linspace(min_val, max_val, n_bins + 1)

        # Create representatives as bin centers
        reps = [(edges[i] + edges[i + 1]) / 2 for i in range(n_bins)]

        return list(edges), reps

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        params = super().get_params(deep=deep)
        params.update(
            {
                "n_bins": self.n_bins,
                "bin_range": self.bin_range,
                "joint_range_method": self.joint_range_method,
            }
        )
        return params

    def set_params(self, **params) -> "EqualWidthBinning":
        """Set parameters for this estimator."""
        # Handle equal-width specific parameters that should reset fitted state
        reset_fitted = False

        if "n_bins" in params:
            self.n_bins = params.pop("n_bins")
            reset_fitted = True

        if "bin_range" in params:
            self.bin_range = params.pop("bin_range")
            reset_fitted = True

        if "joint_range_method" in params:
            self.joint_range_method = params.pop("joint_range_method")
            reset_fitted = True

        if "fit_jointly" in params:
            reset_fitted = True  # Will be handled by parent

        if reset_fitted:
            self._fitted = False

        super().set_params(**params)
        return self
    
    def _validate_params(self) -> None:
        """Validate parameters for sklearn compatibility."""
        # Validate n_bins
        if isinstance(self.n_bins, int):
            if self.n_bins < 1:
                raise ConfigurationError(
                    "n_bins must be positive",
                    suggestions=["Set n_bins to a positive integer (e.g., n_bins=10)"]
                )
        elif isinstance(self.n_bins, dict):
            for col, n in self.n_bins.items():
                if not isinstance(n, int) or n < 1:
                    raise ConfigurationError(
                        f"n_bins for column {col} must be a positive integer",
                        suggestions=[f"Set n_bins[{col}] to a positive integer"]
                    )
        else:
            raise ConfigurationError(
                "n_bins must be int or dict",
                suggestions=["Use int for same bins across columns or dict for per-column bins"]
            )
        
        # Validate bin_range if provided
        if self.bin_range is not None:
            if isinstance(self.bin_range, tuple):
                if len(self.bin_range) != 2 or self.bin_range[0] >= self.bin_range[1]:
                    raise ConfigurationError(
                        "bin_range must be a tuple (min, max) with min < max",
                        suggestions=["Example: bin_range=(0, 100)"]
                    )
            elif isinstance(self.bin_range, dict):
                for col, range_val in self.bin_range.items():
                    if not isinstance(range_val, tuple) or len(range_val) != 2 or range_val[0] >= range_val[1]:
                        raise ConfigurationError(
                            f"bin_range for column {col} must be a tuple (min, max) with min < max",
                            suggestions=[f"Set bin_range[{col}] = (min_val, max_val)"]
                        )
