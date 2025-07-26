import numpy as np

from ._pandas_config import PANDAS_AVAILABLE, pd
from ._interval_binning_base import IntervalBinningBase
from ._binning_utils import ensure_dict_format, default_representatives

from typing import Any, Dict, List, Optional, Union


class GuidedBinningBase(IntervalBinningBase):
    """
    Base class for guided/supervised binning.
    Allows specifying which columns are used as guidance (not binned).
    Uses column labels as keys for bins_ and bin_representatives_ for both DataFrames and ndarrays.
    """

    guidance_columns: Optional[List[Any]]
    _bin_cols: Optional[List[Any]]
    _guidance_cols: Optional[List[Any]]
    _use_labels: bool

    def __init__(
        self,
        guidance_columns: Optional[List[Any]] = None,
        bin_spec: Optional[Dict[Any, Any]] = None,
        bin_representatives: Optional[Dict[Any, List[float]]] = None,
        clip: bool = True,
        preserve_dataframe: bool = False,
        fit_jointly: bool = False,
        **kwargs,
    ):
        super().__init__(
            bin_spec=bin_spec,
            bin_representatives=bin_representatives,
            clip=clip,
            preserve_dataframe=preserve_dataframe,
            fit_jointly=fit_jointly,
            **kwargs,
        )
        self.guidance_columns = guidance_columns
        self._bin_cols = None
        self._guidance_cols = None
        self._use_labels = False

    def fit(self, X: Any, y: Any = None, sample_weight: Any = None):
        """
        Fit the binning using only the columns not in guidance_columns.
        """
        if PANDAS_AVAILABLE and hasattr(X, "columns"):
            columns = list(X.columns)
            guidance_cols = self.guidance_columns or []
            bin_cols = [col for col in columns if col not in guidance_cols]
            X_binning = X[bin_cols]
            guidance = X[guidance_cols] if guidance_cols else None
            col_labels = bin_cols
            use_labels = True
        else:
            n_cols = X.shape[1]
            guidance_cols = self.guidance_columns or []
            bin_cols = [i for i in range(n_cols) if i not in guidance_cols]
            X_binning = X[:, bin_cols]
            guidance = X[:, guidance_cols] if guidance_cols else None
            col_labels = list(range(len(bin_cols)))  # consecutive ints
            use_labels = False

        self._bin_cols = bin_cols
        self._guidance_cols = guidance_cols
        self._use_labels = use_labels

        x_array, feature_names, index = self._prepare_input_array(X_binning, require_fitted=False)
        self.n_features_in_ = x_array.shape[1]
        self.bins_ = {}
        self.bin_representatives_ = {}

        for i in range(len(col_labels)):
            column_data = x_array[:, i]
            # Pass guidance and sample_weight to _calculate_bins if needed
            edges = self._calculate_bins(
                column_data,
                col_labels[i] if use_labels else i,
                guidance=guidance,
                sample_weight=sample_weight,
            )
            self.bins_[col_labels[i] if use_labels else i] = np.asarray(edges)

        self.feature_names_in_ = np.array(col_labels)
        self.feature_names_out_ = np.array(col_labels)
        self.is_fitted_ = True
        return self

    def _calculate_bins(self, x_col, col_idx=None, guidance=None, sample_weight=None):
        """
        Abstract method to calculate interval bin edges for a column, possibly using guidance columns.
        Must be implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement _calculate_bins for guided interval binning."
        )

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        """String representation of the estimator."""
        class_name = self.__class__.__name__
        params = []

        if self.preserve_dataframe:
            params.append("preserve_dataframe=True")
        if self.fit_jointly:
            params.append("fit_jointly=True")
        if self.guidance_columns is not None:
            if isinstance(self.guidance_columns, list):
                params.append(f"guidance_columns={self.guidance_columns}")
            else:
                params.append(f"guidance_columns=[{self.guidance_columns}]")

        param_str = ", ".join(params)
        result = f"{class_name}({param_str})"

        # Truncate if exceeds N_CHAR_MAX
        if len(result) > N_CHAR_MAX:
            result = result[: N_CHAR_MAX - 3] + "..."

        return result
