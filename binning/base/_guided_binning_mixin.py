"""
Mixin for guided binning methods that require target/guidance data validation.
"""

from typing import Any, Dict, List, Tuple, Optional, Union
import numpy as np
from ..errors import ValidationMixin, InvalidDataError, DataQualityWarning


class GuidedBinningMixin(ValidationMixin):
    """
    Mixin providing common validation and processing patterns for guided binning methods.
    
    This mixin factors out common patterns from supervised binning methods:
    - Guidance data validation and preprocessing
    - Missing value handling for feature-target pairs
    - Dimensionality checks and reshaping
    - Data quality validation for both features and targets
    """
    
    def validate_guidance_data(
        self, 
        guidance_data: np.ndarray,
        expected_columns: int = 1,
        name: str = "guidance_data"
    ) -> np.ndarray:
        """
        Validate and preprocess guidance data for supervised binning.
        
        Parameters
        ----------
        guidance_data : np.ndarray
            Raw guidance/target data to validate
        expected_columns : int, default=1
            Expected number of guidance columns (1 for most methods)
        name : str, default="guidance_data"
            Name for error messages
            
        Returns
        -------
        np.ndarray
            Validated and potentially reshaped guidance data (always 1D for single column)
            
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
        
        # Handle dimensionality
        if guidance_validated.ndim == 1:
            if expected_columns != 1:
                raise ValueError(
                    f"{name} is 1D but {expected_columns} columns expected"
                )
            return guidance_validated
        elif guidance_validated.ndim == 2:
            if guidance_validated.shape[1] != expected_columns:
                raise ValueError(
                    f"{name} has {guidance_validated.shape[1]} columns, "
                    f"expected exactly {expected_columns}. "
                    f"Please specify the correct guidance column(s)."
                )
            # For single column, flatten to 1D for easier processing
            if expected_columns == 1:
                return guidance_validated.ravel()
            return guidance_validated
        else:
            raise ValueError(
                f"{name} has {guidance_validated.ndim} dimensions, "
                f"expected 1D or 2D array"
            )
    
    def validate_feature_target_pair(
        self,
        x_col: np.ndarray,
        guidance_data: np.ndarray,
        col_id: Any = None
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
        
        # Check data quality
        self.check_data_quality(x_col_validated, f"feature column {col_id}")
        self.check_data_quality(guidance_data_validated, "guidance data")
        
        # Convert feature to float for numeric operations
        x_col = np.asarray(x_col_validated, dtype=float)
        
        # Create valid data mask (both feature and target must be non-missing)
        feature_finite = np.isfinite(x_col)
        
        if guidance_data_validated.dtype == object:
            # Handle object dtype (e.g., strings, mixed types)
            target_valid = guidance_data_validated != None
            # Also check for string representations of missing values
            if hasattr(guidance_data_validated, '__iter__'):
                string_missing = np.array([
                    str(val).lower() in {'nan', 'none', 'null', ''} 
                    if val is not None else True
                    for val in guidance_data_validated
                ])
                target_valid = target_valid & ~string_missing
        else:
            # Numeric guidance data
            try:
                target_valid = np.isfinite(guidance_data_validated.astype(float))
            except (ValueError, TypeError):
                # Fallback for non-numeric data
                target_valid = guidance_data_validated != None
        
        valid_mask = feature_finite & target_valid
        
        return x_col, guidance_data_validated, valid_mask
    
    def handle_insufficient_data(
        self,
        x_col: np.ndarray, 
        valid_mask: np.ndarray,
        min_samples: int,
        col_id: Any = None
    ) -> Optional[Tuple[List[float], List[float]]]:
        """
        Handle cases with insufficient valid data for meaningful binning.
        
        Parameters
        ----------
        x_col : np.ndarray
            Feature column data
        valid_mask : np.ndarray
            Boolean mask for valid data points
        min_samples : int
            Minimum number of samples required
        col_id : Any, optional
            Column identifier for warnings
            
        Returns
        -------
        Tuple[List[float], List[float]]
            Default bin edges and representatives (single bin)
        """
        n_valid = valid_mask.sum()
        
        if n_valid == 0:
            # No valid data - create default range
            min_val = np.nanmin(x_col) if not np.isnan(x_col).all() else 0.0
            max_val = np.nanmax(x_col) if not np.isnan(x_col).all() else 1.0
            if min_val == max_val:
                max_val = min_val + 1.0
            
            if col_id is not None:
                import warnings
                warnings.warn(
                    f"Column {col_id} has no valid data points. Using default bin range [{min_val}, {max_val}]",
                    DataQualityWarning
                )
            
            return [min_val, max_val], [(min_val + max_val) / 2]
        
        elif n_valid < min_samples:
            # Insufficient data for complex binning
            valid_data = x_col[valid_mask]
            min_val = np.min(valid_data)
            max_val = np.max(valid_data)
            if min_val == max_val:
                max_val = min_val + 1.0
                
            if col_id is not None:
                import warnings
                warnings.warn(
                    f"Column {col_id} has only {n_valid} valid samples (minimum {min_samples} required). "
                    f"Creating single bin.",
                    DataQualityWarning
                )
            
            return [min_val, max_val], [(min_val + max_val) / 2]
        
        # Sufficient data - let caller proceed with normal binning
        return None
    
    def extract_valid_pairs(
        self,
        x_col: np.ndarray,
        guidance_data: np.ndarray, 
        valid_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract valid feature-target pairs for model fitting.
        
        Parameters
        ----------
        x_col : np.ndarray
            Feature column data  
        guidance_data : np.ndarray
            Target/guidance data
        valid_mask : np.ndarray
            Boolean mask for valid pairs
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - x_valid: Valid feature data reshaped for sklearn (n_samples, 1)
            - y_valid: Valid target data (1D)
        """
        x_valid = x_col[valid_mask].reshape(-1, 1)
        y_valid = guidance_data[valid_mask]
        return x_valid, y_valid
    
    def require_guidance_data(self, guidance_data: Optional[np.ndarray], method_name: str = "guided binning") -> None:
        """
        Ensure guidance data is provided for methods that require it.
        
        Parameters
        ----------
        guidance_data : Optional[np.ndarray]
            Guidance data to check
        method_name : str, default="guided binning"
            Name of the method for error messages
            
        Raises
        ------
        ValueError
            If guidance_data is None
        """
        if guidance_data is None:
            raise ValueError(
                f"{method_name} requires guidance_data (target values) to be provided. "
                f"Please specify guidance_columns when creating the transformer."
            )
    
    def validate_task_type(self, task_type: str, valid_types: List[str]) -> None:
        """
        Validate task type parameter.
        
        Parameters
        ----------
        task_type : str
            Task type to validate
        valid_types : List[str]  
            List of valid task types
            
        Raises
        ------
        ValueError
            If task_type is not in valid_types
        """
        if task_type not in valid_types:
            raise ValueError(
                f"task_type must be one of {valid_types}, got '{task_type}'"
            )
    
    def create_fallback_bins(
        self,
        x_col: np.ndarray,
        default_range: Optional[Tuple[float, float]] = None
    ) -> Tuple[List[float], List[float]]:
        """
        Create fallback bins when normal binning fails.
        
        Parameters
        ----------
        x_col : np.ndarray
            Feature column data
        default_range : Optional[Tuple[float, float]]
            Optional default range to use
            
        Returns
        -------
        Tuple[List[float], List[float]]
            Fallback bin edges and representatives
        """
        if default_range is not None:
            min_val, max_val = default_range
        else:
            min_val = np.nanmin(x_col) if not np.isnan(x_col).all() else 0.0
            max_val = np.nanmax(x_col) if not np.isnan(x_col).all() else 1.0
            if min_val == max_val:
                max_val = min_val + 1.0
        
        return [min_val, max_val], [(min_val + max_val) / 2]
