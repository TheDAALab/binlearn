"""
Comprehensive test suite for FlexibleBinningBase covering critical code paths.
"""

import pytest
import numpy as np
from typing import Any, List, Dict, Tuple
from unittest.mock import Mock, patch, MagicMock

from binning.base._flexible_binning_base import (
    FlexibleBinningBase,
    FlexibleBinSpec,
    FlexibleBinReps,
)
from binning.base._constants import MISSING_VALUE


class ConcreteFlexibleBinning(FlexibleBinningBase):
    """Concrete implementation for testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._calculate_flexible_bins_calls = []
        self._calculate_flexible_bins_jointly_calls = []
        self._joint_params_calls = []

    def _calculate_flexible_bins(
        self, x_col: np.ndarray, col_id: Any
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Mock flexible bin calculation."""
        self._calculate_flexible_bins_calls.append((x_col.copy(), col_id))

        # Create simple bins: singleton for unique values, interval for range
        unique_vals = np.unique(x_col[~np.isnan(x_col)])

        if len(unique_vals) == 0:
            # All NaN data
            bin_defs = [{"singleton": 0.0}]
            reps = [0.0]
        elif len(unique_vals) <= 2:
            # Few unique values - use singletons
            bin_defs = [{"singleton": float(val)} for val in unique_vals]
            reps = [float(val) for val in unique_vals]
        else:
            # Many values - use intervals
            min_val, max_val = float(unique_vals[0]), float(unique_vals[-1])
            mid_val = (min_val + max_val) / 2
            bin_defs = [{"interval": [min_val, mid_val]}, {"interval": [mid_val, max_val]}]
            reps = [(min_val + mid_val) / 2, (mid_val + max_val) / 2]

        return bin_defs, reps

    def _calculate_joint_parameters(self, X: np.ndarray, columns: List[Any]) -> Dict[str, Any]:
        """Mock joint parameter calculation."""
        self._joint_params_calls.append((X.copy(), columns.copy()))
        return {"global_min": np.nanmin(X), "global_max": np.nanmax(X)}

    def _calculate_flexible_bins_jointly(
        self, x_col: np.ndarray, col_id: Any, joint_params: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Mock joint flexible bin calculation."""
        self._calculate_flexible_bins_jointly_calls.append(
            (x_col.copy(), col_id, joint_params.copy())
        )

        # Use global range from joint params
        global_min = joint_params.get("global_min", 0)
        global_max = joint_params.get("global_max", 1)

        # Create intervals based on global range
        mid_val = (global_min + global_max) / 2
        bin_defs = [{"interval": [global_min, mid_val]}, {"interval": [mid_val, global_max]}]
        reps = [(global_min + mid_val) / 2, (mid_val + global_max) / 2]

        return bin_defs, reps


class TestInitialization:
    """Test initialization and parameter handling."""

    def test_default_initialization(self):
        """Test default parameter values."""
        binner = ConcreteFlexibleBinning()

        assert binner.preserve_dataframe is False
        assert binner.bin_spec is None
        assert binner.bin_representatives is None
        assert binner.fit_jointly is False
        assert binner._user_bin_spec is None
        assert binner._user_bin_reps is None
        assert binner._bin_spec == {}
        assert binner._bin_reps == {}

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        bin_spec = {0: [{"singleton": 1}, {"interval": [2, 4]}]}
        bin_reps = {0: [1.0, 3.0]}

        binner = ConcreteFlexibleBinning(
            preserve_dataframe=True,
            bin_spec=bin_spec,
            bin_representatives=bin_reps,
            fit_jointly=True,
        )

        assert binner.preserve_dataframe is True
        assert binner.bin_spec == bin_spec
        assert binner.bin_representatives == bin_reps
        assert binner.fit_jointly is True
        assert binner._user_bin_spec == bin_spec
        assert binner._user_bin_reps == bin_reps


class TestFittingRouting:
    """Test fitting routing between per-column and joint methods."""

    def test_per_column_fitting(self):
        """Test per-column fitting path."""
        binner = ConcreteFlexibleBinning(fit_jointly=False)
        X = np.array([[1, 10], [2, 20], [3, 30]])

        binner.fit(X)

        # Should call _calculate_flexible_bins for each column
        assert len(binner._calculate_flexible_bins_calls) == 2
        assert len(binner._calculate_flexible_bins_jointly_calls) == 0
        assert len(binner._joint_params_calls) == 0

        # Check that bins were created for both columns
        assert 0 in binner._bin_spec
        assert 1 in binner._bin_spec
        assert 0 in binner._bin_reps
        assert 1 in binner._bin_reps

    def test_joint_fitting(self):
        """Test joint fitting path."""
        binner = ConcreteFlexibleBinning(fit_jointly=True)
        X = np.array([[1, 10], [2, 20], [3, 30]])

        binner.fit(X)

        # Should call joint methods
        assert len(binner._calculate_flexible_bins_calls) == 0
        assert len(binner._calculate_flexible_bins_jointly_calls) == 2
        assert len(binner._joint_params_calls) == 1

        # Check that bins were created for both columns using joint params
        assert 0 in binner._bin_spec
        assert 1 in binner._bin_spec

        # Both columns should use same global range (1-30)
        for col in [0, 1]:
            for bin_def in binner._bin_spec[col]:
                if "interval" in bin_def:
                    interval = bin_def["interval"]
                    assert interval[0] >= 1  # global min
                    assert interval[1] <= 30  # global max


class TestUserSpecifications:
    """Test handling of user-provided bin specifications."""

    @patch("binning.base._flexible_binning_base.ensure_bin_dict")
    def test_user_provided_specs_skip_calculation(self, mock_ensure):
        """Test that user-provided specs skip calculation."""
        mock_ensure.side_effect = lambda x: x  # Return input unchanged

        bin_spec = {0: [{"singleton": 1}, {"singleton": 2}]}
        bin_reps = {0: [1.0, 2.0]}

        binner = ConcreteFlexibleBinning(bin_spec=bin_spec, bin_representatives=bin_reps)
        X = np.array([[1, 2]])

        binner.fit(X)

        # Should not call calculation methods
        assert len(binner._calculate_flexible_bins_calls) == 0
        assert len(binner._calculate_flexible_bins_jointly_calls) == 0

        # Should use provided specs
        assert binner._bin_spec == bin_spec
        assert binner._bin_reps == bin_reps

    def test_missing_representatives_generated(self):
        """Test that missing representatives are generated."""
        bin_spec = {0: [{"singleton": 1}, {"interval": [2, 4]}]}

        binner = ConcreteFlexibleBinning(bin_spec=bin_spec)  # No representatives provided
        X = np.array([[1, 3]])

        binner.fit(X)

        # Should generate default representatives
        assert 0 in binner._bin_reps
        reps = binner._bin_reps[0]
        assert len(reps) == 2
        assert reps[0] == 1.0  # Singleton value
        assert reps[1] == 3.0  # Interval midpoint


class TestFlexibleBinValidation:
    """Test flexible bin validation logic."""

    def test_valid_singleton_bins(self):
        """Test validation of singleton bins."""
        binner = ConcreteFlexibleBinning()

        bin_spec = {0: [{"singleton": 1}, {"singleton": 2.5}]}
        bin_reps = {0: [1.0, 2.5]}

        # Should not raise error
        binner._validate_flexible_bins(bin_spec, bin_reps)

    def test_valid_interval_bins(self):
        """Test validation of interval bins."""
        binner = ConcreteFlexibleBinning()

        bin_spec = {0: [{"interval": [0, 5]}, {"interval": [5, 10]}]}
        bin_reps = {0: [2.5, 7.5]}

        # Should not raise error
        binner._validate_flexible_bins(bin_spec, bin_reps)

    def test_mismatched_bin_rep_counts(self):
        """Test validation fails when bin and rep counts don't match."""
        binner = ConcreteFlexibleBinning()

        bin_spec = {0: [{"singleton": 1}, {"singleton": 2}]}
        bin_reps = {0: [1.0]}  # Missing one rep

        with pytest.raises(ValueError, match="Number of bin definitions.*must match"):
            binner._validate_flexible_bins(bin_spec, bin_reps)

    def test_invalid_bin_definition_format(self):
        """Test validation fails for invalid bin definitions."""
        binner = ConcreteFlexibleBinning()

        # Test multiple keys
        bin_spec = {0: [{"singleton": 1, "interval": [0, 2]}]}
        bin_reps = {0: [1.0]}

        with pytest.raises(ValueError, match="must have only.*key"):
            binner._validate_flexible_bins(bin_spec, bin_reps)

        # Test invalid interval
        bin_spec = {0: [{"interval": [5, 2]}]}  # min > max
        bin_reps = {0: [3.5]}

        with pytest.raises(ValueError, match="min must be <= max"):
            binner._validate_flexible_bins(bin_spec, bin_reps)

        # Test unknown bin type
        bin_spec = {0: [{"unknown": 1}]}
        bin_reps = {0: [1.0]}

        with pytest.raises(ValueError, match="must have 'singleton' or 'interval' key"):
            binner._validate_flexible_bins(bin_spec, bin_reps)


class TestTransformation:
    """Test the flexible bin transformation logic."""

    def test_singleton_bin_transformation(self):
        """Test transformation with singleton bins."""
        bin_spec = {0: [{"singleton": 1}, {"singleton": 2}, {"singleton": 3}]}
        bin_reps = {0: [1.0, 2.0, 3.0]}

        binner = ConcreteFlexibleBinning(bin_spec=bin_spec, bin_representatives=bin_reps)
        X_fit = np.array([[1], [2], [3]])
        X_transform = np.array([[1], [2], [5]])  # 5 doesn't match any singleton

        binner.fit(X_fit)
        result = binner.transform(X_transform)

        assert result[0, 0] == 0  # Matches first singleton
        assert result[1, 0] == 1  # Matches second singleton
        assert result[2, 0] == MISSING_VALUE  # No match

    def test_interval_bin_transformation(self):
        """Test transformation with interval bins."""
        bin_spec = {0: [{"interval": [0, 5]}, {"interval": [5, 10]}]}
        bin_reps = {0: [2.5, 7.5]}

        binner = ConcreteFlexibleBinning(bin_spec=bin_spec, bin_representatives=bin_reps)
        X_fit = np.array([[2], [7]])
        X_transform = np.array([[1], [6], [15]])  # 15 doesn't match any interval

        binner.fit(X_fit)
        result = binner.transform(X_transform)

        assert result[0, 0] == 0  # Falls in first interval
        assert result[1, 0] == 1  # Falls in second interval
        assert result[2, 0] == MISSING_VALUE  # No match

    def test_mixed_bin_transformation(self):
        """Test transformation with mixed singleton and interval bins."""
        bin_spec = {0: [{"singleton": 0}, {"interval": [1, 5]}, {"singleton": 10}]}
        bin_reps = {0: [0.0, 3.0, 10.0]}

        binner = ConcreteFlexibleBinning(bin_spec=bin_spec, bin_representatives=bin_reps)
        X_fit = np.array([[0], [3], [10]])
        X_transform = np.array([[0], [2.5], [10], [7]])

        binner.fit(X_fit)
        result = binner.transform(X_transform)

        assert result[0, 0] == 0  # Singleton match
        assert result[1, 0] == 1  # Interval match
        assert result[2, 0] == 2  # Singleton match
        assert result[3, 0] == MISSING_VALUE  # No match

    def test_transformation_with_nan_values(self):
        """Test transformation with NaN values."""
        bin_spec = {0: [{"singleton": 1}, {"singleton": 2}]}
        bin_reps = {0: [1.0, 2.0]}

        binner = ConcreteFlexibleBinning(bin_spec=bin_spec, bin_representatives=bin_reps)
        X_fit = np.array([[1], [2]])
        X_transform = np.array([[np.nan], [1]])

        binner.fit(X_fit)
        result = binner.transform(X_transform)

        assert result[0, 0] == MISSING_VALUE  # NaN -> MISSING_VALUE
        assert result[1, 0] == 0  # Normal match


class TestInverseTransformation:
    """Test inverse transformation logic."""

    def test_basic_inverse_transformation(self):
        """Test basic inverse transformation."""
        bin_spec = {0: [{"singleton": 1}, {"interval": [2, 4]}]}
        bin_reps = {0: [1.0, 3.0]}

        binner = ConcreteFlexibleBinning(bin_spec=bin_spec, bin_representatives=bin_reps)
        X = np.array([[1], [3]])

        binner.fit(X)
        transformed = binner.transform(X)
        reconstructed = binner.inverse_transform(transformed)

        assert reconstructed.shape == X.shape
        assert reconstructed[0, 0] == 1.0  # Representative for singleton
        assert reconstructed[1, 0] == 3.0  # Representative for interval

    def test_inverse_transform_missing_values(self):
        """Test inverse transformation with missing values."""
        bin_spec = {0: [{"singleton": 1}]}
        bin_reps = {0: [1.0]}

        binner = ConcreteFlexibleBinning(bin_spec=bin_spec, bin_representatives=bin_reps)
        X = np.array([[1]])

        binner.fit(X)

        # Create transformed data with missing values
        missing_data = np.array([[MISSING_VALUE]])

        result = binner.inverse_transform(missing_data)

        assert np.isnan(result[0, 0])  # MISSING_VALUE -> NaN


class TestUtilityMethods:
    """Test utility methods."""

    def test_lookup_bin_widths_singleton(self):
        """Test bin width lookup for singleton bins."""
        bin_spec = {0: [{"singleton": 1}, {"singleton": 2}]}
        bin_reps = {0: [1.0, 2.0]}

        binner = ConcreteFlexibleBinning(bin_spec=bin_spec, bin_representatives=bin_reps)
        X = np.array([[1], [2]])

        binner.fit(X)
        bin_indices = binner.transform(X)
        widths = binner.lookup_bin_widths(bin_indices)

        assert widths[0, 0] == 0.0  # Singleton has zero width
        assert widths[1, 0] == 0.0  # Singleton has zero width

    def test_lookup_bin_widths_interval(self):
        """Test bin width lookup for interval bins."""
        bin_spec = {0: [{"interval": [0, 5]}, {"interval": [5, 8]}]}
        bin_reps = {0: [2.5, 6.5]}

        binner = ConcreteFlexibleBinning(bin_spec=bin_spec, bin_representatives=bin_reps)
        X = np.array([[2], [6]])

        binner.fit(X)
        bin_indices = binner.transform(X)
        widths = binner.lookup_bin_widths(bin_indices)

        assert widths[0, 0] == 5.0  # Width of [0, 5]
        assert widths[1, 0] == 3.0  # Width of [5, 8]

    def test_lookup_bin_ranges(self):
        """Test bin range lookup."""
        bin_spec = {0: [{"singleton": 1}, {"singleton": 2}], 1: [{"interval": [0, 10]}]}
        bin_reps = {0: [1.0, 2.0], 1: [5.0]}

        binner = ConcreteFlexibleBinning(bin_spec=bin_spec, bin_representatives=bin_reps)
        X = np.array([[1, 5], [2, 7]])

        binner.fit(X)
        ranges = binner.lookup_bin_ranges()

        assert ranges[0] == 2  # Two bins for column 0
        assert ranges[1] == 1  # One bin for column 1


class TestParameterManagement:
    """Test parameter getting and setting."""

    def test_get_params_unfitted(self):
        """Test get_params when not fitted."""
        bin_spec = {0: [{"singleton": 1}]}
        binner = ConcreteFlexibleBinning(bin_spec=bin_spec)

        params = binner.get_params()

        # Should return constructor values
        assert params["bin_spec"] == bin_spec
        assert "bin_representatives" in params
        assert "fit_jointly" in params

    def test_get_params_fitted(self):
        """Test get_params when fitted."""
        binner = ConcreteFlexibleBinning()
        X = np.array([[1, 2]])

        binner.fit(X)
        params = binner.get_params()

        # Should return fitted values
        assert "bin_spec" in params
        assert params["bin_spec"] == binner._bin_spec

    def test_set_params_resets_fitted_state(self):
        """Test that setting bin params resets fitted state."""
        binner = ConcreteFlexibleBinning()
        X = np.array([[1, 2]])

        binner.fit(X)
        assert binner._fitted is True

        # Setting bin_spec should reset fitted state
        binner.set_params(bin_spec={0: [{"singleton": 5}]})
        assert binner._fitted is False


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_bin_spec_dict(self):
        """Test behavior with empty bin specifications."""
        binner = ConcreteFlexibleBinning(bin_spec={})
        X = np.array([[1, 2]])

        # Should fall back to calculated bins
        binner.fit(X)
        assert len(binner._bin_spec) == 2  # Should create bins for both columns

    def test_all_nan_data(self):
        """Test behavior with all-NaN data."""
        binner = ConcreteFlexibleBinning()
        X = np.array([[np.nan, np.nan], [np.nan, np.nan]])

        binner.fit(X)
        result = binner.transform(X)

        # All NaN input should produce MISSING_VALUE output
        assert np.all(result == MISSING_VALUE)

    def test_single_value_data(self):
        """Test behavior with single value repeated."""
        binner = ConcreteFlexibleBinning()
        X = np.array([[5, 5], [5, 5], [5, 5]])

        binner.fit(X)
        result = binner.transform(X)

        # Should create singleton bins and all values should map to bin 0
        assert np.all(result == 0)


class TestAbstractMethodEnforcement:
    """Test that abstract methods are enforced."""

    def test_missing_calculate_flexible_bins_raises_error(self):
        """Test that missing _calculate_flexible_bins raises error."""

        class IncompleteBinning(FlexibleBinningBase):
            pass  # Missing _calculate_flexible_bins

        with pytest.raises(TypeError):
            IncompleteBinning()


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_joint_vs_individual_fitting_different_results(self):
        """Test that joint fitting produces different results than individual."""
        X = np.array([[1, 100], [2, 200], [3, 300]])  # Different scales

        # Individual fitting
        binner_individual = ConcreteFlexibleBinning(fit_jointly=False)
        binner_individual.fit(X)

        # Joint fitting
        binner_joint = ConcreteFlexibleBinning(fit_jointly=True)
        binner_joint.fit(X)

        # Should use different approaches
        assert binner_individual._calculate_flexible_bins_calls
        assert binner_joint._calculate_flexible_bins_jointly_calls
        assert not binner_individual._calculate_flexible_bins_jointly_calls
        assert not binner_joint._calculate_flexible_bins_calls

    def test_complete_workflow_with_mixed_bins(self):
        """Test complete workflow with mixed singleton and interval bins."""
        # Complex bin specification
        bin_spec = {
            0: [{"singleton": 0}, {"interval": [1, 5]}, {"singleton": 10}],
            1: [{"interval": [0, 50]}, {"interval": [50, 100]}],
        }
        bin_reps = {0: [0.0, 3.0, 10.0], 1: [25.0, 75.0]}

        binner = ConcreteFlexibleBinning(
            bin_spec=bin_spec, bin_representatives=bin_reps, fit_jointly=True
        )

        # Fit with normal data
        X_fit = np.array([[0, 25], [3, 75], [10, 25]])
        binner.fit(X_fit)

        # Transform with challenging data
        X_transform = np.array(
            [
                [0, 25],  # Normal matches
                [2, 60],  # Interval matches
                [10, 75],  # Mixed matches
                [7, 110],  # No matches
                [np.nan, np.nan],  # Missing values
            ]
        )

        # Transform
        transformed = binner.transform(X_transform)

        # Inverse transform
        reconstructed = binner.inverse_transform(transformed)

        # Check results
        assert transformed[0, 0] == 0 and transformed[0, 1] == 0  # Exact matches
        assert transformed[1, 0] == 1 and transformed[1, 1] == 1  # Interval matches
        assert transformed[2, 0] == 2 and transformed[2, 1] == 1  # Mixed
        assert (
            transformed[3, 0] == MISSING_VALUE and transformed[3, 1] == MISSING_VALUE
        )  # No matches
        assert transformed[4, 0] == MISSING_VALUE and transformed[4, 1] == MISSING_VALUE  # NaN

        # Check inverse transform handles missing values
        assert np.isnan(reconstructed[3, 0]) and np.isnan(reconstructed[3, 1])
        assert np.isnan(reconstructed[4, 0]) and np.isnan(reconstructed[4, 1])

        # Check utility methods work
        widths = binner.lookup_bin_widths(transformed)
        ranges = binner.lookup_bin_ranges()

        assert widths.shape == transformed.shape
        assert ranges[0] == 3 and ranges[1] == 2
