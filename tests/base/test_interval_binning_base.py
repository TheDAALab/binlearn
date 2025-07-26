"""
Comprehensive test suite for IntervalBinningBase covering critical code paths.
"""

import pytest
import numpy as np
from typing import Any, List, Dict, Tuple
from unittest.mock import Mock, patch, MagicMock

from binning.base._interval_binning_base import IntervalBinningBase
from binning.base._constants import MISSING_VALUE, ABOVE_RANGE, BELOW_RANGE


class ConcreteIntervalBinning(IntervalBinningBase):
    """Concrete implementation for testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._calculate_bins_calls = []
        self._calculate_bins_jointly_calls = []
        self._joint_params_calls = []

    def _calculate_bins(self, x_col: np.ndarray, col_id: Any) -> Tuple[List[float], List[float]]:
        """Mock bin calculation."""
        self._calculate_bins_calls.append((x_col.copy(), col_id))

        # Create simple bins based on data range
        if len(x_col) == 0 or np.all(np.isnan(x_col)):
            edges = [0.0, 1.0, 2.0]
            reps = [0.5, 1.5]
        else:
            min_val, max_val = np.nanmin(x_col), np.nanmax(x_col)
            if min_val == max_val:
                edges = [min_val - 0.5, min_val + 0.5]
                reps = [min_val]
            else:
                edges = [min_val, (min_val + max_val) / 2, max_val]
                reps = [
                    (min_val + (min_val + max_val) / 2) / 2,
                    ((min_val + max_val) / 2 + max_val) / 2,
                ]

        return edges, reps

    def _calculate_joint_parameters(self, X: np.ndarray, columns: List[Any]) -> Dict[str, Any]:
        """Mock joint parameter calculation."""
        self._joint_params_calls.append((X.copy(), columns.copy()))
        return {"global_min": np.nanmin(X), "global_max": np.nanmax(X)}

    def _calculate_bins_jointly(
        self, x_col: np.ndarray, col_id: Any, joint_params: Dict[str, Any]
    ) -> Tuple[List[float], List[float]]:
        """Mock joint bin calculation."""
        self._calculate_bins_jointly_calls.append((x_col.copy(), col_id, joint_params.copy()))

        # Use global range from joint params
        global_min = joint_params.get("global_min", 0)
        global_max = joint_params.get("global_max", 1)

        edges = [global_min, (global_min + global_max) / 2, global_max]
        reps = [
            (global_min + (global_min + global_max) / 2) / 2,
            ((global_min + global_max) / 2 + global_max) / 2,
        ]

        return edges, reps


class TestInitialization:
    """Test initialization and parameter handling."""

    def test_default_initialization(self):
        """Test default parameter values."""
        binner = ConcreteIntervalBinning()

        assert binner.clip is True
        assert binner.preserve_dataframe is False
        assert binner.bin_edges is None
        assert binner.bin_representatives is None
        assert binner.fit_jointly is False
        assert binner._user_bin_edges is None
        assert binner._user_bin_reps is None
        assert binner._bin_edges == {}
        assert binner._bin_reps == {}

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        edges = {0: [0, 1, 2], 1: [10, 20, 30]}
        reps = {0: [0.5, 1.5], 1: [15, 25]}

        binner = ConcreteIntervalBinning(
            clip=False,
            preserve_dataframe=True,
            bin_edges=edges,
            bin_representatives=reps,
            fit_jointly=True,
        )

        assert binner.clip is False
        assert binner.preserve_dataframe is True
        assert binner.bin_edges == edges
        assert binner.bin_representatives == reps
        assert binner.fit_jointly is True
        assert binner._user_bin_edges == edges
        assert binner._user_bin_reps == reps


class TestFittingRouting:
    """Test fitting routing between per-column and joint methods."""

    def test_per_column_fitting(self):
        """Test per-column fitting path."""
        binner = ConcreteIntervalBinning(fit_jointly=False)
        X = np.array([[1, 10], [2, 20], [3, 30]])

        binner.fit(X)

        # Should call _calculate_bins for each column
        assert len(binner._calculate_bins_calls) == 2
        assert len(binner._calculate_bins_jointly_calls) == 0
        assert len(binner._joint_params_calls) == 0

        # Check that bins were created for both columns
        assert 0 in binner._bin_edges
        assert 1 in binner._bin_edges
        assert 0 in binner._bin_reps
        assert 1 in binner._bin_reps

    def test_joint_fitting(self):
        """Test joint fitting path."""
        binner = ConcreteIntervalBinning(fit_jointly=True)
        X = np.array([[1, 10], [2, 20], [3, 30]])

        binner.fit(X)

        # Should call joint methods
        assert len(binner._calculate_bins_calls) == 0
        assert len(binner._calculate_bins_jointly_calls) == 2
        assert len(binner._joint_params_calls) == 1

        # Check that bins were created for both columns using joint params
        assert 0 in binner._bin_edges
        assert 1 in binner._bin_edges

        # Both columns should use same global range (1-30)
        edges_0 = binner._bin_edges[0]
        edges_1 = binner._bin_edges[1]
        assert edges_0[0] == 1  # global min
        assert edges_0[-1] == 30  # global max
        assert edges_1[0] == 1  # global min
        assert edges_1[-1] == 30  # global max


class TestUserSpecifications:
    """Test handling of user-provided bin specifications."""

    @patch("binning.base._interval_binning_base.ensure_bin_dict")
    @patch("binning.base._interval_binning_base.validate_bins")
    def test_user_provided_edges_skip_calculation(self, mock_validate, mock_ensure):
        """Test that user-provided edges skip calculation."""
        mock_ensure.side_effect = lambda x: x  # Return input unchanged

        edges = {0: [0, 1, 2], 1: [10, 20, 30]}
        binner = ConcreteIntervalBinning(bin_edges=edges)
        X = np.array([[0.5, 15], [1.5, 25]])

        binner.fit(X)

        # Should not call calculation methods
        assert len(binner._calculate_bins_calls) == 0
        assert len(binner._calculate_bins_jointly_calls) == 0

        # Should call ensure_bin_dict and validate_bins
        mock_ensure.assert_called()
        mock_validate.assert_called_once()

    @patch("binning.base._interval_binning_base.default_representatives")
    def test_missing_representatives_generated(self, mock_default_reps):
        """Test that missing representatives are generated."""
        mock_default_reps.return_value = [0.5, 1.5]

        # Use bin_edges without bin_representatives to force generation
        edges = {0: [0, 1, 2]}
        binner = ConcreteIntervalBinning(bin_edges=edges)  # No representatives provided
        X = np.array([[1, 2]])

        binner.fit(X)

        # Should call default_representatives for columns missing reps
        assert mock_default_reps.call_count >= 1


class TestTransformation:
    """Test the complex transformation logic."""

    def test_basic_transformation(self):
        """Test basic bin index transformation."""
        binner = ConcreteIntervalBinning()
        X_fit = np.array([[1, 10], [2, 20], [3, 30]])
        X_transform = np.array([[1.5, 15], [2.5, 25]])

        binner.fit(X_fit)
        result = binner.transform(X_transform)

        assert result.shape == X_transform.shape
        assert result.dtype == int
        # Values should be valid bin indices
        assert np.all(result >= 0)

    def test_transformation_with_clipping(self):
        """Test transformation with clipping enabled."""
        binner = ConcreteIntervalBinning(clip=True)
        X_fit = np.array([[1, 10], [2, 20], [3, 30]])
        X_transform = np.array([[-10, 5], [100, 50]])  # Values outside range

        binner.fit(X_fit)
        result = binner.transform(X_transform)

        # With clipping, all values should be valid bin indices
        max_bins = [len(binner._bin_edges[col]) - 2 for col in [0, 1]]
        assert np.all(result[:, 0] >= 0) and np.all(result[:, 0] <= max_bins[0])
        assert np.all(result[:, 1] >= 0) and np.all(result[:, 1] <= max_bins[1])

    def test_transformation_without_clipping(self):
        """Test transformation without clipping."""
        binner = ConcreteIntervalBinning(clip=False)
        X_fit = np.array([[1, 10], [2, 20], [3, 30]])
        X_transform = np.array([[-10, 5], [100, 50]])  # Values outside range

        binner.fit(X_fit)
        result = binner.transform(X_transform)

        # Without clipping, should have special values for out-of-range
        assert BELOW_RANGE in result.flatten() or ABOVE_RANGE in result.flatten()

    def test_transformation_with_nan_values(self):
        """Test transformation with NaN values."""
        binner = ConcreteIntervalBinning()
        X_fit = np.array([[1, 10], [2, 20], [3, 30]])
        X_transform = np.array([[np.nan, 15], [2.5, np.nan]])

        binner.fit(X_fit)
        result = binner.transform(X_transform)

        # NaN values should be mapped to MISSING_VALUE
        assert result[0, 0] == MISSING_VALUE
        assert result[1, 1] == MISSING_VALUE


class TestInverseTransformation:
    """Test inverse transformation logic."""

    def test_basic_inverse_transformation(self):
        """Test basic inverse transformation."""
        binner = ConcreteIntervalBinning()
        X = np.array([[1, 10], [2, 20], [3, 30]])

        binner.fit(X)
        transformed = binner.transform(X)
        reconstructed = binner.inverse_transform(transformed)

        assert reconstructed.shape == X.shape
        assert reconstructed.dtype == float
        # Should be finite representative values
        assert np.all(np.isfinite(reconstructed[~np.isnan(reconstructed)]))

    def test_inverse_transform_special_values(self):
        """Test inverse transformation with special values."""
        binner = ConcreteIntervalBinning()
        X = np.array([[1, 10], [2, 20]])

        binner.fit(X)

        # Create transformed data with special values
        special_data = np.array([[MISSING_VALUE, 0], [BELOW_RANGE, 1], [ABOVE_RANGE, 0]])

        result = binner.inverse_transform(special_data)

        # Check special value handling
        assert np.isnan(result[0, 0])  # MISSING_VALUE -> NaN
        assert result[1, 0] == -np.inf  # BELOW_RANGE -> -inf
        assert result[2, 0] == np.inf  # ABOVE_RANGE -> +inf


class TestColumnKeyMatching:
    """Test the complex column key matching logic."""

    def test_direct_column_match(self):
        """Test direct column name matching."""
        binner = ConcreteIntervalBinning()

        available_keys = ["A", "B", "C"]

        # Direct match should work
        key = binner._get_column_key("B", available_keys, 1)
        assert key == "B"

    def test_index_based_fallback(self):
        """Test fallback to index-based matching."""
        binner = ConcreteIntervalBinning()

        available_keys = ["A", "B", "C"]

        # When target not found, should fall back to index
        key = binner._get_column_key("NOT_FOUND", available_keys, 1)
        assert key == "B"  # Should get index 1

    def test_no_match_raises_error(self):
        """Test error when no match found."""
        binner = ConcreteIntervalBinning()

        available_keys = ["A", "B"]

        # Index out of range and no direct match
        with pytest.raises(ValueError, match="No bin specification found"):
            binner._get_column_key("NOT_FOUND", available_keys, 5)


class TestUtilityMethods:
    """Test utility methods."""

    def test_lookup_bin_widths(self):
        """Test bin width lookup."""
        binner = ConcreteIntervalBinning()
        X = np.array([[1, 10], [2, 20], [3, 30]])

        binner.fit(X)
        bin_indices = binner.transform(X)
        widths = binner.lookup_bin_widths(bin_indices)

        assert widths.shape == X.shape
        assert widths.dtype == float
        assert np.all(widths >= 0)  # Widths should be non-negative

    def test_lookup_bin_ranges(self):
        """Test bin range lookup."""
        binner = ConcreteIntervalBinning()
        X = np.array([[1, 10], [2, 20], [3, 30]])

        binner.fit(X)
        ranges = binner.lookup_bin_ranges()

        assert isinstance(ranges, dict)
        assert 0 in ranges and 1 in ranges
        assert all(isinstance(v, int) and v > 0 for v in ranges.values())


class TestParameterManagement:
    """Test parameter getting and setting."""

    def test_get_params_unfitted(self):
        """Test get_params when not fitted."""
        edges = {0: [0, 1, 2]}
        binner = ConcreteIntervalBinning(bin_edges=edges)

        params = binner.get_params()

        # Should return constructor values
        assert params["bin_edges"] == edges
        assert "clip" in params
        assert "fit_jointly" in params

    def test_get_params_fitted(self):
        """Test get_params when fitted."""
        binner = ConcreteIntervalBinning()
        X = np.array([[1, 2]])

        binner.fit(X)
        params = binner.get_params()

        # Should return fitted values
        assert "bin_edges" in params
        assert params["bin_edges"] == binner._bin_edges

    def test_set_params_resets_fitted_state(self):
        """Test that setting bin params resets fitted state."""
        binner = ConcreteIntervalBinning()
        X = np.array([[1, 2]])

        binner.fit(X)
        assert binner._fitted is True

        # Setting bin_edges should reset fitted state
        binner.set_params(bin_edges={0: [0, 1, 2]})
        assert binner._fitted is False


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_data_handling(self):
        """Test behavior with edge case data."""
        binner = ConcreteIntervalBinning()

        # Single point data
        X = np.array([[5]])
        binner.fit(X)
        result = binner.transform(X)

        assert result.shape == (1, 1)
        assert binner._fitted is True

    def test_constant_data_handling(self):
        """Test behavior with constant data."""
        binner = ConcreteIntervalBinning()

        # All same values
        X = np.array([[5, 5], [5, 5], [5, 5]])
        binner.fit(X)
        result = binner.transform(X)

        assert result.shape == X.shape
        # Should handle constant data gracefully
        assert np.all(np.isfinite(result))

    def test_all_nan_data(self):
        """Test behavior with all-NaN data."""
        binner = ConcreteIntervalBinning()

        X = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        binner.fit(X)
        result = binner.transform(X)

        # All NaN input should produce MISSING_VALUE output
        assert np.all(result == MISSING_VALUE)


class TestAbstractMethodEnforcement:
    """Test that abstract methods are enforced."""

    def test_missing_calculate_bins_raises_error(self):
        """Test that missing _calculate_bins raises error."""

        class IncompleteBinning(IntervalBinningBase):
            pass  # Missing _calculate_bins

        with pytest.raises(TypeError):
            IncompleteBinning()


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_joint_fitting_produces_different_results(self):
        """Test that joint fitting produces different results than per-column."""
        X = np.array([[1, 100], [2, 200], [3, 300]])  # Different scales

        # Per-column fitting
        binner_individual = ConcreteIntervalBinning(fit_jointly=False)
        binner_individual.fit(X)

        # Joint fitting
        binner_joint = ConcreteIntervalBinning(fit_jointly=True)
        binner_joint.fit(X)

        # Should produce different bin edges
        edges_ind = binner_individual._bin_edges
        edges_joint = binner_joint._bin_edges

        # At least one column should have different edges
        different = False
        for col in edges_ind:
            if edges_ind[col] != edges_joint[col]:
                different = True
                break

        assert different, "Joint fitting should produce different results"

    def test_complete_workflow_with_special_values(self):
        """Test complete workflow including special value handling."""
        binner = ConcreteIntervalBinning(clip=False, fit_jointly=True)

        # Fit with normal data
        X_fit = np.array([[1, 10], [2, 20], [3, 30]])
        binner.fit(X_fit)

        # Transform with challenging data
        X_transform = np.array(
            [
                [1.5, 15],  # Normal values
                [-10, 5],  # Below range
                [100, 50],  # Above range
                [np.nan, np.nan],  # Missing values
            ]
        )

        # Transform
        transformed = binner.transform(X_transform)

        # Inverse transform
        reconstructed = binner.inverse_transform(transformed)

        # Check that special values are handled correctly
        assert np.isnan(reconstructed[3, 0]) and np.isnan(reconstructed[3, 1])
        assert reconstructed[1, 0] == -np.inf or reconstructed[1, 1] == -np.inf
        assert reconstructed[2, 0] == np.inf or reconstructed[2, 1] == np.inf

        # Check utility methods work
        widths = binner.lookup_bin_widths(transformed)
        ranges = binner.lookup_bin_ranges()

        assert widths.shape == transformed.shape
        assert len(ranges) == 2
