"""
Minimal test suite for OneHotBinning covering critical code paths.
"""

import pytest
import numpy as np
from binning.methods._onehot_binning import OneHotBinning
from binning.base._constants import MISSING_VALUE


class TestOneHotBinningCore:
    """Test core functionality of OneHotBinning."""

    def test_basic_numeric_binning(self):
        """Test basic singleton bin creation for numeric data."""
        binner = OneHotBinning()
        X = np.array([[1, 10], [2, 20], [1, 10], [3, 30]])

        binner.fit(X)
        result = binner.transform(X)

        # Should create singleton bins for each unique value
        assert result.shape == X.shape  # Same shape as input
        assert result[0, 0] == 0 and result[2, 0] == 0  # Same value -> same bin
        assert result[1, 0] == 1  # Different value -> different bin
        assert result[3, 0] == 2  # Third unique value -> third bin

        # Check bin specifications
        assert 0 in binner._bin_spec and 1 in binner._bin_spec
        assert len(binner._bin_spec[0]) == 3  # Three unique values in col 0
        assert len(binner._bin_spec[1]) == 3  # Three unique values in col 1

        # Check all bins are singletons
        for bin_def in binner._bin_spec[0]:
            assert "singleton" in bin_def
        for bin_def in binner._bin_spec[1]:
            assert "singleton" in bin_def

    def test_per_column_vs_joint_fitting(self):
        """Test difference between per-column and joint fitting."""
        X = np.array([[1, 10], [2, 20]])  # Different unique values per column

        # Per-column fitting
        binner_individual = OneHotBinning(fit_jointly=False)
        binner_individual.fit(X)

        # Joint fitting
        binner_joint = OneHotBinning(fit_jointly=True)
        binner_joint.fit(X)

        # Joint fitting should create more bins per column (global unique values)
        individual_bins_col0 = len(binner_individual._bin_spec[0])
        joint_bins_col0 = len(binner_joint._bin_spec[0])

        assert individual_bins_col0 == 2  # Only [1, 2] for column 0
        assert joint_bins_col0 == 4  # [1, 2, 10, 20] for both columns

        # Both columns should have same bins in joint mode
        assert binner_joint._bin_spec[0] == binner_joint._bin_spec[1]

    def test_transform_with_missing_values(self):
        """Test transformation handles values not in training data."""
        binner = OneHotBinning()
        X_fit = np.array([[1, 10], [2, 20]])
        X_transform = np.array([[1, 10], [3, 30], [np.nan, 15]])  # 3, 30, 15 not in fit data

        binner.fit(X_fit)
        result = binner.transform(X_transform)

        # Known values should map correctly
        assert result[0, 0] == 0 and result[0, 1] == 0  # 1->bin0, 10->bin0

        # Unknown values should map to MISSING_VALUE
        assert result[1, 0] == MISSING_VALUE  # 3 not in training data
        assert result[1, 1] == MISSING_VALUE  # 30 not in training data
        assert result[2, 1] == MISSING_VALUE  # 15 not in training data

        # NaN should map to MISSING_VALUE
        assert result[2, 0] == MISSING_VALUE

    def test_inverse_transform(self):
        """Test inverse transformation returns representatives."""
        binner = OneHotBinning()
        X = np.array([[1, 10], [2, 20], [3, 30]])

        binner.fit(X)
        transformed = binner.transform(X)
        reconstructed = binner.inverse_transform(transformed)

        # Should return the original unique values (representatives)
        assert reconstructed.shape == X.shape
        assert np.allclose(reconstructed, X, equal_nan=True)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_max_unique_values_limit(self):
        """Test max_unique_values parameter enforcement."""
        # Create data with too many unique values
        X = np.arange(150).reshape(-1, 1)  # 150 unique values

        binner = OneHotBinning(max_unique_values=100)

        with pytest.raises(ValueError, match="exceeds max_unique_values"):
            binner.fit(X)

    def test_max_unique_values_joint_limit(self):
        """Test max_unique_values with joint fitting."""
        # Create data where joint unique values exceed limit
        X = np.array([[i, i + 50] for i in range(60)])  # 120 total unique values

        binner = OneHotBinning(fit_jointly=True, max_unique_values=100)

        with pytest.raises(ValueError, match="Joint fitting found.*exceeds max_unique_values"):
            binner.fit(X)

    def test_all_nan_data(self):
        """Test behavior with all-NaN data."""
        binner = OneHotBinning()
        X = np.array([[np.nan, np.nan], [np.nan, np.nan]])

        binner.fit(X)
        result = binner.transform(X)

        # Should create default bins and map NaN to MISSING_VALUE
        assert np.all(result == MISSING_VALUE)
        assert len(binner._bin_spec[0]) == 1  # Default bin
        assert binner._bin_spec[0][0] == {"singleton": 0.0}

    def test_single_unique_value(self):
        """Test behavior with only one unique value per column."""
        binner = OneHotBinning()
        X = np.array([[5, 5], [5, 5], [5, 5]])

        binner.fit(X)
        result = binner.transform(X)

        # All values should map to bin 0
        assert np.all(result == 0)
        assert len(binner._bin_spec[0]) == 1
        assert binner._bin_spec[0][0] == {"singleton": 5.0}


class TestDataTypes:
    """Test handling of different data types."""

    def test_mixed_numeric_types(self):
        """Test handling of integers and floats."""
        binner = OneHotBinning()
        X = np.array([[1, 1.5], [2, 2.5], [1, 1.5]])  # Mix of int and float

        binner.fit(X)
        result = binner.transform(X)

        # Should handle both types and create appropriate bins
        assert result.shape == X.shape
        assert result[0, 0] == result[2, 0]  # Same values -> same bin

        # Check representatives are float (converted from mixed types)
        for col in [0, 1]:
            for rep in binner._bin_reps[col]:
                assert isinstance(rep, float)

    def test_object_dtype_handling(self):
        """Test handling of object arrays with None values."""
        binner = OneHotBinning()
        X = np.array([["A", "X"], ["B", "Y"], [None, "X"]], dtype=object)

        binner.fit(X)
        result = binner.transform(X)

        # Should handle None values and create appropriate bins
        assert result.shape == X.shape
        assert len(binner._bin_spec[0]) == 2  # 'A', 'B' (None excluded)
        assert len(binner._bin_spec[1]) == 2  # 'X', 'Y'


class TestUserSpecifications:
    """Test user-provided bin specifications."""

    def test_user_provided_bin_spec(self):
        """Test using pre-defined bin specifications."""
        bin_spec = {
            0: [{"singleton": 1}, {"singleton": 2}],
            1: [{"singleton": 10}, {"singleton": 20}],
        }
        bin_reps = {0: [1.0, 2.0], 1: [10.0, 20.0]}

        binner = OneHotBinning(bin_spec=bin_spec, bin_representatives=bin_reps)
        X = np.array([[1, 10], [2, 20]])

        binner.fit(X)
        result = binner.transform(X)

        # Should use provided specifications
        assert binner._bin_spec == bin_spec
        assert binner._bin_reps == bin_reps
        assert result[0, 0] == 0 and result[0, 1] == 0
        assert result[1, 0] == 1 and result[1, 1] == 1

    def test_partial_user_specifications(self):
        """Test with bin_spec but no bin_representatives."""
        bin_spec = {0: [{"singleton": 1}, {"singleton": 2}]}

        binner = OneHotBinning(bin_spec=bin_spec)
        X = np.array([[1, 10], [2, 20]])

        binner.fit(X)

        # Should use provided bin_spec and generate representatives
        assert binner._bin_spec[0] == bin_spec[0]
        assert binner._bin_reps[0] == [1.0, 2.0]  # Generated from singletons


class TestUtilityMethods:
    """Test utility methods."""

    def test_lookup_bin_widths(self):
        """Test bin width lookup (should be 0 for all singletons)."""
        binner = OneHotBinning()
        X = np.array([[1, 10], [2, 20]])

        binner.fit(X)
        transformed = binner.transform(X)
        widths = binner.lookup_bin_widths(transformed)

        # All singleton bins should have zero width
        assert np.all(widths == 0.0)

    def test_lookup_bin_ranges(self):
        """Test bin range lookup."""
        binner = OneHotBinning()
        X = np.array([[1, 10], [2, 20], [3, 30]])

        binner.fit(X)
        ranges = binner.lookup_bin_ranges()

        # Should return number of bins per column
        assert ranges[0] == 3  # Three unique values
        assert ranges[1] == 3  # Three unique values

    def test_repr(self):
        """Test string representation."""
        binner = OneHotBinning(max_unique_values=50, fit_jointly=True)
        repr_str = repr(binner)

        assert "OneHotBinning" in repr_str
        assert "max_unique_values=50" in repr_str
        assert "fit_jointly=True" in repr_str


class TestIntegration:
    """Integration test combining multiple features."""

    def test_complete_workflow(self):
        """Test complete workflow with various challenges."""
        binner = OneHotBinning(max_unique_values=10, fit_jointly=False)

        # Fit with mixed data
        X_fit = np.array([[1, 10], [2, 20], [1, 10]])
        binner.fit(X_fit)

        # Transform with challenging data
        X_transform = np.array(
            [
                [1, 10],  # Known values
                [2, 20],  # Known values
                [3, 30],  # Unknown values
                [np.nan, np.nan],  # Missing values
            ]
        )

        transformed = binner.transform(X_transform)
        reconstructed = binner.inverse_transform(transformed)

        # Check transformations
        assert transformed[0, 0] == 0 and transformed[0, 1] == 0  # Known -> bins
        assert transformed[1, 0] == 1 and transformed[1, 1] == 1  # Known -> bins
        assert transformed[2, 0] == MISSING_VALUE  # Unknown -> missing
        assert transformed[2, 1] == MISSING_VALUE  # Unknown -> missing
        assert transformed[3, 0] == MISSING_VALUE  # NaN -> missing
        assert transformed[3, 1] == MISSING_VALUE  # NaN -> missing

        # Check inverse transform
        assert reconstructed[0, 0] == 1.0 and reconstructed[0, 1] == 10.0
        assert reconstructed[1, 0] == 2.0 and reconstructed[1, 1] == 20.0
        assert np.isnan(reconstructed[2, 0]) and np.isnan(reconstructed[2, 1])
        assert np.isnan(reconstructed[3, 0]) and np.isnan(reconstructed[3, 1])

        # Check utility methods
        widths = binner.lookup_bin_widths(transformed)
        ranges = binner.lookup_bin_ranges()

        assert widths.shape == transformed.shape
        assert np.all(widths[~np.isnan(widths)] == 0.0)  # All singleton widths are 0
        assert ranges[0] == 2 and ranges[1] == 2  # Two unique values each
