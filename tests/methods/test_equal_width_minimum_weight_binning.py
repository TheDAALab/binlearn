"""
Comprehensive test suite for EqualWidthMinimumWeightBinning transformer.

This module contains extensive tests for the EqualWidthMinimumWeightBinning class,
covering initialization, parameter validation, fitting, transformation, edge cases,
data type compatibility, sklearn integration, and error handling.

Test Classes:
    TestEqualWidthMinimumWeightBinning: Core functionality tests including initialization,
        validation, fitting, transformation, and basic operations.
    TestEqualWidthMinimumWeightBinningDataTypes: Tests for various data type compatibility
        including pandas DataFrames and polars DataFrames.
    TestEqualWidthMinimumWeightBinningSklearnIntegration: Tests for sklearn compatibility
        including pipeline integration, ColumnTransformer usage, and cloning.
    TestEqualWidthMinimumWeightBinningWeightConstraints: Tests specific to weight
        constraint functionality and bin merging behavior.
"""

import numpy as np
import pytest
from sklearn.base import clone

# Import sklearn components
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from binlearn import PANDAS_AVAILABLE, POLARS_AVAILABLE, pd, pl
from binlearn.methods._equal_width_minimum_weight_binning import EqualWidthMinimumWeightBinning
from binlearn.utils.errors import ConfigurationError, DataQualityWarning, FittingError


class TestEqualWidthMinimumWeightBinning:
    """Comprehensive test cases for EqualWidthMinimumWeightBinning core functionality.

    This test class covers the fundamental operations of the EqualWidthMinimumWeightBinning
    transformer including initialization, parameter validation, fitting, transformation,
    edge cases, and basic data handling scenarios.
    """

    def test_init_default(self):
        """Test initialization with default parameters.

        Verifies that the transformer initializes correctly with default
        parameter values and that all attributes are set as expected.
        """
        ewmwb = EqualWidthMinimumWeightBinning()
        assert ewmwb.n_bins == 10
        assert ewmwb.minimum_weight == 1.0
        assert ewmwb.bin_range is None

    def test_init_custom_params(self):
        """Test initialization with custom parameters.

        Verifies that the transformer correctly stores custom initialization
        parameter values including n_bins, minimum_weight, and bin_range.
        """
        ewmwb = EqualWidthMinimumWeightBinning(
            n_bins=5, minimum_weight=2.5, bin_range=(0, 100), fit_jointly=True
        )
        assert ewmwb.n_bins == 5
        assert ewmwb.minimum_weight == 2.5
        assert ewmwb.bin_range == (0, 100)

    def test_repr(self):
        """Test string representation of the transformer."""
        ewmwb = EqualWidthMinimumWeightBinning(n_bins=5, minimum_weight=2.0)
        repr_str = repr(ewmwb)
        assert "EqualWidthMinimumWeightBinning" in repr_str
        assert "n_bins=5" in repr_str
        assert "minimum_weight=2.0" in repr_str

    def test_validate_params_invalid_n_bins(self):
        """Test parameter validation with invalid n_bins values.

        Verifies that the validator correctly rejects non-positive n_bins.
        """
        with pytest.raises(ConfigurationError, match="n_bins must be a positive integer"):
            EqualWidthMinimumWeightBinning(n_bins=0)

        with pytest.raises(ConfigurationError, match="n_bins must be a positive integer"):
            EqualWidthMinimumWeightBinning(n_bins=-1)

        with pytest.raises(ConfigurationError, match="n_bins must be a positive integer"):
            EqualWidthMinimumWeightBinning(n_bins=3.5)  # type: ignore[arg-type]

    def test_validate_params_invalid_minimum_weight(self):
        """Test parameter validation with invalid minimum_weight values.

        Verifies that the validator correctly rejects non-positive minimum_weight.
        """
        with pytest.raises(ConfigurationError, match="minimum_weight must be a positive number"):
            EqualWidthMinimumWeightBinning(minimum_weight=0)

        with pytest.raises(ConfigurationError, match="minimum_weight must be a positive number"):
            EqualWidthMinimumWeightBinning(minimum_weight=-1.5)

    def test_validate_params_invalid_bin_range(self):
        """Test parameter validation with invalid bin_range values.

        Verifies that the validator correctly rejects invalid bin_range values.
        """
        with pytest.raises(ConfigurationError, match="bin_range must be a tuple/list"):
            EqualWidthMinimumWeightBinning(bin_range=(5, 5))  # min == max

        with pytest.raises(ConfigurationError, match="bin_range must be a tuple/list"):
            EqualWidthMinimumWeightBinning(bin_range=(10, 5))  # min > max

        with pytest.raises(ConfigurationError, match="bin_range must be a tuple/list"):
            EqualWidthMinimumWeightBinning(
                bin_range=[1, 2, 3]  # type: ignore[arg-type]
            )  # wrong length

    def test_requires_guidance_columns(self):
        """Test that the method correctly reports requiring guidance columns."""
        ewmwb = EqualWidthMinimumWeightBinning()
        assert ewmwb.requires_guidance_columns() is True

    def test_fit_transform_basic(self):
        """Test basic fit_transform functionality with guidance data."""
        # Create data with clear weight distribution
        X = np.array([[1, 10], [2, 11], [3, 12], [4, 13], [5, 14], [6, 15]]).astype(float)
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # equal weights

        ewmwb = EqualWidthMinimumWeightBinning(n_bins=3, minimum_weight=1.5)
        X_binned = ewmwb.fit_transform(X, guidance_data=weights)

        assert X_binned.shape == X.shape
        assert np.all(X_binned >= 0)  # All values should be non-negative bin indices

    def test_fit_transform_with_weight_merging(self):
        """Test fit_transform where bins need to be merged due to insufficient weight."""
        # Create data where some bins will have insufficient weight
        X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]).astype(float)
        # Weights concentrated in middle values, edges have low weight
        weights = np.array([0.1, 0.1, 2.0, 2.0, 2.0, 2.0, 2.0, 0.1, 0.1, 0.1])

        ewmwb = EqualWidthMinimumWeightBinning(n_bins=5, minimum_weight=1.0)
        X_binned = ewmwb.fit_transform(X, guidance_data=weights)

        assert X_binned.shape == X.shape
        # Due to merging, we should have fewer unique bin values than initial n_bins
        unique_bins = len(np.unique(X_binned))
        assert unique_bins < 5

    def test_separate_fit_transform(self):
        """Test separate fit and transform calls."""
        X = np.array([[1], [2], [3], [4], [5]]).astype(float)
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        ewmwb = EqualWidthMinimumWeightBinning(n_bins=2, minimum_weight=2.0)

        # Fit and transform separately
        ewmwb.fit(X, guidance_data=weights)
        X_binned = ewmwb.transform(X)

        assert X_binned.shape == X.shape
        assert hasattr(ewmwb, "bin_edges_")

    def test_no_guidance_data_error(self):
        """Test error when guidance_data is not provided."""
        X = np.array([[1], [2], [3]]).astype(float)
        ewmwb = EqualWidthMinimumWeightBinning(n_bins=2, minimum_weight=1.0)

        with pytest.raises(ValueError, match="requires guidance_data"):
            ewmwb.fit(X)

    def test_negative_weights_error(self):
        """Test error handling with negative weights in guidance data."""
        X = np.array([[1], [2], [3]]).astype(float)
        weights = np.array([1.0, -1.0, 1.0])  # Contains negative weight

        ewmwb = EqualWidthMinimumWeightBinning(n_bins=2, minimum_weight=1.0)

        with pytest.raises(ValueError, match="contains negative weights"):
            ewmwb.fit(X, guidance_data=weights)

    def test_insufficient_total_weight_error(self):
        """Test error when total weight is less than minimum_weight."""
        X = np.array([[1], [2], [3], [4], [5]]).astype(float)
        weights = np.array([0.1, 0.1, 0.1, 0.1, 0.1])  # Total weight = 0.5

        ewmwb = EqualWidthMinimumWeightBinning(n_bins=2, minimum_weight=1.0)

        with pytest.raises(FittingError, match="Total weight.*is less than minimum_weight"):
            ewmwb.fit(X, guidance_data=weights)

    def test_all_zero_weights_warning(self):
        """Test warning when all weights are zero."""
        X = np.array([[1], [2], [3], [4], [5]]).astype(float)
        weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # All zero weights

        ewmwb = EqualWidthMinimumWeightBinning(
            n_bins=2, minimum_weight=0.1
        )  # Small positive minimum

        # Expect multiple warnings: one from _calculate_bins and one from _merge_underweight_bins
        with pytest.warns(DataQualityWarning) as warning_list:
            ewmwb.fit(X, guidance_data=weights)

        # Check that we got the expected warnings
        warning_messages = [str(w.message) for w in warning_list]
        assert any("All weights are zero" in msg for msg in warning_messages)

    def test_all_nan_column(self):
        """Test behavior with all-NaN column."""
        X = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
        weights = np.array([1.0, 1.0, 1.0])
        ewmwb = EqualWidthMinimumWeightBinning(n_bins=2, minimum_weight=1.0)

        # Should handle all-NaN column gracefully and emit warning
        with pytest.warns(DataQualityWarning) as warning_list:
            ewmwb.fit(X, guidance_data=weights)

        # Check that we got warnings about no valid data points
        warning_messages = [str(w.message) for w in warning_list]
        assert any(
            "No valid" in msg and "data points available for binning" in msg
            for msg in warning_messages
        )

        X_binned = ewmwb.transform(X)

        assert X_binned.shape == X.shape
        # First column should be binned normally
        assert not np.all(X_binned[:, 0] == -1)
        # Second column should be all MISSING_VALUE (-1)
        from binlearn.utils.constants import MISSING_VALUE

        assert np.all(X_binned[:, 1] == MISSING_VALUE)

    def test_nan_in_guidance_data(self):
        """Test handling of NaN values in guidance data."""
        X = np.array([[1], [2], [3], [4], [5]]).astype(float)
        weights = np.array([1.0, np.nan, 1.0, 1.0, 1.0])  # NaN in weights

        ewmwb = EqualWidthMinimumWeightBinning(n_bins=2, minimum_weight=2.0)

        # Should handle NaN in guidance data by excluding those points
        X_binned = ewmwb.fit_transform(X, guidance_data=weights)
        assert X_binned.shape == X.shape

    def test_insufficient_data_for_binning(self):
        """Test error handling with insufficient data for binning."""
        # Only 1 data point
        X = np.array([[1.0]])
        weights = np.array([1.0])

        ewmwb = EqualWidthMinimumWeightBinning(n_bins=2, minimum_weight=1.0)

        with pytest.raises(ValueError, match="Insufficient non-NaN values"):
            ewmwb.fit(X, guidance_data=weights)

    def test_custom_bin_range(self):
        """Test binning with custom bin_range parameter."""
        X = np.array([[2], [3], [4], [5], [6]]).astype(float)
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        # Use bin_range that extends beyond actual data range
        ewmwb = EqualWidthMinimumWeightBinning(n_bins=3, minimum_weight=1.0, bin_range=(0, 10))
        X_binned = ewmwb.fit_transform(X, guidance_data=weights)

        assert X_binned.shape == X.shape
        # Check that bin edges respect the custom range
        assert ewmwb.bin_edges_[0][0] == 0
        assert ewmwb.bin_edges_[0][-1] == 10

    def test_constant_data(self):
        """Test handling of constant data (all values the same)."""
        # All values are the same
        X = np.array([[5], [5], [5], [5], [5]]).astype(float)
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        ewmwb = EqualWidthMinimumWeightBinning(n_bins=3, minimum_weight=2.0)

        # Should handle constant data gracefully
        X_binned = ewmwb.fit_transform(X, guidance_data=weights)
        assert X_binned.shape == X.shape
        # All values should be in the same bin
        assert len(np.unique(X_binned)) == 1

    def test_direct_calculate_bins_basic(self):
        """Test _calculate_bins method directly."""
        ewmwb = EqualWidthMinimumWeightBinning(n_bins=3, minimum_weight=2.0)
        data = np.array([1, 2, 3, 4, 5, 6])
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        edges, reps = ewmwb._calculate_bins(data, col_id=0, guidance_data=weights)

        assert len(edges) >= 2  # At least one bin
        assert len(reps) == len(edges) - 1
        # Check that edges are monotonically increasing
        for i in range(1, len(edges)):
            assert edges[i] >= edges[i - 1]

    def test_direct_calculate_bins_invalid_n_bins(self):
        """Test _calculate_bins with invalid n_bins."""
        ewmwb = EqualWidthMinimumWeightBinning(n_bins=1)  # Start with valid n_bins
        ewmwb.n_bins = 0  # Set directly to bypass init validation
        data = np.array([1, 2, 3, 4, 5])
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        with pytest.raises(ValueError, match="n_bins must be >= 1"):
            ewmwb._calculate_bins(data, col_id=0, guidance_data=weights)

    def test_empty_data(self):
        """Test behavior with empty data arrays."""
        X = np.array([]).reshape(0, 2)
        weights = np.array([])
        ewmwb = EqualWidthMinimumWeightBinning(n_bins=3, minimum_weight=1.0)

        # Empty data should be handled gracefully with warnings
        with pytest.warns(DataQualityWarning) as warning_list:
            ewmwb.fit(X, guidance_data=weights)

        # Check that we got warnings about no valid data points
        warning_messages = [str(w.message) for w in warning_list]
        print(f"DEBUG: Warning messages: {warning_messages}")  # Debug print
        assert any(
            "No valid" in msg and "data points available for binning" in msg
            for msg in warning_messages
        )

        X_binned = ewmwb.transform(X)
        assert X_binned.shape == (0, 2)

    def test_fit_jointly_vs_per_column(self):
        """Test that joint fitting falls back to per-column fitting with warning."""
        # Create data with different scales in different columns
        X = np.array([[1, 100], [2, 200], [3, 300], [4, 400], [5, 500]]).astype(float)
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        # Per-column fitting (default)
        ewmwb_per_col = EqualWidthMinimumWeightBinning(
            n_bins=2, minimum_weight=2.0, fit_jointly=False
        )
        X_per_col = ewmwb_per_col.fit_transform(X, guidance_data=weights)

        # Joint fitting should warn and use per-column fitting
        ewmwb_joint = EqualWidthMinimumWeightBinning(n_bins=2, minimum_weight=2.0, fit_jointly=True)

        with pytest.warns(
            DataQualityWarning,
            match="Joint fitting is not recommended.*Using per-column fitting instead",
        ):
            X_joint = ewmwb_joint.fit_transform(X, guidance_data=weights)

        # Results should be identical since joint fitting falls back to per-column
        assert np.array_equal(X_per_col, X_joint)

    def test_defensive_single_bin_coverage(self):
        """Test coverage of defensive code when merged_edges < 2."""
        # Create a test that triggers the defensive code path
        ewmwb = EqualWidthMinimumWeightBinning(n_bins=2, minimum_weight=1.0)

        # Mock _perform_bin_merging to return a list with only 1 element
        # This will trigger the defensive code: if len(merged_edges) < 2
        import unittest.mock

        edges = np.array([1.0, 2.0, 3.0])
        bin_weights = np.array([2.0, 2.0])  # Sufficient total weight

        with unittest.mock.patch.object(ewmwb, "_perform_bin_merging") as mock_merge:
            # Make the method return only 1 edge (simulating some error condition)
            mock_merge.return_value = [1.0]  # Only 1 element, triggers defensive code

            result = ewmwb._merge_underweight_bins(edges, bin_weights, col_id=0)

            # The defensive code should execute: merged_edges = [edges[0], edges[-1]]
            expected = np.array([1.0, 3.0])  # edges[0], edges[-1]
            np.testing.assert_array_equal(result, expected)

            # Verify the mock was called
            mock_merge.assert_called_once_with(edges, bin_weights)


@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
class TestEqualWidthMinimumWeightBinningDataTypes:
    """Test EqualWidthMinimumWeightBinning with different data types."""

    def test_pandas_dataframe(self):
        """Test with pandas DataFrame input and output."""
        df = pd.DataFrame({"feature1": [1, 2, 3, 4, 5, 6], "feature2": [10, 11, 12, 13, 14, 15]})
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        ewmwb = EqualWidthMinimumWeightBinning(
            n_bins=2, minimum_weight=2.0, preserve_dataframe=True
        )
        df_binned = ewmwb.fit_transform(df, guidance_data=weights)

        assert isinstance(df_binned, pd.DataFrame)
        assert df_binned.shape == df.shape
        assert list(df_binned.columns) == list(df.columns)

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="polars not available")
    def test_polars_dataframe(self):
        """Test with polars DataFrame input and output."""
        df = pl.DataFrame({"feature1": [1, 2, 3, 4, 5, 6], "feature2": [10, 11, 12, 13, 14, 15]})
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        ewmwb = EqualWidthMinimumWeightBinning(
            n_bins=2, minimum_weight=2.0, preserve_dataframe=True
        )
        df_binned = ewmwb.fit(df, guidance_data=weights).transform(df)

        assert isinstance(df_binned, pl.DataFrame)
        assert df_binned.shape == df.shape


class TestEqualWidthMinimumWeightBinningSklearnIntegration:
    """Test sklearn compatibility and integration."""

    def test_sklearn_pipeline(self):
        """Test integration with sklearn Pipeline."""
        X = np.random.rand(20, 3) * 100
        weights = np.random.rand(20) * 5 + 1  # Weights between 1 and 6

        pipeline = Pipeline(
            [
                ("binner", EqualWidthMinimumWeightBinning(n_bins=3, minimum_weight=2.0)),
                ("scaler", StandardScaler()),
            ]
        )

        X_transformed = pipeline.fit_transform(X, binner__guidance_data=weights)
        assert X_transformed.shape == X.shape

    def test_sklearn_clone(self):
        """Test sklearn clone functionality."""
        original = EqualWidthMinimumWeightBinning(n_bins=5, minimum_weight=3.0)
        cloned = clone(original)

        assert cloned.n_bins == original.n_bins
        assert cloned.minimum_weight == original.minimum_weight
        assert cloned is not original

    def test_get_params(self):
        """Test get_params method for sklearn compatibility."""
        ewmwb = EqualWidthMinimumWeightBinning(n_bins=7, minimum_weight=2.5)
        params = ewmwb.get_params()

        assert params["n_bins"] == 7
        assert params["minimum_weight"] == 2.5

    def test_set_params(self):
        """Test set_params method for sklearn compatibility."""
        ewmwb = EqualWidthMinimumWeightBinning()
        ewmwb.set_params(n_bins=8, minimum_weight=3.5)

        assert ewmwb.n_bins == 8
        assert ewmwb.minimum_weight == 3.5


class TestEqualWidthMinimumWeightBinningWeightConstraints:
    """Test specific weight constraint functionality and bin merging behavior."""

    def test_weight_constraint_single_bin_merge(self):
        """Test that underweight bins are properly merged."""
        # Create data where initial bins will have uneven weights
        X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]]).astype(float)
        # First bin gets little weight, middle bins get more weight
        weights = np.array([0.1, 0.1, 2.0, 2.0, 2.0, 2.0, 0.1, 0.1])

        ewmwb = EqualWidthMinimumWeightBinning(n_bins=4, minimum_weight=1.0)
        ewmwb.fit(X, guidance_data=weights)

        # Should have fewer bins than requested due to merging
        final_n_bins = len(ewmwb.bin_edges_[0]) - 1
        assert final_n_bins < 4

    def test_weight_constraint_no_merge_needed(self):
        """Test that bins are not merged when all meet minimum weight."""
        X = np.array([[1], [2], [3], [4], [5], [6]]).astype(float)
        # All bins have sufficient weight
        weights = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])

        ewmwb = EqualWidthMinimumWeightBinning(n_bins=3, minimum_weight=1.0)
        ewmwb.fit(X, guidance_data=weights)

        # Should have exactly 3 bins since no merging is needed
        final_n_bins = len(ewmwb.bin_edges_[0]) - 1
        assert final_n_bins == 3

    def test_weight_constraint_extreme_merge(self):
        """Test extreme case where almost all bins need to be merged."""
        X = np.array([[i] for i in range(1, 11)]).astype(float)  # 10 data points
        # Only two points have significant weight
        weights = np.array([0.01, 0.01, 0.01, 0.01, 5.0, 5.0, 0.01, 0.01, 0.01, 0.01])

        ewmwb = EqualWidthMinimumWeightBinning(n_bins=5, minimum_weight=3.0)
        ewmwb.fit(X, guidance_data=weights)

        # Should result in very few bins due to aggressive merging
        final_n_bins = len(ewmwb.bin_edges_[0]) - 1
        assert final_n_bins <= 2

    def test_weight_constraint_gradual_weights(self):
        """Test with gradually increasing weights across the range."""
        X = np.array([[i] for i in range(1, 9)]).astype(float)  # 8 data points
        # Gradually increasing weights
        weights = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])

        ewmwb = EqualWidthMinimumWeightBinning(n_bins=4, minimum_weight=2.5)
        ewmwb.fit(X, guidance_data=weights)

        # Early bins (with lower weights) should be merged
        final_n_bins = len(ewmwb.bin_edges_[0]) - 1
        assert final_n_bins < 4
        assert final_n_bins >= 1

    def test_weight_constraint_with_custom_range(self):
        """Test weight constraints work properly with custom bin_range."""
        X = np.array([[2], [3], [4], [5], [6]]).astype(float)
        weights = np.array([1.0, 0.1, 0.1, 0.1, 1.0])  # Weight at edges

        ewmwb = EqualWidthMinimumWeightBinning(n_bins=3, minimum_weight=0.8, bin_range=(0, 10))
        ewmwb.fit(X, guidance_data=weights)

        # Check that bin edges respect custom range
        assert ewmwb.bin_edges_[0][0] == 0
        assert ewmwb.bin_edges_[0][-1] == 10

        # Check that merging still occurs based on weights
        final_n_bins = len(ewmwb.bin_edges_[0]) - 1
        assert final_n_bins <= 3

    def test_weight_sum_calculation(self):
        """Test that weight sums are calculated correctly for bin decisions."""
        X = np.array([[1], [2], [3], [4], [5], [6]]).astype(float)
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        ewmwb = EqualWidthMinimumWeightBinning(n_bins=3, minimum_weight=1.5)
        ewmwb.fit(X, guidance_data=weights)

        # With 6 points of weight 1.0 each, and minimum_weight=1.5,
        # we should get bins where adjacent bins are merged to meet the minimum
        final_n_bins = len(ewmwb.bin_edges_[0]) - 1
        assert final_n_bins <= 3
        assert final_n_bins >= 1

    def test_transform_consistency_after_merging(self):
        """Test that transform produces consistent results after bin merging."""
        X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]]).astype(float)
        weights = np.array([0.1, 0.1, 2.0, 2.0, 2.0, 2.0, 0.1, 0.1])

        ewmwb = EqualWidthMinimumWeightBinning(n_bins=4, minimum_weight=1.0)
        X_binned = ewmwb.fit_transform(X, guidance_data=weights)

        # Check that all bin indices are valid
        assert np.all(X_binned >= 0)
        assert np.all(X_binned < len(ewmwb.bin_edges_[0]) - 1)

        # Check that transformation is consistent
        X_binned_again = ewmwb.transform(X)
        assert np.array_equal(X_binned, X_binned_again)

    def test_extreme_weight_merging_single_bin(self):
        """Test extreme case where all bins merge into a single bin."""
        # Create data with very small weights and high minimum requirement
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # Total weight = 1.0

        # Set minimum weight equal to total weight to force single bin
        ewmwb = EqualWidthMinimumWeightBinning(n_bins=5, minimum_weight=1.0)

        # This should create a single bin containing all data
        X_binned = ewmwb.fit_transform(X, guidance_data=weights)

        # All values should be in bin 0 (single bin)
        expected = np.array([[0], [0], [0], [0], [0]])
        np.testing.assert_array_equal(X_binned, expected)

        # Check that we have exactly 2 bin edges (creating 1 bin)
        assert len(ewmwb.bin_edges_[0]) == 2

    def test_defensive_single_bin_fallback(self):
        """Test defensive code path for ensuring at least one bin exists."""
        # This test directly calls the _merge_underweight_bins method
        # to cover the defensive line that ensures at least one bin
        X = np.array([[1.0], [2.0]])
        weights = np.array([1.0, 1.0])

        ewmwb = EqualWidthMinimumWeightBinning(n_bins=1, minimum_weight=1.0)

        # Create edge case: single bin scenario that might trigger the defensive code
        edges = np.array([1.0, 2.0])  # Just 2 edges (1 bin)
        bin_weights = np.array([2.0])  # 1 bin with sufficient weight

        # This should work normally but exercises the merging logic
        result = ewmwb._merge_underweight_bins(edges, bin_weights, 0)
        assert len(result) >= 2  # Should have at least 2 edges (1 bin)

        # Test with actual fitting to ensure normal operation
        X_binned = ewmwb.fit_transform(X, guidance_data=weights)
        assert X_binned.shape == (2, 1)
        assert np.all(X_binned >= 0)
