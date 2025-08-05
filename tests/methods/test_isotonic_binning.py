"""
Comprehensive tests for IsotonicBinning functionality.
"""

import warnings

import numpy as np
import pytest

from binlearn.methods._isotonic_binning import IsotonicBinning
from binlearn.utils.errors import ConfigurationError, DataQualityWarning, FittingError


class TestIsotonicBinningInitialization:
    """Test IsotonicBinning initialization and parameter handling."""

    def test_default_initialization(self) -> None:
        """Test default parameter initialization."""
        binning = IsotonicBinning()
        assert binning.max_bins == 10
        assert binning.min_samples_per_bin == 5
        assert binning.increasing is True
        assert binning.y_min is None
        assert binning.y_max is None
        assert binning.min_change_threshold == 0.01
        assert binning.preserve_dataframe is False
        assert binning.guidance_columns is None

    def test_custom_initialization(self) -> None:
        """Test initialization with custom parameters."""
        binning = IsotonicBinning(
            max_bins=8,
            min_samples_per_bin=10,
            increasing=False,
            y_min=0.0,
            y_max=1.0,
            min_change_threshold=0.05,
            preserve_dataframe=True,
            guidance_columns=[2],
        )
        assert binning.max_bins == 8
        assert binning.min_samples_per_bin == 10
        assert binning.increasing is False
        assert binning.y_min == 0.0
        assert binning.y_max == 1.0
        assert binning.min_change_threshold == 0.05
        assert binning.preserve_dataframe is True
        assert binning.guidance_columns

    def test_invalid_max_bins(self) -> None:
        """Test initialization with invalid max_bins."""
        with pytest.raises(ConfigurationError, match="max_bins must be a positive integer"):
            IsotonicBinning(max_bins=0)

        with pytest.raises(ConfigurationError, match="max_bins must be a positive integer"):
            IsotonicBinning(max_bins=-1)

        with pytest.raises(ConfigurationError, match="max_bins must be a positive integer"):
            IsotonicBinning(max_bins=1.5)  # type: ignore[arg-type]

    def test_invalid_min_samples_per_bin(self) -> None:
        """Test initialization with invalid min_samples_per_bin."""
        with pytest.raises(
            ConfigurationError, match="min_samples_per_bin must be a positive integer"
        ):
            IsotonicBinning(min_samples_per_bin=0)

        with pytest.raises(
            ConfigurationError, match="min_samples_per_bin must be a positive integer"
        ):
            IsotonicBinning(min_samples_per_bin=-5)

    def test_invalid_increasing_parameter(self) -> None:
        """Test initialization with invalid increasing parameter."""
        with pytest.raises(ConfigurationError, match="increasing must be a boolean"):
            IsotonicBinning(increasing="yes")  # type: ignore[arg-type]

        with pytest.raises(ConfigurationError, match="increasing must be a boolean"):
            IsotonicBinning(increasing=1)  # type: ignore[arg-type]

    def test_invalid_y_bounds(self) -> None:
        """Test initialization with invalid y bounds."""
        with pytest.raises(ConfigurationError, match="y_min must be less than y_max"):
            IsotonicBinning(y_min=1.0, y_max=0.0)

        with pytest.raises(ConfigurationError, match="y_min must be less than y_max"):
            IsotonicBinning(y_min=1.0, y_max=1.0)

    def test_invalid_min_change_threshold(self) -> None:
        """Test initialization with invalid min_change_threshold."""
        with pytest.raises(
            ConfigurationError, match="min_change_threshold must be a positive number"
        ):
            IsotonicBinning(min_change_threshold=0)

        with pytest.raises(
            ConfigurationError, match="min_change_threshold must be a positive number"
        ):
            IsotonicBinning(min_change_threshold=-0.1)


class TestIsotonicBinningFitting:
    """Test IsotonicBinning fitting functionality."""

    def test_basic_monotonic_data(self) -> None:
        """Test fitting with basic monotonic data."""
        np.random.seed(42)
        X = np.linspace(0, 10, 100).reshape(-1, 1)
        y = X.ravel() + 0.1 * np.random.randn(100)  # Monotonic relationship with noise

        binning = IsotonicBinning(max_bins=5)
        binning.fit(X, guidance_data=y)

        assert hasattr(binning, "bin_edges_")
        assert len(binning.bin_edges_) == 1
        assert len(binning.bin_edges_[0]) >= 2  # At least 2 edges for 1+ bins
        assert len(binning.bin_representatives_[0]) >= 1

    def test_decreasing_monotonic_data(self) -> None:
        """Test fitting with decreasing monotonic data."""
        np.random.seed(42)
        X = np.linspace(0, 10, 100).reshape(-1, 1)
        y = -X.ravel() + 0.1 * np.random.randn(100)  # Decreasing relationship

        binning = IsotonicBinning(max_bins=5, increasing=False)
        binning.fit(X, guidance_data=y)

        assert hasattr(binning, "bin_edges_")
        assert 0 in binning._isotonic_models
        assert binning._isotonic_models[0] is not None

    def test_classification_data(self) -> None:
        """Test fitting with classification target data."""
        np.random.seed(42)
        X = np.linspace(0, 10, 100).reshape(-1, 1)
        y = (X.ravel() > 5).astype(int)  # Binary classification

        binning = IsotonicBinning(max_bins=3)
        binning.fit(X, guidance_data=y)

        assert hasattr(binning, "bin_edges_")
        assert len(binning.bin_edges_[0]) >= 2

    def test_categorical_target(self) -> None:
        """Test fitting with categorical target data."""
        np.random.seed(42)
        X = np.linspace(0, 10, 100).reshape(-1, 1)
        y = np.array(["low"] * 30 + ["medium"] * 40 + ["high"] * 30)

        binning = IsotonicBinning(max_bins=4)
        binning.fit(X, guidance_data=y)

        assert hasattr(binning, "bin_edges_")
        assert 0 in binning._isotonic_models

    def test_constant_feature_data(self) -> None:
        """Test fitting with constant feature data."""
        X = np.ones((50, 1)) * 5.0
        y = np.random.randn(50)

        binning = IsotonicBinning(max_bins=3)
        binning.fit(X, guidance_data=y)

        assert hasattr(binning, "bin_edges_")
        # Should create fallback bins for constant data
        assert len(binning.bin_edges_[0]) == 2

    def test_insufficient_data(self) -> None:
        """Test fitting with insufficient data."""
        X = np.array([[1], [2]])
        y = np.array([0, 1])

        binning = IsotonicBinning(min_samples_per_bin=5)
        # Should handle insufficient data gracefully and issue warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            binning.fit(X, guidance_data=y)
            assert len(w) > 0
            assert "Data in column 0 has only 2 valid samples" in str(w[0].message)
        assert hasattr(binning, "bin_edges_")

    def test_missing_guidance_data(self) -> None:
        """Test fitting without guidance data."""
        X = np.random.randn(50, 2)

        binning = IsotonicBinning()
        with pytest.raises(ValueError, match="requires guidance_data"):
            binning.fit(X)

    def test_data_with_nans(self) -> None:
        """Test fitting with NaN values in data."""
        np.random.seed(42)
        X = np.linspace(0, 10, 100).reshape(-1, 1)
        X[::10] = np.nan  # Add some NaN values
        y = np.linspace(0, 5, 100)
        y[::15] = np.nan  # Add some NaN values to target

        binning = IsotonicBinning(max_bins=4, min_samples_per_bin=5)
        binning.fit(X, guidance_data=y)

        assert hasattr(binning, "bin_edges_")
        # Should handle NaN values appropriately

    def test_multiple_columns(self) -> None:
        """Test fitting with multiple feature columns."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X[:, 0] + 0.5 * X[:, 1] + 0.1 * np.random.randn(100)

        binning = IsotonicBinning(max_bins=4)
        binning.fit(X, guidance_data=y)

        assert hasattr(binning, "bin_edges_")
        assert len(binning.bin_edges_) == 3  # 3 feature columns
        assert len(binning._isotonic_models) == 3


class TestIsotonicBinningTransform:
    """Test IsotonicBinning transformation functionality."""

    def test_basic_transform(self) -> None:
        """Test basic transformation."""
        np.random.seed(42)
        X = np.linspace(0, 10, 100).reshape(-1, 1)
        y = X.ravel() + 0.1 * np.random.randn(100)

        binning = IsotonicBinning(max_bins=5)
        binning.fit(X, guidance_data=y)
        X_binned = binning.transform(X)

        assert X_binned.shape == X.shape
        assert np.all(X_binned >= 0)  # Bin indices should be non-negative

    def test_fit_transform(self) -> None:
        """Test fit_transform method."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = X[:, 0] + X[:, 1] + 0.1 * np.random.randn(100)

        binning = IsotonicBinning(max_bins=4)
        X_binned = binning.fit_transform(X, guidance_data=y)

        assert X_binned.shape == X.shape
        assert hasattr(binning, "bin_edges_")

    def test_transform_with_new_data(self) -> None:
        """Test transformation with new data points."""
        np.random.seed(42)
        X_train = np.linspace(0, 10, 100).reshape(-1, 1)
        y_train = X_train.ravel() + 0.1 * np.random.randn(100)

        binning = IsotonicBinning(max_bins=5)
        binning.fit(X_train, guidance_data=y_train)

        # Transform new data
        X_new = np.array([[2.5], [7.5], [12.0], [-1.0]])  # Including out-of-bounds values
        X_binned = binning.transform(X_new)

        assert X_binned.shape == X_new.shape


class TestIsotonicBinningEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_unique_target_value(self) -> None:
        """Test with constant target values."""
        X = np.linspace(0, 10, 50).reshape(-1, 1)
        y = np.ones(50) * 5.0  # Constant target

        binning = IsotonicBinning(max_bins=5)
        binning.fit(X, guidance_data=y)

        assert hasattr(binning, "bin_edges_")
        # Should still create some bins even with constant target

    def test_very_small_change_threshold(self) -> None:
        """Test with very small change threshold."""
        np.random.seed(42)
        X = np.linspace(0, 10, 100).reshape(-1, 1)
        y = X.ravel() + 0.01 * np.random.randn(100)

        binning = IsotonicBinning(max_bins=10, min_change_threshold=0.001)
        binning.fit(X, guidance_data=y)

        assert hasattr(binning, "bin_edges_")
        # Should create more bins with smaller threshold

    def test_large_change_threshold(self) -> None:
        """Test with large change threshold."""
        np.random.seed(42)
        X = np.linspace(0, 10, 100).reshape(-1, 1)
        y = X.ravel() + 0.1 * np.random.randn(100)

        binning = IsotonicBinning(max_bins=10, min_change_threshold=0.5)
        binning.fit(X, guidance_data=y)

        assert hasattr(binning, "bin_edges_")
        # Should create fewer bins with larger threshold

    def test_with_bounds(self) -> None:
        """Test with y_min and y_max bounds."""
        np.random.seed(42)
        X = np.linspace(0, 10, 100).reshape(-1, 1)
        y = np.random.rand(100) * 10  # Random values 0-10

        binning = IsotonicBinning(max_bins=5, y_min=2.0, y_max=8.0)
        binning.fit(X, guidance_data=y)

        assert hasattr(binning, "bin_edges_")
        # Just verify that the binning worked with bounds
        assert binning.y_min == 2.0
        assert binning.y_max == 8.0

    def test_empty_arrays(self) -> None:
        """Test with empty arrays."""
        X = np.array([]).reshape(0, 1)
        y = np.array([])

        binning = IsotonicBinning()
        # Should handle empty data gracefully and issue warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            binning.fit(X, guidance_data=y)
            # Check for both warnings that might be issued
            warning_messages = [str(warning.message) for warning in w]
            assert any("only NaN values" in msg for msg in warning_messages)
            assert any("no valid data points" in msg for msg in warning_messages)
        assert hasattr(binning, "bin_edges_")


class TestIsotonicBinningIntegration:
    """Integration tests for IsotonicBinning."""

    def test_sklearn_compatibility(self) -> None:
        """Test scikit-learn compatibility."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = X[:, 0] + X[:, 1] + 0.1 * np.random.randn(100)

        # Create pipeline
        pipeline = Pipeline([("scaler", StandardScaler()), ("binner", IsotonicBinning(max_bins=4))])

        # This should work if IsotonicBinning is sklearn-compatible
        X_transformed = pipeline.fit_transform(X, y)
        assert X_transformed.shape == X.shape

    def test_with_pandas_like_input(self) -> None:
        """Test with pandas-like input (if available)."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X[:, 0] + 0.5 * X[:, 1] + 0.1 * np.random.randn(100)

        binning = IsotonicBinning(max_bins=4, preserve_dataframe=True)
        X_binned = binning.fit_transform(X, guidance_data=y)

        assert X_binned.shape == X.shape

    def test_reproducibility(self) -> None:
        """Test that results are reproducible."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = X[:, 0] + X[:, 1] + 0.1 * np.random.randn(100)

        binning1 = IsotonicBinning(max_bins=5)
        binning2 = IsotonicBinning(max_bins=5)

        X_binned1 = binning1.fit_transform(X, guidance_data=y)
        X_binned2 = binning2.fit_transform(X, guidance_data=y)

        # Results should be the same with same random seed and parameters
        np.testing.assert_array_equal(X_binned1, X_binned2)


class TestIsotonicBinningCoverage:
    """Tests to achieve complete coverage of missing lines."""

    def test_empty_unique_values_in_categorical(self) -> None:
        """Test with empty categorical guidance data."""
        X = np.array([[1], [2], [3], [4], [5], [6]])
        # Create guidance data that will result in empty unique values after filtering
        y = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], dtype=object)

        binning = IsotonicBinning(min_samples_per_bin=1)
        # This will trigger the categorical handling path but with empty unique values
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            binning.fit(X, guidance_data=y)  # Should handle gracefully

    def test_isotonic_regression_failure(self) -> None:
        """Test handling of isotonic regression fitting failure."""
        # This is hard to trigger naturally, so we'll patch it
        import unittest.mock

        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([1, 2, 3, 4, 5, 6])

        binning = IsotonicBinning(min_samples_per_bin=1)

        # Mock IsotonicRegression to raise an exception
        with unittest.mock.patch(
            "binlearn.methods._isotonic_binning.IsotonicRegression"
        ) as mock_iso:
            mock_iso.return_value.fit_transform.side_effect = RuntimeError("Mocked failure")

            with pytest.raises(ValueError, match="Isotonic regression failed"):
                binning.fit(X, guidance_data=y)

    def test_single_value_fitted_function(self) -> None:
        """Test when fitted function has only one value."""
        # Create data that will result in y_fitted with length 1
        X = np.array([[5], [5.1], [5.2], [5.3], [5.4], [5.5]])
        y = np.array([10, 10, 10, 10, 10, 10])

        binning = IsotonicBinning(max_bins=3, min_samples_per_bin=1)
        binning.fit(X, guidance_data=y)

        # Should handle single point gracefully
        assert hasattr(binning, "bin_edges_")
        assert len(binning.bin_edges_[0]) == 2

    def test_zero_range_fitted_function(self) -> None:
        """Test when fitted function has zero range (constant values)."""
        # Create data that results in constant fitted values
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([10, 10, 10, 10, 10])  # Constant target

        binning = IsotonicBinning(max_bins=5)
        binning.fit(X, guidance_data=y)

        # Should handle zero range gracefully
        assert hasattr(binning, "bin_edges_")

    def test_single_cut_point_bin_creation(self) -> None:
        """Test bin creation with only one cut point."""
        # Create minimal data that results in single cut point
        X = np.array([[1], [1.1]])  # Very close values
        y = np.array([5, 5])  # Same target

        binning = IsotonicBinning(max_bins=10, min_samples_per_bin=1)
        binning.fit(X, guidance_data=y)

        # Should create single bin
        assert len(binning.bin_edges_[0]) == 2
        assert len(binning.bin_representatives_[0]) == 1

    def test_equal_min_max_in_single_cut(self) -> None:
        """Test when x_min equals x_max in single cut point scenario."""
        # Create data where all feature values are identical
        X = np.array([[5], [5], [5]])
        y = np.array([1, 2, 3])

        binning = IsotonicBinning()

        # Suppress DataQualityWarning for insufficient data
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DataQualityWarning)
            binning.fit(X, guidance_data=y)

        # Should handle equal min/max by adding offset
        edges = binning.bin_edges_[0]
        assert edges[1] > edges[0]  # x_max should be x_min + 1.0

    def test_final_bin_representative_single_cut(self) -> None:
        """Test final bin representative calculation with single cut point."""
        # This tests the 'else' branch in line 380
        X = np.array([[1], [1.1], [1.2], [1.3], [1.4], [1.5]])
        y = np.array([5, 5, 5, 5, 5, 5])  # Constant target to force single cut

        binning = IsotonicBinning(min_samples_per_bin=1)
        binning.fit(X, guidance_data=y)

        # Should handle correctly
        assert len(binning.bin_representatives_[0]) >= 1

    def test_categorical_target_processing_in_prepare_values(self) -> None:
        """Test categorical target processing in _prepare_target_values."""
        # This tests lines 276-278 directly
        X = np.array([[1], [2], [3]])
        y = np.array(["cat", "dog", "bird"], dtype=object)

        binning = IsotonicBinning()

        # Suppress DataQualityWarning for insufficient data
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DataQualityWarning)
            binning.fit(X, guidance_data=y)

        # Should handle categorical targets properly
        assert hasattr(binning, "bin_edges_")

    def test_numeric_target_processing_in_prepare_values(self) -> None:
        """Test numeric target processing in _prepare_target_values."""
        # This tests the 'else' branch in _prepare_target_values
        X = np.array([[1], [2], [3]])
        y = np.array([1.5, 2.5, 3.5])  # Already numeric

        binning = IsotonicBinning()

        # Suppress DataQualityWarning for insufficient data
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DataQualityWarning)
            binning.fit(X, guidance_data=y)

        # Should handle numeric targets properly
        assert hasattr(binning, "bin_edges_")

    def test_sufficient_data_path(self) -> None:
        """Test the path when we have sufficient data (no fallback)."""
        # Create data with enough samples to avoid insufficient data handling
        np.random.seed(42)
        X = np.linspace(0, 10, 20).reshape(-1, 1)
        y = X.ravel() + 0.1 * np.random.randn(20)

        binning = IsotonicBinning(min_samples_per_bin=5)
        binning.fit(X, guidance_data=y)

        # Should have proper bins without fallback
        assert len(binning.bin_edges_[0]) > 2  # More than just fallback bins

    def test_actual_insufficient_data_error(self) -> None:
        """Test the actual insufficient data error path."""
        # Create a scenario where handle_insufficient_data returns None
        # This is tricky, so let's patch it
        import unittest.mock

        X = np.array([[1], [2]])
        y = np.array([1, 2])

        binning = IsotonicBinning(min_samples_per_bin=10)

        # Mock handle_insufficient_data to return None instead of fallback
        with unittest.mock.patch.object(binning, "handle_insufficient_data", return_value=None):
            with pytest.raises(FittingError, match="Insufficient valid data points"):
                binning.fit(X, guidance_data=y)

    def test_branch_coverage_direct_calls(self) -> None:
        """Test specific method calls to ensure branch coverage."""
        binning = IsotonicBinning()

        # Test _prepare_target_values with object dtype
        y_cat = np.array(["a", "b", "c"], dtype=object)
        result = binning._prepare_target_values(y_cat)
        assert result.dtype == float

        # Test _prepare_target_values with numeric dtype
        y_num = np.array([1.5, 2.5, 3.5])
        result = binning._prepare_target_values(y_num)
        assert result.dtype == float

        # Test _find_cut_points with length 1
        x_single = np.array([1.0])
        y_single = np.array([1.0])
        cuts = binning._find_cut_points(x_single, y_single)
        assert cuts == [0]

        # Test _find_cut_points with zero range
        x_multi = np.array([1.0, 2.0, 3.0])
        y_zero_range = np.array([5.0, 5.0, 5.0])
        cuts = binning._find_cut_points(x_multi, y_zero_range)
        assert cuts == [0]

    def test_direct_method_coverage(self) -> None:
        """Test direct method calls for remaining coverage."""
        binning = IsotonicBinning()

        # Test _create_bins_from_cuts with equal min/max scenario
        x_sorted = np.array([5.0])
        y_fitted = np.array([10.0])
        cut_indices = [0]  # Single cut point

        edges, reps = binning._create_bins_from_cuts(x_sorted, y_fitted, cut_indices, 0)
        assert len(edges) == 2
        assert len(reps) == 1
        assert edges[1] > edges[0]  # Should add offset for equal values

    def test_fallback_scenario_comprehensive(self) -> None:
        """Test comprehensive fallback scenarios."""
        # Create data that will definitely trigger insufficient data handling
        X = np.array([[1]])  # Single data point
        y = np.array([5])

        binning = IsotonicBinning(min_samples_per_bin=10)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            binning.fit(X, guidance_data=y)
            # Should get warning about insufficient data
            assert len(w) > 0

    def test_validation_edge_cases(self) -> None:
        """Test parameter validation edge cases."""
        # Test that _validate_params is called during initialization
        with pytest.raises(ConfigurationError):
            IsotonicBinning(max_bins=0)  # Invalid max_bins

    def test_empty_unique_values_error(self) -> None:
        """Test the specific error when no unique values found in categorical guidance."""
        # Create a scenario where np.unique returns empty array
        # This is very hard to achieve naturally, so we'll patch it
        import unittest.mock

        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array(["a", "b", "c", "a", "b", "c"], dtype=object)

        binning = IsotonicBinning(min_samples_per_bin=1)

        # Mock np.unique to return empty array for the specific call
        with unittest.mock.patch("numpy.unique") as mock_unique:
            # First call is for checking unique values, second call should return empty
            mock_unique.side_effect = [np.array([], dtype=object), np.array(["a", "b", "c"])]

            with pytest.raises(FittingError, match="No valid guidance data found"):
                binning.fit(X, guidance_data=y)

    def test_single_cut_point_else_branch(self) -> None:
        """Test the else branch in final bin representative calculation."""
        # Create a scenario that results in exactly one cut point
        # but forces the else branch (line 380)

        # This is complex to achieve naturally, so let's call the method directly
        binning = IsotonicBinning()

        # Create scenario with single cut point where len(cut_indices) == 1
        x_sorted = np.array([1.0, 2.0, 3.0])
        y_fitted = np.array([5.0, 5.0, 5.0])
        cut_indices = [0]  # Single cut point

        edges, reps = binning._create_bins_from_cuts(x_sorted, y_fitted, cut_indices, 0)

        # Should use the else branch and calculate mean of all x_sorted values
        assert len(reps) == 1
        assert reps[0] == float(np.mean(x_sorted))  # Should be 2.0

    def test_cut_index_not_greater_than_previous(self) -> None:
        """Test the branch where cut_idx <= prev_cut_idx (line 364->355)."""
        # This is also hard to achieve naturally, so we'll use direct method call

        binning = IsotonicBinning()

        # Create a scenario where cut indices are not properly ordered
        # This could happen if cut_indices has duplicate values
        x_sorted = np.array([1.0, 2.0, 3.0, 4.0])
        y_fitted = np.array([1.0, 2.0, 3.0, 4.0])
        cut_indices = [0, 1, 1]  # Duplicate cut index to trigger the condition

        edges, reps = binning._create_bins_from_cuts(x_sorted, y_fitted, cut_indices, 0)

        # Should handle the case gracefully
        assert len(edges) >= 2
        assert len(reps) >= 1

    def test_actual_single_cut_scenario(self) -> None:
        """Test a real scenario that results in single cut point (else branch line 380)."""
        # Create data that will naturally result in single cut point
        # Use minimal data with constant target that forces minimal cut points
        X = np.array([[1.0], [1.1]])  # Very close feature values
        y = np.array([5.0, 5.0])  # Identical targets

        binning = IsotonicBinning(
            max_bins=10,
            min_samples_per_bin=1,
            min_change_threshold=0.1,  # High threshold to prevent multiple cuts
        )

        binning.fit(X, guidance_data=y)

        # This should trigger the else branch in _create_bins_from_cuts
        # because the isotonic regression will result in minimal cut points
        assert hasattr(binning, "bin_edges_")
        assert len(binning.bin_representatives_[0]) >= 1

    def test_force_else_branch_with_mock(self) -> None:
        """Force the else branch using method mocking."""
        import unittest.mock

        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([1, 2, 3, 4, 5, 6])

        binning = IsotonicBinning(min_samples_per_bin=1)

        # Mock _find_cut_points to return single cut point
        def mock_find_cut_points(x_sorted, y_fitted) -> list[int]:
            return [0]  # Always return single cut point

        with unittest.mock.patch.object(
            binning, "_find_cut_points", side_effect=mock_find_cut_points
        ):
            binning.fit(X, guidance_data=y)

        # This should have triggered the else branch (line 380)
        assert hasattr(binning, "bin_edges_")
        assert len(binning.bin_representatives_[0]) == 1

    def test_zero_cut_indices_else_branch(self) -> None:
        """Test to cover the else branch at line 380 with zero cut indices."""
        X = np.array([[1, 2, 3, 4, 5]]).T
        y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        binner = IsotonicBinning(n_bins=2)
        binner.fit(X, y)

        # Call _create_bins_from_cuts directly with empty cut_indices (len=0)
        x_sorted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_fitted = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        edges, reps = binner._create_bins_from_cuts(x_sorted, y_fitted, [], 0)

        # With zero cut indices, for loop doesn't execute, only final part runs
        # Line 377: bin_edges.append(float(x_sorted[-1])) adds final edge
        # Line 378-382: if len(cut_indices) > 1 is False (0 > 1), so goes to else at line 380
        assert len(edges) == 1  # Only final edge from x_sorted[-1]
        assert len(reps) == 1  # One representative from else branch
        assert reps[0] == 3.0  # mean of [1, 2, 3, 4, 5] from line 380

    def test_one_cut_index_else_branch(self) -> None:
        """Test to cover the else branch at line 380 with exactly one cut index."""
        X = np.array([[1, 2, 3, 4, 5]]).T
        y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        binner = IsotonicBinning(n_bins=2)
        binner.fit(X, y)

        # Call _create_bins_from_cuts directly with one cut index (len=1)
        x_sorted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_fitted = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        cut_indices = [2]  # Only one cut
        edges, reps = binner._create_bins_from_cuts(x_sorted, y_fitted, cut_indices, 0)

        # With one cut index, len(cut_indices) <= 1 is True,
        # so should hit the else branch and use mean of all x_sorted
        assert len(reps) >= 1
        # The last representative should be the mean of all x_sorted
        # when len(cut_indices) <= 1
