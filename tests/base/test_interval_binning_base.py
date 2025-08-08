"""
Comprehensive tests for IntervalBinningBase"""

import numpy as np
import pytest

from binlearn.base._interval_binning_base import IntervalBinningBase
from binlearn.utils import ConfigurationError, FittingError


class MockIntervalBinner(IntervalBinningBase):
    """Mock concrete implementation of IntervalBinningBase for testing."""

    def __init__(self, n_bins=3, **kwargs):
        self.n_bins = n_bins
        super().__init__(**kwargs)

    def _calculate_bins(self, x_col, col_id, guidance_data=None):
        """Simple implementation that creates n_bins equal-width bins."""
        if len(x_col) == 0:
            return [0.0, 1.0], [0.5]

        # Handle NaN values by filtering them out for bin calculation
        finite_values = x_col[np.isfinite(x_col)]
        if len(finite_values) == 0:
            return [0.0, 1.0], [0.5]

        min_val = np.min(finite_values)
        max_val = np.max(finite_values)

        # Handle constant data
        if min_val == max_val:
            edges = [min_val - 0.5, min_val + 0.5]
            representatives = [min_val]
        else:
            edges = np.linspace(min_val, max_val, self.n_bins + 1).tolist()
            representatives = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]

        return edges, representatives


class TestIntervalBinningBase:
    """Test suite for IntervalBinningBase."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        binner = MockIntervalBinner()
        assert binner.clip is True  # default from config
        assert binner.preserve_dataframe is False  # default from config
        assert binner.fit_jointly is False  # default from config
        assert binner.guidance_columns is None
        assert binner.bin_edges is None
        assert binner.bin_representatives is None
        assert binner.bin_edges_ == {}
        assert binner.bin_representatives_ == {}
        assert binner.n_bins == 3

    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        binner = MockIntervalBinner(
            n_bins=5,
            clip=False,
            preserve_dataframe=True,
            fit_jointly=False,  # Changed to False to avoid incompatibility
            guidance_columns=["guide"],
        )
        assert binner.clip is False
        assert binner.preserve_dataframe is True
        assert binner.fit_jointly is False
        assert binner.guidance_columns == ["guide"]
        assert binner.n_bins == 5

    def test_init_with_bin_edges(self):
        """Test initialization with predefined bin edges."""
        bin_edges = {"col1": [0.0, 1.0, 2.0]}
        binner = MockIntervalBinner(bin_edges=bin_edges)
        assert binner.bin_edges_ == bin_edges
        assert "col1" in binner.bin_representatives_
        # Should have generated default representatives: len(edges) - 1
        assert len(binner.bin_representatives_["col1"]) == 2  # 3 edges - 1 = 2 representatives

    def test_init_with_bin_edges_and_representatives(self):
        """Test initialization with both bin edges and representatives."""
        bin_edges = {"col1": [0.0, 1.0, 2.0]}
        bin_representatives = {"col1": [0.5, 1.5]}
        binner = MockIntervalBinner(bin_edges=bin_edges, bin_representatives=bin_representatives)
        assert binner.bin_edges_ == bin_edges
        assert binner.bin_representatives_ == bin_representatives
        # Should have set sklearn attributes
        assert hasattr(binner, "_feature_names_in")
        assert hasattr(binner, "_n_features_in")

    def test_validate_params_invalid_clip(self):
        """Test parameter validation with invalid clip parameter."""
        with pytest.raises(TypeError, match="clip must be a boolean"):
            MockIntervalBinner(clip="invalid")

    def test_validate_params_invalid_bin_edges_format(self):
        """Test parameter validation with invalid bin edges format."""
        with pytest.raises(ConfigurationError):
            MockIntervalBinner(bin_edges="invalid")

    def test_set_sklearn_attributes_from_specs_basic(self):
        """Test sklearn attribute setting from bin specifications."""
        bin_edges = {"col1": [0.0, 1.0, 2.0], "col2": [0.0, 0.5, 1.0]}
        binner = MockIntervalBinner(bin_edges=bin_edges)

        assert binner._feature_names_in == ["col1", "col2"]
        assert binner._n_features_in == 2

    def test_set_sklearn_attributes_from_specs_with_guidance(self):
        """Test sklearn attribute setting with guidance columns."""
        bin_edges = {"col1": [0.0, 1.0, 2.0]}
        binner = MockIntervalBinner(bin_edges=bin_edges, guidance_columns=["guide1", "guide2"])

        # Should include both binning and guidance columns
        expected_features = ["col1", "guide1", "guide2"]
        assert binner._feature_names_in == expected_features
        assert binner._n_features_in == 3

    def test_set_sklearn_attributes_single_guidance_column(self):
        """Test sklearn attribute setting with single guidance column (not list)."""
        bin_edges = {"col1": [0.0, 1.0, 2.0]}
        binner = MockIntervalBinner(
            bin_edges=bin_edges, guidance_columns="guide"  # Single string, not list
        )

        expected_features = ["col1", "guide"]
        assert binner._feature_names_in == expected_features
        assert binner._n_features_in == 2

    def test_set_sklearn_attributes_guidance_already_in_binning(self):
        """Test sklearn attributes when guidance column is already in binning columns."""
        bin_edges = {"col1": [0.0, 1.0, 2.0], "guide": [0.0, 0.5, 1.0]}
        binner = MockIntervalBinner(
            bin_edges=bin_edges, guidance_columns=["guide"]  # Already in bin_edges
        )

        # Should not duplicate the guidance column
        expected_features = ["col1", "guide"]
        assert binner._feature_names_in == expected_features
        assert binner._n_features_in == 2

    def test_fit_per_column_independently(self):
        """Test fitting columns independently."""
        binner = MockIntervalBinner(n_bins=2)
        X = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        columns = ["col1", "col2"]

        binner._fit_per_column_independently(X, columns)

        assert "col1" in binner.bin_edges_
        assert "col2" in binner.bin_edges_
        assert "col1" in binner.bin_representatives_
        assert "col2" in binner.bin_representatives_
        assert len(binner.bin_edges_["col1"]) == 3  # n_bins + 1
        assert len(binner.bin_representatives_["col1"]) == 2  # n_bins

    def test_fit_per_column_independently_with_guidance(self):
        """Test fitting with guidance data."""
        binner = MockIntervalBinner(n_bins=2)
        X = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        columns = ["col1", "col2"]
        guidance_data = np.array([[10.0], [20.0], [30.0]])

        binner._fit_per_column_independently(X, columns, guidance_data)

        # Should still fit normally (guidance is passed to _calculate_bins)
        assert "col1" in binner.bin_edges_
        assert "col2" in binner.bin_edges_

    def test_fit_jointly_across_columns(self):
        """Test joint fitting (should be same as independent for intervals)."""
        binner = MockIntervalBinner(n_bins=2)
        X = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        columns = ["col1", "col2"]

        binner._fit_jointly_across_columns(X, columns)

        assert "col1" in binner.bin_edges_
        assert "col2" in binner.bin_edges_
        assert "col1" in binner.bin_representatives_
        assert "col2" in binner.bin_representatives_

    def test_transform_columns_to_bins_empty_data(self):
        """Test transformation with empty data."""
        binner = MockIntervalBinner()
        X = np.empty((0, 0))
        columns = []

        result = binner._transform_columns_to_bins(X, columns)
        assert result.shape == (0, 0)

    def test_transform_columns_to_bins_basic(self):
        """Test basic bin transformation."""
        binner = MockIntervalBinner(n_bins=2)
        # Set up fitted state
        binner.bin_edges_ = {"col1": [0.0, 1.0, 2.0]}

        X = np.array([[0.5], [1.5]])
        columns = ["col1"]

        result = binner._transform_columns_to_bins(X, columns)
        assert result.shape == (2, 1)
        assert result[0, 0] == 0  # First bin
        assert result[1, 0] == 1  # Second bin

    def test_transform_columns_to_bins_with_clipping_enabled(self):
        """Test transformation with clipping enabled."""
        binner = MockIntervalBinner(clip=True)
        binner.bin_edges_ = {"col1": [1.0, 2.0, 3.0]}

        # Values outside the range should be clipped
        X = np.array([[0.5], [3.5]])  # Below and above range
        columns = ["col1"]

        result = binner._transform_columns_to_bins(X, columns)
        assert result.shape == (2, 1)
        # Both should be valid bin indices after clipping
        assert 0 <= result[0, 0] <= 1
        assert 0 <= result[1, 0] <= 1

    def test_transform_columns_to_bins_with_clipping_disabled(self):
        """Test transformation with clipping disabled."""
        binner = MockIntervalBinner(clip=False)
        binner.bin_edges_ = {"col1": [1.0, 2.0, 3.0]}

        X = np.array([[0.5], [3.5]])  # Outside range
        columns = ["col1"]

        result = binner._transform_columns_to_bins(X, columns)
        assert result.shape == (2, 1)
        # Should still get valid indices due to clipping in the method
        assert 0 <= result[0, 0] <= 1
        assert 0 <= result[1, 0] <= 1

    def test_transform_columns_to_bins_with_special_values(self):
        """Test transformation with NaN and inf values."""
        binner = MockIntervalBinner()
        binner.bin_edges_ = {"col1": [1.0, 2.0, 3.0]}

        X = np.array([[1.5], [np.nan], [np.inf], [-np.inf]])
        columns = ["col1"]

        result = binner._transform_columns_to_bins(X, columns)
        assert result.shape == (4, 1)
        # Normal value should get proper bin
        assert result[0, 0] == 0
        # Special values should be assigned to last bin (len(edges) - 2 = 1)
        assert result[1, 0] == 1  # NaN
        assert result[2, 0] == 1  # inf
        assert result[3, 0] == 1  # -inf

    def test_inverse_transform_bins_to_values_empty(self):
        """Test inverse transformation with empty data."""
        binner = MockIntervalBinner()
        X = np.empty((0, 0))
        columns = []

        result = binner._inverse_transform_bins_to_values(X, columns)
        assert result.shape == (0, 0)

    def test_inverse_transform_bins_to_values_basic(self):
        """Test basic inverse transformation."""
        binner = MockIntervalBinner()
        binner.bin_representatives_ = {"col1": [0.5, 1.5]}

        X = np.array([[0], [1]])  # Bin indices
        columns = ["col1"]

        result = binner._inverse_transform_bins_to_values(X, columns)
        assert result.shape == (2, 1)
        assert result[0, 0] == 0.5  # Representative of bin 0
        assert result[1, 0] == 1.5  # Representative of bin 1

    def test_inverse_transform_bins_to_values_with_clipping(self):
        """Test inverse transformation with out-of-range indices."""
        binner = MockIntervalBinner()
        binner.bin_representatives_ = {"col1": [0.5, 1.5]}

        # Out of range indices should be clipped
        X = np.array([[-1], [5]])  # Invalid indices
        columns = ["col1"]

        result = binner._inverse_transform_bins_to_values(X, columns)
        assert result.shape == (2, 1)
        assert result[0, 0] == 0.5  # Clipped to bin 0
        assert result[1, 0] == 1.5  # Clipped to bin 1 (last valid)

    def test_validate_and_preprocess_column_valid_data(self):
        """Test column validation with valid data."""
        binner = MockIntervalBinner()
        x_col = np.array([1.0, 2.0, 3.0])

        result = binner._validate_and_preprocess_column(x_col, "test_col")
        np.testing.assert_array_equal(result, x_col)  # Should return unchanged

    def test_validate_and_preprocess_column_all_nan(self):
        """Test column validation with all NaN values."""
        binner = MockIntervalBinner()
        x_col = np.array([np.nan, np.nan, np.nan])

        with pytest.raises(FittingError, match="Column test_col contains only NaN values"):
            binner._validate_and_preprocess_column(x_col, "test_col")

    def test_validate_and_preprocess_column_mixed_with_nan(self):
        """Test column validation with mixed values including NaN."""
        binner = MockIntervalBinner()
        x_col = np.array([1.0, np.nan, 3.0])

        result = binner._validate_and_preprocess_column(x_col, "test_col")
        np.testing.assert_array_equal(result, x_col)  # Should return unchanged

    def test_validate_and_preprocess_column_with_inf(self):
        """Test column validation with infinite values."""
        binner = MockIntervalBinner()
        x_col = np.array([1.0, np.inf, -np.inf])

        result = binner._validate_and_preprocess_column(x_col, "test_col")
        np.testing.assert_array_equal(result, x_col)  # Should return unchanged

    def test_integration_fit_transform_workflow(self):
        """Test complete fit and transform workflow."""
        binner = MockIntervalBinner(n_bins=3)

        # Fit data
        X_fit = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
        columns = ["col1", "col2"]
        binner._fit_per_column_independently(X_fit, columns)

        # Transform data
        X_transform = np.array([[1.5, 15.0], [3.5, 35.0]])
        result = binner._transform_columns_to_bins(X_transform, columns)

        assert result.shape == (2, 2)
        assert isinstance(result[0, 0], int | np.integer)
        assert isinstance(result[1, 1], int | np.integer)

        # Inverse transform
        inverse_result = binner._inverse_transform_bins_to_values(result, columns)
        assert inverse_result.shape == (2, 2)
        assert isinstance(inverse_result[0, 0], float | np.floating)

    def test_column_key_resolution_workflow(self):
        """Test that column key resolution works in transform methods."""
        binner = MockIntervalBinner()

        # Set up with integer keys
        binner.bin_edges_ = {0: [1.0, 2.0, 3.0], 1: [10.0, 20.0, 30.0]}
        binner.bin_representatives_ = {0: [1.5, 2.5], 1: [15.0, 25.0]}

        X = np.array([[1.5, 15.0]])
        columns = ["feature_0", "feature_1"]  # Should map to 0, 1

        # Should work without errors using column key resolution
        result = binner._transform_columns_to_bins(X, columns)
        assert result.shape == (1, 2)

        inverse_result = binner._inverse_transform_bins_to_values(result, columns)
        assert inverse_result.shape == (1, 2)

    def test_edge_case_single_column_single_value(self):
        """Test edge case with single column and single value."""
        binner = MockIntervalBinner(n_bins=2)
        X = np.array([[5.0]])  # Single value
        columns = ["col1"]

        binner._fit_per_column_independently(X, columns)

        # Should handle constant data gracefully
        assert "col1" in binner.bin_edges_
        assert "col1" in binner.bin_representatives_

        # Transform the same value
        result = binner._transform_columns_to_bins(X, columns)
        assert result.shape == (1, 1)
        assert isinstance(result[0, 0], int | np.integer)

    def test_edge_case_empty_column_list(self):
        """Test edge case with empty column list."""
        binner = MockIntervalBinner()
        X = np.array([[1.0], [2.0], [3.0]])  # Has data but no columns to process
        columns = []

        binner._fit_per_column_independently(X, columns)
        assert binner.bin_edges_ == {}
        assert binner.bin_representatives_ == {}

    def test_fitted_attributes_configuration(self):
        """Test that fitted attributes are properly configured."""
        binner = MockIntervalBinner()

        expected_attributes = ["bin_edges_", "bin_representatives_"]
        assert binner._fitted_attributes == expected_attributes

        # Test that the attributes exist
        assert hasattr(binner, "bin_edges_")
        assert hasattr(binner, "bin_representatives_")

    def test_init_with_explicit_clip_true(self):
        """Test initialization with clip explicitly set to True (not None)."""
        # This tests the negation of 'if clip is None:' branch (line 57)
        binner = MockIntervalBinner(clip=True)
        assert binner.clip is True

    def test_validate_params_no_bin_edges(self):
        """Test _validate_params when bin_edges is None."""
        # This tests the negation of 'if self.bin_edges is not None:' branch (line 86)
        binner = MockIntervalBinner()  # No bin_edges provided
        # Should complete without error
        assert binner.bin_edges is None
        assert binner.bin_edges_ == {}

    def test_validate_params_empty_bin_edges_dict(self):
        """Test _validate_params when bin_edges_ is empty after assignment."""
        # This tests the negation of 'elif self.bin_edges_:' branch (line 96)
        binner = MockIntervalBinner(bin_edges={})  # Empty dict
        assert binner.bin_edges_ == {}
        assert binner.bin_representatives_ == {}
        # Should not call _set_sklearn_attributes_from_specs

    def test_validate_params_bin_edges_none_branch(self):
        """Test the branch where bin_edges is None and no processing happens."""
        # This tests the branch path where bin_edges processing is skipped
        binner = MockIntervalBinner()  # No bin_edges provided

        # The validation should complete without processing bin_edges
        assert binner.bin_edges is None
        assert binner.bin_edges_ == {}
        assert binner.bin_representatives_ == {}

    def test_transform_columns_to_bins_clipping_disabled_branch(self):
        """Test transformation with clipping explicitly disabled."""
        # This tests the negation of 'if self.clip:' branch (line 183)
        binner = MockIntervalBinner(clip=False)
        binner.bin_edges_ = {"col1": [1.0, 2.0, 3.0]}

        X = np.array([[1.5]])  # Normal value within range
        columns = ["col1"]

        result = binner._transform_columns_to_bins(X, columns)
        # Should work normally even without clipping
        assert result.shape == (1, 1)
        assert isinstance(result[0, 0], int | np.integer)

    def test_validate_and_preprocess_column_not_all_nan(self):
        """Test column validation when NOT all values are NaN."""
        # This tests the negation of 'if np.all(np.isnan(x_col)):' branch (line 238)
        binner = MockIntervalBinner()
        x_col = np.array([1.0, np.nan, 3.0])  # Mixed values, not all NaN

        result = binner._validate_and_preprocess_column(x_col, "test_col")
        # Should pass through without error and return unchanged
        np.testing.assert_array_equal(result, x_col)

    def test_set_sklearn_attributes_no_guidance_columns(self):
        """Test _set_sklearn_attributes_from_specs when guidance_columns is None."""
        # This tests the negation of 'if self.guidance_columns is not None:' branch (line 118)
        bin_edges = {"col1": [0.0, 1.0, 2.0]}
        binner = MockIntervalBinner(bin_edges=bin_edges)  # No guidance_columns

        # Should only include binning columns
        assert binner._feature_names_in == ["col1"]
        assert binner._n_features_in == 1

    def test_set_sklearn_attributes_bin_edges_none(self):
        """Test _set_sklearn_attributes_from_specs with empty bin_edges."""
        # Test the method behavior with empty bin_edges_
        binner = MockIntervalBinner()  # No bin_edges provided

        # With empty bin_edges_, the method should still set empty feature names
        binner._set_sklearn_attributes_from_specs()

        # Should set empty feature names list due to empty bin_edges_
        assert binner._feature_names_in == []
        assert binner._n_features_in == 0

    def test_guidance_column_already_in_binning_features(self):
        """Test when guidance column is already in all_features."""
        # This tests the negation of 'if col not in all_features:' branch (line 126)
        bin_edges = {"col1": [0.0, 1.0], "guide": [0.0, 1.0]}
        binner = MockIntervalBinner(
            bin_edges=bin_edges, guidance_columns=["guide"]  # This column is already in bin_edges
        )

        # Should not add duplicate - guide is already in binning columns
        expected_features = ["col1", "guide"]  # No duplicate
        assert binner._feature_names_in == expected_features
        assert binner._n_features_in == 2

    def test_set_sklearn_attributes_bin_edges_none_branch(self):
        """Test _set_sklearn_attributes_from_specs when bin_edges_ is actually None."""
        # This tests the line 112->exit branch when bin_edges_ is None
        binner = MockIntervalBinner()

        # Remove any existing sklearn attributes that might have been set during init
        if hasattr(binner, "_feature_names_in"):
            delattr(binner, "_feature_names_in")
        if hasattr(binner, "_n_features_in"):
            delattr(binner, "_n_features_in")

        # Temporarily override the type annotation to set bin_edges_ to None
        # This simulates a scenario where bin_edges_ might be None
        binner.__dict__["bin_edges_"] = None  # Bypass type checking

        # Call the method - should exit early without setting attributes
        binner._set_sklearn_attributes_from_specs()

        # Should not have set any sklearn attributes due to early exit
        assert not hasattr(binner, "_feature_names_in")
        assert not hasattr(binner, "_n_features_in")

    def test_transform_columns_column_count_validation(self):
        """Test column count validation in _transform_columns_to_bins."""
        # Create binner with specification for only 1 column
        bin_edges = {0: [0.0, 5.0, 10.0]}
        binner = MockIntervalBinner(bin_edges=bin_edges)

        # Try to transform data with 2 columns
        X_multi = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

        # Should raise ValueError for column count mismatch
        with pytest.raises(
            ValueError,
            match="Input data has 2 columns but bin specifications are provided for 1 columns",
        ):
            binner._transform_columns_to_bins(X_multi, [0, 1])

        # Should work fine with matching column count
        X_single = np.array([[1.0], [2.0], [3.0]])
        result = binner._transform_columns_to_bins(X_single, [0])
        assert result.shape == X_single.shape

    def test_transform_empty_data(self):
        """Test _transform_columns_to_bins with empty data."""
        bin_edges = {0: [0.0, 5.0, 10.0]}
        binner = MockIntervalBinner(bin_edges=bin_edges)

        # Empty data should return empty result
        X_empty = np.array([]).reshape(0, 1)
        result = binner._transform_columns_to_bins(X_empty, [0])
        assert result.shape == (0, 0)
