"""
Comprehensive tests for FlexibleBinningBase class.
"""

import numpy as np
import pytest

from binlearn.base._flexible_binning_base import FlexibleBinningBase
from binlearn.utils import MISSING_VALUE, ConfigurationError


class MockFlexibleBinner(FlexibleBinningBase):
    """Mock concrete implementation of FlexibleBinningBase for testing."""

    def __init__(self, n_bins=3, **kwargs):
        self.n_bins = n_bins
        super().__init__(**kwargs)

    def _calculate_flexible_bins(self, x_col, col_id, guidance_data=None):
        """Simple implementation that creates n_bins unique value bins."""
        if len(x_col) == 0:
            return [0.0], [0.0]

        # Handle NaN values by filtering them out for bin calculation
        finite_values = x_col[np.isfinite(x_col)]
        if len(finite_values) == 0:
            return [0.0], [0.0]

        # Get unique values
        unique_values = np.unique(finite_values)

        # Limit to n_bins if we have too many unique values
        if len(unique_values) > self.n_bins:
            # Take evenly spaced values
            indices = np.linspace(0, len(unique_values) - 1, self.n_bins, dtype=int)
            unique_values = unique_values[indices]

        bin_values = unique_values.tolist()
        representatives = (
            bin_values.copy()
        )  # For flexible binning, representatives are the values themselves

        return bin_values, representatives


class TestFlexibleBinningBase:
    """Test suite for FlexibleBinningBase."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        binner = MockFlexibleBinner()
        assert binner.preserve_dataframe is False  # default from config
        assert binner.fit_jointly is False  # default from config
        assert binner.guidance_columns is None
        assert binner.bin_spec is None
        assert binner.bin_representatives is None
        assert binner.bin_spec_ == {}
        assert binner.bin_representatives_ == {}
        assert binner.n_bins == 3

    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        binner = MockFlexibleBinner(
            n_bins=5,
            preserve_dataframe=True,
            fit_jointly=False,  # Avoid incompatibility with guidance_columns
            guidance_columns=["guide"],
        )
        assert binner.preserve_dataframe is True
        assert binner.fit_jointly is False
        assert binner.guidance_columns == ["guide"]
        assert binner.n_bins == 5

    def test_init_with_bin_spec(self):
        """Test initialization with predefined bin specification."""
        bin_spec = {"col1": [1, 2, 3]}
        binner = MockFlexibleBinner(bin_spec=bin_spec)
        assert binner.bin_spec_ == bin_spec
        # Should have generated default representatives (same as bin_spec for flexible)
        assert binner.bin_representatives_["col1"] == [1, 2, 3]

    def test_init_with_bin_spec_and_representatives(self):
        """Test initialization with both bin spec and representatives."""
        bin_spec = {"col1": [1, 2, 3]}
        bin_representatives = {"col1": [1.1, 2.1, 3.1]}
        binner = MockFlexibleBinner(bin_spec=bin_spec, bin_representatives=bin_representatives)
        assert binner.bin_spec_ == bin_spec
        assert binner.bin_representatives_ == bin_representatives
        # Should have set sklearn attributes
        assert hasattr(binner, "_feature_names_in")
        assert hasattr(binner, "_n_features_in")

    def test_validate_params_invalid_bin_spec_format(self):
        """Test parameter validation with invalid bin_spec format."""
        with pytest.raises(ConfigurationError):
            MockFlexibleBinner(bin_spec="invalid")  # Should be a dict

    def test_validate_params_invalid_bin_representatives_format(self):
        """Test parameter validation with invalid bin representatives format."""
        bin_spec = {"col1": [1, 2, 3]}
        with pytest.raises(ConfigurationError):
            MockFlexibleBinner(bin_spec=bin_spec, bin_representatives="invalid")  # Should be a dict

    def test_set_sklearn_attributes_from_specs_basic(self):
        """Test sklearn attribute setting from bin specifications."""
        bin_spec = {"col1": [1, 2, 3], "col2": [4, 5, 6]}
        binner = MockFlexibleBinner(bin_spec=bin_spec)

        assert binner._feature_names_in == ["col1", "col2"]
        assert binner._n_features_in == 2

    def test_set_sklearn_attributes_from_specs_with_guidance(self):
        """Test sklearn attribute setting with guidance columns."""
        bin_spec = {"col1": [1, 2, 3]}
        binner = MockFlexibleBinner(bin_spec=bin_spec, guidance_columns=["guide1", "guide2"])

        # Should include both binning and guidance columns
        expected_features = ["col1", "guide1", "guide2"]
        assert binner._feature_names_in == expected_features
        assert binner._n_features_in == 3

    def test_set_sklearn_attributes_single_guidance_column(self):
        """Test sklearn attribute setting with single guidance column (not list)."""
        bin_spec = {"col1": [1, 2, 3]}
        binner = MockFlexibleBinner(
            bin_spec=bin_spec, guidance_columns="guide"  # Single string, not list
        )

        expected_features = ["col1", "guide"]
        assert binner._feature_names_in == expected_features
        assert binner._n_features_in == 2

    def test_set_sklearn_attributes_guidance_already_in_binning(self):
        """Test sklearn attributes when guidance column is already in binning columns."""
        bin_spec = {"col1": [1, 2, 3], "guide": [4, 5, 6]}
        binner = MockFlexibleBinner(
            bin_spec=bin_spec, guidance_columns=["guide"]  # Already in bin_spec
        )

        # Should not duplicate the guidance column
        expected_features = ["col1", "guide"]
        assert binner._feature_names_in == expected_features
        assert binner._n_features_in == 2

    def test_fit_per_column_independently(self):
        """Test fitting columns independently."""
        binner = MockFlexibleBinner(n_bins=2)
        X = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        columns = ["col1", "col2"]

        binner._fit_per_column_independently(X, columns)

        assert "col1" in binner.bin_spec_
        assert "col2" in binner.bin_spec_
        assert "col1" in binner.bin_representatives_
        assert "col2" in binner.bin_representatives_
        # Should have at most n_bins values
        assert len(binner.bin_spec_["col1"]) <= 2
        assert len(binner.bin_representatives_["col1"]) <= 2

    def test_fit_per_column_independently_with_guidance(self):
        """Test fitting with guidance data."""
        binner = MockFlexibleBinner(n_bins=2)
        X = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        columns = ["col1", "col2"]
        guidance_data = np.array([[10.0], [20.0], [30.0]])

        binner._fit_per_column_independently(X, columns, guidance_data)

        # Should still fit normally (guidance is passed to _calculate_flexible_bins)
        assert "col1" in binner.bin_spec_
        assert "col2" in binner.bin_spec_

    def test_fit_jointly_across_columns(self):
        """Test joint fitting (should be same as independent for flexible)."""
        binner = MockFlexibleBinner(n_bins=2)
        X = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        columns = ["col1", "col2"]

        binner._fit_jointly_across_columns(X, columns)

        assert "col1" in binner.bin_spec_
        assert "col2" in binner.bin_spec_
        assert "col1" in binner.bin_representatives_
        assert "col2" in binner.bin_representatives_

    def test_transform_columns_to_bins_empty_data(self):
        """Test transformation with empty data."""
        binner = MockFlexibleBinner()
        X = np.empty((0, 0))
        columns = []

        result = binner._transform_columns_to_bins(X, columns)
        assert result.shape == (0, 0)

    def test_transform_columns_to_bins_basic(self):
        """Test basic bin transformation."""
        binner = MockFlexibleBinner()
        # Set up fitted state
        binner.bin_spec_ = {"col1": [1.0, 2.0, 3.0]}

        X = np.array([[1.0], [2.0], [3.0], [1.5]])
        columns = ["col1"]

        result = binner._transform_columns_to_bins(X, columns)
        assert result.shape == (4, 1)
        # Values should map to their bin indices or MISSING_VALUE
        assert result[0, 0] == 0  # 1.0 -> bin 0
        assert result[1, 0] == 1  # 2.0 -> bin 1
        assert result[2, 0] == 2  # 3.0 -> bin 2
        # 1.5 doesn't match any bin exactly, should get MISSING_VALUE
        assert result[3, 0] == MISSING_VALUE

    def test_transform_columns_to_bins_with_missing_values(self):
        """Test transformation with NaN and inf values."""
        binner = MockFlexibleBinner()
        binner.bin_spec_ = {"col1": [1.0, 2.0, 3.0]}

        X = np.array([[1.0], [np.nan], [np.inf], [-np.inf]])
        columns = ["col1"]

        result = binner._transform_columns_to_bins(X, columns)
        assert result.shape == (4, 1)
        # Normal value should get proper bin
        assert result[0, 0] == 0  # 1.0 -> bin 0
        # Special values should get MISSING_VALUE
        assert result[1, 0] == MISSING_VALUE  # NaN
        assert result[2, 0] == MISSING_VALUE  # inf
        assert result[3, 0] == MISSING_VALUE  # -inf

    def test_inverse_transform_bins_to_values_empty(self):
        """Test inverse transformation with empty data."""
        binner = MockFlexibleBinner()
        X = np.empty((0, 0))
        columns = []

        result = binner._inverse_transform_bins_to_values(X, columns)
        assert result.shape == (0, 0)

    def test_inverse_transform_bins_to_values_basic(self):
        """Test basic inverse transformation."""
        binner = MockFlexibleBinner()
        binner.bin_representatives_ = {"col1": [0.5, 1.5, 2.5]}

        X = np.array([[0], [1], [2]])  # Bin indices
        columns = ["col1"]

        result = binner._inverse_transform_bins_to_values(X, columns)
        assert result.shape == (3, 1)
        assert result[0, 0] == 0.5  # Representative of bin 0
        assert result[1, 0] == 1.5  # Representative of bin 1
        assert result[2, 0] == 2.5  # Representative of bin 2

    def test_inverse_transform_bins_to_values_with_clipping(self):
        """Test inverse transformation with out-of-range indices."""
        binner = MockFlexibleBinner()
        binner.bin_representatives_ = {"col1": [0.5, 1.5]}

        # Out of range indices should be clipped
        X = np.array([[-1], [5]])  # Invalid indices
        columns = ["col1"]

        result = binner._inverse_transform_bins_to_values(X, columns)
        assert result.shape == (2, 1)
        assert result[0, 0] == 0.5  # Clipped to bin 0
        assert result[1, 0] == 1.5  # Clipped to bin 1 (last valid)

    def test_integration_fit_transform_workflow(self):
        """Test complete fit and transform workflow."""
        binner = MockFlexibleBinner(n_bins=3)

        # Fit data
        X_fit = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
        columns = ["col1", "col2"]
        binner._fit_per_column_independently(X_fit, columns)

        # Transform data
        X_transform = np.array([[1.0, 10.0], [3.0, 30.0]])
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
        binner = MockFlexibleBinner()

        # Set up with integer keys
        binner.bin_spec_ = {0: [1.0, 2.0], 1: [10.0, 20.0]}
        binner.bin_representatives_ = {0: [1.0, 2.0], 1: [10.0, 20.0]}

        X = np.array([[1.0, 10.0]])
        columns = ["feature_0", "feature_1"]  # Should map to 0, 1

        # Should work without errors using column key resolution
        result = binner._transform_columns_to_bins(X, columns)
        assert result.shape == (1, 2)

        inverse_result = binner._inverse_transform_bins_to_values(result, columns)
        assert inverse_result.shape == (1, 2)

    def test_edge_case_single_column_single_value(self):
        """Test edge case with single column and single value."""
        binner = MockFlexibleBinner(n_bins=2)
        X = np.array([[5.0]])  # Single value
        columns = ["col1"]

        binner._fit_per_column_independently(X, columns)

        # Should handle single value gracefully
        assert "col1" in binner.bin_spec_
        assert "col1" in binner.bin_representatives_

        # Transform the same value
        result = binner._transform_columns_to_bins(X, columns)
        assert result.shape == (1, 1)
        assert isinstance(result[0, 0], int | np.integer)

    def test_edge_case_empty_column_list(self):
        """Test edge case with empty column list."""
        binner = MockFlexibleBinner()
        X = np.array([[1.0], [2.0], [3.0]])  # Has data but no columns to process
        columns = []

        binner._fit_per_column_independently(X, columns)
        assert binner.bin_spec_ == {}
        assert binner.bin_representatives_ == {}

    def test_fitted_attributes_configuration(self):
        """Test that fitted attributes are properly configured."""
        binner = MockFlexibleBinner()

        expected_attributes = ["bin_spec_", "bin_representatives_"]
        assert binner._fitted_attributes == expected_attributes

        # Test that the attributes exist
        assert hasattr(binner, "bin_spec_")
        assert hasattr(binner, "bin_representatives_")

    # Branch coverage tests

    def test_validate_params_no_bin_spec(self):
        """Test _validate_params when bin_spec is None."""
        # This tests the negation of 'if self.bin_spec is not None:' branch
        binner = MockFlexibleBinner()  # No bin_spec provided
        # Should complete without error
        assert binner.bin_spec is None
        assert binner.bin_spec_ == {}

    def test_validate_params_empty_bin_spec_dict(self):
        """Test _validate_params when bin_spec_ is empty after assignment."""
        binner = MockFlexibleBinner(bin_spec={})  # Empty dict
        assert binner.bin_spec_ == {}
        assert binner.bin_representatives_ == {}

    def test_validate_params_bin_spec_without_representatives(self):
        """Test when we have bin_spec but no bin_representatives."""
        bin_spec = {"col1": [1, 2, 3]}
        binner = MockFlexibleBinner(bin_spec=bin_spec)  # No bin_representatives

        # Should generate default representatives from bin_spec
        assert binner.bin_spec_ == bin_spec
        assert binner.bin_representatives_["col1"] == [1, 2, 3]  # Same as bin_spec for flexible

    def test_set_sklearn_attributes_no_guidance_columns(self):
        """Test _set_sklearn_attributes_from_specs when guidance_columns is None."""
        bin_spec = {"col1": [1, 2, 3]}
        binner = MockFlexibleBinner(bin_spec=bin_spec)  # No guidance_columns

        # Should only include binning columns
        assert binner._feature_names_in == ["col1"]
        assert binner._n_features_in == 1

    def test_set_sklearn_attributes_guidance_is_list(self):
        """Test _set_sklearn_attributes_from_specs when guidance_columns is already a list."""
        bin_spec = {"col1": [1, 2, 3]}
        binner = MockFlexibleBinner(
            bin_spec=bin_spec, guidance_columns=["guide1", "guide2"]  # Already a list
        )

        expected_features = ["col1", "guide1", "guide2"]
        assert binner._feature_names_in == expected_features
        assert binner._n_features_in == 3

    def test_set_sklearn_attributes_bin_spec_none_branch(self):
        """Test _set_sklearn_attributes_from_specs when bin_spec_ is None."""
        binner = MockFlexibleBinner()

        # Remove any existing sklearn attributes that might have been set during init
        if hasattr(binner, "_feature_names_in"):
            delattr(binner, "_feature_names_in")
        if hasattr(binner, "_n_features_in"):
            delattr(binner, "_n_features_in")

        # Temporarily set bin_spec_ to None to test the branch
        binner.__dict__["bin_spec_"] = None  # Bypass type checking

        # Call the method - should exit early without setting attributes
        binner._set_sklearn_attributes_from_specs()

        # Should not have set any sklearn attributes due to early exit
        assert not hasattr(binner, "_feature_names_in")
        assert not hasattr(binner, "_n_features_in")

    def test_validate_numeric_data_called(self):
        """Test that _validate_numeric_data is called during fitting."""
        binner = MockFlexibleBinner()

        # Test with string data that should cause validation to fail
        X = np.array([["a", "b"], ["c", "d"]], dtype=object)
        columns = ["col1", "col2"]

        # This should raise an error during numeric validation
        with pytest.raises((TypeError, ValueError)):
            binner._fit_per_column_independently(X, columns)

    def test_bin_spec_with_non_list_values_branch(self):
        """Test the branch where bin_spec contains non-list values."""
        # This tests the negation of 'if isinstance(spec, list):' at line 88
        bin_spec = {"col1": "not_a_list"}  # Non-list spec value

        binner = MockFlexibleBinner(bin_spec=bin_spec)

        # Should have processed bin_spec but not created representatives for non-list values
        assert binner.bin_spec_ == {"col1": "not_a_list"}
        # The non-list spec should not have generated representatives
        assert (
            "col1" not in binner.bin_representatives_
            or binner.bin_representatives_["col1"] != "not_a_list"
        )
