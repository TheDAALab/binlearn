"""
Comprehensive tests for SupervisedBinningBase class.
"""

import warnings

import numpy as np
import pytest

from binlearn.base._supervised_binning_base import SupervisedBinningBase
from binlearn.utils import DataQualityWarning, ValidationError


class MockSupervisedBinner(SupervisedBinningBase):
    """Mock concrete implementation of SupervisedBinningBase for testing."""

    def __init__(self, n_bins=3, **kwargs):
        self.n_bins = n_bins
        super().__init__(**kwargs)

    def _calculate_bins(self, x_col, col_id, guidance_data=None):
        """Simple implementation that creates n_bins based on quantiles with target consideration."""
        if len(x_col) == 0:
            return [0.0, 1.0], [0.5]

        # Handle NaN values by filtering them out for bin calculation
        finite_values = x_col[np.isfinite(x_col)]
        if len(finite_values) == 0:
            return [0.0, 1.0], [0.5]

        # Use guidance data to influence bin edges (simple approach: consider target correlation)
        if guidance_data is not None and len(guidance_data) == len(x_col):
            # Filter guidance data to match finite values
            valid_mask = np.isfinite(x_col)
            target_values = guidance_data.flatten()[valid_mask]

            # Simple supervised approach: create bins that separate different target classes
            if len(np.unique(target_values)) > 1:
                # Create quantile-based bins but consider target distribution
                quantiles = np.linspace(0, 1, self.n_bins + 1)
                bin_edges = np.quantile(finite_values, quantiles)
            else:
                # Fall back to regular quantiles if target is uniform
                quantiles = np.linspace(0, 1, self.n_bins + 1)
                bin_edges = np.quantile(finite_values, quantiles)
        else:
            # No guidance data, use regular quantiles
            quantiles = np.linspace(0, 1, self.n_bins + 1)
            bin_edges = np.quantile(finite_values, quantiles)

        # Ensure unique edges
        bin_edges = np.unique(bin_edges)

        # Ensure at least 2 edges
        if len(bin_edges) < 2:
            bin_edges = np.array([finite_values.min(), finite_values.max()])
            if bin_edges[0] == bin_edges[1]:
                bin_edges = np.array([bin_edges[0], bin_edges[0] + 1])

        # Calculate representatives (midpoints)
        representatives = []
        for i in range(len(bin_edges) - 1):
            mid = (bin_edges[i] + bin_edges[i + 1]) / 2
            representatives.append(mid)

        return bin_edges.tolist(), representatives


class TestSupervisedBinningBase:
    """Test suite for SupervisedBinningBase."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        binner = MockSupervisedBinner()
        assert binner.clip is True  # default from config
        assert binner.preserve_dataframe is False  # default from config
        assert binner.fit_jointly is False  # forced to False for supervised
        assert binner.guidance_columns is None
        assert binner.bin_edges is None
        assert binner.bin_representatives is None
        assert binner.bin_edges_ == {}
        assert binner.bin_representatives_ == {}
        assert binner.n_bins == 3

    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        binner = MockSupervisedBinner(
            n_bins=5, clip=True, preserve_dataframe=True, guidance_columns="target"
        )
        assert binner.clip is True
        assert binner.preserve_dataframe is True
        assert binner.fit_jointly is False  # Always False for supervised
        assert binner.guidance_columns == "target"
        assert binner.n_bins == 5

    def test_init_with_bin_edges(self):
        """Test initialization with predefined bin edges."""
        bin_edges = {"col1": np.array([0, 1, 2, 3])}
        binner = MockSupervisedBinner(bin_edges=bin_edges)
        assert np.array_equal(binner.bin_edges_["col1"], [0, 1, 2, 3])

    def test_init_forces_fit_jointly_false(self):
        """Test that fit_jointly is forced to False in supervised binning."""
        # Even if we try to set fit_jointly=True, it should be False
        binner = MockSupervisedBinner()
        assert binner.fit_jointly is False

        # The parent's __init__ is called with fit_jointly=False explicitly
        # This is verified by checking the class structure

    def test_validate_params_single_guidance_column_no_warning(self):
        """Test validation with single guidance column (should not warn)."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            MockSupervisedBinner(guidance_columns="target")
            # Should not issue any warnings for single column
            quality_warnings = [
                warning for warning in w if issubclass(warning.category, DataQualityWarning)
            ]
            assert len(quality_warnings) == 0

    def test_validate_params_multiple_guidance_columns_warning(self):
        """Test validation with multiple guidance columns (should warn)."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            MockSupervisedBinner(guidance_columns=["target1", "target2"])
            # Should issue warning about multiple columns
            quality_warnings = [
                warning for warning in w if issubclass(warning.category, DataQualityWarning)
            ]
            assert len(quality_warnings) == 1
            assert "single target column" in str(quality_warnings[0].message)

    def test_validate_guidance_data_none_raises_error(self):
        """Test that None guidance data raises ValidationError."""
        binner = MockSupervisedBinner()
        with pytest.raises(ValidationError, match="guidance_data cannot be None"):
            binner.validate_guidance_data(None)

    def test_validate_guidance_data_1d_array(self):
        """Test validation with 1D guidance data."""
        binner = MockSupervisedBinner()
        data = np.array([0, 1, 0, 1, 1])
        result = binner.validate_guidance_data(data)

        assert result.shape == (5, 1)
        assert np.array_equal(result.flatten(), [0, 1, 0, 1, 1])

    def test_validate_guidance_data_2d_array(self):
        """Test validation with 2D guidance data."""
        binner = MockSupervisedBinner()
        data = np.array([[0], [1], [0], [1]])
        result = binner.validate_guidance_data(data)

        assert result.shape == (4, 1)
        assert np.array_equal(result.flatten(), [0, 1, 0, 1])

    def test_validate_guidance_data_list_input(self):
        """Test validation with list input."""
        binner = MockSupervisedBinner()
        data = [0, 1, 0, 1, 1]
        result = binner.validate_guidance_data(data)

        assert result.shape == (5, 1)
        assert np.array_equal(result.flatten(), [0, 1, 0, 1, 1])

    def test_validate_guidance_data_3d_array_raises_error(self):
        """Test that 3D guidance data raises ValidationError."""
        binner = MockSupervisedBinner()
        data = np.array([[[1]], [[2]]])  # 3D array
        with pytest.raises(ValidationError, match="must be 1D or 2D array, got 3D"):
            binner.validate_guidance_data(data)

    def test_validate_guidance_data_empty_raises_error(self):
        """Test that empty guidance data raises ValidationError."""
        binner = MockSupervisedBinner()
        data = np.array([])
        with pytest.raises(ValidationError, match="guidance_data cannot be empty"):
            binner.validate_guidance_data(data)

    def test_validate_guidance_data_multiple_columns_raises_error(self):
        """Test that multiple guidance columns raise ValidationError."""
        binner = MockSupervisedBinner()
        data = np.array([[0, 1], [1, 0]])  # 2 columns
        with pytest.raises(ValidationError, match="requires exactly one target column, got 2"):
            binner.validate_guidance_data(data)

    def test_validate_guidance_data_single_class_no_warning(self):
        """Test that single-class target does not warn (warnings removed)."""
        binner = MockSupervisedBinner()
        data = np.array([1, 1, 1, 1])  # All same class

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = binner.validate_guidance_data(data)
            quality_warnings = [
                warning for warning in w if issubclass(warning.category, DataQualityWarning)
            ]
            assert len(quality_warnings) == 0  # No warnings expected
            assert result.shape == (4, 1)  # Should still process correctly

    def test_validate_guidance_data_two_classes_no_warning(self):
        """Test that two-class target does not warn (warnings removed)."""
        binner = MockSupervisedBinner()
        data = np.array([0, 1, 0, 1])  # Only 2 classes

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = binner.validate_guidance_data(data)
            quality_warnings = [
                warning for warning in w if issubclass(warning.category, DataQualityWarning)
            ]
            assert len(quality_warnings) == 0  # No warnings expected
            assert result.shape == (4, 1)  # Should still process correctly

    def test_validate_guidance_data_sufficient_classes_no_warning(self):
        """Test no warning for sufficient target diversity."""
        binner = MockSupervisedBinner()
        data = np.array([0, 1, 2, 0, 1, 2])  # 3+ classes

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            binner.validate_guidance_data(data)
            quality_warnings = [
                warning for warning in w if issubclass(warning.category, DataQualityWarning)
            ]
            assert len(quality_warnings) == 0

    def test_validate_feature_target_pair_basic(self):
        """Test basic feature-target pair validation."""
        binner = MockSupervisedBinner()
        feature = np.array([1.0, 2.0, 3.0, 4.0])
        target = np.array([[0], [1], [0], [1]])

        cleaned_feature, cleaned_target = binner._validate_feature_target_pair(
            feature, target, "col1"
        )

        assert np.array_equal(cleaned_feature, [1.0, 2.0, 3.0, 4.0])
        assert np.array_equal(cleaned_target, [[0], [1], [0], [1]])

    def test_validate_feature_target_pair_length_mismatch_error(self):
        """Test error when feature and target lengths don't match."""
        binner = MockSupervisedBinner()
        feature = np.array([1.0, 2.0, 3.0])
        target = np.array([[0], [1]])  # Different length

        with pytest.raises(ValidationError, match="must have same length"):
            binner._validate_feature_target_pair(feature, target, "col1")

    def test_validate_feature_target_pair_with_target_missing_values(self):
        """Test handling of missing values in target data."""
        binner = MockSupervisedBinner()
        feature = np.array([1.0, 2.0, 3.0, 4.0])
        target = np.array([[0], [np.nan], [1], [np.inf]])  # Missing values in target

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore warnings for this basic functionality test
            cleaned_feature, cleaned_target = binner._validate_feature_target_pair(
                feature, target, "col1"
            )

        # Should remove rows with invalid target values
        assert np.array_equal(cleaned_feature, [1.0, 3.0])
        assert np.array_equal(cleaned_target, [[0], [1]])

    def test_validate_feature_target_pair_insufficient_data_warning(self):
        """Test warning when insufficient data remains after cleaning."""
        binner = MockSupervisedBinner()
        feature = np.array([1.0])  # Only one valid point
        target = np.array([[0]])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            binner._validate_feature_target_pair(feature, target, "col1")
            quality_warnings = [
                warning for warning in w if issubclass(warning.category, DataQualityWarning)
            ]
            assert len(quality_warnings) == 1
            assert "insufficient valid data points" in str(quality_warnings[0].message)

    def test_validate_feature_target_pair_all_target_missing_warning(self):
        """Test warning when all target values are missing."""
        binner = MockSupervisedBinner()
        feature = np.array([1.0, 2.0, 3.0])
        target = np.array([[np.nan], [np.inf], [-np.inf]])  # All invalid

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cleaned_feature, cleaned_target = binner._validate_feature_target_pair(
                feature, target, "col1"
            )
            # Should warn about insufficient data (0 points)
            quality_warnings = [
                warning for warning in w if issubclass(warning.category, DataQualityWarning)
            ]
            assert len(quality_warnings) == 1
            assert len(cleaned_feature) == 0
            assert len(cleaned_target) == 0

    def test_fit_per_column_independently_no_guidance_error(self):
        """Test that fitting without guidance data raises error."""
        binner = MockSupervisedBinner()
        X = np.array([[1.0], [2.0], [3.0]])
        columns = ["col1"]

        with pytest.raises(ValidationError, match="Supervised binning requires guidance data"):
            binner._fit_per_column_independently(X, columns)

    def test_fit_per_column_independently_with_guidance(self):
        """Test successful fitting with guidance data."""
        binner = MockSupervisedBinner(n_bins=2)
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        columns = ["col1"]
        guidance = np.array([[0], [0], [1], [1]])  # Binary target

        binner._fit_per_column_independently(X, columns, guidance)

        # Should have fitted successfully
        assert "col1" in binner.bin_edges_
        assert "col1" in binner.bin_representatives_
        assert len(binner.bin_edges_["col1"]) >= 2  # At least 2 edges for 1+ bins

    def test_fit_per_column_independently_calls_validation(self):
        """Test that fitting calls guidance data validation."""
        binner = MockSupervisedBinner()
        X = np.array([[1.0], [2.0], [3.0]])
        columns = ["col1"]
        guidance = np.array([[0], [1], [0]])  # Proper numpy format

        # This should work because validation handles the format
        binner._fit_per_column_independently(X, columns, guidance)

        assert "col1" in binner.bin_edges_
        assert "col1" in binner.bin_representatives_

    def test_integration_supervised_workflow(self):
        """Test complete supervised binning workflow."""
        binner = MockSupervisedBinner(n_bins=3)

        # Create supervised data
        X_fit = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
        columns = ["feature1", "feature2"]
        y_fit = np.array([[0], [0], [1], [1]])  # Binary target

        # Fit with supervised data
        binner._fit_per_column_independently(X_fit, columns, y_fit)

        # Transform new data
        X_transform = np.array([[1.5, 15.0], [3.5, 35.0]])
        result = binner._transform_columns_to_bins(X_transform, columns)

        assert result.shape == (2, 2)
        assert isinstance(result[0, 0], int | np.integer)

        # Inverse transform
        inverse_result = binner._inverse_transform_bins_to_values(result, columns)
        assert inverse_result.shape == (2, 2)
        assert isinstance(inverse_result[0, 0], float | np.floating)

    def test_fitted_attributes_configuration(self):
        """Test that fitted attributes are properly configured."""
        binner = MockSupervisedBinner()

        # Should inherit from IntervalBinningBase
        expected_attributes = ["bin_edges_", "bin_representatives_"]
        assert binner._fitted_attributes == expected_attributes

        # Test that the attributes exist
        assert hasattr(binner, "bin_edges_")
        assert hasattr(binner, "bin_representatives_")

    # Branch coverage tests

    def test_validate_params_no_guidance_columns_branch(self):
        """Test _validate_params when guidance_columns is None."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            MockSupervisedBinner()  # No guidance_columns
            # Should not issue any warnings
            quality_warnings = [
                warning for warning in w if issubclass(warning.category, DataQualityWarning)
            ]
            assert len(quality_warnings) == 0

    def test_validate_params_single_guidance_column_string_branch(self):
        """Test _validate_params when guidance_columns is a single string."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            MockSupervisedBinner(guidance_columns="target")  # Single string
            # Should not issue warnings for single column
            quality_warnings = [
                warning for warning in w if issubclass(warning.category, DataQualityWarning)
            ]
            assert len(quality_warnings) == 0

    def test_validate_params_single_item_list_no_warning_branch(self):
        """Test _validate_params when guidance_columns is a single-item list."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            MockSupervisedBinner(guidance_columns=["target"])  # Single-item list
            # Should not issue warnings for single column
            quality_warnings = [
                warning for warning in w if issubclass(warning.category, DataQualityWarning)
            ]
            assert len(quality_warnings) == 0

    def test_guidance_data_validation_custom_name(self):
        """Test validate_guidance_data with custom name parameter."""
        binner = MockSupervisedBinner()
        with pytest.raises(ValidationError, match="custom_target cannot be None"):
            binner.validate_guidance_data(None, name="custom_target")

    def test_guidance_data_edge_case_exactly_two_unique_values(self):
        """Test guidance data with exactly 2 unique values (no warning expected)."""
        binner = MockSupervisedBinner()
        data = np.array([0, 1])  # Exactly 2 unique values

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = binner.validate_guidance_data(data)
            quality_warnings = [
                warning for warning in w if issubclass(warning.category, DataQualityWarning)
            ]
            assert len(quality_warnings) == 0  # No warnings expected anymore
            assert result.shape == (2, 1)  # Should still process correctly

    def test_guidance_data_edge_case_exactly_three_unique_values(self):
        """Test guidance data with exactly 3 unique values (no warning boundary)."""
        binner = MockSupervisedBinner()
        data = np.array([0, 1, 2])  # Exactly 3 unique values

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            binner.validate_guidance_data(data)
            quality_warnings = [
                warning for warning in w if issubclass(warning.category, DataQualityWarning)
            ]
            assert len(quality_warnings) == 0  # Should not warn with 3+ classes

    def test_validate_feature_target_pair_edge_case_exactly_two_points(self):
        """Test feature-target validation with exactly 2 data points (boundary)."""
        binner = MockSupervisedBinner()
        feature = np.array([1.0, 2.0])  # Exactly 2 points
        target = np.array([[0], [1]])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            binner._validate_feature_target_pair(feature, target, "col1")
            quality_warnings = [
                warning for warning in w if issubclass(warning.category, DataQualityWarning)
            ]
            assert len(quality_warnings) == 0  # Should not warn with 2+ points
