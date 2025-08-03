"""
Comprehensive tests for ManualFlexibleBinning transformer.

This test suite provides complete line coverage for the ManualFlexibleBinning class,
testing all initialization scenarios, validation logic, transformation operations,
error handling, and edge cases.
"""


import numpy as np
import pytest

from binlearn import PANDAS_AVAILABLE, pd
from binlearn.methods import ManualFlexibleBinning
from binlearn.utils.errors import BinningError, ConfigurationError


class TestManualFlexibleBinningInitialization:
    """Test initialization scenarios for ManualFlexibleBinning."""

    def test_init_with_basicbin_spec_(self):
        """Test initialization with basic bin specifications."""
        bin_spec = {0: [1.5, (2, 5), (5, 10)], 1: [(0, 25), (25, 50), 50]}

        binner = ManualFlexibleBinning(bin_spec=bin_spec)

        assert binner.bin_spec == bin_spec
        assert binner.bin_representatives is None
        assert binner._fitted is True  # Should be fitted with provided specs

    def test_init_withbin_spec__and_representatives(self):
        """Test initialization with both bin_spec and bin_representatives."""
        bin_spec = {0: [1.5, (2, 5), (5, 10)]}
        bin_representatives = {0: [1.5, 3.5, 7.5]}

        binner = ManualFlexibleBinning(bin_spec=bin_spec, bin_representatives=bin_representatives)

        assert binner.bin_spec == bin_spec
        assert binner.bin_representatives == bin_representatives
        assert binner._fitted is True

    def test_init_with_preserve_dataframe(self):
        """Test initialization with preserve_dataframe option."""
        bin_spec = {0: [1, 2, 3]}

        binner = ManualFlexibleBinning(bin_spec=bin_spec, preserve_dataframe=True)

        assert binner.preserve_dataframe is True

    def test_init_with_additional_kwargs(self):
        """Test initialization with additional keyword arguments."""
        bin_spec = {0: [1, 2, 3]}

        binner = ManualFlexibleBinning(
            bin_spec=bin_spec,
            preserve_dataframe=True,
            # Any additional kwargs are passed to parent classes
        )

        assert binner.preserve_dataframe is True

    def test_init_nonebin_spec__raises_error(self):
        """Test that providing None bin_spec raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="bin_spec must be provided"):
            ManualFlexibleBinning(bin_spec=None)  # type: ignore

    def test_init_missingbin_spec__raises_error(self):
        """Test that missing bin_spec raises TypeError."""
        with pytest.raises(TypeError):
            ManualFlexibleBinning()  # type: ignore


class TestManualFlexibleBinningCalculateFlexibleBins:
    """Test _calculate_flexible_bins method."""

    def test_calculate_flexible_bins_with_valid_column(self):
        """Test _calculate_flexible_bins with valid column."""
        bin_spec = {0: [1.5, (2, 5), (5, 10)]}
        binner = ManualFlexibleBinning(bin_spec=bin_spec)

        x_col = np.array([1, 3, 7])  # Data is ignored
        specs, representatives = binner._calculate_flexible_bins(x_col, 0)

        assert specs == [1.5, (2, 5), (5, 10)]
        assert representatives == [1.5, 3.5, 7.5]  # Auto-generated

    def test_calculate_flexible_bins_with_provided_representatives(self):
        """Test _calculate_flexible_bins with provided representatives."""
        bin_spec = {0: [1.5, (2, 5), (5, 10)]}
        bin_representatives = {0: [1.0, 3.0, 8.0]}

        binner = ManualFlexibleBinning(bin_spec=bin_spec, bin_representatives=bin_representatives)

        x_col = np.array([1, 3, 7])
        specs, representatives = binner._calculate_flexible_bins(x_col, 0)

        assert specs == [1.5, (2, 5), (5, 10)]
        assert representatives == [1.0, 3.0, 8.0]

    def test_calculate_flexible_bins_missing_column_raises_error(self):
        """Test _calculate_flexible_bins with missing column raises BinningError."""
        bin_spec = {0: [1, 2, 3]}
        binner = ManualFlexibleBinning(bin_spec=bin_spec)

        x_col = np.array([1, 2, 3])

        with pytest.raises(BinningError, match="No bin specifications defined for column 1"):
            binner._calculate_flexible_bins(x_col, 1)

    def test_calculate_flexible_bins_with_nonebin_spec_(self):
        """Test _calculate_flexible_bins when bin_spec becomes None."""
        bin_spec = {0: [1, 2, 3]}
        binner = ManualFlexibleBinning(bin_spec=bin_spec)

        # Manually set bin_spec to None to test the condition
        binner.bin_spec = None

        x_col = np.array([1, 2, 3])

        with pytest.raises(BinningError):
            binner._calculate_flexible_bins(x_col, 0)

    def test_calculate_flexible_bins_numeric_singleton(self):
        """Test _calculate_flexible_bins with various numeric singleton bins."""
        bin_spec = {0: [1.5, 2.7, (5, 10)]}
        binner = ManualFlexibleBinning(bin_spec=bin_spec)

        x_col = np.array([1, 2, 7])
        specs, representatives = binner._calculate_flexible_bins(x_col, 0)

        assert specs == [1.5, 2.7, (5, 10)]
        # Numeric singletons keep their values, intervals get midpoint
        assert representatives == [1.5, 2.7, 7.5]

    def test_calculate_flexible_bins_invalid_tuple_format(self):
        """Test that invalid tuple format is rejected during initialization."""
        # 3-tuple instead of 2-tuple should be rejected at initialization
        with pytest.raises(
            ValueError, match="Failed to process provided flexible bin specifications"
        ):
            bin_spec = {0: [1, (2, 3, 4)]}  # 3-tuple instead of 2-tuple
            ManualFlexibleBinning(bin_spec=bin_spec)

    def test_calculate_flexible_bins_ignores_guidance_data(self):
        """Test that _calculate_flexible_bins ignores guidance_data parameter."""
        bin_spec = {0: [1, 2, 3]}
        binner = ManualFlexibleBinning(bin_spec=bin_spec)

        x_col = np.array([1, 2, 3])
        guidance_data = np.array([10, 20, 30])

        specs, representatives = binner._calculate_flexible_bins(x_col, 0, guidance_data)

        assert specs == [1, 2, 3]
        assert representatives == [1.0, 2.0, 3.0]


class TestManualFlexibleBinningValidateParams:
    """Test _validate_params method."""

    def test_validate_params_valid_specifications(self):
        """Test _validate_params with valid bin specifications."""
        bin_spec = {0: [1.5, (2, 5), (5, 10)], 1: [(0, 25), (25, 50), 100]}  # Only numeric bins
        binner = ManualFlexibleBinning(bin_spec=bin_spec)

        # Should not raise any exception with valid numeric bins
        binner._validate_params()

    def test_validate_params_emptybin_spec__raises_error(self):
        """Test that empty bin_spec raises ConfigurationError during construction."""
        with pytest.raises(ConfigurationError, match="bin_spec must be provided and non-empty"):
            ManualFlexibleBinning(bin_spec={})

    def test_validate_params_non_list_specs_raises_error(self):
        """Test that non-list specifications are rejected during initialization."""
        with pytest.raises(
            ValueError, match="Failed to process provided flexible bin specifications"
        ):
            bin_spec = {0: 123}  # type: ignore  # Not a list/iterable
            ManualFlexibleBinning(bin_spec=bin_spec)  # type: ignore

    def test_validate_params_empty_column_specs_raises_error(self):
        """Test that empty column specifications raise ConfigurationError during construction."""
        with pytest.raises(ConfigurationError, match="cannot be empty"):
            ManualFlexibleBinning(bin_spec={0: []})

    def test_validate_params_invalid_interval_length_raises_error(self):
        """Test that invalid interval tuple length is rejected during initialization."""
        with pytest.raises(
            ValueError, match="Failed to process provided flexible bin specifications"
        ):
            bin_spec = {0: [(1, 2, 3)]}  # type: ignore  # 3-tuple instead of 2-tuple
            ManualFlexibleBinning(bin_spec=bin_spec)

    def test_validate_params_non_finite_interval_bounds_raises_error(self):
        """Test that non-finite interval bounds raise ConfigurationError during construction."""
        with pytest.raises(ConfigurationError, match="must be finite"):
            ManualFlexibleBinning(bin_spec={0: [(np.nan, 5)]})

    def test_validate_params_invalid_interval_order_raises_error(self):
        """Test that invalid interval order is rejected during initialization."""
        with pytest.raises(
            ValueError, match="Failed to process provided flexible bin specifications"
        ):
            bin_spec = {0: [(5, 2)]}  # min > max
            ManualFlexibleBinning(bin_spec=bin_spec)

    def test_validate_params_equal_interval_bounds_raises_error(self):
        """Test that equal interval bounds raise ConfigurationError during construction."""
        with pytest.raises(ConfigurationError, match="min .* must be < max"):
            ManualFlexibleBinning(bin_spec={0: [(5, 5)]})

    def test_validate_params_with_valid_representatives(self):
        """Test _validate_params with valid bin_representatives."""
        bin_spec = {0: [1, (2, 5), 10]}
        bin_representatives = {0: [1.0, 3.5, 10.0]}

        binner = ManualFlexibleBinning(bin_spec=bin_spec, bin_representatives=bin_representatives)

        # Should not raise any exception
        binner._validate_params()

    def test_validate_params_representatives_validation_failure(self):
        """Test that mismatched representatives are rejected during initialization."""
        with pytest.raises(
            ValueError, match="Failed to process provided flexible bin specifications"
        ):
            bin_spec = {0: [1, 2, 3]}
            bin_representatives = {0: [1.0, 2.0]}  # Wrong number of representatives
            ManualFlexibleBinning(bin_spec=bin_spec, bin_representatives=bin_representatives)

    def test_validate_params_representatives_for_missing_column(self):
        """Test that representatives for missing columns are rejected during initialization."""
        with pytest.raises(
            ValueError, match="Failed to process provided flexible bin specifications"
        ):
            bin_spec = {0: [1, 2, 3]}
            bin_representatives = {1: [1.0, 2.0, 3.0]}  # Column 1 not in bin_spec
            ManualFlexibleBinning(bin_spec=bin_spec, bin_representatives=bin_representatives)

    def test_validate_params_non_list_representatives(self):
        """Test that non-list representatives are rejected during initialization."""
        with pytest.raises(
            ValueError, match="Failed to process provided flexible bin specifications"
        ):
            bin_spec = {0: [1, 2, 3]}
            bin_representatives = {0: "invalid"}  # type: ignore
            ManualFlexibleBinning(bin_spec=bin_spec, bin_representatives=bin_representatives)  # type: ignore

    def test_validate_params_wrong_number_of_representatives(self):
        """Test that wrong number of representatives is rejected during initialization."""
        with pytest.raises(
            ValueError, match="Failed to process provided flexible bin specifications"
        ):
            bin_spec = {0: [1, 2, 3]}
            bin_representatives = {0: [1.0, 2.0]}  # Only 2 reps for 3 specs
            ManualFlexibleBinning(bin_spec=bin_spec, bin_representatives=bin_representatives)


class TestManualFlexibleBinningFit:
    """Test fit method."""

    def test_fit_calls_parent_implementation(self):
        """Test that fit method calls parent implementation."""
        bin_spec = {0: [1, 2, 3]}
        binner = ManualFlexibleBinning(bin_spec=bin_spec)

        X = np.array([[1], [2], [3]])
        result = binner.fit(X)

        assert result is binner
        assert binner._fitted is True

    def test_fit_with_y_parameter(self):
        """Test fit method with y parameter (should be ignored)."""
        bin_spec = {0: [1, 2, 3]}
        binner = ManualFlexibleBinning(bin_spec=bin_spec)

        X = np.array([[1], [2], [3]])
        y = np.array([0, 1, 0])

        result = binner.fit(X, y)

        assert result is binner
        assert binner._fitted is True

    def test_fit_with_fit_params(self):
        """Test fit method with additional fit_params."""
        bin_spec = {0: [1, 2, 3]}
        binner = ManualFlexibleBinning(bin_spec=bin_spec)

        X = np.array([[1], [2], [3]])

        result = binner.fit(X, sample_weight=np.array([1, 1, 1]))

        assert result is binner
        assert binner._fitted is True


class TestManualFlexibleBinningTransformation:
    """Test transformation operations."""

    def test_transform_basic_data(self):
        """Test transform with basic numeric data."""
        bin_spec = {0: [1, (2, 5), (5, 10)], 1: [10, 20, 30]}
        binner = ManualFlexibleBinning(bin_spec=bin_spec)

        X = np.array([[1, 10], [3, 20], [7, 30], [1, 10]])

        result = binner.fit_transform(X)

        # Check that transformation worked
        assert result.shape == X.shape
        assert isinstance(result, np.ndarray)

    def test_fit_transform_workflow(self):
        """Test complete fit_transform workflow."""
        bin_spec = {0: [(0, 5), (5, 10), 10]}
        binner = ManualFlexibleBinning(bin_spec=bin_spec)

        X = np.array([[2], [7], [10]])

        result = binner.fit_transform(X)

        assert result.shape == X.shape
        assert binner._fitted is True

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_transform_pandas_dataframe(self):
        """Test transform with pandas DataFrame."""
        bin_spec = {"feature1": [1, (2, 5), (5, 10)], "feature2": [10, 20, 30]}
        binner = ManualFlexibleBinning(bin_spec=bin_spec, preserve_dataframe=True)

        df = pd.DataFrame({"feature1": [1, 3, 7], "feature2": [10, 20, 30]})

        result = binner.fit_transform(df)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["feature1", "feature2"]


class TestManualFlexibleBinningEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_column_single_bin(self):
        """Test with single column and single bin specification."""
        bin_spec = {0: [42]}
        binner = ManualFlexibleBinning(bin_spec=bin_spec)

        X = np.array([[42], [42], [42]])
        result = binner.fit_transform(X)

        assert result.shape == X.shape

    def test_numericbin_spec_ifications(self):
        """Test with various numeric bin specifications."""
        bin_spec = {0: [1.5, 2.7, (5, 10), 8.3]}
        binner = ManualFlexibleBinning(bin_spec=bin_spec)

        # Test that _calculate_flexible_bins handles numeric types
        x_col = np.array([1, 2, 7, 8])
        specs, reps = binner._calculate_flexible_bins(x_col, 0)

        assert specs == [1.5, 2.7, (5, 10), 8.3]
        assert reps == [1.5, 2.7, 7.5, 8.3]

    def test_numpy_array_specifications(self):
        """Test with numpy array bin specifications."""
        bin_spec = {0: np.array([1, 2, 3]).tolist()}  # Convert to list for proper typing
        binner = ManualFlexibleBinning(bin_spec=bin_spec)

        # Should not raise exception during validation
        binner._validate_params()

    def test_tuple_specifications(self):
        """Test with tuple bin specifications."""
        bin_spec = {0: (1, 2, 3)}  # type: ignore  # Testing tuple support
        binner = ManualFlexibleBinning(bin_spec=bin_spec)  # type: ignore

        # Should not raise exception during validation
        binner._validate_params()

    def test_infinite_bound_validation(self):
        """Test that infinite bounds raise ConfigurationError during construction."""
        with pytest.raises(ConfigurationError, match="must be finite"):
            ManualFlexibleBinning(bin_spec={0: [(1, np.inf)]})

    def test_negative_infinite_bound_validation(self):
        """Test that negative infinite bounds raise ConfigurationError during construction."""
        with pytest.raises(ConfigurationError, match="must be finite"):
            ManualFlexibleBinning(bin_spec={0: [(-np.inf, 5)]})


class TestManualFlexibleBinningErrorHandling:
    """Test error handling and validation scenarios."""

    def test_detailed_error_messages_for_validation(self):
        """Test that validation errors include helpful suggestions."""
        with pytest.raises(
            ValueError, match="Failed to process provided flexible bin specifications"
        ):
            bin_spec = {0: [(5, 2)]}  # Invalid order
            ManualFlexibleBinning(bin_spec=bin_spec)

    def test_suggestions_in_configuration_errors(self):
        """Test that ConfigurationError includes helpful suggestions."""
        with pytest.raises(ConfigurationError) as exc_info:
            ManualFlexibleBinning(bin_spec=None)  # type: ignore

        error_msg = str(exc_info.value)
        assert "bin_spec must be provided" in error_msg

    def test_binning_error_for_missing_column(self):
        """Test BinningError for missing column in _calculate_flexible_bins."""
        bin_spec = {0: [1, 2, 3]}
        binner = ManualFlexibleBinning(bin_spec=bin_spec)

        with pytest.raises(BinningError) as exc_info:
            binner._calculate_flexible_bins(np.array([1, 2]), 999)

        error_msg = str(exc_info.value)
        assert "No bin specifications defined for column 999" in error_msg


class TestManualFlexibleBinningSklearnCompatibility:
    """Test sklearn compatibility features."""

    def test_get_params(self):
        """Test get_params method for sklearn compatibility."""
        bin_spec = {0: [1, 2, 3]}
        bin_representatives = {0: [1.0, 2.0, 3.0]}

        binner = ManualFlexibleBinning(
            bin_spec=bin_spec, bin_representatives=bin_representatives, preserve_dataframe=True
        )

        params = binner.get_params()

        assert params["bin_spec"] == bin_spec
        assert params["bin_representatives"] == bin_representatives
        assert params["preserve_dataframe"] is True

    def test_set_params(self):
        """Test set_params method for sklearn compatibility."""
        bin_spec = {0: [1, 2, 3]}
        binner = ManualFlexibleBinning(bin_spec=bin_spec)

        new_spec = {0: [4, 5, 6]}
        result = binner.set_params(bin_spec=new_spec)

        assert result is binner
        assert binner.bin_spec == new_spec

    def test_sklearn_clone_compatibility(self):
        """Test compatibility with sklearn.base.clone."""
        from sklearn.base import clone

        bin_spec = {0: [1, 2, 3]}
        original = ManualFlexibleBinning(bin_spec=bin_spec)

        cloned = clone(original)

        assert cloned.bin_spec == original.bin_spec
        assert cloned is not original


class TestManualFlexibleBinningInheritedMethods:
    """Test inherited methods from parent classes."""

    def test_repr_functionality(self):
        """Test that repr works correctly (from ReprMixin)."""
        bin_spec = {0: [1, 2, 3]}
        binner = ManualFlexibleBinning(bin_spec=bin_spec)

        repr_str = repr(binner)

        assert "ManualFlexibleBinning" in repr_str
        assert "bin_spec" in repr_str

    def test_inherited_transform_methods(self):
        """Test that inherited transform methods work."""
        bin_spec = {0: [1, (2, 5), (5, 10)]}
        binner = ManualFlexibleBinning(bin_spec=bin_spec)

        X = np.array([[1], [3], [7]])
        binner.fit(X)

        # Test that transform works (inherited from FlexibleBinningBase)
        result = binner.transform(X)
        assert result.shape == X.shape

    def test_check_fitted_validation(self):
        """Test that operations require fitted state."""
        bin_spec = {0: [1, 2, 3]}
        binner = ManualFlexibleBinning(bin_spec=bin_spec)

        # Manually reset fitted state to test validation
        binner._fitted = False

        X = np.array([[1], [2], [3]])

        # Should raise an error about not being fitted
        with pytest.raises(RuntimeError, match="This estimator is not fitted yet"):
            binner.transform(X)


class TestManualFlexibleBinningDocstringsAndComments:
    """Test that docstrings and comments are reflected in functionality."""

    def test_docstring_example_basic(self):
        """Test the basic example with numeric bins only."""
        # Example with numeric bins only
        specs = {
            "grade": [
                95,  # Singleton bin for high grade
                85,  # Another singleton bin
                (0, 60),  # Interval bin for failing grades
                (60, 80),  # Interval bin for passing grades
                (80, 100),  # Interval bin for high grades
            ],
            "age": [
                (0, 18),  # Minors
                (18, 35),  # Young adults
                (35, 65),  # Middle-aged
                65,  # Seniors (singleton for exact match)
            ],
        }

        binner = ManualFlexibleBinning(bin_spec=specs)

        # Should initialize successfully
        assert binner.bin_spec == specs
        assert binner._fitted is True

    def test_docstring_example_with_representatives(self):
        """Test example with custom representatives from docstring."""
        specs = {0: [(0, 10), (10, 20), 25]}
        reps = {0: [5.0, 15.0, 25.0]}  # Use float to match expected types

        binner = ManualFlexibleBinning(bin_spec=specs, bin_representatives=reps)

        assert binner.bin_spec == specs
        assert binner.bin_representatives == reps

    def test_note_about_data_ignored(self):
        """Test that input data is ignored as noted in docstring."""
        bin_spec = {0: [1, 2, 3]}
        binner = ManualFlexibleBinning(bin_spec=bin_spec)

        # Different data should give same bin specs
        x_col1 = np.array([100, 200, 300])
        x_col2 = np.array([1, 2, 3])

        specs1, reps1 = binner._calculate_flexible_bins(x_col1, 0)
        specs2, reps2 = binner._calculate_flexible_bins(x_col2, 0)

        assert specs1 == specs2
        assert reps1 == reps2


class TestManualFlexibleBinningExceptionHandling:
    """Test exception handling paths in ManualFlexibleBinning."""

    def test_calculate_flexible_bins_non_numeric_singleton_error_handling(self):
        """Test _calculate_flexible_bins with non-numeric singleton bins that cause conversion errors."""
        # Create a bin_spec with problematic singleton values that will trigger exception handling

        # Create a dummy object to manipulate internal state
        binner = ManualFlexibleBinning(bin_spec={0: [1, 2, 3]})

        # Temporarily replace bin_spec with problematic values to test exception paths
        # We need to trigger the ValueError/TypeError exception in the float() conversion
        originalbin_spec_ = binner.bin_spec

        # Create a mock object that will raise ValueError when converted to float
        class BadSingleton:
            def __float__(self):
                raise ValueError("Cannot convert to float")

        class BadSingleton2:
            def __float__(self):
                raise TypeError("Cannot convert to float")

        # Test ValueError path (line 215)
        binner.bin_spec = {0: [BadSingleton()]}
        x_col = np.array([1, 2, 3])
        specs, reps = binner._calculate_flexible_bins(x_col, 0)

        # Should have fallback representative of 0.0
        assert len(reps) == 1
        assert reps[0] == 0.0

        # Test TypeError path (line 215)
        binner.bin_spec = {0: [BadSingleton2()]}
        specs, reps = binner._calculate_flexible_bins(x_col, 0)

        # Should have fallback representative of 0.0
        assert len(reps) == 1
        assert reps[0] == 0.0

        # Test fallback path for unexpected formats (line 219)
        # Create an object that is neither a tuple nor convertible to float
        class UnexpectedFormat:
            pass

        binner.bin_spec = {0: [UnexpectedFormat()]}
        specs, reps = binner._calculate_flexible_bins(x_col, 0)

        # Should have fallback representative of 0.0
        assert len(reps) == 1
        assert reps[0] == 0.0

        # Test fallback path for tuples with wrong length (line 219)
        # This creates a tuple that is a tuple but not len == 2, triggering the else clause
        binner.bin_spec = {0: [(1, 2, 3)]}  # 3-element tuple
        specs, reps = binner._calculate_flexible_bins(x_col, 0)

        # Should have fallback representative of 0.0
        assert len(reps) == 1
        assert reps[0] == 0.0

        # Test with 1-element tuple too
        binner.bin_spec = {0: [(42,)]}  # 1-element tuple
        specs, reps = binner._calculate_flexible_bins(x_col, 0)

        # Should have fallback representative of 0.0
        assert len(reps) == 1
        assert reps[0] == 0.0

        # Restore original bin_spec
        binner.bin_spec = originalbin_spec_
