"""
Tests for error handling utilities.
"""

import warnings
from unittest.mock import Mock
import pytest

from binlearn.utils._error_handling import (
    handle_sklearn_import_error,
    handle_insufficient_data_error,
    handle_convergence_warning,
    handle_parameter_bounds_error,
    safe_sklearn_call,
    validate_fitted_state,
)
from binlearn.utils._errors import ConfigurationError, BinningError


class TestHandleSklearnImportError:
    """Test the handle_sklearn_import_error function."""

    def test_basic_functionality(self):
        """Test basic sklearn import error handling."""
        import_error = ImportError("No module named 'sklearn'")
        method_name = "TestMethod"

        result = handle_sklearn_import_error(import_error, method_name)

        assert isinstance(result, ConfigurationError)
        assert "TestMethod requires scikit-learn" in str(result)
        assert len(result.suggestions) >= 3
        assert any("pip install scikit-learn" in suggestion for suggestion in result.suggestions)

    def test_different_method_names(self):
        """Test with different method names."""
        import_error = ImportError("sklearn not found")

        result1 = handle_sklearn_import_error(import_error, "KMeans")
        result2 = handle_sklearn_import_error(import_error, "DBSCAN")

        assert "KMeans requires scikit-learn" in str(result1)
        assert "DBSCAN requires scikit-learn" in str(result2)


class TestHandleInsufficientDataError:
    """Test the handle_insufficient_data_error function."""

    def test_basic_functionality(self):
        """Test basic insufficient data error handling."""
        data_size = 5
        min_required = 10
        method_name = "TestMethod"

        result = handle_insufficient_data_error(data_size, min_required, method_name)

        assert isinstance(result, ConfigurationError)
        assert "TestMethod requires at least 10 data points, got 5" in str(result)
        assert len(result.suggestions) >= 3

    def test_suggestions_content(self):
        """Test that suggestions contain expected content."""
        result = handle_insufficient_data_error(3, 5, "TestMethod")

        suggestions_text = " ".join(result.suggestions)
        assert "5 points" in suggestions_text
        assert "equal-width" in suggestions_text or "equal-frequency" in suggestions_text
        assert "bins" in suggestions_text


class TestHandleConvergenceWarning:
    """Test the handle_convergence_warning function."""

    def test_with_fallback_suggestion(self):
        """Test convergence warning with fallback suggestion."""
        method_name = "TestMethod"
        max_iterations = 100

        with pytest.warns(UserWarning, match="TestMethod did not converge within 100 iterations"):
            handle_convergence_warning(method_name, max_iterations, suggest_fallback=True)

    def test_without_fallback_suggestion(self):
        """Test convergence warning without fallback suggestion."""
        method_name = "TestMethod"
        max_iterations = 50

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            handle_convergence_warning(method_name, max_iterations, suggest_fallback=False)

            assert len(w) == 1
            warning_msg = str(w[0].message)
            assert "TestMethod did not converge within 50 iterations" in warning_msg
            assert "fallback" not in warning_msg.lower()

    def test_default_fallback_suggestion(self):
        """Test default behavior (suggest_fallback=True)."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            handle_convergence_warning("TestMethod", 100)  # Default suggest_fallback=True

            assert len(w) == 1
            assert "equal-width binning as a fallback" in str(w[0].message)


class TestHandleParameterBoundsError:
    """Test the handle_parameter_bounds_error function."""

    def test_both_min_max_bounds(self):
        """Test with both min and max bounds."""
        result = handle_parameter_bounds_error("test_param", 15, min_val=0, max_val=10)

        assert isinstance(result, ConfigurationError)
        assert "test_param must be between 0 and 10, got 15" in str(result)
        assert "test_param=5" in result.suggestions[0]  # Middle value

    def test_only_min_bound(self):
        """Test with only minimum bound."""
        result = handle_parameter_bounds_error("test_param", -5, min_val=0)

        assert "test_param must be at least 0, got -5" in str(result)
        assert any("test_param=" in suggestion for suggestion in result.suggestions)

    def test_only_max_bound(self):
        """Test with only maximum bound."""
        result = handle_parameter_bounds_error("test_param", 15, max_val=10)

        assert "test_param must be at most 10, got 15" in str(result)

    def test_no_bounds_specified(self):
        """Test with no bounds specified."""
        result = handle_parameter_bounds_error("test_param", "invalid")

        assert "test_param must be within valid range, got invalid" in str(result)

    def test_custom_suggestions(self):
        """Test with custom suggestions."""
        custom_suggestions = ["Use positive values", "Try test_param=1.0"]
        result = handle_parameter_bounds_error(
            "test_param", -1, min_val=0, suggestions=custom_suggestions
        )

        assert result.suggestions == custom_suggestions

    def test_edge_case_bounds(self):
        """Test edge cases with bounds calculation."""
        # Test where min_val is large
        result = handle_parameter_bounds_error("test_param", 0, min_val=10, max_val=20)
        assert "test_param=15" in result.suggestions[0]


class TestSafeSklearnCall:
    """Test the safe_sklearn_call function."""

    def test_successful_call(self):
        """Test successful sklearn function call."""
        mock_func = Mock(return_value="success")

        result = safe_sklearn_call(
            mock_func, "arg1", "arg2", method_name="TestMethod", kwarg1="value1"
        )

        assert result == "success"
        mock_func.assert_called_once_with("arg1", "arg2", kwarg1="value1")

    def test_import_error_with_fallback(self):
        """Test ImportError handling with fallback function."""
        mock_func = Mock(side_effect=ImportError("No sklearn"))
        mock_fallback = Mock(return_value="fallback_success")

        with pytest.warns(UserWarning, match="falling back to simpler implementation"):
            result = safe_sklearn_call(
                mock_func, "arg1", method_name="TestMethod", fallback_func=mock_fallback
            )

        assert result == "fallback_success"
        mock_fallback.assert_called_once_with("arg1")

    def test_import_error_without_fallback(self):
        """Test ImportError handling without fallback function."""
        mock_func = Mock(side_effect=ImportError("No sklearn"))

        with pytest.raises(ConfigurationError, match="TestMethod requires scikit-learn"):
            safe_sklearn_call(mock_func, method_name="TestMethod")

    def test_general_exception_with_fallback(self):
        """Test general exception handling with fallback."""
        mock_func = Mock(side_effect=ValueError("Some error"))
        mock_fallback = Mock(return_value="fallback_result")

        with pytest.warns(UserWarning, match="TestMethod failed with sklearn"):
            result = safe_sklearn_call(
                mock_func, "arg1", method_name="TestMethod", fallback_func=mock_fallback
            )

        assert result == "fallback_result"

    def test_general_exception_without_fallback(self):
        """Test general exception handling without fallback."""
        mock_func = Mock(side_effect=ValueError("Some error"))

        with pytest.raises(BinningError, match="TestMethod failed: Some error"):
            safe_sklearn_call(mock_func, method_name="TestMethod")

    def test_default_method_name(self):
        """Test with default method name."""
        mock_func = Mock(side_effect=ValueError("Error"))

        with pytest.raises(BinningError, match="method failed"):
            safe_sklearn_call(mock_func)  # Default method_name="method"

    def test_fallback_with_args_and_kwargs(self):
        """Test fallback function receives correct arguments."""
        mock_func = Mock(side_effect=ValueError("Error"))
        mock_fallback = Mock(return_value="result")

        with pytest.warns(UserWarning):
            safe_sklearn_call(
                mock_func,
                "arg1",
                "arg2",
                method_name="TestMethod",
                fallback_func=mock_fallback,
                kwarg1="value1",
                kwarg2="value2",
            )

        mock_fallback.assert_called_once_with("arg1", "arg2", kwarg1="value1", kwarg2="value2")


class TestValidateFittedState:
    """Test the validate_fitted_state function."""

    def test_fitted_object_bin_edges(self):
        """Test with fitted object having bin_edges_."""
        mock_obj = Mock()
        mock_obj.bin_edges_ = [1, 2, 3]

        # Should not raise any exception
        validate_fitted_state(mock_obj, "transform")

    def test_fitted_object_bins(self):
        """Test with fitted object having bins_."""
        mock_obj = Mock()
        mock_obj.bins_ = [1, 2, 3]

        # Should not raise any exception
        validate_fitted_state(mock_obj, "transform")

    def test_fitted_object_boundaries(self):
        """Test with fitted object having boundaries_."""
        mock_obj = Mock()
        mock_obj.boundaries_ = [1, 2, 3]

        # Should not raise any exception
        validate_fitted_state(mock_obj, "transform")

    def test_fitted_object_fitted_flag(self):
        """Test with fitted object having fitted_ flag."""
        mock_obj = Mock()
        mock_obj.fitted_ = True

        # Should not raise any exception
        validate_fitted_state(mock_obj, "transform")

    def test_unfitted_object(self):
        """Test with unfitted object."""
        mock_obj = Mock()
        # Remove any fitted attributes
        del mock_obj.bin_edges_
        del mock_obj.bins_
        del mock_obj.boundaries_
        del mock_obj.fitted_

        with pytest.raises(
            BinningError, match="Cannot call transform\\(\\) before calling fit\\(\\)"
        ):
            validate_fitted_state(mock_obj, "transform")

    def test_custom_method_name(self):
        """Test with custom method name."""
        mock_obj = Mock()
        # Remove fitted attributes
        del mock_obj.bin_edges_
        del mock_obj.bins_
        del mock_obj.boundaries_
        del mock_obj.fitted_

        with pytest.raises(
            BinningError, match="Cannot call predict\\(\\) before calling fit\\(\\)"
        ):
            validate_fitted_state(mock_obj, "predict")

    def test_default_method_name(self):
        """Test with default method name."""
        mock_obj = Mock()
        # Remove fitted attributes
        del mock_obj.bin_edges_
        del mock_obj.bins_
        del mock_obj.boundaries_
        del mock_obj.fitted_

        with pytest.raises(
            BinningError, match="Cannot call transform\\(\\) before calling fit\\(\\)"
        ):
            validate_fitted_state(mock_obj)  # Default method_name="transform"

    def test_suggestions_content(self):
        """Test that error suggestions contain expected content."""
        mock_obj = Mock()
        # Remove fitted attributes
        del mock_obj.bin_edges_
        del mock_obj.bins_
        del mock_obj.boundaries_
        del mock_obj.fitted_

        try:
            validate_fitted_state(mock_obj)
        except BinningError as e:
            assert len(e.suggestions) >= 2
            suggestions_text = " ".join(e.suggestions)
            assert "fit()" in suggestions_text
            assert "fit_transform()" in suggestions_text
