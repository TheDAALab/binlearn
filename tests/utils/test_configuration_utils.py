"""
Tests for configuration utilities.
"""

import warnings
from unittest.mock import Mock, patch
import pytest
import numpy as np

from binlearn.utils._configuration_utils import (
    standardize_init_pattern,
    create_param_dict_for_config,
    get_effective_n_bins,
    prepare_sklearn_estimator_params,
    handle_common_warnings,
)


class TestStandardizeInitPattern:
    """Test the standardize_init_pattern function."""

    def test_basic_functionality(self):
        """Test basic parameter setting without validations."""
        mock_obj = Mock()
        method_name = "test_method"
        user_params = {"param1": "value1", "param2": "value2"}

        with patch("binlearn.utils._configuration_utils.apply_config_defaults") as mock_apply:
            mock_apply.return_value = {"param1": "value1", "param2": "value2", "param3": "default"}

            standardize_init_pattern(mock_obj, method_name, user_params)

            # Check that apply_config_defaults was called correctly
            mock_apply.assert_called_once_with(method_name, user_params)

            # Check that attributes were set
            assert mock_obj.param1 == "value1"
            assert mock_obj.param2 == "value2"
            assert mock_obj.param3 == "default"

    def test_with_validations(self):
        """Test with custom validations."""

        # Create a real object instead of Mock to control attribute existence
        class TestObject:
            pass

        mock_obj = TestObject()
        method_name = "test_method"
        user_params = {"param1": "value1"}

        # Mock validator that just returns True
        validator1 = Mock(return_value=None)
        validator2 = Mock(return_value=None)
        validator_nonexistent = Mock(return_value=None)

        extra_validations = {
            "param1": validator1,
            "param2": validator2,
            "nonexistent": validator_nonexistent,  # Should not be called since attribute doesn't exist
        }

        with patch("binlearn.utils._configuration_utils.apply_config_defaults") as mock_apply:
            mock_apply.return_value = {
                "param1": "value1",
                "param2": "value2",
            }  # Only these get set as attributes

            standardize_init_pattern(mock_obj, method_name, user_params, extra_validations)

            # Check validators were called for existing attributes
            validator1.assert_called_once_with("value1")
            validator2.assert_called_once_with("value2")

            # Check nonexistent param validator was not called (no such attribute exists on mock_obj)
            assert validator_nonexistent.call_count == 0

    def test_without_extra_validations(self):
        """Test when extra_validations is None."""
        mock_obj = Mock()
        method_name = "test_method"
        user_params = {"param1": "value1"}

        with patch("binlearn.utils._configuration_utils.apply_config_defaults") as mock_apply:
            mock_apply.return_value = {"param1": "value1"}

            # Should not raise any errors
            standardize_init_pattern(mock_obj, method_name, user_params, None)

            assert mock_obj.param1 == "value1"


class TestCreateParamDictForConfig:
    """Test the create_param_dict_for_config function."""

    def test_all_none_values(self):
        """Test when all values are None."""
        result = create_param_dict_for_config()
        assert result == {}

    def test_with_n_bins_and_random_state(self):
        """Test with n_bins and random_state provided."""
        result = create_param_dict_for_config(n_bins=5, random_state=42)
        expected = {"n_bins": 5, "random_state": 42}
        assert result == expected

    def test_with_kwargs(self):
        """Test with additional keyword arguments."""
        result = create_param_dict_for_config(
            n_bins=10, random_state=None, clip=True, eps=0.5, min_samples=None
        )
        expected = {"n_bins": 10, "clip": True, "eps": 0.5}
        assert result == expected

    def test_mixed_none_and_values(self):
        """Test with mix of None and actual values."""
        result = create_param_dict_for_config(n_bins=None, random_state=123, other_param="value")
        expected = {"random_state": 123, "other_param": "value"}
        assert result == expected


class TestGetEffectiveNBins:
    """Test the get_effective_n_bins function."""

    def test_integer_input(self):
        """Test with integer n_bins."""
        assert get_effective_n_bins(5, 100) == 5
        assert get_effective_n_bins(1, 10) == 1
        assert get_effective_n_bins(100, 50) == 100

    def test_auto_method(self):
        """Test 'auto' method."""
        # auto = min(50, max(2, int(data_size**0.5)))
        assert get_effective_n_bins("auto", 4) == 2  # max(2, 2)
        assert get_effective_n_bins("auto", 100) == 10  # max(2, 10)
        assert get_effective_n_bins("auto", 10000) == 50  # min(50, 100)

    def test_sqrt_method(self):
        """Test 'sqrt' method."""
        assert get_effective_n_bins("sqrt", 4) == 2  # max(2, 2)
        assert get_effective_n_bins("sqrt", 100) == 10  # max(2, 10)
        assert get_effective_n_bins("sqrt", 1) == 2  # max(2, 1)

    def test_log_method(self):
        """Test 'log' method."""
        assert get_effective_n_bins("log", 4) == 2  # max(2, 2)
        assert get_effective_n_bins("log", 256) == 8  # max(2, 8)
        assert get_effective_n_bins("log", 1) == 2  # max(2, 0)

    def test_unknown_method(self):
        """Test unknown string method falls back to 10."""
        assert get_effective_n_bins("unknown", 100) == 10
        assert get_effective_n_bins("invalid", 50) == 10


class TestPrepareSklearnEstimatorParams:
    """Test the prepare_sklearn_estimator_params function."""

    def test_basic_mapping(self):
        """Test basic parameter mapping."""
        user_params = {"n_bins": 10, "random_state": 42, "other": "value"}
        mapping = {"n_bins": "n_clusters", "random_state": "random_state"}

        result = prepare_sklearn_estimator_params(user_params, mapping)
        expected = {"n_clusters": 10, "random_state": 42}
        assert result == expected

    def test_missing_parameters(self):
        """Test when user_params doesn't have all mapped parameters."""
        user_params = {"n_bins": 10}
        mapping = {"n_bins": "n_clusters", "random_state": "random_state"}

        result = prepare_sklearn_estimator_params(user_params, mapping)
        expected = {"n_clusters": 10}
        assert result == expected

    def test_empty_mapping(self):
        """Test with empty mapping."""
        user_params = {"n_bins": 10, "random_state": 42}
        mapping = {}

        result = prepare_sklearn_estimator_params(user_params, mapping)
        assert result == {}

    def test_empty_user_params(self):
        """Test with empty user params."""
        user_params = {}
        mapping = {"n_bins": "n_clusters"}

        result = prepare_sklearn_estimator_params(user_params, mapping)
        assert result == {}


class TestHandleCommonWarnings:
    """Test the handle_common_warnings function."""

    def test_small_data_warning(self):
        """Test warning for small data."""
        with pytest.warns(UserWarning, match="may not work well with very small datasets"):
            handle_common_warnings("TestMethod", small_data=True)

    def test_few_bins_warning(self):
        """Test warning for few bins."""
        with pytest.warns(UserWarning, match="with very few bins may not capture data patterns"):
            handle_common_warnings("TestMethod", few_bins=True)

    def test_many_bins_warning(self):
        """Test warning for many bins."""
        with pytest.warns(UserWarning, match="with many bins may overfit to the data"):
            handle_common_warnings("TestMethod", many_bins=True)

    def test_multiple_warnings(self):
        """Test multiple warnings at once."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            handle_common_warnings("TestMethod", small_data=True, few_bins=True)

            assert len(w) == 2
            assert "small datasets" in str(w[0].message)
            assert "few bins" in str(w[1].message)

    def test_no_warnings(self):
        """Test when no conditions are met."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            handle_common_warnings("TestMethod")

            assert len(w) == 0

    def test_false_conditions(self):
        """Test with explicitly False conditions."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            handle_common_warnings("TestMethod", small_data=False, few_bins=False)

            assert len(w) == 0
