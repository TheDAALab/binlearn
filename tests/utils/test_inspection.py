"""
Test the inspection utilities for comprehensive coverage.
"""

import inspect
from unittest.mock import MagicMock, patch

import pytest

from binlearn.base import GeneralBinningBase
from binlearn.methods import EqualWidthBinning
from binlearn.utils import (
    get_class_parameters,
    get_constructor_info,
    safe_get_class_parameters,
    safe_get_constructor_info,
)


# pylint: disable=too-few-public-methods
class MockClass:
    """Mock class for inspection."""

    def __init__(self, param1: int, param2: str = "default", **kwargs):
        _ = kwargs
        self.param1 = param1
        self.param2 = param2


# pylint: disable=too-few-public-methods
class MockBaseClass:
    """Base class for inheritance testing."""

    def __init__(self, base_param: int = 10, **kwargs):
        _ = kwargs
        self.base_param = base_param


# pylint: disable=too-few-public-methods
class MockDerivedClass(MockBaseClass):
    """Derived class for inheritance testing."""

    def __init__(self, derived_param: str = "derived", **kwargs):
        _ = kwargs
        super().__init__(**kwargs)
        self.derived_param = derived_param


class TestNoInit:
    """Class without explicit __init__ method."""


class TestGetClassParameters:
    """Test get_class_parameters function."""

    def test_basic_functionality(self):
        """Test basic parameter extraction."""
        params = get_class_parameters(MockClass)
        assert set(params) == {"param1", "param2"}

    def test_exclude_params(self):
        """Test parameter exclusion."""
        params = get_class_parameters(MockClass, exclude_params={"param1", "self", "kwargs"})
        assert set(params) == {"param2"}

    def test_exclude_base_class(self):
        """Test base class parameter exclusion."""
        params = get_class_parameters(MockDerivedClass, exclude_base_class="MockBaseClass")
        assert set(params) == {"derived_param"}

    def test_exclude_base_class_not_found(self):
        """Test when base class is not in MRO."""
        params = get_class_parameters(MockClass, exclude_base_class="NonExistentClass")
        assert set(params) == {"param1", "param2"}

    def test_no_init_method(self):
        """Test with class that has no explicit __init__."""
        params = get_class_parameters(TestNoInit)
        # object.__init__ has *args parameter, so we expect that
        assert "args" in params

    @patch("inspect.signature")
    def test_signature_value_error(self, mock_signature):
        """Test ValueError from inspect.signature on current class."""
        mock_signature.side_effect = ValueError("Invalid signature")
        with pytest.raises(
            ValueError, match="Failed to inspect MockClass.__init__: Invalid signature"
        ):
            get_class_parameters(MockClass)

    @patch("inspect.signature")
    def test_signature_type_error(self, mock_signature):
        """Test TypeError from inspect.signature on current class."""
        mock_signature.side_effect = TypeError("Type error")
        with pytest.raises(TypeError, match="Failed to inspect MockClass.__init__: Type error"):
            get_class_parameters(MockClass)

    @patch("inspect.signature")
    def test_base_class_signature_error(self, mock_signature):
        """Test exception from inspect.signature on base class."""
        # First call (current class) succeeds, second call (base class) fails
        mock_signature.side_effect = [
            MagicMock(parameters={"self": None, "derived_param": None, "kwargs": None}),
            ValueError("Base class signature error"),
        ]
        with pytest.raises(
            ValueError, match="Failed to inspect MockBaseClass.__init__: Base class signature error"
        ):
            get_class_parameters(MockDerivedClass, exclude_base_class="MockBaseClass")


class TestGetConstructorInfo:
    """Test get_constructor_info function."""

    def test_basic_functionality(self):
        """Test basic constructor info extraction."""
        info = get_constructor_info(MockClass)
        assert "param1" in info
        assert "param2" in info
        assert info["param1"] == inspect.Parameter.empty  # Required parameter
        assert info["param2"] == "default"  # Default value

    def test_concrete_only_true(self):
        """Test concrete_only=True behavior."""

        # Create a test class with __init__ in its dict
        class TestConcreteClass:
            """Test class with concrete parameters."""

            def __init__(self, concrete_param: str = "test"):
                self.concrete_param = concrete_param

        # Instantiate to ensure coverage
        test_obj = TestConcreteClass()
        assert test_obj.concrete_param == "test"

        info = get_constructor_info(TestConcreteClass, concrete_only=True)
        assert "concrete_param" in info

    def test_concrete_only_false(self):
        """Test concrete_only=False behavior."""
        info = get_constructor_info(MockDerivedClass, concrete_only=False)
        assert "derived_param" in info

    def test_no_explicit_init(self):
        """Test with class that has no explicit __init__."""
        info = get_constructor_info(TestNoInit)
        # object.__init__ has *args parameter
        assert "args" in info

    @patch("inspect.signature")
    def test_signature_value_error_concrete(self, mock_signature):
        """Test ValueError from inspect.signature with concrete_only=True."""

        # Create a test class with __init__ in its dict
        class TestConcreteErrorClass:
            """Test class for signature error handling."""

            def __init__(self, param: str):
                self.param = param

        # Instantiate to ensure coverage
        test_obj = TestConcreteErrorClass("test")
        assert test_obj.param == "test"

        mock_signature.side_effect = ValueError("Signature error")
        with pytest.raises(
            ValueError,
            match="Failed to get constructor info for TestConcreteErrorClass: Signature error",
        ):
            get_constructor_info(TestConcreteErrorClass, concrete_only=True)

    @patch("inspect.signature")
    def test_signature_type_error_normal(self, mock_signature):
        """Test TypeError from inspect.signature with concrete_only=False."""
        mock_signature.side_effect = TypeError("Type error")
        with pytest.raises(
            TypeError, match="Failed to get constructor info for MockClass: Type error"
        ):
            get_constructor_info(MockClass, concrete_only=False)


class TestSafeGetClassParameters:
    """Test safe_get_class_parameters function."""

    def test_successful_call(self):
        """Test successful parameter extraction."""
        params = safe_get_class_parameters(MockClass)
        assert set(params) == {"param1", "param2"}

    def test_with_custom_fallback(self):
        """Test with custom fallback value."""
        params = safe_get_class_parameters(MockClass, fallback=["custom_fallback"])
        assert set(params) == {"param1", "param2"}  # Should still succeed

    @patch("binlearn.utils.inspection.get_class_parameters")
    def test_fallback_on_value_error(self, mock_get_params):
        """Test fallback when ValueError is raised."""
        mock_get_params.side_effect = ValueError("Test error")
        params = safe_get_class_parameters(MockClass)
        assert params == []  # Default fallback

    @patch("binlearn.utils.inspection.get_class_parameters")
    def test_fallback_on_type_error(self, mock_get_params):
        """Test fallback when TypeError is raised."""
        mock_get_params.side_effect = TypeError("Test error")
        params = safe_get_class_parameters(MockClass, fallback=["fallback_param"])
        assert params == ["fallback_param"]

    @patch("binlearn.utils.inspection.get_class_parameters")
    def test_other_exceptions_propagate(self, mock_get_params):
        """Test that other exceptions are not caught."""
        mock_get_params.side_effect = RuntimeError("Test error")
        with pytest.raises(RuntimeError):
            safe_get_class_parameters(MockClass)


class TestSafeGetConstructorInfo:
    """Test safe_get_constructor_info function."""

    def test_successful_call(self):
        """Test successful constructor info extraction."""
        info = safe_get_constructor_info(MockClass)
        assert "param1" in info
        assert "param2" in info

    def test_with_custom_fallback(self):
        """Test with custom fallback value."""
        info = safe_get_constructor_info(MockClass, fallback={"custom": "fallback"})
        assert "param1" in info  # Should still succeed

    @patch("binlearn.utils.inspection.get_constructor_info")
    def test_fallback_on_value_error(self, mock_get_info):
        """Test fallback when ValueError is raised."""
        mock_get_info.side_effect = ValueError("Test error")
        info = safe_get_constructor_info(MockClass)
        assert info == {}  # Default fallback

    @patch("binlearn.utils.inspection.get_constructor_info")
    def test_fallback_on_type_error(self, mock_get_info):
        """Test fallback when TypeError is raised."""
        mock_get_info.side_effect = TypeError("Test error")
        info = safe_get_constructor_info(MockClass, fallback={"fallback": "value"})
        assert info == {"fallback": "value"}

    @patch("binlearn.utils.inspection.get_constructor_info")
    def test_other_exceptions_propagate(self, mock_get_info):
        """Test that other exceptions are not caught."""
        mock_get_info.side_effect = RuntimeError("Test error")
        with pytest.raises(RuntimeError):
            safe_get_constructor_info(MockClass)


class TestIntegration:
    """Integration tests with real binning classes."""

    def test_with_general_binning_base(self):
        """Test inspection utilities with GeneralBinningBase."""
        # This should work without errors using the safe versions
        params = safe_get_class_parameters(GeneralBinningBase)
        info = safe_get_constructor_info(GeneralBinningBase)

        # Should include GeneralBinningBase parameters
        expected_params = {"preserve_dataframe", "fit_jointly", "guidance_columns"}
        assert set(params) >= expected_params
        assert all(param in info for param in expected_params)

    def test_exclude_general_binning_base(self):
        """Test excluding GeneralBinningBase parameters."""
        params = safe_get_class_parameters(
            EqualWidthBinning, exclude_base_class="GeneralBinningBase"
        )

        # Should exclude GeneralBinningBase parameters
        general_params = {"preserve_dataframe", "fit_jointly", "guidance_columns"}
        assert not any(param in params for param in general_params)


def test_mock_classes_instantiation():
    """Test that mock classes can be instantiated to ensure coverage."""
    # Test MockClass instantiation
    mock_obj = MockClass(param1=42, param2="test")
    assert mock_obj.param1 == 42
    assert mock_obj.param2 == "test"

    # Test MockBaseClass instantiation
    base_obj = MockBaseClass(base_param=20)
    assert base_obj.base_param == 20

    # Test MockDerivedClass instantiation
    derived_obj = MockDerivedClass(derived_param="custom", base_param=30)
    assert derived_obj.derived_param == "custom"
    assert derived_obj.base_param == 30

    # Test TestNoInit class
    no_init_obj = TestNoInit()
    assert no_init_obj is not None
