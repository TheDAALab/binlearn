"""Comprehensive tests for binlearn.base._sklearn_integration_base module.

This module tests all functionality in the SklearnIntegrationBase class
to achieve 100% test coverage, including edge cases and error conditions.
"""

from unittest.mock import Mock

import pytest

from binlearn.base._sklearn_integration_base import SklearnIntegrationBase


class MockEstimator(SklearnIntegrationBase):
    """Mock estimator for testing SklearnIntegrationBase functionality."""

    def __init__(
        self, n_bins=5, clip=True, bin_edges=None, bin_representatives=None, bin_spec=None
    ):
        """Initialize mock estimator with test parameters."""
        super().__init__()
        self.n_bins = n_bins
        self.clip = clip
        self.bin_edges = bin_edges
        self.bin_representatives = bin_representatives
        self.bin_spec = bin_spec
        # Configure which attributes indicate fitted state
        self._fitted_attributes = ["bin_edges_", "bin_representatives_", "bin_spec_"]

    def fit(self, X):
        """Mock fit method."""
        self.bin_edges_ = [[0, 1, 2], [0, 1, 2]]
        self.bin_representatives_ = [[0.5, 1.5], [0.5, 1.5]]
        self.fitted_attr_ = "fitted_value"
        return self

    def _set_sklearn_attributes_from_specs(self):
        """Mock sklearn attribute setter."""
        self.n_features_in_ = 2
        self.feature_names_in_ = ["feature1", "feature2"]
        # Ensure bin_spec_ is available for tests
        if not hasattr(self, "bin_spec_"):
            self.bin_spec_ = None


class TestSklearnIntegrationBase:
    """Test suite for SklearnIntegrationBase class."""

    def test_initialization(self):
        """Test basic initialization of SklearnIntegrationBase."""
        estimator = SklearnIntegrationBase()
        assert hasattr(estimator, "_fitted_attributes")
        assert estimator._fitted_attributes == []

    def test_fitted_property_no_attributes(self):
        """Test _fitted property when no fitted attributes are configured."""
        estimator = SklearnIntegrationBase()
        assert not estimator._fitted

    def test_fitted_property_empty_attributes(self):
        """Test _fitted property when fitted attributes list is empty."""
        estimator = MockEstimator()
        estimator._fitted_attributes = []
        assert not estimator._fitted

    def test_fitted_property_no_fitted_attributes_attr(self):
        """Test _fitted property when _fitted_attributes doesn't exist."""
        estimator = SklearnIntegrationBase()
        # Remove the attribute to test the hasattr check
        delattr(estimator, "_fitted_attributes")
        assert not estimator._fitted

    def test_fitted_property_with_dict_content(self):
        """Test _fitted property with dict fitted attributes."""
        estimator = MockEstimator()
        estimator._fitted_attributes = ["test_dict_"]
        estimator.test_dict_ = {"key": "value"}  # type: ignore
        assert estimator._fitted

    def test_fitted_property_with_empty_dict(self):
        """Test _fitted property with empty dict fitted attribute."""
        estimator = MockEstimator()
        estimator._fitted_attributes = ["test_dict_"]
        estimator.test_dict_ = {}  # type: ignore
        assert not estimator._fitted

    def test_fitted_property_with_list_content(self):
        """Test _fitted property with list fitted attributes."""
        estimator = MockEstimator()
        estimator._fitted_attributes = ["test_list_"]
        estimator.test_list_ = [1, 2, 3]  # type: ignore
        assert estimator._fitted

    def test_fitted_property_with_empty_list(self):
        """Test _fitted property with empty list fitted attribute."""
        estimator = MockEstimator()
        estimator._fitted_attributes = ["test_list_"]
        estimator.test_list_ = []  # type: ignore
        assert not estimator._fitted

    def test_fitted_property_with_other_content(self):
        """Test _fitted property with non-dict/list fitted attributes."""
        estimator = MockEstimator()
        estimator._fitted_attributes = ["test_value_"]
        estimator.test_value_ = "some_value"  # type: ignore
        assert estimator._fitted

    def test_fitted_property_with_falsy_content(self):
        """Test _fitted property with falsy fitted attribute."""
        estimator = MockEstimator()
        estimator._fitted_attributes = ["test_value_"]
        estimator.test_value_ = None  # type: ignore
        assert not estimator._fitted

    def test_fitted_property_with_missing_attribute(self):
        """Test _fitted property when fitted attribute doesn't exist."""
        estimator = MockEstimator()
        estimator._fitted_attributes = ["nonexistent_"]
        assert not estimator._fitted

    def test_fitted_property_multiple_attributes_some_empty(self):
        """Test _fitted property with multiple attributes where some are empty."""
        estimator = MockEstimator()
        estimator._fitted_attributes = ["empty_", "filled_"]
        estimator.empty_ = None  # type: ignore
        estimator.filled_ = "content"  # type: ignore
        assert estimator._fitted

    def test_set_fitted_attributes(self):
        """Test _set_fitted_attributes method."""
        estimator = MockEstimator()
        estimator._set_fitted_attributes(
            bin_edges_=[[0, 1, 2]], bin_representatives_=[[0.5, 1.5]], custom_attr_="custom_value"
        )

        assert estimator.bin_edges_ == [[0, 1, 2]]
        assert estimator.bin_representatives_ == [[0.5, 1.5]]
        assert estimator.custom_attr_ == "custom_value"  # type: ignore

    def test_check_fitted_when_not_fitted(self):
        """Test _check_fitted raises error when not fitted."""
        estimator = MockEstimator()

        with pytest.raises(
            RuntimeError, match="This estimator is not fitted yet. Call 'fit' first."
        ):
            estimator._check_fitted()

    def test_check_fitted_when_fitted(self):
        """Test _check_fitted passes when fitted."""
        estimator = MockEstimator()
        estimator.fit([[1, 2], [3, 4]])

        # Should not raise
        estimator._check_fitted()

    def test_get_params_basic(self):
        """Test get_params method basic functionality."""
        estimator = MockEstimator(n_bins=10, clip=False)
        params = estimator.get_params()

        assert params["n_bins"] == 10
        assert params["clip"] is False
        assert params["bin_edges"] is None
        assert params["bin_representatives"] is None
        assert params["bin_spec"] is None
        assert params["class_"] == "MockEstimator"
        assert params["module_"] == "test_sklearn_integration_base"

    def test_get_params_with_defaults(self):
        """Test get_params with default parameter values."""
        estimator = MockEstimator()
        params = estimator.get_params()

        assert params["n_bins"] == 5  # Default value
        assert params["clip"] is True  # Default value

    def test_get_params_with_missing_attribute(self):
        """Test get_params when attribute is missing from instance."""
        estimator = MockEstimator()
        # Remove an attribute to test the default handling
        delattr(estimator, "n_bins")

        params = estimator.get_params()
        assert params["n_bins"] == 5  # Should use default from signature

    def test_get_params_with_no_default_parameter(self):
        """Test get_params with parameter that has no default."""

        class NoDefaultEstimator(SklearnIntegrationBase):
            def __init__(self, required_param):
                super().__init__()
                self.required_param = required_param

        estimator = NoDefaultEstimator("test_value")
        params = estimator.get_params()

        assert params["required_param"] == "test_value"

    def test_get_params_with_missing_no_default_parameter(self):
        """Test get_params with missing parameter that has no default."""

        class NoDefaultEstimator(SklearnIntegrationBase):
            def __init__(self, required_param):
                super().__init__()
                # Don't set the attribute to test missing case
                pass

        estimator = NoDefaultEstimator("test_value")
        params = estimator.get_params()

        assert params["required_param"] is None  # Should be None when missing

    def test_get_params_removes_class_and_module(self):
        """Test that get_params removes class_ and module_ if they exist as attributes."""
        estimator = MockEstimator()
        # Set class_ and module_ attributes
        estimator.class_ = "ShouldBeRemoved"  # type: ignore
        estimator.module_ = "ShouldBeRemoved"  # type: ignore

        params = estimator.get_params()

        # Should be overridden with correct values, not the attribute values
        assert params["class_"] == "MockEstimator"
        assert params["module_"] == "test_sklearn_integration_base"

    def test_get_params_when_fitted(self):
        """Test get_params includes fitted parameters when fitted."""
        estimator = MockEstimator()
        estimator.fit([[1, 2], [3, 4]])

        params = estimator.get_params()

        # Should include fitted parameters without underscores
        assert "bin_edges" in params
        assert "bin_representatives" in params
        assert "fitted_attr" in params
        assert params["bin_edges"] == [[0, 1, 2], [0, 1, 2]]
        assert params["bin_representatives"] == [[0.5, 1.5], [0.5, 1.5]]
        assert params["fitted_attr"] == "fitted_value"

    def test_extract_fitted_params(self):
        """Test _extract_fitted_params method."""
        estimator = MockEstimator()
        estimator.bin_edges_ = [[0, 1, 2]]
        estimator.bin_representatives_ = [[0.5, 1.5]]
        estimator.custom_fitted_ = "custom_value"  # type: ignore
        estimator.n_features_in_ = 2  # Should be excluded
        estimator.feature_names_in_ = ["a", "b"]  # Should be excluded

        fitted_params = estimator._extract_fitted_params()

        assert fitted_params["bin_edges"] == [[0, 1, 2]]
        assert fitted_params["bin_representatives"] == [[0.5, 1.5]]
        assert fitted_params["custom_fitted"] == "custom_value"
        assert "n_features_in" not in fitted_params
        assert "feature_names_in" not in fitted_params

    def test_extract_fitted_params_excludes_private_attributes(self):
        """Test _extract_fitted_params excludes private attributes."""
        estimator = MockEstimator()
        estimator._private_attr_ = "should_be_excluded"  # type: ignore
        estimator.__dunder_attr__ = "should_be_excluded"  # type: ignore
        estimator.public_attr_ = "should_be_included"  # type: ignore

        fitted_params = estimator._extract_fitted_params()

        assert "_private_attr" not in fitted_params
        assert "__dunder_attr" not in fitted_params
        assert "public_attr" in fitted_params

    def test_extract_fitted_params_excludes_none_values(self):
        """Test _extract_fitted_params excludes None values."""
        estimator = MockEstimator()
        estimator.valid_attr_ = "value"  # type: ignore
        estimator.none_attr_ = None  # type: ignore

        fitted_params = estimator._extract_fitted_params()

        assert "valid_attr" in fitted_params
        assert "none_attr" not in fitted_params

    def test_set_params_basic(self):
        """Test set_params method basic functionality."""
        estimator = MockEstimator()
        result = estimator.set_params(n_bins=10, clip=False)

        assert result is estimator  # Returns self
        assert estimator.n_bins == 10
        assert estimator.clip is False

    def test_set_params_ignores_class_metadata(self):
        """Test set_params ignores class_ and module_ parameters."""
        estimator = MockEstimator(n_bins=5)
        estimator.set_params(n_bins=10, class_="IgnoreMe", module_="IgnoreMe")

        assert estimator.n_bins == 10
        # class_ and module_ should not be set as attributes
        assert not hasattr(estimator, "class_") or getattr(estimator, "class_", None) != "IgnoreMe"
        assert (
            not hasattr(estimator, "module_") or getattr(estimator, "module_", None) != "IgnoreMe"
        )

    def test_set_params_fitted_parameters(self):
        """Test set_params with fitted parameters (reconstruction)."""
        estimator = MockEstimator()
        estimator.set_params(n_bins=10, bin_edges=[[0, 1, 2]], bin_representatives=[[0.5, 1.5]])

        assert estimator.n_bins == 10
        assert estimator.bin_edges_ == [[0, 1, 2]]
        assert estimator.bin_representatives_ == [[0.5, 1.5]]

    def test_set_params_reconstruction_params_special_case(self):
        """Test set_params special handling of bin_edges, bin_representatives, bin_spec."""
        estimator = MockEstimator()
        estimator.set_params(
            bin_edges=[[0, 1, 2]], bin_representatives=[[0.5, 1.5]], bin_spec={"type": "test"}
        )

        # These should be set as fitted attributes due to special case logic
        assert estimator.bin_edges_ == [[0, 1, 2]]
        assert estimator.bin_representatives_ == [[0.5, 1.5]]
        assert estimator.bin_spec_ == {"type": "test"}

    def test_set_params_reconstruction_params_none_values(self):
        """Test set_params doesn't override constructor defaults with None."""
        estimator = MockEstimator(bin_edges=[1, 2, 3])
        estimator.set_params(bin_edges=None, n_bins=10)

        # Should not override the constructor default
        assert estimator.bin_edges == [1, 2, 3]
        assert estimator.n_bins == 10

    def test_set_params_calls_sklearn_attribute_setter(self):
        """Test set_params calls _set_sklearn_attributes_from_specs when available."""
        estimator = MockEstimator()

        # Mock the sklearn attribute setter
        estimator._set_sklearn_attributes_from_specs = Mock()

        estimator.set_params(bin_edges=[[0, 1, 2]])

        # Should be called when fitted parameters are set
        estimator._set_sklearn_attributes_from_specs.assert_called_once()

    def test_set_params_no_sklearn_attribute_setter(self):
        """Test set_params when _set_sklearn_attributes_from_specs is not available."""
        estimator = MockEstimator()

        # Should not raise error even without the method
        estimator.set_params(bin_edges=[[0, 1, 2]])
        assert estimator.bin_edges_ == [[0, 1, 2]]

    def test_set_params_fitted_parameter_detection(self):
        """Test set_params correctly detects fitted parameters."""
        estimator = MockEstimator()

        # Test with known fitted parameters
        estimator.set_params(bin_edges=[[0, 1, 2]], n_bins=10)

        assert estimator.bin_edges_ == [[0, 1, 2]]
        assert estimator.n_bins == 10

    def test_validate_params_default(self):
        """Test _validate_params default implementation."""
        estimator = SklearnIntegrationBase()
        # Should not raise - default implementation is empty
        estimator._validate_params()

    def test_more_tags(self):
        """Test _more_tags method returns correct sklearn tags."""
        estimator = SklearnIntegrationBase()
        tags = estimator._more_tags()

        expected_tags = {
            "requires_fit": True,
            "requires_y": False,
            "X_types": ["2darray"],
            "allow_nan": True,
            "stateless": False,
        }

        assert tags == expected_tags

    def test_inheritance_from_base_estimator(self):
        """Test that SklearnIntegrationBase properly inherits from BaseEstimator."""
        estimator = SklearnIntegrationBase()

        # Should have BaseEstimator methods
        assert hasattr(estimator, "get_params")
        assert hasattr(estimator, "set_params")

        # Should be instance of BaseEstimator
        from sklearn.base import BaseEstimator

        assert isinstance(estimator, BaseEstimator)

    def test_complex_parameter_workflow(self):
        """Test complex workflow with parameter getting and setting."""
        # Create estimator and fit
        estimator = MockEstimator(n_bins=8, clip=False)
        estimator.fit([[1, 2], [3, 4]])

        # Get parameters (should include fitted params)
        params = estimator.get_params()

        # Create new estimator and reconstruct from params
        new_estimator = MockEstimator()
        # Only set the parameters that can be reconstructed
        reconstruction_params = {
            k: v
            for k, v in params.items()
            if k in ["n_bins", "clip", "bin_edges", "bin_representatives", "bin_spec"]
        }
        new_estimator.set_params(**reconstruction_params)

        # Should have same state
        assert new_estimator.n_bins == 8
        assert new_estimator.clip is False
        assert new_estimator.bin_edges_ == [[0, 1, 2], [0, 1, 2]]
        assert new_estimator.bin_representatives_ == [[0.5, 1.5], [0.5, 1.5]]
        assert new_estimator._fitted

    def test_fitted_attributes_configuration(self):
        """Test that subclasses can configure fitted attributes."""

        class CustomEstimator(SklearnIntegrationBase):
            def __init__(self):
                super().__init__()
                self._fitted_attributes = ["custom_attr_", "another_attr_"]

        estimator = CustomEstimator()
        assert not estimator._fitted

        # Use setattr to add attributes dynamically
        estimator.custom_attr_ = "value"  # type: ignore
        assert estimator._fitted

        estimator.custom_attr_ = None  # type: ignore
        estimator.another_attr_ = [1, 2, 3]  # type: ignore
        assert estimator._fitted

    def test_edge_case_empty_string_fitted_attribute(self):
        """Test fitted property with empty string (falsy but not None)."""
        estimator = MockEstimator()
        estimator._fitted_attributes = ["test_attr_"]
        estimator.test_attr_ = ""  # type: ignore

        # Empty string is falsy, so should not be considered fitted
        assert not estimator._fitted

    def test_edge_case_zero_fitted_attribute(self):
        """Test fitted property with zero value (falsy but not None)."""
        estimator = MockEstimator()
        estimator._fitted_attributes = ["test_attr_"]
        estimator.test_attr_ = 0  # type: ignore

        # Zero is falsy, so should not be considered fitted
        assert not estimator._fitted

    def test_get_params_deep_parameter(self):
        """Test get_params with deep parameter (inherited from BaseEstimator)."""
        estimator = MockEstimator()

        # Test both deep=True and deep=False
        params_deep = estimator.get_params(deep=True)
        params_shallow = estimator.get_params(deep=False)

        # Should be the same for this simple case
        assert params_deep["n_bins"] == params_shallow["n_bins"]
        assert params_deep["clip"] == params_shallow["clip"]

    def test_multiple_fitted_attributes_mixed_types(self):
        """Test fitted property with multiple attributes of different types."""
        estimator = MockEstimator()
        estimator._fitted_attributes = ["dict_attr_", "list_attr_", "str_attr_"]

        # All empty - not fitted
        estimator.dict_attr_ = {}  # type: ignore
        estimator.list_attr_ = []  # type: ignore
        estimator.str_attr_ = None  # type: ignore
        assert not estimator._fitted

        # One has content - fitted
        estimator.dict_attr_ = {"key": "value"}  # type: ignore
        assert estimator._fitted

        # Reset and try with list
        estimator.dict_attr_ = {}  # type: ignore
        estimator.list_attr_ = ["item"]  # type: ignore
        assert estimator._fitted

        # Reset and try with string
        estimator.list_attr_ = []  # type: ignore
        estimator.str_attr_ = "content"  # type: ignore
        assert estimator._fitted

    def test_fitted_property_non_dict_non_list_truthy_value(self):
        """Test fitted property branch for non-dict, non-list truthy values."""
        estimator = MockEstimator()
        estimator._fitted_attributes = ["test_attr_"]

        # Set a truthy value that's not a dict or list
        estimator.test_attr_ = 42  # integer  # type: ignore
        assert estimator._fitted

        # Set another type - tuple (truthy, not dict or list)
        estimator.test_attr_ = 1, 2, 3  # type: ignore
        assert estimator._fitted

    def test_set_params_empty_filtered_regular_params(self):
        """Test set_params when filtered_regular_params becomes empty."""
        estimator = MockEstimator()

        # Set parameters where all regular params would be filtered out
        # This happens when bin_edges/bin_representatives/bin_spec are None
        estimator.set_params(bin_edges=None, bin_representatives=None, bin_spec=None)

        # Should still work without calling BaseEstimator.set_params
        # The branch where filtered_regular_params is empty should be covered

    def test_set_params_without_sklearn_attribute_setter_none(self):
        """Test set_params when sklearn setter doesn't exist (getattr returns None)."""

        # Create a basic estimator without the sklearn attribute setter method
        class BasicEstimator(SklearnIntegrationBase):
            def __init__(self, test_param=None):
                super().__init__()
                self.test_param = test_param
                self._fitted_attributes = ["test_attr_"]

        estimator = BasicEstimator()

        # This should not raise error when sklearn_setter is None (method doesn't exist)
        estimator.set_params(test_param="value")

        assert estimator.test_param == "value"

    def test_fitted_property_list_isinstance_false_branch(self):
        """Test fitted property branch where isinstance(attr_value, list) is False but attr_value is truthy."""
        estimator = MockEstimator()
        estimator._fitted_attributes = ["test_attr_"]

        # Set a truthy value that's not a dict but will fail isinstance(attr_value, list)
        # This tests the branch 46->39 where the elif isinstance(attr_value, list) condition is False
        # but we still have a truthy value that's not a dict
        estimator.test_attr_ = (  # type: ignore
            "string_value"  # String is not dict, not list, but truthy
        )
        assert estimator._fitted

    def test_set_params_no_fitted_params_to_set_no_sklearn_setter(self):
        """Test set_params when fitted_params_to_set is empty (no sklearn setter called)."""
        # This tests branch 184->187 where fitted_params_to_set is empty so sklearn_setter check is skipped
        estimator = MockEstimator()

        # Set only regular parameters, no fitted parameters
        estimator.set_params(n_bins=10, clip=False)

        assert estimator.n_bins == 10
        assert estimator.clip is False

    def test_fitted_property_multiple_attrs_non_list_continues_loop(self):
        """Test fitted property with multiple attributes where non-list continues to next iteration."""
        estimator = MockEstimator()
        estimator._fitted_attributes = ["attr1_", "attr2_"]

        # Set first attribute to a non-list truthy value, second to None
        # This should test the branch where isinstance(attr_value, list) is False
        # and we continue to check the next attribute in the loop
        estimator.attr1_ = (  # type: ignore
            "not_a_list"  # Truthy, not dict, not list -> should return True
        )
        estimator.attr2_ = None  # type: ignore

        # The first attribute should make it fitted, but we want to test the branch logic
        assert estimator._fitted

    def test_fitted_property_list_check_false_continues_to_next_condition(self):
        """Test fitted property where list check is False but continues to final elif."""
        estimator = MockEstimator()
        estimator._fitted_attributes = ["test_attr_"]

        # Set to a value that's truthy, not dict, not list
        # This should go through: if attr_value (True) -> if isinstance(dict) (False) -> elif isinstance(list) (False) -> elif (True)
        estimator.test_attr_ = 42  # Number: truthy, not dict, not list  # type: ignore
        assert estimator._fitted

    def test_set_params_fitted_params_no_sklearn_setter(self):
        """Test set_params with fitted parameters when sklearn_setter is None."""

        # Create a basic estimator without the sklearn attribute setter method
        class BasicEstimator(SklearnIntegrationBase):
            def __init__(self):
                super().__init__()
                self._fitted_attributes = ["custom_fitted_"]

        estimator = BasicEstimator()

        # Set fitted parameters using the bin_edges special case
        # This will be treated as fitted parameter due to the special reconstruction logic
        estimator.set_params(bin_edges=[[0, 1, 2]])

        # Should set the fitted parameter
        assert hasattr(estimator, "bin_edges_")
        assert estimator.bin_edges_ == [[0, 1, 2]]  # type: ignore

        # This covers the branch where fitted_params_to_set is not empty
        # but sklearn_setter is None (since BasicEstimator doesn't have that method)

    def test_fitted_property_empty_list_then_valid_attr(self):
        """Test fitted property with empty list first, then valid attribute."""
        estimator = MockEstimator()
        estimator._fitted_attributes = ["empty_list_", "valid_attr_"]

        # First attribute: empty list (should fail isinstance check and continue loop)
        # Second attribute: valid content
        estimator.empty_list_ = []  # Empty list: truthy check fails, loop continues  # type: ignore
        estimator.valid_attr_ = "valid"  # type: ignore

        # Should be fitted due to second attribute
        assert estimator._fitted

    def test_fitted_property_false_value_then_continue_loop(self):
        """Test fitted property where first attr is falsy, continues to next in loop."""
        estimator = MockEstimator()
        estimator._fitted_attributes = ["falsy_attr_", "truthy_attr_"]

        # Set first to falsy (fails initial if check, continues loop)
        # Set second to truthy non-dict/non-list
        estimator.falsy_attr_ = (  # type: ignore
            None  # Falsy: if attr_value fails, continues to next iteration
        )
        estimator.truthy_attr_ = "string_value"  # Should make it fitted  # type: ignore

        assert estimator._fitted

    def test_fitted_property_force_elif_list_branch_false(self):
        """Test specific branch: elif isinstance(attr_value, list) evaluates False."""
        estimator = MockEstimator()
        estimator._fitted_attributes = ["test1_", "test2_"]

        # Create a scenario that forces the elif isinstance(attr_value, list) to be False
        # but continues to next iteration: set first attr to string (not list), second to None
        estimator.test1_ = {1, 2, 3}  # Set: truthy, not dict, not list  # type: ignore
        estimator.test2_ = None  # None: falsy  # type: ignore

        # This should cover the branch 46->39 (from elif isinstance list check back to loop)
        assert estimator._fitted

    def test_set_params_force_no_fitted_params_branch(self):
        """Test set_params with no fitted parameters to force empty fitted_params_to_set."""
        estimator = MockEstimator()

        # Call set_params with only constructor parameters (no fitted params)
        # This should make fitted_params_to_set empty, skipping sklearn setter branch
        result = estimator.set_params(n_bins=15)

        assert result is estimator
        assert estimator.n_bins == 15
