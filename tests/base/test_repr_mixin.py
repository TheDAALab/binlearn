"""Tests for ReprMixin to achieve 100% coverage."""


from binlearn.base._repr_mixin import ReprMixin


class TestReprMixin:
    """Test class for ReprMixin functionality."""

    def test_repr_mixin_fallback_to_class_init(self):
        """Test ReprMixin when __init__ is not in cls.__dict__ to trigger line 28."""

        class BaseClass(ReprMixin):
            def __init__(self, param1=None):
                self.param1 = param1

        class DerivedClass(BaseClass):
            # No __init__ defined - should fallback to BaseClass.__init__
            pass

        instance = DerivedClass(param1="test")
        repr_str = repr(instance)
        assert "DerivedClass" in repr_str

    def test_repr_mixin_required_parameter_handling(self):
        """Test ReprMixin with required parameters (no defaults) to trigger line 38."""

        class RequiredParamsClass(ReprMixin):
            def __init__(self, required_param, optional_param="default"):
                self.required_param = required_param
                self.optional_param = optional_param

        instance = RequiredParamsClass("test_value", "custom")
        repr_str = repr(instance)
        assert "RequiredParamsClass" in repr_str
        assert "required_param='test_value'" in repr_str
        assert "optional_param='custom'" in repr_str

    def test_repr_mixin_exception_handling(self):
        """Test ReprMixin exception handling to trigger lines 40-41."""
        import inspect

        class ProblematicClass(ReprMixin):
            def __init__(self, param1=None):
                self.param1 = param1

        # Create an instance first
        instance = ProblematicClass(param1="test")

        # Temporarily replace inspect.signature to make it fail
        original_signature = inspect.signature

        def failing_signature(*args, **kwargs):
            raise ValueError("Signature inspection failed")

        try:
            inspect.signature = failing_signature
            # This should trigger the exception handling in _get_constructor_info (lines 40-41)
            repr_str = repr(instance)
            # Should not crash and should return basic class name due to exception handling
            assert "ProblematicClass" in repr_str

        finally:
            # Restore inspect.signature
            inspect.signature = original_signature

    def test_repr_mixin_missing_attribute(self):
        """Test ReprMixin when object is missing expected attributes."""

        class IncompleteClass(ReprMixin):
            def __init__(self, param1=None, param2="default"):
                # Intentionally not setting param1 to test hasattr check
                self.param2 = param2

        instance = IncompleteClass(param2="test")

        # This should handle missing attributes gracefully
        repr_str = repr(instance)
        assert "IncompleteClass" in repr_str

    def test_repr_mixin_default_values_handling(self):
        """Test ReprMixin handling of default values and empty containers."""

        class DefaultsClass(ReprMixin):
            def __init__(self, param1=None, param2=None, param3=None):
                if param3 is None:
                    param3 = {}
                if param2 is None:
                    param2 = []
                self.param1 = param1
                self.param2 = param2 or []
                self.param3 = param3 or {}

        # Test with default None value
        instance1 = DefaultsClass(param1=None)
        repr_str1 = repr(instance1)
        assert "DefaultsClass" in repr_str1

        # Test with empty containers that match defaults
        instance2 = DefaultsClass(param2=[], param3={})
        repr_str2 = repr(instance2)
        assert "DefaultsClass" in repr_str2

    def test_repr_mixin_normal_operation(self):
        """Test ReprMixin normal operation with non-default values."""

        class NormalClass(ReprMixin):
            def __init__(self, param1="default", param2=42):
                self.param1 = param1
                self.param2 = param2

        instance = NormalClass(param1="custom", param2=100)
        repr_str = repr(instance)

        assert "NormalClass" in repr_str
        assert "param1='custom'" in repr_str
        assert "param2=100" in repr_str
