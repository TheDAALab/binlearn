import pytest
import inspect
from binning.base._repr_mixin import ReprMixin

class DirectInitTop(ReprMixin):
    def __init__(self, x, y=5):
        self.x = x
        self.y = y

def test_direct_init_top_coverage():
    obj = DirectInitTop(42, y=99)
    info = obj._get_constructor_info()
    assert info == {'x': inspect.Parameter.empty, 'y': 5}
    assert obj.x == 42
    assert obj.y == 99
    r = repr(obj)
    assert 'DirectInitTop' in r and 'x=42' in r and 'y=99' in r

class Dummy(ReprMixin):
    def __init__(self, a, b=2):
        self.a = a
        self.b = b

def test_get_constructor_info_full_coverage():
    obj = Dummy(1, b=3)
    info = obj._get_constructor_info()
    assert 'a' in info and 'b' in info
    assert info['b'] == 2  # default value

    # Should handle missing __init__ gracefully
    class NoInit(ReprMixin):
        pass
    noinit = NoInit()
    info2 = noinit._get_constructor_info()
    assert isinstance(info2, dict)

def test_repr_full_coverage():
    obj = Dummy(5, b=7)
    r = repr(obj)
    assert 'Dummy' in r and 'a=5' in r and 'b=7' in r

    # Should handle edge case with no __init__
    class NoInit(ReprMixin):
        pass
    noinit = NoInit()
    r2 = repr(noinit)
    assert 'NoInit' in r2

def test_repr_edge_cases():
    # Test with default values - should not show in repr
    obj = Dummy(1)  # b uses default value 2
    r = repr(obj)
    assert 'a=1' in r
    assert 'b=' not in r  # Should not show default value

    # Test with None values
    class WithNone(ReprMixin):
        def __init__(self, x=None, y=None):
            self.x = x
            self.y = y
    
    obj_none = WithNone()
    r = repr(obj_none)
    assert 'WithNone()' == r  # Should not show None defaults

    # Test with empty containers
    class WithContainers(ReprMixin):
        def __init__(self, bin_edges=None, data=None):
            self.bin_edges = bin_edges or {}
            self.data = data or []
    
    obj_empty = WithContainers()
    r = repr(obj_empty)
    assert 'WithContainers()' == r

    # Test with large objects that should be abbreviated
    class WithLargeObjects(ReprMixin):
        def __init__(self, bin_edges=None, bin_representatives=None, bin_spec=None):
            self.bin_edges = bin_edges or {1: [0, 1, 2]}
            self.bin_representatives = bin_representatives or {1: [0.5, 1.5]}
            self.bin_spec = bin_spec or {1: {'singleton': 1}}
    
    obj_large = WithLargeObjects()
    r = repr(obj_large)
    assert 'bin_edges=...' in r
    assert 'bin_representatives=...' in r
    assert 'bin_spec=...' in r

def test_repr_missing_attributes():
    # Test when object doesn't have the expected attribute
    class MissingAttr(ReprMixin):
        def __init__(self, x=1):
            # Don't set self.x
            pass
    
    obj = MissingAttr()
    r = repr(obj)
    assert 'MissingAttr()' == r  # Should handle missing attribute gracefully

def test_repr_string_values():
    class WithString(ReprMixin):
        def __init__(self, name='default'):
            self.name = name
    
    obj = WithString('test')
    r = repr(obj)
    assert "name='test'" in r  # String should be quoted

def test_get_constructor_info_exception_handling():
    # Test exception handling in _get_constructor_info
    class BadSignature(ReprMixin):
        pass
    
    obj = BadSignature()
    # Mock inspect.signature to trigger exception
    original_signature = inspect.signature
    def mock_signature(*args, **kwargs):
        raise Exception("Test exception")
    
    inspect.signature = mock_signature
    try:
        info = obj._get_constructor_info()
        assert info == {}  # Should return empty dict on exception
    finally:
        inspect.signature = original_signature

def test_repr_kwargs_handling():
    # Test kwargs parameter is skipped
    class WithKwargs(ReprMixin):
        def __init__(self, a, b=2, **kwargs):
            self.a = a
            self.b = b
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    obj = WithKwargs(1, extra=3)
    info = obj._get_constructor_info()
    assert 'kwargs' not in info
    assert 'self' not in info

def test_constructor_info_fallback_path():
    # Test fallback to class resolution when no __init__ in concrete class
    class Parent(ReprMixin):
        def __init__(self, x=1):
            self.x = x
    
    class Child(Parent):
        pass  # No __init__ defined
    
    obj = Child()
    info = obj._get_constructor_info()
    assert 'x' in info
    assert info['x'] == 1
