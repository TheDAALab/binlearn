"""Tests for types module."""

from typing import Any

import numpy as np

from binlearn.utils import _types


class TestTypeAliases:
    """Test that type aliases are properly defined."""

    def test_column_types_exist(self):
        """Test that column-related types exist."""
        assert hasattr(_types, "ColumnId")
        assert hasattr(_types, "ColumnList")
        assert hasattr(_types, "OptionalColumnList")
        assert hasattr(_types, "GuidanceColumns")
        assert hasattr(_types, "ArrayLike")

    def test_interval_binning_types_exist(self):
        """Test that interval binning types exist."""
        assert hasattr(_types, "BinEdges")
        assert hasattr(_types, "BinEdgesDict")
        assert hasattr(_types, "BinReps")
        assert hasattr(_types, "BinRepsDict")
        assert hasattr(_types, "OptionalBinEdgesDict")
        assert hasattr(_types, "OptionalBinRepsDict")

    def test_flexible_binning_types_exist(self):
        """Test that flexible binning types exist."""
        assert hasattr(_types, "FlexibleBinDef")
        assert hasattr(_types, "FlexibleBinDefs")
        assert hasattr(_types, "FlexibleBinSpec")
        assert hasattr(_types, "OptionalFlexibleBinSpec")

    def test_calculation_types_exist(self):
        """Test that calculation return types exist."""
        assert hasattr(_types, "IntervalBinCalculationResult")
        assert hasattr(_types, "FlexibleBinCalculationResult")

    def test_count_types_exist(self):
        """Test that count and validation types exist."""
        assert hasattr(_types, "BinCountDict")

    def test_numpy_array_types_exist(self):
        """Test that numpy array types exist."""
        assert hasattr(_types, "Array1D")
        assert hasattr(_types, "Array2D")
        assert hasattr(_types, "BooleanMask")

    def test_parameter_types_exist(self):
        """Test that parameter types exist."""
        assert hasattr(_types, "FitParams")
        assert hasattr(_types, "JointParams")

    def test_numpy_array_types_are_ndarray(self):
        """Test that numpy array types are actually ndarray."""
        assert _types.Array1D == np.ndarray[Any, Any]
        assert _types.Array2D == np.ndarray[Any, Any]
        assert _types.BooleanMask == np.ndarray[Any, Any]

    def test_types_module_imports(self):
        """Test that the module imports work correctly."""
        # Test that we can import from typing (only Any is needed in Python 3.10+)
        assert hasattr(_types, "Any")

        # Python 3.10+ uses built-in types instead of typing equivalents
        # We should verify the type aliases are defined correctly
        assert hasattr(_types, "ColumnId")
        assert hasattr(_types, "ColumnList")
        assert hasattr(_types, "BinEdges")
        assert hasattr(_types, "BinEdgesDict")

    def test_numpy_import(self):
        """Test that numpy is imported."""
        assert hasattr(_types, "np")
        assert _types.np is np
