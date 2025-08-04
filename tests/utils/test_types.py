"""Tests for types module."""

import numpy as np

from binlearn.utils import types


class TestTypeAliases:
    """Test that type aliases are properly defined."""

    def test_column_types_exist(self):
        """Test that column-related types exist."""
        assert hasattr(types, "ColumnId")
        assert hasattr(types, "ColumnList")
        assert hasattr(types, "OptionalColumnList")
        assert hasattr(types, "GuidanceColumns")
        assert hasattr(types, "ArrayLike")

    def test_interval_binning_types_exist(self):
        """Test that interval binning types exist."""
        assert hasattr(types, "BinEdges")
        assert hasattr(types, "BinEdgesDict")
        assert hasattr(types, "BinReps")
        assert hasattr(types, "BinRepsDict")
        assert hasattr(types, "OptionalBinEdgesDict")
        assert hasattr(types, "OptionalBinRepsDict")

    def test_flexible_binning_types_exist(self):
        """Test that flexible binning types exist."""
        assert hasattr(types, "FlexibleBinDef")
        assert hasattr(types, "FlexibleBinDefs")
        assert hasattr(types, "FlexibleBinSpec")
        assert hasattr(types, "OptionalFlexibleBinSpec")

    def test_calculation_types_exist(self):
        """Test that calculation return types exist."""
        assert hasattr(types, "IntervalBinCalculationResult")
        assert hasattr(types, "FlexibleBinCalculationResult")

    def test_count_types_exist(self):
        """Test that count and validation types exist."""
        assert hasattr(types, "BinCountDict")

    def test_numpy_array_types_exist(self):
        """Test that numpy array types exist."""
        assert hasattr(types, "Array1D")
        assert hasattr(types, "Array2D")
        assert hasattr(types, "BooleanMask")

    def test_parameter_types_exist(self):
        """Test that parameter types exist."""
        assert hasattr(types, "FitParams")
        assert hasattr(types, "JointParams")

    def test_numpy_array_types_are_ndarray(self):
        """Test that numpy array types are actually ndarray."""
        assert types.Array1D == np.ndarray
        assert types.Array2D == np.ndarray
        assert types.BooleanMask == np.ndarray

    def test_types_module_imports(self):
        """Test that the module imports work correctly."""
        # Test that we can import from typing (only Any is needed in Python 3.10+)
        assert hasattr(types, "Any")

        # Python 3.10+ uses built-in types instead of typing equivalents
        # We should verify the type aliases are defined correctly
        assert hasattr(types, "ColumnId")
        assert hasattr(types, "ColumnList")
        assert hasattr(types, "BinEdges")
        assert hasattr(types, "BinEdgesDict")

    def test_numpy_import(self):
        """Test that numpy is imported."""
        assert hasattr(types, "np")
        assert types.np is np
