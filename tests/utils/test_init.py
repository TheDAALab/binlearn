"""Tests for utils __init__.py module."""
import pytest
from binning import utils


class TestUtilsImports:
    """Test that all expected imports are available in utils module."""

    def test_constants_imported(self):
        """Test that constants are imported."""
        assert hasattr(utils, 'MISSING_VALUE')
        assert hasattr(utils, 'ABOVE_RANGE')
        assert hasattr(utils, 'BELOW_RANGE')

    def test_type_aliases_imported(self):
        """Test that type aliases are imported."""
        # Column and data types
        assert hasattr(utils, 'ColumnId')
        assert hasattr(utils, 'ColumnList')
        assert hasattr(utils, 'OptionalColumnList')
        assert hasattr(utils, 'GuidanceColumns')
        assert hasattr(utils, 'ArrayLike')
        
        # Interval binning types
        assert hasattr(utils, 'BinEdges')
        assert hasattr(utils, 'BinEdgesDict')
        assert hasattr(utils, 'BinReps')
        assert hasattr(utils, 'BinRepsDict')
        assert hasattr(utils, 'OptionalBinEdgesDict')
        assert hasattr(utils, 'OptionalBinRepsDict')
        
        # Flexible binning types
        assert hasattr(utils, 'FlexibleBinDef')
        assert hasattr(utils, 'FlexibleBinDefs')
        assert hasattr(utils, 'FlexibleBinSpec')
        assert hasattr(utils, 'OptionalFlexibleBinSpec')
        
        # Calculation return types
        assert hasattr(utils, 'IntervalBinCalculationResult')
        assert hasattr(utils, 'FlexibleBinCalculationResult')
        
        # Count and validation types
        assert hasattr(utils, 'BinCountDict')
        
        # Numpy array types
        assert hasattr(utils, 'Array1D')
        assert hasattr(utils, 'Array2D')
        assert hasattr(utils, 'BooleanMask')
        
        # Parameter types
        assert hasattr(utils, 'FitParams')
        assert hasattr(utils, 'JointParams')

    def test_error_classes_imported(self):
        """Test that error classes are imported."""
        assert hasattr(utils, 'BinningError')
        assert hasattr(utils, 'InvalidDataError')
        assert hasattr(utils, 'ConfigurationError')
        assert hasattr(utils, 'FittingError')
        assert hasattr(utils, 'TransformationError')
        assert hasattr(utils, 'ValidationError')
        assert hasattr(utils, 'ValidationMixin')
        assert hasattr(utils, 'DataQualityWarning')

    def test_sklearn_integration_imported(self):
        """Test that sklearn integration utilities are imported."""
        assert hasattr(utils, 'SklearnCompatibilityMixin')
        assert hasattr(utils, 'BinningPipeline')
        assert hasattr(utils, 'BinningFeatureSelector')
        assert hasattr(utils, 'make_binning_scorer')

    def test_bin_operations_imported(self):
        """Test that bin operations are imported."""
        assert hasattr(utils, 'ensure_bin_dict')
        assert hasattr(utils, 'validate_bins')
        assert hasattr(utils, 'default_representatives')
        assert hasattr(utils, 'create_bin_masks')

    def test_flexible_binning_imported(self):
        """Test that flexible binning utilities are imported."""
        assert hasattr(utils, 'ensure_flexible_bin_spec')
        assert hasattr(utils, 'generate_default_flexible_representatives')
        assert hasattr(utils, 'validate_flexible_bins')
        assert hasattr(utils, 'is_missing_value')
        assert hasattr(utils, 'find_flexible_bin_for_value')
        assert hasattr(utils, 'calculate_flexible_bin_width')
        assert hasattr(utils, 'transform_value_to_flexible_bin')
        assert hasattr(utils, 'get_flexible_bin_count')

    def test_data_handling_imported(self):
        """Test that data handling utilities are imported."""
        assert hasattr(utils, 'prepare_array')
        assert hasattr(utils, 'return_like_input')
        assert hasattr(utils, 'prepare_input_with_columns')

    def test_all_list_completeness(self):
        """Test that __all__ list includes all expected exports."""
        expected_exports = {
            # Constants
            "MISSING_VALUE", "ABOVE_RANGE", "BELOW_RANGE",
            
            # Type aliases
            "ColumnId", "ColumnList", "OptionalColumnList", "GuidanceColumns", "ArrayLike",
            "BinEdges", "BinEdgesDict", "BinReps", "BinRepsDict",
            "OptionalBinEdgesDict", "OptionalBinRepsDict",
            "FlexibleBinDef", "FlexibleBinDefs", "FlexibleBinSpec", "OptionalFlexibleBinSpec",
            "IntervalBinCalculationResult", "FlexibleBinCalculationResult",
            "BinCountDict", "Array1D", "Array2D", "BooleanMask",
            "FitParams", "JointParams",
            
            # Error classes
            "BinningError", "InvalidDataError", "ConfigurationError",
            "FittingError", "TransformationError", "ValidationError",
            "ValidationMixin", "DataQualityWarning",
            
            # Sklearn integration
            "SklearnCompatibilityMixin", "BinningPipeline", "BinningFeatureSelector",
            "make_binning_scorer",
            
            # Interval binning utilities
            "ensure_bin_dict", "validate_bins", "default_representatives", "create_bin_masks",
            
            # Flexible binning utilities
            "ensure_flexible_bin_spec", "generate_default_flexible_representatives",
            "validate_flexible_bins", "is_missing_value", "find_flexible_bin_for_value",
            "calculate_flexible_bin_width", "transform_value_to_flexible_bin",
            "get_flexible_bin_count",
            
            # Data handling utilities
            "prepare_array", "return_like_input", "prepare_input_with_columns",
        }
        
        # Check that all expected exports are in __all__
        actual_exports = set(utils.__all__)
        assert expected_exports == actual_exports

    def test_constants_values(self):
        """Test that constants have correct values."""
        assert utils.MISSING_VALUE == -1
        assert utils.ABOVE_RANGE == -2
        assert utils.BELOW_RANGE == -3

    def test_error_hierarchy(self):
        """Test that error classes have correct inheritance."""
        assert issubclass(utils.InvalidDataError, utils.BinningError)
        assert issubclass(utils.ConfigurationError, utils.BinningError)
        assert issubclass(utils.FittingError, utils.BinningError)
        assert issubclass(utils.TransformationError, utils.BinningError)
        assert issubclass(utils.ValidationError, utils.BinningError)
        assert issubclass(utils.DataQualityWarning, UserWarning)

    def test_direct_import_works(self):
        """Test that direct imports work correctly."""
        # Test importing specific functions
        from binning.utils import ensure_bin_dict, validate_bins
        from binning.utils import MISSING_VALUE, ABOVE_RANGE
        from binning.utils import BinningError, InvalidDataError
        
        # Test that they are callable/usable
        assert callable(ensure_bin_dict)
        assert callable(validate_bins)
        assert isinstance(MISSING_VALUE, int)
        assert isinstance(ABOVE_RANGE, int)
        assert issubclass(InvalidDataError, BinningError)
