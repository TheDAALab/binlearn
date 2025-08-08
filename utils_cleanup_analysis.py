#!/usr/bin/env python3
"""
Compile actually used utils functions based on manual analysis.
"""

# Based on grep analysis, these are the actually used functions from utils:

USED_FUNCTIONS = {
    # From bin_operations.py
    'bin_operations': {
        'create_bin_masks',
        'default_representatives', 
        'validate_bin_edges_format',
        'validate_bin_representatives_format',
        'validate_bins',
    },
    
    # From constants.py
    'constants': {
        'ABOVE_RANGE',
        'BELOW_RANGE', 
        'MISSING_VALUE',
    },
    
    # From data_handling.py
    'data_handling': {
        'prepare_array',
        'prepare_input_with_columns',
        'return_like_input',
    },
    
    # From errors.py
    'errors': {
        'BinningError',
        'ConfigurationError',
        'DataQualityWarning',
        'FittingError',
        'TransformationError',
        'ValidationError',
        'ValidationMixin',
    },
    
    # From flexible_bin_operations.py
    'flexible_bin_operations': {
        'calculate_flexible_bin_width',
        'find_flexible_bin_for_value', 
        'generate_default_flexible_representatives',
        'get_flexible_bin_count',
        'is_missing_value',
        'transform_value_to_flexible_bin',
        'validate_flexible_bin_spec_format',
        'validate_flexible_bins',
    },
    
    # From inspection.py
    'inspection': {
        'safe_get_class_parameters',
        'safe_get_constructor_info',
    },
    
    # From parameter_conversion.py
    'parameter_conversion': {
        'resolve_n_bins_parameter',
        'resolve_string_parameter',
        'validate_bin_number_for_calculation',
        'validate_bin_number_parameter',
        'validate_numeric_parameter',
    },
    
    # From types.py (all types are potentially used)
    'types': {
        'Array1D', 'Array2D', 'ArrayLike', 'BinCountDict', 'BinEdges', 
        'BinEdgesDict', 'BinReps', 'BinRepsDict', 'BooleanMask',
        'ColumnId', 'ColumnList', 'FitParams', 'FlexibleBinCalculationResult',
        'FlexibleBinDef', 'FlexibleBinDefs', 'FlexibleBinSpec', 
        'GuidanceColumns', 'IntervalBinCalculationResult', 'JointParams',
        'OptionalBinEdgesDict', 'OptionalBinRepsDict', 'OptionalColumnList',
        'OptionalFlexibleBinSpec',
    },
}

# Utils files that exist
ALL_UTILS_FILES = {
    'bin_operations.py',
    'constants.py', 
    'data_handling.py',
    'errors.py',
    'flexible_bin_operations.py',
    'flexible_bin_operations_new.py',  # potentially unused
    'inspection.py',
    'integration.py',  # potentially unused
    'parameter_conversion.py',
    'sklearn_integration.py',  # potentially unused
    'types.py',
    'validation.py',  # potentially unused
}

USED_FILES = set()
for module_name in USED_FUNCTIONS.keys():
    USED_FILES.add(f'{module_name}.py')

UNUSED_FILES = ALL_UTILS_FILES - USED_FILES

print("UTILS CLEANUP ANALYSIS:")
print("=" * 50)

print(f"\nUSED FILES ({len(USED_FILES)}):")
for f in sorted(USED_FILES):
    print(f"  ✅ {f}")

print(f"\nUNUSED FILES ({len(UNUSED_FILES)}):")
for f in sorted(UNUSED_FILES):
    print(f"  ❌ {f}")

print(f"\nSUMMARY:")
print(f"  Total files: {len(ALL_UTILS_FILES)}")  
print(f"  Used files: {len(USED_FILES)}")
print(f"  Unused files: {len(UNUSED_FILES)}")

if UNUSED_FILES:
    print(f"\nFiles that can be safely removed:")
    for f in sorted(UNUSED_FILES):
        print(f"  - binlearn/utils/{f}")
