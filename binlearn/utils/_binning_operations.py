"""
Binning operations utilities for interval and flexible binning.

This module provides utility functions for working with both traditional interval bins
and flexible bins that can contain singleton values and interval ranges.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._types import (
    ABOVE_RANGE,
    BELOW_RANGE,
    MISSING_VALUE,
    BinCountDict,
    BinEdges,
    BinEdgesDict,
    BinReps,
    BinRepsDict,
    BooleanMask,
    ColumnId,
    FlexibleBinDef,
    FlexibleBinDefs,
    FlexibleBinSpec,
)

# =============================================================================
# INTERVAL BINNING OPERATIONS
# =============================================================================


def validate_bin_edges_format(bin_edges: Any) -> None:
    """Validate bin edges format without transformation.

    This function performs comprehensive validation of bin edges format to ensure
    they meet the requirements for interval binning operations. It checks data
    structure, numeric validity, sorting, and other constraints without modifying
    the input data.

    Args:
        bin_edges: Input bin edges to validate. Expected format is a dictionary
            mapping column identifiers to lists/arrays of edge values. Can be None
            (which is valid and results in no validation).

    Raises:
        ValueError: If validation fails with specific error details:
            - bin_edges is not a dictionary when not None
            - Edge values for any column are not array-like
            - Any column has fewer than 2 edges (minimum for 1 bin)
            - Edge values contain non-numeric data
            - Edge values are not sorted in ascending order
            - Edge values contain infinite or NaN values

    Example:
        >>> # Valid bin edges
        >>> edges = {
        ...     'feature1': [0.0, 1.0, 2.0, 3.0],
        ...     'feature2': [-1.5, 0.0, 1.5]
        ... }
        >>> validate_bin_edges_format(edges)  # No exception
        >>>
        >>> # None is valid (no validation needed)
        >>> validate_bin_edges_format(None)  # No exception
        >>>
        >>> # Invalid cases
        >>> validate_bin_edges_format([1, 2, 3])  # ValueError: must be dictionary
        >>> validate_bin_edges_format({'col': [1.0]})  # ValueError: need >= 2 edges
        >>> validate_bin_edges_format({'col': [3.0, 1.0, 2.0]})  # ValueError: not sorted

    Note:
        - This function only validates format and constraints, it doesn't transform data
        - Bin edges define interval boundaries: [a, b, c] creates bins [a,b) and [b,c]
        - Edge values are converted to float internally for numeric validation
        - String and bytes are explicitly excluded from array-like check
        - Allows any hashable type as column identifier (string, int, etc.)
    """
    if bin_edges is None:
        return

    if not isinstance(bin_edges, dict):
        raise ValueError("bin_edges must be a dictionary mapping column identifiers to edge lists")

    for col_id, edges in bin_edges.items():
        if not hasattr(edges, "__iter__") or isinstance(edges, str | bytes):
            raise ValueError(
                f"Edges for column {col_id} must be array-like (list, tuple, or numpy array)"
            )

        edges_list = list(edges)
        if len(edges_list) < 2:
            raise ValueError(f"Column {col_id} needs at least 2 bin edges")

        # Check if all values are numeric
        try:
            float_edges = [float(x) for x in edges_list]
        except (ValueError, TypeError) as exc:
            raise ValueError(f"All edges for column {col_id} must be numeric") from exc

        # Check if edges are sorted
        if not all(float_edges[i] <= float_edges[i + 1] for i in range(len(float_edges) - 1)):
            raise ValueError(f"Bin edges for column {col_id} must be sorted in ascending order")

        # Check for invalid values
        if any(not np.isfinite(x) for x in float_edges):
            raise ValueError(f"Bin edges for column {col_id} must be finite values")


def validate_bin_representatives_format(bin_representatives: Any, bin_edges: Any = None) -> None:
    """Validate bin representatives format without transformation.

    This function validates that bin representatives are properly formatted for use
    in interval binning operations. Representatives are the values that represent
    each bin (often the midpoint) and are used as the transformed output values.

    Args:
        bin_representatives: Input bin representatives to validate. Expected format
            is a dictionary mapping column identifiers to lists/arrays of representative
            values. Can be None (which is valid and results in no validation).
        bin_edges: Optional bin edges for compatibility checking. If provided,
            validates that the number of representatives matches the number of bins
            implied by the edges. Currently not fully implemented.

    Raises:
        ValueError: If validation fails with specific error details:
            - bin_representatives is not a dictionary when not None
            - Representative values for any column are not array-like
            - Representative values contain non-numeric data
            - Representative values contain infinite or NaN values
            - Number of representatives doesn't match number of bins (when bin_edges provided)

    Example:
        >>> # Valid representatives
        >>> reps = {
        ...     'feature1': [0.5, 1.5, 2.5],  # 3 representatives for 3 bins
        ...     'feature2': [-0.75, 0.75]     # 2 representatives for 2 bins
        ... }
        >>> validate_bin_representatives_format(reps)  # No exception
        >>>
        >>> # None is valid (no validation needed)
        >>> validate_bin_representatives_format(None)  # No exception
        >>>
        >>> # Invalid cases
        >>> validate_bin_representatives_format([0.5, 1.5])  # ValueError: must be dictionary
        >>> validate_bin_representatives_format({'col': ['a', 'b']})  # ValueError: non-numeric

    Note:
        - This function only validates format and basic constraints
        - Representatives typically correspond to bin centers or other meaningful values
        - Number of representatives should equal number of bins (len(edges) - 1)
        - String and bytes are explicitly excluded from array-like check
        - The bin_edges parameter is reserved for future compatibility validation
    """
    if bin_representatives is None:
        return

    if not isinstance(bin_representatives, dict):
        raise ValueError(
            "bin_representatives must be a dictionary mapping column identifiers to"
            " representative lists"
        )

    for col_id, reps in bin_representatives.items():
        if not hasattr(reps, "__iter__") or isinstance(reps, str | bytes):
            raise ValueError(
                f"Representatives for column {col_id} must be array-like (list, tuple,"
                " or numpy array)"
            )

        reps_list = list(reps)

        # Check if all values are numeric
        try:
            float_reps = [float(x) for x in reps_list]
        except (ValueError, TypeError) as exc:
            raise ValueError(f"All representatives for column {col_id} must be numeric") from exc

        # Check for invalid values
        if any(not np.isfinite(x) for x in float_reps):
            raise ValueError(f"Representatives for column {col_id} must be finite values")

        # Check compatibility with bin edges if provided
        if bin_edges is not None and col_id in bin_edges:
            expected_bins = len(list(bin_edges[col_id])) - 1
            if len(reps_list) != expected_bins:
                raise ValueError(
                    f"Column {col_id}: {len(reps_list)} representatives provided, but "
                    f"{expected_bins} expected"
                )


def validate_bins(bin_spec: BinEdgesDict | None, bin_reps: BinRepsDict | None) -> None:
    """Validate bin specifications and representatives for consistency.

    This function performs integrated validation of bin edges and representatives
    to ensure they are compatible and properly configured for binning operations.
    It checks both individual validity and cross-compatibility between edges and
    representatives.

    Args:
        bin_spec: Dictionary mapping column identifiers to lists of bin edges.
            Each edge list defines interval boundaries for that column. Can be None
            if no validation is needed.
        bin_reps: Dictionary mapping column identifiers to lists of bin representatives.
            Each representative list contains values that represent the bins for that
            column. Can be None if no representatives are provided.

    Raises:
        ValueError: If validation fails with specific error details:
            - Any column has fewer than 2 bin edges (minimum for 1 bin)
            - Number of representatives doesn't match number of bins per column
            - Cross-compatibility issues between edges and representatives

    Example:
        >>> # Valid bin specification
        >>> edges = {'feature1': [0, 1, 2, 3], 'feature2': [-1, 0, 1]}
        >>> reps = {'feature1': [0.5, 1.5, 2.5], 'feature2': [-0.5, 0.5]}
        >>> validate_bins(edges, reps)  # No exception
        >>>
        >>> # Only edges (representatives will be generated automatically)
        >>> validate_bins(edges, None)  # No exception
        >>>
        >>> # Mismatched counts
        >>> bad_reps = {'feature1': [0.5, 1.5]}  # Only 2 reps for 3 bins
        >>> validate_bins(edges, bad_reps)  # ValueError: mismatched counts

    Note:
        - This function coordinates validation between edges and representatives
        - It complements the individual format validation functions
        - None values are handled gracefully (no validation performed)
        - Number of bins = number of edges - 1
        - Number of representatives must equal number of bins for each column
    """
    if bin_spec is None:
        return

    for col, edges in bin_spec.items():
        edges_list = list(edges)
        if len(edges_list) < 2:
            raise ValueError(f"Column {col} needs at least 2 bin edges")

        # Check if edges are sorted
        float_edges = [float(x) for x in edges_list]
        if not all(float_edges[i] <= float_edges[i + 1] for i in range(len(float_edges) - 1)):
            raise ValueError(f"Bin edges for column {col} must be non-decreasing")

        # Check representatives match
        if bin_reps is not None and col in bin_reps:
            n_bins = len(edges_list) - 1
            reps_list = list(bin_reps[col])
            if len(reps_list) != n_bins:
                raise ValueError(
                    f"Column {col}: {len(reps_list)} representatives for {n_bins} bins"
                )


def default_representatives(edges: BinEdges) -> BinReps:
    """Compute default bin representatives (bin centers) from bin edges.

    This function calculates representative values for each bin based on the bin
    edges. Representatives are typically used as the transformed output values
    and usually represent the center or a meaningful value within each bin.

    Args:
        edges: List of bin edges defining interval boundaries. Must contain at least
            2 values to define at least 1 bin. Can contain infinite values for
            unbounded bins.

    Returns:
        List of representative values, one for each bin. The number of representatives
        equals len(edges) - 1. Each representative is computed as:
        - For finite intervals [a, b]: midpoint (a + b) / 2
        - For left-unbounded intervals (-∞, b]: b - 1.0
        - For right-unbounded intervals [a, +∞): a + 1.0
        - For fully unbounded interval (-∞, +∞): 0.0

    Example:
        >>> # Regular intervals
        >>> edges = [0.0, 1.0, 2.0, 3.0]
        >>> reps = default_representatives(edges)
        >>> print(reps)
        [0.5, 1.5, 2.5]
        >>>
        >>> # With unbounded intervals
        >>> edges = [float('-inf'), 0.0, 1.0, float('inf')]
        >>> reps = default_representatives(edges)
        >>> print(reps)
        [-1.0, 0.5, 2.0]

    Note:
        - Handles infinite edge values gracefully for unbounded bins
        - For unbounded intervals, uses ±1.0 offset from finite boundary
        - Midpoint calculation handles mixed int/float edge types
        - Common use case: converting interval edges to point representatives
    """
    reps = []
    for i in range(len(edges) - 1):
        left, right = edges[i], edges[i + 1]
        if np.isneginf(left) and np.isposinf(right):
            reps.append(0.0)
        elif np.isneginf(left):
            reps.append(float(right) - 1.0)
        elif np.isposinf(right):
            reps.append(float(left) + 1.0)
        else:
            reps.append((left + right) / 2.0)
    return reps


def create_bin_masks(
    bin_indices: np.ndarray[Any, Any], n_bins: int
) -> tuple[BooleanMask, BooleanMask, BooleanMask, BooleanMask]:
    """Create boolean masks for different bin index types.

    This function creates boolean masks that identify different categories of
    data points based on their bin indices. These masks are useful for
    handling special cases and filtering data during binning operations.

    Args:
        bin_indices: Array of bin indices where each value represents:
            - 0 to n_bins-1: Valid bin index
            - MISSING_VALUE (-1): Missing/NaN values
            - BELOW_RANGE (-3): Values below the binning range
            - ABOVE_RANGE (-2): Values above the binning range
        n_bins: Total number of valid bins (for defining the valid range).

    Returns:
        A tuple of four boolean masks:
        - valid_mask: True for indices in [0, n_bins-1] (valid bins)
        - missing_mask: True for indices == MISSING_VALUE (-1)
        - below_mask: True for indices == BELOW_RANGE (-3)
        - above_mask: True for indices == ABOVE_RANGE (-2)

    Example:
        >>> import numpy as np
        >>> from binlearn.utils import MISSING_VALUE, BELOW_RANGE, ABOVE_RANGE
        >>>
        >>> # Sample bin indices with special values
        >>> indices = np.array([0, 1, 2, MISSING_VALUE, BELOW_RANGE, ABOVE_RANGE])
        >>> valid, missing, below, above = create_bin_masks(indices, n_bins=3)
        >>>
        >>> print(valid)   # [True, True, True, False, False, False]
        >>> print(missing) # [False, False, False, True, False, False]
        >>> print(below)   # [False, False, False, False, True, False]
        >>> print(above)   # [False, False, False, False, False, True]

    Note:
        - Masks are mutually exclusive (each element belongs to exactly one category)
        - Useful for separate handling of different data point categories
        - Compatible with the binning framework's special index value conventions
        - All returned arrays have the same shape as input bin_indices
    """
    # Create masks for special values first
    nan_mask = bin_indices == MISSING_VALUE
    below_mask = bin_indices == BELOW_RANGE
    above_mask = bin_indices == ABOVE_RANGE

    # Valid indices are non-negative, less than n_bins, and not special values
    valid = (bin_indices >= 0) & (bin_indices < n_bins) & ~nan_mask & ~below_mask & ~above_mask

    return valid, nan_mask, below_mask, above_mask


# =============================================================================
# FLEXIBLE BINNING OPERATIONS
# =============================================================================


def generate_default_flexible_representatives(bin_defs: FlexibleBinDefs) -> BinReps:
    """Generate default representatives for flexible bins.

    This function creates representative values for flexible bin definitions, which
    can contain both singleton values (for categorical/exact matching) and interval
    ranges. Representatives are used as the transformed output values.

    Args:
        bin_defs: List of flexible bin definitions where each definition is either:
            - Scalar value (int/float): Singleton bin that matches exactly that value
            - Tuple (start, end): Interval bin that matches values in [start, end]
            - Mixed types are supported within the same bin list

    Returns:
        List of representative values, one for each bin definition:
        - For singleton bins: the singleton value itself
        - For interval bins: the midpoint (start + end) / 2.0
        - All representatives are converted to float for consistency

    Example:
        >>> # Mix of singletons and intervals
        >>> bin_defs = [
        ...     42,           # Singleton bin
        ...     (1.0, 3.0),   # Interval bin [1.0, 3.0]
        ...     'category_A', # Singleton bin (non-numeric)
        ...     (10, 20)      # Interval bin [10, 20]
        ... ]
        >>> reps = generate_default_flexible_representatives(bin_defs)
        >>> print(reps)
        [42.0, 2.0, 'category_A', 15.0]

    Note:
        - Singleton values are preserved as-is (converted to float if numeric)
        - Interval midpoints are calculated as (start + end) / 2.0
        - Non-numeric singleton values (like strings) are preserved unchanged
        - This function handles mixed numeric/non-numeric bin definitions gracefully
    """
    reps = []
    for bin_def in bin_defs:
        if isinstance(bin_def, int | float):
            # Numeric singleton bin
            reps.append(float(bin_def))
        elif isinstance(bin_def, tuple) and len(bin_def) == 2:
            # Interval bin
            left, right = bin_def
            reps.append((left + right) / 2)  # Midpoint
        else:
            raise ValueError(f"Unknown bin definition: {bin_def}")
    return reps


def validate_flexible_bins(bin_spec: FlexibleBinSpec, bin_reps: BinRepsDict) -> None:
    """Validate flexible bin specifications and representatives for consistency.

    This function validates that flexible bin specifications and their corresponding
    representatives are properly formatted and compatible. It ensures that each
    column has matching numbers of bin definitions and representatives.

    Args:
        bin_spec: Dictionary mapping column identifiers to lists of flexible bin
            definitions. Each bin definition can be either a singleton value or
            an interval tuple.
        bin_reps: Dictionary mapping column identifiers to lists of representative
            values. Must contain entries for all columns in bin_spec, with matching
            numbers of representatives.

    Raises:
        ValueError: If validation fails with specific error details:
            - Mismatched number of bin definitions and representatives for any column
            - Invalid bin definition formats (detected by _validate_single_flexible_bin_def)
            - Missing representative entries for columns present in bin_spec

    Example:
        >>> # Valid flexible bins
        >>> bin_spec = {
        ...     'feature1': [42, (1.0, 3.0), 'category'],
        ...     'feature2': [(0, 10), (10, 20)]
        ... }
        >>> bin_reps = {
        ...     'feature1': [42.0, 2.0, 'category'],
        ...     'feature2': [5.0, 15.0]
        ... }
        >>> validate_flexible_bins(bin_spec, bin_reps)  # No exception
        >>>
        >>> # Mismatched counts
        >>> bad_reps = {'feature1': [42.0, 2.0]}  # Missing one representative
        >>> validate_flexible_bins(bin_spec, bad_reps)  # ValueError

    Note:
        - This function coordinates validation between bin specs and representatives
        - Each bin definition is individually validated using _validate_single_flexible_bin_def
        - Representatives dictionary must contain entries for all columns in bin_spec
        - Number of representatives must exactly match number of bin definitions per column
    """
    for col in bin_spec:
        bin_defs = bin_spec[col]
        reps = bin_reps.get(col, [])

        if len(bin_defs) != len(reps):
            raise ValueError(
                f"Column {col}: Number of bin definitions ({len(bin_defs)}) "
                f"must match number of representatives ({len(reps)})"
            )

        # Validate bin definition format
        for bin_idx, bin_def in enumerate(bin_defs):
            _validate_single_flexible_bin_def(
                bin_def, col, bin_idx, check_finite_bounds=False, strict=True
            )


def validate_flexible_bin_spec_format(
    bin_spec: FlexibleBinSpec, check_finite_bounds: bool = False, strict: bool = True
) -> None:
    """Validate the format and content of flexible bin specifications.

    This function performs comprehensive validation of flexible bin specifications
    to ensure they meet the requirements for flexible binning operations. It
    validates both the overall structure and individual bin definitions.

    Args:
        bin_spec: Dictionary mapping column identifiers to lists of flexible bin
            definitions. Each bin definition can be either a singleton value or
            an interval tuple (start, end).
        check_finite_bounds: Whether to enforce that interval bounds are finite
            (not infinite). Set to False to allow unbounded intervals like
            (-inf, 10) or (10, inf).
        strict: Whether to perform strict validation. When False:
            - Allows empty bin definition lists
            - Allows equal interval bounds (like (5, 5))
            - More lenient validation for edge cases

    Raises:
        ValueError: If validation fails with specific error details:
            - bin_spec is not a dictionary
            - Any column's bin definitions are not list/tuple
            - Empty bin definitions when strict=True
            - Invalid individual bin definition formats
            - Infinite bounds when check_finite_bounds=True

    Example:
        >>> # Valid flexible bin spec
        >>> bin_spec = {
        ...     'numeric_col': [42, (1.0, 5.0), (10, 20)],
        ...     'string_col': ['A', 'B', 'C'],
        ...     'mixed_col': [100, (200, 300), 'special']
        ... }
        >>> validate_flexible_bin_spec_format(bin_spec)  # No exception
        >>>
        >>> # With finite bounds check
        >>> bin_spec_with_inf = {'col': [(-float('inf'), 0), (0, float('inf'))]}
        >>> validate_flexible_bin_spec_format(bin_spec_with_inf, check_finite_bounds=True)  # ValueError
        >>>
        >>> # Lenient validation
        >>> empty_spec = {'col': []}
        >>> validate_flexible_bin_spec_format(empty_spec, strict=False)  # No exception

    Note:
        - Each bin definition is validated individually using _validate_single_flexible_bin_def
        - Supports mixed data types within bin definitions (numeric, string, etc.)
        - Default behavior allows infinite bounds for unbounded intervals
        - Strict mode enforces non-empty bin definition lists
    """
    if not isinstance(bin_spec, dict):
        raise ValueError("bin_spec must be a dictionary mapping columns to bin definitions")

    for col, bin_defs in bin_spec.items():
        if not isinstance(bin_defs, list | tuple):
            raise ValueError(f"Bin definitions for column {col} must be a list or tuple")

        if strict and len(bin_defs) == 0:
            raise ValueError(f"Bin specifications for column {col} cannot be empty")

        # Validate each bin definition
        for bin_idx, bin_def in enumerate(bin_defs):
            _validate_single_flexible_bin_def(bin_def, col, bin_idx, check_finite_bounds, strict)


def _validate_single_flexible_bin_def(
    bin_def: FlexibleBinDef,
    col: ColumnId,
    bin_idx: int,
    check_finite_bounds: bool = False,
    strict: bool = True,
) -> None:
    """Validate a single flexible bin definition for format and constraints.

    This internal function performs detailed validation of an individual flexible
    bin definition. It's used by higher-level validation functions to ensure
    each bin definition meets the format requirements and constraint rules.

    Args:
        bin_def: Single bin definition to validate. Can be either:
            - Scalar value (int/float): Singleton bin for exact value matching
            - Tuple (start, end): Interval bin for range matching
        col: Column identifier for generating informative error messages.
            Used to identify which column contains the problematic bin definition.
        bin_idx: Index of this bin definition within the column's bin list.
            Used for generating specific error messages.
        check_finite_bounds: Whether to enforce that interval bounds and singleton
            values are finite (not infinite or NaN). When True, raises ValueError
            for infinite values. Defaults to False for backward compatibility.
        strict: Whether to perform strict validation. When False:
            - Allows interval bounds where start == end (zero-width intervals)
            - More lenient validation for edge cases
            When True, enforces start < end for intervals. Defaults to True.

    Raises:
        ValueError: If bin definition is invalid with specific error details:
            - Non-tuple interval with wrong length (must be exactly 2 elements)
            - Non-numeric values in interval bounds
            - Infinite values when check_finite_bounds=True
            - Invalid interval ordering (start >= end when strict=True)
            - Invalid interval ordering (start > end when strict=False)
            - Unsupported bin definition type (not scalar or 2-tuple)

    Example:
        >>> # Valid singleton bin
        >>> _validate_single_flexible_bin_def(42, 'feature1', 0)  # No exception
        >>>
        >>> # Valid interval bin
        >>> _validate_single_flexible_bin_def((10, 20), 'feature1', 1)  # No exception
        >>>
        >>> # Invalid cases
        >>> _validate_single_flexible_bin_def((10, 5), 'feature1', 2)  # ValueError: start >= end
        >>> _validate_single_flexible_bin_def([10, 20], 'feature1', 3)  # ValueError: not tuple
        >>> _validate_single_flexible_bin_def((10,), 'feature1', 4)     # ValueError: wrong length

    Note:
        - This is an internal function called by validate_flexible_bin_spec_format
        - Provides detailed error messages including column and bin index context
        - Handles both strict and lenient validation modes
        - The check_finite_bounds parameter is useful for applications requiring finite bins
        - Singleton bins can be any numeric type (int, float)
    """
    if isinstance(bin_def, int | float):
        # Numeric singleton bin - optionally check if finite
        if check_finite_bounds and not np.isfinite(bin_def):
            raise ValueError(f"Column {col}, bin {bin_idx}: Singleton value must be finite")
        return
    if isinstance(bin_def, tuple):
        if len(bin_def) != 2:
            raise ValueError(f"Column {col}, bin {bin_idx}: Interval must be (min, max)")

        left, right = bin_def
        if not isinstance(left, int | float) or not isinstance(right, int | float):
            raise ValueError(f"Column {col}, bin {bin_idx}: Interval values must be numeric")

        # Check for finite bounds if required
        if check_finite_bounds:
            if not (np.isfinite(left) and np.isfinite(right)):
                raise ValueError(f"Column {col}, bin {bin_idx}: Interval bounds must be finite")

        # Check for proper ordering - be less strict when not in strict mode
        if strict and left >= right:
            raise ValueError(
                f"Column {col}, bin {bin_idx}: Interval min ({left}) must be < max ({right})"
            )
        if not strict and left > right:
            raise ValueError(
                f"Column {col}, bin {bin_idx}: Interval min ({left}) must be <= max ({right})"
            )
    else:
        raise ValueError(
            f"Column {col}, bin {bin_idx}: Bin must be either a numeric scalar (singleton) or "
            f"tuple (interval)"
        )


def is_missing_value(value: Any) -> bool:
    """Check if a value represents a missing value.

    This function identifies missing values according to the binning framework's
    conventions. It recognizes both explicit None values and numeric NaN values
    as missing data that should receive special handling during binning.

    Args:
        value: Value to check for missing status. Can be any type, but only
            None and numeric types are checked for missing status.

    Returns:
        True if the value is considered missing, False otherwise:
        - None values: always considered missing
        - Numeric values (int/float): missing if NaN
        - All other types: not considered missing

    Example:
        >>> # None is missing
        >>> is_missing_value(None)
        True
        >>>
        >>> # NaN is missing
        >>> import numpy as np
        >>> is_missing_value(np.nan)
        True
        >>> is_missing_value(float('nan'))
        True
        >>>
        >>> # Regular values are not missing
        >>> is_missing_value(0)
        False
        >>> is_missing_value(42.5)
        False
        >>> is_missing_value('string')
        False
        >>> is_missing_value([1, 2, 3])
        False

    Note:
        - Only None and numeric NaN values are treated as missing
        - String values like 'NaN' or 'null' are NOT considered missing
        - Empty containers ([], {}, '') are NOT considered missing
        - This function is used internally to assign MISSING_VALUE (-1) bin index
        - Missing values receive special treatment during transformation
    """
    if value is None:
        return True

    if isinstance(value, int | float):
        return bool(np.isnan(value))

    return False


def find_flexible_bin_for_value(value: Any, bin_defs: FlexibleBinDefs) -> int:
    """Find the bin index for a given value in flexible bin definitions.

    This function searches through a list of flexible bin definitions to find
    which bin contains the given value. It handles both singleton bins (exact
    value matching) and interval bins (range matching) within the same search.

    Args:
        value: Value to find the appropriate bin for. Should be numeric for
            interval bins, but any type is accepted for singleton bins.
            Missing values (None, NaN) are handled by returning MISSING_VALUE.
        bin_defs: List of flexible bin definitions to search through. Each
            definition can be either:
            - Scalar value: Creates singleton bin matching exactly that value
            - Tuple (start, end): Creates interval bin matching [start, end]

    Returns:
        Bin index (0-based) if a matching bin is found, or MISSING_VALUE (-1)
        if no bin matches the value. Returns the index of the first matching
        bin if multiple bins could contain the value.

    Example:
        >>> from binlearn.utils import MISSING_VALUE
        >>>
        >>> # Define flexible bins
        >>> bin_defs = [42, (10.0, 20.0), 'category_A', (100, 200)]
        >>>
        >>> # Find bins for various values
        >>> find_flexible_bin_for_value(42, bin_defs)      # Returns 0 (singleton match)
        >>> find_flexible_bin_for_value(15.5, bin_defs)    # Returns 1 (interval match)
        >>> find_flexible_bin_for_value('category_A', bin_defs)  # Returns 2 (singleton match)
        >>> find_flexible_bin_for_value(150, bin_defs)     # Returns 3 (interval match)
        >>> find_flexible_bin_for_value(999, bin_defs)     # Returns MISSING_VALUE (no match)

    Note:
        - Returns the index of the FIRST matching bin if multiple matches exist
        - For interval bins, uses inclusive bounds: [start, end]
        - Non-numeric values can only match singleton bins
        - Values that don't match any bin return MISSING_VALUE (-1)
        - This function is used internally during flexible binning transformation
    """
    for bin_idx, bin_def in enumerate(bin_defs):
        if isinstance(bin_def, int | float):
            # Singleton bin - direct comparison
            if value == bin_def:
                return bin_idx
        elif isinstance(bin_def, tuple) and len(bin_def) == 2:
            # Interval bin - only for numeric values
            left, right = bin_def
            if isinstance(value, int | float) and left <= value <= right:
                return bin_idx

    # Value doesn't match any bin - treat as missing
    return MISSING_VALUE


def calculate_flexible_bin_width(bin_def: FlexibleBinDef) -> float:
    """Calculate the width of a flexible bin definition.

    This function computes the width (range span) of a flexible bin definition.
    For interval bins, this is the difference between the upper and lower bounds.
    For singleton bins, the width is always zero since they represent exact values.

    Args:
        bin_def: Flexible bin definition to calculate width for. Can be either:
            - Scalar value (int/float): Singleton bin representing an exact value
            - Tuple (start, end): Interval bin representing a range [start, end]

    Returns:
        Width of the bin as a float:
        - For singleton bins: always 0.0
        - For interval bins: end - start (can be negative if end < start)

    Raises:
        ValueError: If bin_def is not a recognized format (not scalar or 2-tuple).

    Example:
        >>> # Singleton bin has zero width
        >>> calculate_flexible_bin_width(42)
        0.0
        >>> calculate_flexible_bin_width(3.14)
        0.0
        >>>
        >>> # Interval bin width is the range
        >>> calculate_flexible_bin_width((10.0, 20.0))
        10.0
        >>> calculate_flexible_bin_width((5, 15))
        10.0
        >>>
        >>> # Zero-width interval
        >>> calculate_flexible_bin_width((5.0, 5.0))
        0.0
        >>>
        >>> # Invalid format
        >>> calculate_flexible_bin_width([1, 2, 3])  # ValueError

    Note:
        - Width calculation does not validate that start <= end for intervals
        - Negative widths are possible if interval bounds are reversed
        - This function is useful for bin analysis and balancing operations
        - Compatible with both integer and float numeric types
    """
    if isinstance(bin_def, int | float):
        # Singleton bin has zero width
        return 0.0
    if isinstance(bin_def, tuple) and len(bin_def) == 2:
        # Interval bin
        left, right = bin_def
        return right - left  # type: ignore[no-any-return]

    raise ValueError(f"Unknown bin definition: {bin_def}")


def transform_value_to_flexible_bin(value: Any, bin_defs: FlexibleBinDefs) -> int:
    """Transform a single value to its corresponding flexible bin index.

    This function performs the core transformation operation for flexible binning,
    taking an individual value and returning the index of the bin it belongs to.
    It handles missing value detection and delegates to the bin finding logic.

    Args:
        value: Value to transform to a bin index. Can be any type, including
            numeric values, strings, or None. Missing values (None, NaN) are
            handled specially.
        bin_defs: List of flexible bin definitions to match against. Each
            definition can be either a singleton value or an interval tuple.

    Returns:
        Bin index (0-based integer) if a matching bin is found, or MISSING_VALUE
        (-1) if the value is missing or doesn't match any bin definition.

    Example:
        >>> from binlearn.utils import MISSING_VALUE
        >>>
        >>> # Define flexible bins
        >>> bin_defs = [10, (20, 30), 'category', (40, 50)]
        >>>
        >>> # Transform various values
        >>> transform_value_to_flexible_bin(10, bin_defs)        # Returns 0
        >>> transform_value_to_flexible_bin(25, bin_defs)        # Returns 1
        >>> transform_value_to_flexible_bin('category', bin_defs) # Returns 2
        >>> transform_value_to_flexible_bin(45, bin_defs)        # Returns 3
        >>>
        >>> # Missing and unmatched values
        >>> transform_value_to_flexible_bin(None, bin_defs)      # Returns MISSING_VALUE
        >>> transform_value_to_flexible_bin(100, bin_defs)       # Returns MISSING_VALUE
        >>> import numpy as np
        >>> transform_value_to_flexible_bin(np.nan, bin_defs)    # Returns MISSING_VALUE

    Note:
        - This function combines missing value detection with bin finding
        - Missing values are always assigned MISSING_VALUE (-1) regardless of bin definitions
        - Uses is_missing_value() for robust missing value detection
        - Delegates actual bin finding to find_flexible_bin_for_value()
        - This is typically called during the transform phase of flexible binning
    """
    # Robust missing value check
    if is_missing_value(value):
        return MISSING_VALUE

    # Find matching bin
    return find_flexible_bin_for_value(value, bin_defs)


def get_flexible_bin_count(bin_spec: FlexibleBinSpec) -> BinCountDict:
    """Get the number of bins for each column in a flexible bin specification.

    This utility function extracts bin count information from a flexible bin
    specification dictionary. It provides a convenient way to determine how
    many bins are defined for each column without needing to iterate through
    the specification manually.

    Args:
        bin_spec: Dictionary mapping column identifiers to lists of flexible
            bin definitions. Each column can have a different number of bins,
            and each bin can be either a singleton value or an interval.

    Returns:
        Dictionary mapping each column identifier to the number of bins defined
        for that column. The count equals the length of the bin definitions list
        for each column.

    Example:
        >>> # Define flexible bins for multiple columns
        >>> bin_spec = {
        ...     'numeric_feature': [10, (20, 30), (40, 50)],        # 3 bins
        ...     'category_feature': ['A', 'B', 'C', 'D'],           # 4 bins
        ...     'mixed_feature': [100, (200, 300), 'special']       # 3 bins
        ... }
        >>>
        >>> # Get bin counts
        >>> bin_counts = get_flexible_bin_count(bin_spec)
        >>> print(bin_counts)
        {'numeric_feature': 3, 'category_feature': 4, 'mixed_feature': 3}
        >>>
        >>> # Empty specification
        >>> get_flexible_bin_count({})
        {}
        >>>
        >>> # Single column
        >>> get_flexible_bin_count({'col1': [1, 2, 3, 4, 5]})
        {'col1': 5}

    Note:
        - This function performs a simple length calculation on each bin definitions list
        - Useful for validation, memory allocation, and algorithm initialization
        - Does not validate the bin definitions themselves - use validate_flexible_bin_spec_format for that
        - Returns an empty dictionary if the input specification is empty
        - Compatible with any column identifier type (string, int, etc.)
    """
    return {col: len(bin_defs) for col, bin_defs in bin_spec.items()}
