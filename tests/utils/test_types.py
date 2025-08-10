"""
Comprehensive tests for binlearn.utils._types module.

Tests all type constants and type aliases with 100% coverage.
"""

import numpy as np

from binlearn.utils import (
    ABOVE_RANGE,
    BELOW_RANGE,
    # Constants
    MISSING_VALUE,
    Array1D,
    Array2D,
    ArrayLike,
    BinCountDict,
    # Binning types
    BinEdges,
    BinEdgesDict,
    BinReps,
    BinRepsDict,
    BooleanMask,
    # Basic types
    ColumnId,
    ColumnList,
    FitParams,
    FlexibleBinCalculationResult,
    # Flexible binning types
    FlexibleBinDef,
    FlexibleBinDefs,
    FlexibleBinSpec,
    # Parameter types
    GuidanceColumns,
    # Result types
    IntervalBinCalculationResult,
    JointParams,
    OptionalBinEdgesDict,
    OptionalBinRepsDict,
    OptionalColumnList,
    OptionalFlexibleBinSpec,
)


class TestConstants:
    """Test constant values used throughout binlearn."""

    def test_missing_value_constant(self) -> None:
        """Test MISSING_VALUE constant."""
        assert MISSING_VALUE == -1
        assert isinstance(MISSING_VALUE, int)

    def test_below_range_constant(self) -> None:
        """Test BELOW_RANGE constant."""
        assert BELOW_RANGE == -3
        assert isinstance(BELOW_RANGE, int)

    def test_above_range_constant(self) -> None:
        """Test ABOVE_RANGE constant."""
        assert ABOVE_RANGE == -2
        assert isinstance(ABOVE_RANGE, int)

    def test_constants_are_distinct(self) -> None:
        """Test that all constants have distinct values."""
        constants = [MISSING_VALUE, BELOW_RANGE, ABOVE_RANGE]
        assert len(set(constants)) == len(constants)

    def test_constants_are_negative(self) -> None:
        """Test that constants are negative (to avoid conflicts with bin indices)."""
        assert MISSING_VALUE < 0
        assert BELOW_RANGE < 0
        assert ABOVE_RANGE < 0

    def test_constants_ordering(self) -> None:
        """Test constants have expected ordering."""
        # This ensures they're in expected order for any logic that depends on it
        assert BELOW_RANGE < ABOVE_RANGE < MISSING_VALUE < 0


class TestBasicTypes:
    """Test basic type aliases."""

    def test_column_id_type(self) -> None:
        """Test ColumnId type alias."""

        # ColumnId should accept integers and strings
        def check_column_id(col: ColumnId) -> ColumnId:
            return col

        # These should be valid ColumnId values
        assert check_column_id(0) == 0
        assert check_column_id(42) == 42
        assert check_column_id("column_name") == "column_name"
        assert check_column_id("feature_1") == "feature_1"

    def test_column_list_type(self) -> None:
        """Test ColumnList type alias."""

        def check_column_list(cols: ColumnList) -> ColumnList:
            return cols

        # These should be valid ColumnList values
        assert check_column_list([0, 1, 2]) == [0, 1, 2]
        assert check_column_list(["a", "b", "c"]) == ["a", "b", "c"]
        assert check_column_list([0, "mixed", 2]) == [0, "mixed", 2]
        assert check_column_list([]) == []

    def test_optional_column_list_type(self) -> None:
        """Test OptionalColumnList type alias."""

        def check_optional_column_list(cols: OptionalColumnList) -> OptionalColumnList:
            return cols

        # These should be valid OptionalColumnList values
        assert check_optional_column_list(None) is None
        assert check_optional_column_list([0, 1, 2]) == [0, 1, 2]
        assert check_optional_column_list(["a", "b"]) == ["a", "b"]

    def test_array_like_type(self) -> None:
        """Test ArrayLike type alias."""

        def check_array_like(arr: ArrayLike) -> ArrayLike:
            return arr

        # These should be valid ArrayLike values
        np_array = np.array([[1, 2], [3, 4]])
        python_list = [[1, 2], [3, 4]]
        nested_tuple = ((1, 2), (3, 4))

        assert np.array_equal(check_array_like(np_array), np_array)
        assert check_array_like(python_list) == python_list
        assert check_array_like(nested_tuple) == nested_tuple

    def test_array_1d_type(self) -> None:
        """Test Array1D type alias."""

        def check_array_1d(arr: Array1D) -> Array1D:
            return arr

        # Should work with 1D numpy arrays
        arr_1d = np.array([1, 2, 3, 4])
        result = check_array_1d(arr_1d)
        assert np.array_equal(result, arr_1d)
        assert result.ndim == 1

    def test_array_2d_type(self) -> None:
        """Test Array2D type alias."""

        def check_array_2d(arr: Array2D) -> Array2D:
            return arr

        # Should work with 2D numpy arrays
        arr_2d = np.array([[1, 2], [3, 4]])
        result = check_array_2d(arr_2d)
        assert np.array_equal(result, arr_2d)
        assert result.ndim == 2

    def test_boolean_mask_type(self) -> None:
        """Test BooleanMask type alias."""

        def check_boolean_mask(mask: BooleanMask) -> BooleanMask:
            return mask

        # Should work with boolean numpy arrays
        bool_mask = np.array([True, False, True, False])
        result = check_boolean_mask(bool_mask)
        assert np.array_equal(result, bool_mask)
        assert result.dtype == bool


class TestBinningTypes:
    """Test binning-related type aliases."""

    def test_bin_edges_type(self) -> None:
        """Test BinEdges type alias."""

        def check_bin_edges(edges: BinEdges) -> BinEdges:
            return edges

        # Should work with lists of floats
        edges = [0.0, 1.0, 2.0, 3.0]
        assert check_bin_edges(edges) == edges

    def test_bin_edges_dict_type(self) -> None:
        """Test BinEdgesDict type alias."""

        def check_bin_edges_dict(edges_dict: BinEdgesDict) -> BinEdgesDict:
            return edges_dict

        # Should work with dict mapping columns to edge lists
        edges_dict = {
            0: [0.0, 1.0, 2.0],
            "feature_1": [10.0, 20.0, 30.0, 40.0],
        }
        assert check_bin_edges_dict(edges_dict) == edges_dict

    def test_bin_reps_type(self) -> None:
        """Test BinReps type alias."""

        def check_bin_reps(reps: BinReps) -> BinReps:
            return reps

        # Should work with lists of floats
        reps = [0.5, 1.5, 2.5]
        assert check_bin_reps(reps) == reps

    def test_bin_reps_dict_type(self) -> None:
        """Test BinRepsDict type alias."""

        def check_bin_reps_dict(reps_dict: BinRepsDict) -> BinRepsDict:
            return reps_dict

        # Should work with dict mapping columns to representative lists
        reps_dict = {
            0: [0.5, 1.5],
            "feature_1": [15.0, 25.0, 35.0],
        }
        assert check_bin_reps_dict(reps_dict) == reps_dict

    def test_optional_bin_types(self) -> None:
        """Test optional bin type aliases."""

        def check_optional_edges(edges: OptionalBinEdgesDict) -> OptionalBinEdgesDict:
            return edges

        def check_optional_reps(reps: OptionalBinRepsDict) -> OptionalBinRepsDict:
            return reps

        # Should work with None
        assert check_optional_edges(None) is None
        assert check_optional_reps(None) is None

        # Should work with actual dicts
        edges = {0: [1.0, 2.0]}
        reps = {0: [1.5]}
        assert check_optional_edges(edges) == edges
        assert check_optional_reps(reps) == reps


class TestFlexibleBinningTypes:
    """Test flexible binning type aliases."""

    def test_flexible_bin_def_type(self) -> None:
        """Test FlexibleBinDef type alias."""

        def check_flexible_bin_def(bin_def: FlexibleBinDef) -> FlexibleBinDef:
            return bin_def

        # Should work with various formats
        singleton = 5.0
        interval = (1.0, 3.0)

        assert check_flexible_bin_def(singleton) == singleton
        assert check_flexible_bin_def(interval) == interval

    def test_flexible_bin_defs_type(self) -> None:
        """Test FlexibleBinDefs type alias."""

        def check_flexible_bin_defs(defs: FlexibleBinDefs) -> FlexibleBinDefs:
            return defs

        # Should work with list of flexible bin definitions
        defs = [1.0, (2.0, 4.0), 5.0, (6.0, 8.0)]
        assert check_flexible_bin_defs(defs) == defs

    def test_flexible_bin_spec_type(self) -> None:
        """Test FlexibleBinSpec type alias."""

        def check_flexible_bin_spec(spec: FlexibleBinSpec) -> FlexibleBinSpec:
            return spec

        # Should work with dict mapping columns to flexible bin definitions
        spec = {
            0: [1.0, (2.0, 4.0)],
            "feature_1": [(0.0, 1.0), 2.0, (3.0, 5.0)],
        }
        assert check_flexible_bin_spec(spec) == spec

    def test_optional_flexible_bin_spec_type(self) -> None:
        """Test OptionalFlexibleBinSpec type alias."""

        def check_optional_spec(spec: OptionalFlexibleBinSpec) -> OptionalFlexibleBinSpec:
            return spec

        # Should work with None
        assert check_optional_spec(None) is None

        # Should work with actual spec
        spec = {0: [1.0, (2.0, 3.0)]}
        assert check_optional_spec(spec) == spec


class TestResultTypes:
    """Test result type aliases."""

    def test_interval_bin_calculation_result_type(self) -> None:
        """Test IntervalBinCalculationResult type alias."""

        def check_result(result: IntervalBinCalculationResult) -> IntervalBinCalculationResult:
            return result

        # Should work with tuple of (edges, representatives)
        edges = [0.0, 1.0, 2.0]
        reps = [0.5, 1.5]
        result = (edges, reps)

        assert check_result(result) == result
        assert len(result) == 2

    def test_flexible_bin_calculation_result_type(self) -> None:
        """Test FlexibleBinCalculationResult type alias."""

        def check_result(result: FlexibleBinCalculationResult) -> FlexibleBinCalculationResult:
            return result

        # Should work with tuple of (bin_defs, representatives)
        bin_defs = [1.0, (2.0, 4.0)]
        reps = [1.0, 3.0]
        result = (bin_defs, reps)

        assert check_result(result) == result
        assert len(result) == 2

    def test_bin_count_dict_type(self) -> None:
        """Test BinCountDict type alias."""

        def check_count_dict(counts: BinCountDict) -> BinCountDict:
            return counts

        # Should work with dict mapping columns to integers
        counts = {
            0: 3,
            "feature_1": 5,
            "feature_2": 2,
        }
        assert check_count_dict(counts) == counts


class TestParameterTypes:
    """Test parameter-related type aliases."""

    def test_guidance_columns_type(self) -> None:
        """Test GuidanceColumns type alias."""

        def check_guidance(guidance: GuidanceColumns) -> GuidanceColumns:
            return guidance

        # Should work with various formats
        assert check_guidance(None) is None
        assert check_guidance([0, 1]) == [0, 1]
        assert check_guidance(("a", "b")) == ("a", "b")
        assert check_guidance(5) == 5
        assert check_guidance("target") == "target"

    def test_fit_params_type(self) -> None:
        """Test FitParams type alias."""

        def check_fit_params(params: FitParams) -> FitParams:
            return params

        # Should work with dict mapping strings to any values
        params = {
            "sample_weight": np.array([1, 2, 3]),
            "eval_set": [(np.array([[1, 2]]), np.array([0]))],
            "early_stopping_rounds": 10,
        }
        assert check_fit_params(params) == params

    def test_joint_params_type(self) -> None:
        """Test JointParams type alias."""

        def check_joint_params(params: JointParams) -> JointParams:
            return params

        # Should work with dict mapping strings to any values
        params = {
            "X": np.array([[1, 2], [3, 4]]),
            "y": np.array([0, 1]),
            "sample_weight": np.array([1, 1]),
        }
        assert check_joint_params(params) == params


class TestTypeConsistency:
    """Test that type aliases are consistent and work together."""

    def test_edges_and_reps_consistency(self) -> None:
        """Test that edges and representatives types work together."""
        # Create consistent edges and representatives
        edges_dict: BinEdgesDict = {
            0: [0.0, 1.0, 2.0, 3.0],
            1: [10.0, 20.0, 30.0],
        }

        reps_dict: BinRepsDict = {
            0: [0.5, 1.5, 2.5],  # 3 representatives for 4 edges (3 bins)
            1: [15.0, 25.0],  # 2 representatives for 3 edges (2 bins)
        }

        # Check that they have consistent structure
        for col in edges_dict:
            assert col in reps_dict
            n_bins = len(edges_dict[col]) - 1
            assert len(reps_dict[col]) == n_bins

    def test_flexible_spec_consistency(self) -> None:
        """Test flexible bin spec consistency."""
        spec: FlexibleBinSpec = {
            0: [1.0, (2.0, 4.0), 5.0],  # Mix of singletons and intervals
            "feature": [(0.0, 1.0), (2.0, 3.0)],  # Only intervals
        }

        # Each column should have a list of bin definitions
        for _col, defs in spec.items():
            assert isinstance(defs, list)
            assert len(defs) > 0
            for bin_def in defs:
                # Each definition should be either a number or tuple
                assert isinstance(bin_def, int | float | tuple)

    def test_result_types_match_inputs(self) -> None:
        """Test that result types match their corresponding input types."""
        # Interval binning result should match BinEdges and BinReps
        edges: BinEdges = [0.0, 1.0, 2.0]
        reps: BinReps = [0.5, 1.5]
        interval_result: IntervalBinCalculationResult = (edges, reps)

        # Flexible binning result should match FlexibleBinDefs and BinReps
        defs: FlexibleBinDefs = [1.0, (2.0, 3.0)]
        flex_reps: BinReps = [1.0, 2.5]
        flexible_result: FlexibleBinCalculationResult = (defs, flex_reps)

        # Both should work
        assert len(interval_result) == 2
        assert len(flexible_result) == 2

    def test_column_types_across_contexts(self) -> None:
        """Test that ColumnId works consistently across different contexts."""
        # Use same column identifiers across different type contexts
        col_int: ColumnId = 0
        col_str: ColumnId = "feature_name"

        # Should work in all contexts
        edges_dict: BinEdgesDict = {col_int: [1.0, 2.0], col_str: [3.0, 4.0]}
        reps_dict: BinRepsDict = {col_int: [1.5], col_str: [3.5]}
        flex_spec: FlexibleBinSpec = {col_int: [1.0], col_str: [(3.0, 4.0)]}
        count_dict: BinCountDict = {col_int: 1, col_str: 1}

        # All should have same keys
        assert set(edges_dict.keys()) == set(reps_dict.keys())
        assert set(edges_dict.keys()) == set(flex_spec.keys())
        assert set(edges_dict.keys()) == set(count_dict.keys())


class TestTypeHints:
    """Test type hint behavior and edge cases."""

    def test_any_type_flexibility(self) -> None:
        """Test that Any type allows flexibility where needed."""

        # FlexibleBinDef is Any, so should accept various types
        def accepts_flexible_def(val: FlexibleBinDef) -> bool:  # pylint: disable=unused-argument
            return True

        # Should accept various types without type errors
        assert accepts_flexible_def(1)
        assert accepts_flexible_def(1.5)
        assert accepts_flexible_def((1, 2))
        assert accepts_flexible_def("string")  # Even strings (though not semantically correct)
        assert accepts_flexible_def([1, 2, 3])

    def test_none_handling_in_optional_types(self) -> None:
        """Test None handling in optional types."""
        # Optional types should handle None gracefully
        optional_columns: OptionalColumnList = None
        optional_edges: OptionalBinEdgesDict = None
        optional_reps: OptionalBinRepsDict = None
        optional_spec: OptionalFlexibleBinSpec = None
        guidance: GuidanceColumns = None

        # All should be None
        assert optional_columns is None
        assert optional_edges is None
        assert optional_reps is None
        assert optional_spec is None
        assert guidance is None

    def test_empty_collections_handling(self) -> None:
        """Test handling of empty collections."""
        # Empty but valid collections
        empty_columns: ColumnList = []
        empty_edges_dict: BinEdgesDict = {}
        empty_reps_dict: BinRepsDict = {}
        empty_spec: FlexibleBinSpec = {}
        empty_params: FitParams = {}

        # All should be empty but not None
        assert len(empty_columns) == 0
        assert len(empty_edges_dict) == 0
        assert len(empty_reps_dict) == 0
        assert len(empty_spec) == 0
        assert len(empty_params) == 0
