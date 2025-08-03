"""Tests for constants module."""
from binlearn.utils.constants import ABOVE_RANGE, BELOW_RANGE, MISSING_VALUE


class TestConstants:
    """Test the constant values."""

    def test_missing_value_constant(self):
        """Test MISSING_VALUE constant."""
        assert MISSING_VALUE == -1
        assert isinstance(MISSING_VALUE, int)

    def test_above_range_constant(self):
        """Test ABOVE_RANGE constant."""
        assert ABOVE_RANGE == -2
        assert isinstance(ABOVE_RANGE, int)

    def test_below_range_constant(self):
        """Test BELOW_RANGE constant."""
        assert BELOW_RANGE == -3
        assert isinstance(BELOW_RANGE, int)

    def test_constants_uniqueness(self):
        """Test that all constants have unique values."""
        constants = [MISSING_VALUE, ABOVE_RANGE, BELOW_RANGE]
        assert len(constants) == len(set(constants))

    def test_constants_are_negative(self):
        """Test that all constants are negative integers."""
        assert MISSING_VALUE < 0
        assert ABOVE_RANGE < 0
        assert BELOW_RANGE < 0
