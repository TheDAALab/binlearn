"""
Test to cover lines 147-148 in _flexible_binning_base.py.

This test specifically targets the branch 147->exit in the _validate_params method
where both bin_spec and bin_representatives are provided and validated together.
"""

import numpy as np
import pytest
from binning.methods import ManualFlexibleBinning
from binning.utils.errors import ConfigurationError


class TestSpecificBranchCoverage:
    """Test to cover the specific uncovered branch 147->exit."""

    def test_validate_params_both_spec_and_representatives_147_148(self):
        """Test lines 147-148: validation of bin_spec and bin_representatives compatibility.

        This test creates a ManualFlexibleBinning instance with both bin_spec and
        bin_representatives provided, which should trigger the validation path on
        lines 147-148 in _validate_params method.
        """

        # Create compatible bin_spec and bin_representatives
        # Using the correct format for flexible binning: scalar values for singletons
        bin_spec = {
            "col1": [1.0, 2.0, 3.0],  # Singleton bins (scalar values)
            "col2": [(0.0, 1.0), (1.0, 2.0)],  # Interval bins (tuples)
        }

        bin_representatives = {
            "col1": [1.0, 2.0, 3.0],  # Representatives for singleton bins
            "col2": [0.5, 1.5],  # Representatives for interval bins (midpoints)
        }

        # This should trigger the validation path on lines 147-148
        # The _validate_params method will be called during initialization
        # and should validate compatibility between bin_spec and bin_representatives
        binner = ManualFlexibleBinning(bin_spec=bin_spec, bin_representatives=bin_representatives)

        # Verify the binner was created successfully
        assert binner is not None
        assert binner.bin_spec == bin_spec
        assert binner.bin_representatives == bin_representatives

        # Now explicitly call _validate_params to ensure it covers the branch
        # This should trigger the validation of both bin_spec and bin_representatives
        binner._validate_params()

        # The validation should have passed, covering the branch 147->exit
        # where both bin_spec and bin_representatives are provided and compatible

    def test_validate_params_incompatible_spec_and_representatives(self):
        """Test validation failure when bin_spec and bin_representatives are incompatible."""

        # Create incompatible bin_spec and bin_representatives
        bin_spec = {
            "col1": [1.0, 2.0, 3.0],  # 3 bins
        }

        bin_representatives = {
            "col1": [1.0, 2.0],  # Only 2 representatives - mismatch!
        }

        # This should trigger a ConfigurationError during validation
        with pytest.raises(ConfigurationError):
            ManualFlexibleBinning(bin_spec=bin_spec, bin_representatives=bin_representatives)

    def test_validate_params_only_bin_spec_provided(self):
        """Test validation when only bin_spec is provided."""

        bin_spec = {
            "col1": [1.0, 2.0],
        }

        # This should work fine - no bin_representatives provided
        binner = ManualFlexibleBinning(bin_spec=bin_spec)
        assert binner is not None

        # Explicitly call validation
        binner._validate_params()

    def test_validate_params_only_bin_representatives_provided(self):
        """Test validation when only bin_representatives is provided."""

        bin_representatives = {
            "col1": [1.0, 2.0],
        }

        # ManualFlexibleBinning requires bin_spec, so test the validation behavior
        # by testing that constructor fails when bin_spec is missing
        with pytest.raises(ConfigurationError, match="bin_spec must be provided and non-empty"):
            ManualFlexibleBinning(bin_representatives=bin_representatives)  # type: ignore
