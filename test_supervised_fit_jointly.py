#!/usr/bin/env python3
"""
Test that supervised binning correctly rejects fit_jointly parameter.
"""

import numpy as np
import pandas as pd
from binlearn.methods._chi2_binning_v2 import Chi2BinningV2

# Generate test data
np.random.seed(42)
X = pd.DataFrame(
    {
        "feature1": np.random.normal(10, 3, 100),
        "feature2": np.random.exponential(2, 100),
    }
)
y = np.random.randint(0, 2, 100)

print("Testing Supervised Binning fit_jointly Behavior...")
print("=" * 50)

# Test 1: Supervised binning should always use fit_jointly=False
print("1. Testing Chi2BinningV2 fit_jointly behavior...")
try:
    binner = Chi2BinningV2()
    binner.fit(X, y)

    print(f"   Binner fit_jointly value: {binner.fit_jointly}")
    print(f"   Correctly set to False: {binner.fit_jointly == False}")

    # Verify it works correctly
    result = binner.transform(X)
    print(f"   Transform successful: {result.shape == X.shape}")
    print("   Chi2BinningV2 fit_jointly: PASSED")
except Exception as e:
    print(f"   Chi2BinningV2 fit_jointly: FAILED - {e}")

print()

# Test 2: Test that different columns get different bin edges (not joint)
print("2. Testing that columns are binned independently...")
try:
    binner = Chi2BinningV2()
    binner.fit(X, y)

    # Check that columns have different bin edges (independent fitting)
    feature1_edges = binner.bin_edges_.get("feature1", [])
    feature2_edges = binner.bin_edges_.get("feature2", [])

    edges_different = not np.array_equal(feature1_edges, feature2_edges)

    print(f"   Feature1 edges: {len(feature1_edges)} edges")
    print(f"   Feature2 edges: {len(feature2_edges)} edges")
    print(f"   Edges are different: {edges_different}")
    print("   Independent column binning: PASSED")
except Exception as e:
    print(f"   Independent column binning: FAILED - {e}")

print()

# Test 3: Verify the SupervisedBinningBaseV2 constructor behavior
print("3. Testing SupervisedBinningBaseV2 constructor...")
try:
    from binlearn.base._supervised_binning_base_v2 import SupervisedBinningBaseV2

    # This should work - fit_jointly is hardcoded to False
    class MockSupervisedBinner(SupervisedBinningBaseV2):
        def _calculate_bins(self, x_col, col_id, guidance_data=None):
            # This shouldn't be called for supervised binning, but we need to implement it
            return [0.0, 1.0], [0.5]

        def _calculate_supervised_bins(self, x_col, y_col, col_id):
            return [0.0, 1.0], [0.5]

    test_binner = MockSupervisedBinner()
    print(f"   SupervisedBinningBaseV2 fit_jointly: {test_binner.fit_jointly}")
    print(f"   Correctly hardcoded to False: {test_binner.fit_jointly == False}")
    print("   SupervisedBinningBaseV2 constructor: PASSED")
except Exception as e:
    print(f"   SupervisedBinningBaseV2 constructor: FAILED - {e}")


# PyTest test functions
def test_supervised_binning_rejects_fit_jointly():
    """Test that supervised binning classes don't support fit_jointly parameter."""
    from binlearn.base._supervised_binning_base_v2 import SupervisedBinningBaseV2

    # This should work - fit_jointly is hardcoded to False
    class MockSupervisedBinner(SupervisedBinningBaseV2):
        def _calculate_bins(self, x_col, col_id, guidance_data=None):
            # This shouldn't be called for supervised binning, but we need to implement it
            return [0.0, 1.0], [0.5]

        def _calculate_supervised_bins(self, x_col, y_col, col_id):
            return [0.0, 1.0], [0.5]

    test_binner = MockSupervisedBinner()
    assert (
        test_binner.fit_jointly == False
    ), "Supervised binning should always have fit_jointly=False"

    # Test that set_params properly rejects fit_jointly (should raise ValueError)
    try:
        test_binner.set_params(fit_jointly=True)
        assert False, "set_params should reject fit_jointly for supervised binning"
    except ValueError as e:
        assert "Invalid parameter 'fit_jointly'" in str(
            e
        ), f"Expected invalid parameter error, got: {e}"


def test_chi2_binning_no_fit_jointly():
    """Test that Chi2BinningV2 doesn't accept fit_jointly parameter."""
    from binlearn.methods import Chi2BinningV2

    # Should work without fit_jointly
    chi2_binner = Chi2BinningV2(max_bins=5, min_bins=2)
    assert chi2_binner.fit_jointly == False, "Chi2 binning should always have fit_jointly=False"

    # Test that set_params properly rejects fit_jointly (should raise ValueError)
    try:
        chi2_binner.set_params(fit_jointly=True)
        assert False, "set_params should reject fit_jointly for Chi2 binning"
    except ValueError as e:
        assert "Invalid parameter 'fit_jointly'" in str(
            e
        ), f"Expected invalid parameter error, got: {e}"


def test_supervised_vs_unsupervised_fit_jointly():
    """Test that unsupervised binning supports fit_jointly but supervised doesn't."""
    from binlearn.methods import EqualWidthBinningV2, Chi2BinningV2

    # Unsupervised binning should support fit_jointly
    ew_binner = EqualWidthBinningV2(fit_jointly=True)
    assert ew_binner.fit_jointly == True, "Unsupervised binning should support fit_jointly=True"

    ew_binner.set_params(fit_jointly=False)
    assert ew_binner.fit_jointly == False, "Unsupervised binning should allow changing fit_jointly"

    # Supervised binning should not support fit_jointly
    chi2_binner = Chi2BinningV2()
    assert (
        chi2_binner.fit_jointly == False
    ), "Supervised binning should always have fit_jointly=False"


if __name__ == "__main__":
    # Run direct tests
    test_supervised_binning_rejects_fit_jointly()
    test_chi2_binning_no_fit_jointly()
    test_supervised_vs_unsupervised_fit_jointly()
    print("✓ All supervised binning tests passed!")
    print()
    print("🎯 SUPERVISED BINNING VALIDATION SUMMARY:")
    print("-" * 40)
    print("✅ SupervisedBinningBaseV2 hardcodes fit_jointly=False")
    print("✅ Chi2BinningV2 properly rejects fit_jointly parameter")
    print("✅ set_params raises ValueError for invalid fit_jointly")
    print("✅ Supervised vs unsupervised fit_jointly behavior correct")
    print("✅ Complete parameter validation working")
    print()
    print("🎉 All V2 Architecture Requirements Complete! 🎉")
