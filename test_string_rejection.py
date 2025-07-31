#!/usr/bin/env python3
"""Test script to verify string rejection in flexible binning."""

from binning.methods import ManualFlexibleBinning


def test_string_rejection():
    """Test that string bins are properly rejected."""
    print("Testing string rejection in ManualFlexibleBinning...")

    # Test 1: Numeric bins should work
    try:
        bin_spec = {0: [1, 2, 3]}
        binner = ManualFlexibleBinning(bin_spec=bin_spec)
        print("✓ Test 1 PASSED: Numeric bins work correctly")
    except Exception as e:
        print(f"✗ Test 1 FAILED: Numeric bins rejected: {e}")
        return False

    # Test 2: String bins should be rejected
    try:
        bin_spec = {0: ["A", "B", "C"]}
        binner = ManualFlexibleBinning(bin_spec=bin_spec)
        print("✗ Test 2 FAILED: String bins were accepted (should be rejected)")
        return False
    except Exception as e:
        print(f"✓ Test 2 PASSED: String bins correctly rejected: {type(e).__name__}")

    # Test 3: Mixed bins should be rejected
    try:
        bin_spec = {0: [1, "A", (2, 5)]}
        binner = ManualFlexibleBinning(bin_spec=bin_spec)
        print("✗ Test 3 FAILED: Mixed bins were accepted (should be rejected)")
        return False
    except Exception as e:
        print(f"✓ Test 3 PASSED: Mixed bins correctly rejected: {type(e).__name__}")

    print("\nAll tests passed! String support has been successfully removed.")
    return True


if __name__ == "__main__":
    test_string_rejection()
