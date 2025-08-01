#!/usr/bin/env python3
"""
Test that parameter validation occurs in the constructor.
"""

import sys
import os

sys.path.insert(0, os.path.abspath("."))

from binning.methods import EqualWidthBinning


def test_constructor_validation():
    """Test that invalid parameters are caught at construction time."""

    print("Testing constructor parameter validation...")

    # Test 1: Invalid preserve_dataframe type
    try:
        binner = EqualWidthBinning(preserve_dataframe="invalid")
        print("❌ FAIL: Should have raised TypeError for invalid preserve_dataframe")
        return False
    except TypeError as e:
        print(f"✅ PASS: Caught TypeError for preserve_dataframe: {e}")

    # Test 2: Invalid fit_jointly type
    try:
        binner = EqualWidthBinning(fit_jointly="invalid")
        print("❌ FAIL: Should have raised TypeError for invalid fit_jointly")
        return False
    except TypeError as e:
        print(f"✅ PASS: Caught TypeError for fit_jointly: {e}")

    # Test 3: Invalid guidance_columns type
    try:
        binner = EqualWidthBinning(guidance_columns=123.45)  # float is invalid
        print("❌ FAIL: Should have raised TypeError for invalid guidance_columns")
        return False
    except TypeError as e:
        print(f"✅ PASS: Caught TypeError for guidance_columns: {e}")

    # Test 4: Incompatible guidance_columns + fit_jointly
    try:
        binner = EqualWidthBinning(guidance_columns=["col1"], fit_jointly=True)
        print("❌ FAIL: Should have raised ValueError for incompatible params")
        return False
    except ValueError as e:
        print(f"✅ PASS: Caught ValueError for incompatible params: {e}")

    # Test 5: Valid parameters should work
    try:
        binner = EqualWidthBinning(preserve_dataframe=True, fit_jointly=False)
        print("✅ PASS: Valid parameters accepted")
    except Exception as e:
        print(f"❌ FAIL: Valid parameters rejected: {e}")
        return False

    print("All constructor validation tests passed!")
    return True


def test_set_params_validation():
    """Test that set_params also validates parameters."""

    print("\nTesting set_params parameter validation...")

    # Create a valid binner first
    binner = EqualWidthBinning()

    # Test invalid parameter change
    try:
        binner.set_params(preserve_dataframe="invalid")
        print("❌ FAIL: Should have raised TypeError in set_params")
        return False
    except TypeError as e:
        print(f"✅ PASS: set_params caught TypeError: {e}")

    # Test valid parameter change
    try:
        binner.set_params(preserve_dataframe=True)
        print("✅ PASS: set_params accepted valid parameter")
    except Exception as e:
        print(f"❌ FAIL: set_params rejected valid parameter: {e}")
        return False

    print("All set_params validation tests passed!")
    return True


if __name__ == "__main__":
    success1 = test_constructor_validation()
    success2 = test_set_params_validation()

    if success1 and success2:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)
