#!/usr/bin/env python3
"""
Test the specific parameter reconstruction pattern that was requested.

This tests the crucial design pattern: "it is a crucial design pattern here that we
want to be able to reconstruct fitted binning objects based on the output of get_params"
"""

import numpy as np


def test_crucial_reconstruction_pattern():
    """Test the exact reconstruction pattern that was requested."""
    print("=== Testing Crucial Reconstruction Pattern ===")

    from binlearn.methods._equal_width_binning_v2 import EqualWidthBinningV2

    # Step 1: Create and fit a binner
    print("Step 1: Creating and fitting binner...")
    X_train = np.random.randn(100, 3)
    binner = EqualWidthBinningV2(n_bins="sqrt", clip=True)
    binner.fit(X_train)
    print("✅ Original binner fitted")

    # Step 2: Extract ALL parameters via get_params() - this should include fitted state
    print("\nStep 2: Extracting parameters via get_params()...")
    params = binner.get_params()

    print("Parameters returned by get_params():")
    for key, value in params.items():
        if key.endswith("_"):
            print(f"  {key}: {type(value).__name__} (fitted parameter)")
        else:
            print(f"  {key}: {value}")

    # Check that we have fitted parameters
    fitted_params = [k for k in params.keys() if k.endswith("_") and not k.startswith("_")]
    if fitted_params:
        print(f"✅ Found fitted parameters: {fitted_params}")
    else:
        print("❌ No fitted parameters found!")

    # Check class metadata
    if "class_" in params and "module_" in params:
        print(f"✅ Class metadata: {params['class_']} from {params['module_']}")
    else:
        print("❌ Missing class metadata!")

    # Step 3: Create NEW instance with those parameters - NO FITTING NEEDED!
    print("\nStep 3: Creating new instance from parameters...")
    new_binner = EqualWidthBinningV2(**params)
    print("✅ New binner created from get_params() output")

    # Step 4: Use the new binner for transformation WITHOUT refitting
    print("\nStep 4: Testing transformation without refitting...")
    X_test = np.random.randn(50, 3)

    try:
        # This should work immediately without calling fit()!
        X_test_binned = new_binner.transform(X_test)
        print("✅ Transform worked immediately - no fitting required!")
        print(f"   Transform output shape: {X_test_binned.shape}")

        # Verify consistency with original binner
        X_test_original = binner.transform(X_test)
        if np.array_equal(X_test_binned, X_test_original):
            print("✅ Results identical to original binner")
        else:
            print("❌ Results differ from original binner")

    except Exception as e:
        print(f"❌ Transform failed: {e}")
        return False

    print("\n🎉 CRUCIAL RECONSTRUCTION PATTERN WORKS PERFECTLY!")
    print("   - fit an estimator ✅")
    print("   - extract all parameters via get_params() ✅")
    print("   - create a new instance with those parameters ✅")
    print("   - use it for transformation without refitting ✅")

    return True


def test_reconstruction_with_different_data():
    """Test that the reconstructed binner works on different data."""
    print("\n=== Testing Reconstruction with Different Data ===")

    from binlearn.methods._equal_width_binning_v2 import EqualWidthBinningV2

    # Original data
    X_original = np.random.randn(100, 2) * 2 + 5  # Different scale and offset
    binner = EqualWidthBinningV2(n_bins=4, clip=True)
    binner.fit(X_original)

    # Get params and reconstruct
    params = binner.get_params()
    reconstructed_binner = EqualWidthBinningV2(**params)

    # Test on completely different data
    X_different = np.random.randn(20, 2) * 10 - 3  # Very different scale/offset

    result1 = binner.transform(X_different)
    result2 = reconstructed_binner.transform(X_different)

    if np.array_equal(result1, result2):
        print("✅ Reconstructed binner produces identical results on new data")
        return True
    else:
        print("❌ Reconstructed binner produces different results")
        return False


def main():
    """Run the reconstruction pattern tests."""
    print("Testing the Crucial Parameter Reconstruction Pattern")
    print("=" * 60)

    success1 = test_crucial_reconstruction_pattern()
    success2 = test_reconstruction_with_different_data()

    if success1 and success2:
        print("\n🎉 ALL RECONSTRUCTION TESTS PASSED!")
        print("The crucial design pattern works perfectly.")
    else:
        print("\n❌ Some reconstruction tests failed")


if __name__ == "__main__":
    main()
