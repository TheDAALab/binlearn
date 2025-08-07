#!/usr/bin/env python3
"""Test parameter reconstruction with automatic discovery."""

import numpy as np
import sys

sys.path.append(".")


def test_parameter_reconstruction():
    """Test that parameter reconstruction works with automatic discovery."""
    print("=" * 60)
    print("Testing Parameter Reconstruction with Auto-Discovery")
    print("=" * 60)

    try:
        from binlearn.methods._equal_width_binning_v2 import EqualWidthBinningV2

        # Create and fit original binner
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]])
        print("Sample data shape:", X.shape)

        binner = EqualWidthBinningV2(n_bins=3)
        print("Created EqualWidthBinningV2")

        binner.fit(X)
        print("Fitted successfully")

        # Test get_params
        params = binner.get_params()
        print(f"Parameters: {list(params.keys())}")

        # Check if fitted parameters are included
        fitted_param_keys = [
            k for k in params.keys() if k.endswith("_") or k in ["bin_edges", "bin_representatives"]
        ]
        print(f"Fitted parameters found: {fitted_param_keys}")

        # Print actual values to see what's being passed
        for key in fitted_param_keys:
            value = params[key]
            if isinstance(value, dict) and value:
                print(f"  {key}: dict with {len(value)} entries")
                print(f"    Sample: {list(value.keys())[:2]}")
            else:
                print(f"  {key}: {value}")

        # Test reconstruction
        new_binner = EqualWidthBinningV2(**params)
        print("Reconstructed binner successfully")
        print(f"Reconstructed binner _fitted status: {getattr(new_binner, '_fitted', False)}")
        print(f"Reconstructed binner has bin_edges_: {hasattr(new_binner, 'bin_edges_')}")
        if hasattr(new_binner, "bin_edges_"):
            print(f"  bin_edges_: {new_binner.bin_edges_}")
        print(
            f"Reconstructed binner has bin_representatives_: {hasattr(new_binner, 'bin_representatives_')}"
        )
        if hasattr(new_binner, "bin_representatives_"):
            print(f"  bin_representatives_: {new_binner.bin_representatives_}")

        # Test that it works without refitting
        X_original = binner.transform(X)
        X_reconstructed = new_binner.transform(X)

        print(f"Transform results match: {np.array_equal(X_original, X_reconstructed)}")

        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_parameter_reconstruction()
    print("\n" + "=" * 60)
    print("TEST RESULT:", "SUCCESS" if success else "FAILED")
    print("=" * 60)
