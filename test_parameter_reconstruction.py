#!/usr/bin/env python3
"""Test parameter reconstruction in V2 binning methods."""

import numpy as np
from binlearn.methods._equal_width_binning_v2 import EqualWidthBinningV2
from binlearn.methods._singleton_binning_v2 import SingletonBinningV2


def test_parameter_reconstruction():
    """Test complete parameter reconstruction workflow."""
    print("=" * 80)
    print("Testing Parameter Reconstruction in V2 Binning Methods")
    print("=" * 80)

    # Sample data
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]])

    print("Sample data:")
    print(X)
    print()

    # Test EqualWidthBinningV2
    print("=" * 50)
    print("Testing EqualWidthBinningV2")
    print("=" * 50)

    # Create and fit original binner
    binner_original = EqualWidthBinningV2(n_bins=3)
    X_original = binner_original.fit_transform(X)

    print("Original transform:")
    print(X_original)

    # Get parameters
    params = binner_original.get_params()
    print(f"\nParameters from get_params(): {list(params.keys())}")
    for key, value in params.items():
        if key.startswith("bin_"):
            print(f"  {key}: {value}")

    # Reconstruct
    print(f"\nReconstructing with parameters...")
    binner_reconstructed = EqualWidthBinningV2(**params)

    print(
        f"Reconstructed instance fitted status: {getattr(binner_reconstructed, '_fitted', False)}"
    )

    # Test if it can transform immediately
    try:
        X_reconstructed = binner_reconstructed.transform(X)
        print("Reconstructed transform:")
        print(X_reconstructed)
        print(f"Results match: {np.array_equal(X_original, X_reconstructed)}")

        # Test inverse transform
        X_inverse_original = binner_original.inverse_transform(X_original)
        X_inverse_reconstructed = binner_reconstructed.inverse_transform(X_reconstructed)
        print(f"Inverse results match: {np.allclose(X_inverse_original, X_inverse_reconstructed)}")

    except Exception as e:
        print(f"ERROR during reconstruction: {e}")

    print()

    # Test SingletonBinningV2
    print("=" * 50)
    print("Testing SingletonBinningV2")
    print("=" * 50)

    # Create discrete data for singleton binning
    X_discrete = np.array([[1.0, 10.0], [2.0, 20.0], [1.0, 10.0], [3.0, 30.0]])

    print("Discrete data for singleton binning:")
    print(X_discrete)

    # Create and fit original binner
    singleton_original = SingletonBinningV2(max_unique_values=10)
    X_singleton_original = singleton_original.fit_transform(X_discrete)

    print("Original singleton transform:")
    print(X_singleton_original)

    # Get parameters
    singleton_params = singleton_original.get_params()
    print(f"\nSingleton parameters: {list(singleton_params.keys())}")
    for key, value in singleton_params.items():
        if key.startswith("bin_") or key.startswith("unique_"):
            print(f"  {key}: {value}")

    # Reconstruct
    print(f"\nReconstructing singleton with parameters...")
    singleton_reconstructed = SingletonBinningV2(**singleton_params)

    print(
        f"Singleton reconstructed fitted status: {getattr(singleton_reconstructed, '_fitted', False)}"
    )

    # Test if it can transform immediately
    try:
        X_singleton_reconstructed = singleton_reconstructed.transform(X_discrete)
        print("Reconstructed singleton transform:")
        print(X_singleton_reconstructed)
        print(
            f"Singleton results match: {np.array_equal(X_singleton_original, X_singleton_reconstructed)}"
        )

        # Test inverse transform
        X_singleton_inverse_original = singleton_original.inverse_transform(X_singleton_original)
        X_singleton_inverse_reconstructed = singleton_reconstructed.inverse_transform(
            X_singleton_reconstructed
        )
        print(
            f"Singleton inverse results match: {np.allclose(X_singleton_inverse_original, X_singleton_inverse_reconstructed)}"
        )

    except Exception as e:
        print(f"ERROR during singleton reconstruction: {e}")


if __name__ == "__main__":
    test_parameter_reconstruction()
