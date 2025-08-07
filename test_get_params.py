#!/usr/bin/env python3
"""Test to show get_params output for fitted EqualWidthBinning."""

import numpy as np
import pandas as pd
from binlearn.methods import EqualWidthBinning


def test_get_params():
    """Test get_params output for fitted EqualWidthBinning."""
    print("=" * 60)
    print("Testing get_params() for fitted EqualWidthBinning")
    print("=" * 60)

    # Create sample data
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]])

    print("Sample data:")
    print(X)
    print()

    # Test EqualWidthBinning
    binner = EqualWidthBinning(n_bins=3)
    print("Before fitting:")
    params_before = binner.get_params()
    print(f"Parameters: {list(params_before.keys())}")
    for key, value in params_before.items():
        print(f"  {key}: {value}")
    print()

    # Fit the binner
    binner.fit(X)

    print("After fitting:")
    params_after = binner.get_params()
    print(f"Parameters: {list(params_after.keys())}")
    for key, value in params_after.items():
        if hasattr(value, "shape"):  # Handle numpy arrays
            print(f"  {key}: array with shape {value.shape}")
            print(f"    {value}")
        elif isinstance(value, dict):
            print(f"  {key}: dict with {len(value)} entries")
            for k, v in value.items():
                if hasattr(v, "shape"):
                    print(f"    {k}: array with shape {v.shape} = {v}")
                else:
                    print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    print()

    # Test fitted state attributes
    print("Fitted state attributes:")
    fitted_attrs = [attr for attr in dir(binner) if attr.endswith("_") and not attr.startswith("_")]
    for attr in fitted_attrs:
        value = getattr(binner, attr)
        if hasattr(value, "shape"):
            print(f"  {attr}: array with shape {value.shape}")
        elif isinstance(value, dict):
            print(f"  {attr}: dict with {len(value)} entries")
        else:
            print(f"  {attr}: {value}")


if __name__ == "__main__":
    test_get_params()
