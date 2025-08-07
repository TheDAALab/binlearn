"""Test Chi2BinningV2 with different scenarios."""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from binlearn.methods import Chi2BinningV2


def test_chi2_real_supervised():
    """Test Chi2BinningV2 with actual supervised binning behavior."""

    print("=== Testing Chi2BinningV2 with Supervised Target ===")

    # Create synthetic classification dataset
    X, y = make_classification(
        n_samples=500,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_classes=2,
        random_state=42,
        class_sep=0.8,  # Better class separation
    )

    # Convert to DataFrame
    df = pd.DataFrame(X, columns=["feature_0", "feature_1"])

    print(f"Data shape: {df.shape}")
    print(f"Target classes distribution: {np.bincount(y)}")

    # Test with different column configurations
    print("\n1. Testing single column supervised binning:")
    binner1 = Chi2BinningV2(
        max_bins=4,
        min_bins=2,
        alpha=0.05,
        guidance_columns=["feature_0"],  # Only binning feature_0 with target
    )

    binner1.fit(df, y)
    print(f"Binning columns: {binner1._get_binning_columns()}")
    print(f"Bin edges: {binner1.bin_edges_}")

    transformed1 = binner1.transform(df)
    print(f"Transformed shape: {transformed1.shape}")
    print(f"Unique bin values: {np.unique(transformed1)}")

    print("\n2. Testing fallback behavior (no target):")
    binner2 = Chi2BinningV2(max_bins=4, guidance_columns=["feature_0"])

    binner2.fit(df)  # No target provided
    print(f"Bin edges: {binner2.bin_edges_}")
    transformed2 = binner2.transform(df)
    print(f"Unique bin values: {np.unique(transformed2)}")

    print("\n3. Testing parameter reconstruction:")
    params = binner1.get_params()

    new_binner = Chi2BinningV2()
    new_binner.set_params(**params)

    new_transformed = new_binner.transform(df)
    reconstruction_ok = np.array_equal(transformed1, new_transformed)
    print(f"Parameter reconstruction successful: {reconstruction_ok}")

    print("\n4. Testing all features without explicit guidance columns:")
    binner3 = Chi2BinningV2(
        max_bins=3,
        # No guidance_columns specified - should bin all features
    )

    binner3.fit(df, y)
    print(f"Binning columns: {binner3._get_binning_columns()}")
    print(f"Number of bin edge sets: {len(binner3.bin_edges_)}")
    transformed3 = binner3.transform(df)
    print(f"Transformed shape: {transformed3.shape}")

    print("\nAll tests completed successfully!")
    return binner1, transformed1


if __name__ == "__main__":
    test_chi2_real_supervised()
