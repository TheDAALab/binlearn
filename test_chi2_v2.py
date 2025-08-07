"""Test Chi2BinningV2 implementation."""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from binlearn.methods import Chi2BinningV2


def test_chi2_binning_v2():
    """Test Chi2BinningV2 with synthetic classification data."""

    # Create synthetic classification dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=3,
        n_informative=3,  # Increased to support 3 classes
        n_redundant=0,  # Reduced to keep total features manageable
        n_classes=3,
        random_state=42,
    )

    # Convert to DataFrame for easier testing
    df = pd.DataFrame(X, columns=["feature_0", "feature_1", "feature_2"])

    print("Testing Chi2BinningV2...")
    print(f"Data shape: {df.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Unique classes: {np.unique(y)}")

    # Test basic initialization
    binner = Chi2BinningV2(
        max_bins=5,
        min_bins=2,
        alpha=0.05,
        initial_bins=20,
        guidance_columns=["feature_0", "feature_1"],
        fit_jointly=False,
    )

    print(f"\nInitialized Chi2BinningV2: {binner}")
    print(f"Parameters: {binner.get_params()}")

    # Test fitting with target data
    print("\nFitting with target data...")
    binner.fit(df, y)

    print(f"Fitted: {binner._fitted}")
    print(f"Feature names: {binner.feature_names_in_}")
    print(f"Binning columns: {binner._get_binning_columns()}")

    # Check bin edges
    if hasattr(binner, "bin_edges_"):
        print(f"Bin edges: {binner.bin_edges_}")
    if hasattr(binner, "bin_representatives_"):
        print(f"Bin representatives: {binner.bin_representatives_}")

    # Test transformation
    print("\nTesting transformation...")
    transformed = binner.transform(df)
    print(f"Transformed shape: {transformed.shape}")
    print(f"Transformed type: {type(transformed)}")
    if hasattr(transformed, "head"):
        print(f"Transformed sample:\n{transformed.head()}")
    else:
        print(f"Transformed sample (first 5 rows):\n{transformed[:5]}")

    # Test parameter reconstruction
    print("\nTesting parameter reconstruction...")
    params = binner.get_params()
    print(f"All parameters: {params}")

    # Create new binner from parameters
    new_binner = Chi2BinningV2()
    new_binner.set_params(**params)

    # Test that reconstruction produces same results
    new_transformed = new_binner.transform(df)

    # Compare results
    if hasattr(transformed, "values"):
        transformed_array = transformed.values
        new_transformed_array = (
            new_transformed.values if hasattr(new_transformed, "values") else new_transformed
        )
    else:
        transformed_array = transformed
        new_transformed_array = new_transformed

    are_equal = np.allclose(transformed_array, new_transformed_array, equal_nan=True)
    print(f"Reconstruction successful: {are_equal}")

    # Test fitted parameter preservation
    print(f"Original fitted: {binner._fitted}")
    print(f"Reconstructed fitted: {new_binner._fitted}")

    return binner, transformed


if __name__ == "__main__":
    test_chi2_binning_v2()
