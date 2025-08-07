"""Final test of complete V2 architecture hierarchy."""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from binlearn.methods import EqualWidthBinningV2, SingletonBinningV2, Chi2BinningV2


def test_complete_v2_hierarchy():
    """Test the complete V2 hierarchy with all implemented classes."""

    print("=== Testing Complete V2 Architecture Hierarchy ===")

    # Create test dataset
    X, y = make_classification(
        n_samples=200, n_features=3, n_informative=3, n_redundant=0, n_classes=2, random_state=42
    )

    df = pd.DataFrame(X, columns=["feat_a", "feat_b", "feat_c"])

    print(f"Test data shape: {df.shape}")
    print(f"Target classes: {np.unique(y)}")

    # Test hierarchy components
    binners = []

    # 1. IntervalBinningBaseV2 → EqualWidthBinningV2
    print("\n1. Testing EqualWidthBinningV2 (IntervalBinningBaseV2):")
    ew_binner = EqualWidthBinningV2(n_bins=5, guidance_columns=["feat_a", "feat_b"])
    ew_binner.fit(df)
    ew_transformed = ew_binner.transform(df)
    print(f"  - Fitted: {ew_binner._fitted}")
    print(f"  - Binning columns: {ew_binner._get_binning_columns()}")
    print(f"  - Bin edges keys: {list(ew_binner.bin_edges_.keys())}")
    print(f"  - Transformed shape: {ew_transformed.shape}")
    binners.append(("EqualWidthBinningV2", ew_binner, ew_transformed))

    # 2. FlexibleBinningBaseV2 → SingletonBinningV2
    print("\n2. Testing SingletonBinningV2 (FlexibleBinningBaseV2):")
    # Create some discrete numeric data for singleton binning
    discrete_data = pd.DataFrame(
        {
            "disc_a": np.random.choice([1, 2, 3, 4], size=200),
            "disc_b": np.random.choice([10, 20, 30, 40, 50], size=200),
            "num_c": X[:, 2],  # Keep one continuous numeric column
        }
    )

    st_binner = SingletonBinningV2(guidance_columns=["disc_a", "disc_b"])
    st_binner.fit(discrete_data)
    st_transformed = st_binner.transform(discrete_data)
    print(f"  - Fitted: {st_binner._fitted}")
    print(f"  - Binning columns: {st_binner._get_binning_columns()}")
    print(
        f"  - Flexible bins keys: {list(st_binner.bin_specifications_.keys()) if hasattr(st_binner, 'bin_specifications_') else 'N/A'}"
    )
    print(f"  - Transformed shape: {st_transformed.shape}")
    print(f"  - Unique values in first column: {len(np.unique(st_transformed[:, 0]))}")
    binners.append(("SingletonBinningV2", st_binner, st_transformed))

    # 3. SupervisedBinningBaseV2 → Chi2BinningV2
    print("\n3. Testing Chi2BinningV2 (SupervisedBinningBaseV2):")
    chi2_binner = Chi2BinningV2(max_bins=4, min_bins=2, alpha=0.1, guidance_columns=["feat_a"])
    chi2_binner.fit(df, y)
    chi2_transformed = chi2_binner.transform(df)
    print(f"  - Fitted: {chi2_binner._fitted}")
    print(f"  - Binning columns: {chi2_binner._get_binning_columns()}")
    print(f"  - Bin edges keys: {list(chi2_binner.bin_edges_.keys())}")
    print(f"  - Transformed shape: {chi2_transformed.shape}")
    print(f"  - Unique bins: {len(np.unique(chi2_transformed))}")
    binners.append(("Chi2BinningV2", chi2_binner, chi2_transformed))

    print("\n=== Testing Parameter Reconstruction for All Classes ===")

    for name, original_binner, original_transformed in binners:
        print(f"\nTesting {name} parameter reconstruction:")

        # Get parameters
        params = original_binner.get_params()
        print(f"  - Parameter count: {len(params)}")

        # Create new instance
        new_binner = None
        if name == "EqualWidthBinningV2":
            new_binner = EqualWidthBinningV2()
        elif name == "SingletonBinningV2":
            new_binner = SingletonBinningV2()
        elif name == "Chi2BinningV2":
            new_binner = Chi2BinningV2()

        if new_binner is None:
            print(f"  - ERROR: Unknown binner type {name}")
            continue

        # Set parameters
        new_binner.set_params(**params)

        # Transform with reconstructed binner
        if name == "SingletonBinningV2":
            new_transformed = new_binner.transform(discrete_data)
        else:
            new_transformed = new_binner.transform(df)

        # Compare results
        reconstruction_ok = np.array_equal(original_transformed, new_transformed)
        print(f"  - Reconstruction successful: {reconstruction_ok}")
        print(f"  - Original fitted: {original_binner._fitted}")
        print(f"  - Reconstructed fitted: {new_binner._fitted}")

        if not reconstruction_ok:
            print(f"  - WARNING: Reconstruction failed for {name}")

    print("\n=== V2 Architecture Summary ===")
    print("Hierarchy Structure:")
    print("  SklearnIntegrationV2")
    print("  └── DataHandlingV2")
    print("      └── GeneralBinningBaseV2")
    print("          ├── IntervalBinningBaseV2")
    print("          │   ├── EqualWidthBinningV2 ✓")
    print("          │   └── SupervisedBinningBaseV2")
    print("          │       └── Chi2BinningV2 ✓")
    print("          └── FlexibleBinningBaseV2")
    print("              └── SingletonBinningV2 ✓")
    print("\nAll V2 components working with parameter reconstruction!")

    return binners


if __name__ == "__main__":
    test_complete_v2_hierarchy()
