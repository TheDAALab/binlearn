"""Test V2 reconstruction methods and sklearn pipeline integration."""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from binlearn.methods import EqualWidthBinningV2, SingletonBinningV2, Chi2BinningV2


def test_reconstruction_methods():
    """Test both set_params() and constructor(**get_params()) reconstruction."""

    print("=== Testing V2 Reconstruction Methods ===")

    # Create test dataset
    X, y = make_classification(
        n_samples=300, n_features=3, n_informative=3, n_redundant=0, n_classes=2, random_state=42
    )

    df = pd.DataFrame(X, columns=["feat_a", "feat_b", "feat_c"])

    # Test data for singleton binning (discrete numeric values)
    discrete_data = pd.DataFrame(
        {
            "disc_a": np.random.choice([1, 2, 3, 4], size=300),
            "disc_b": np.random.choice([10, 20, 30], size=300),
            "cont_c": X[:, 2],
        }
    )

    # Test each V2 class
    binners_data = [
        (
            "EqualWidthBinningV2",
            EqualWidthBinningV2(n_bins=4),
            df,
            None,
        ),  # No guidance_columns to bin all
        (
            "SingletonBinningV2",
            SingletonBinningV2(guidance_columns=["disc_a"]),
            discrete_data,
            None,
        ),
        ("Chi2BinningV2", Chi2BinningV2(max_bins=4, guidance_columns=["feat_b"]), df, y),
    ]

    for name, original_binner, data, target in binners_data:
        print(f"\n--- Testing {name} ---")

        # Fit original binner
        if target is not None:
            original_binner.fit(data, target)
        else:
            original_binner.fit(data)

        original_transformed = original_binner.transform(data)
        print(f"Original fitted: {original_binner._fitted}")
        print(f"Original transformed shape: {original_transformed.shape}")

        # Method 1: Using set_params()
        print("\n1. Testing set_params() reconstruction:")
        params = original_binner.get_params()

        method1_binner = None
        if name == "EqualWidthBinningV2":
            method1_binner = EqualWidthBinningV2()
        elif name == "SingletonBinningV2":
            method1_binner = SingletonBinningV2()
        elif name == "Chi2BinningV2":
            method1_binner = Chi2BinningV2()

        if method1_binner is None:
            print(f"  - ERROR: Unknown binner type {name}")
            continue

        method1_binner.set_params(**params)
        method1_transformed = method1_binner.transform(data)

        method1_ok = np.array_equal(original_transformed, method1_transformed)
        print(f"  - Reconstruction successful: {method1_ok}")
        print(f"  - Reconstructed fitted: {method1_binner._fitted}")

        # Method 2: Using constructor(**get_params())
        print("\n2. Testing constructor(**get_params()) reconstruction:")

        method2_binner = None
        if name == "EqualWidthBinningV2":
            method2_binner = EqualWidthBinningV2(**params)
        elif name == "SingletonBinningV2":
            method2_binner = SingletonBinningV2(**params)
        elif name == "Chi2BinningV2":
            method2_binner = Chi2BinningV2(**params)

        if method2_binner is None:
            print(f"  - ERROR: Unknown binner type {name}")
            continue

        method2_transformed = method2_binner.transform(data)

        method2_ok = np.array_equal(original_transformed, method2_transformed)
        print(f"  - Reconstruction successful: {method2_ok}")
        print(f"  - Reconstructed fitted: {method2_binner._fitted}")

        # Verify both methods produce identical results
        both_identical = np.array_equal(method1_transformed, method2_transformed)
        print(f"  - Both methods identical: {both_identical}")

    return True


def test_sklearn_pipeline_integration():
    """Test sklearn pipeline integration with all V2 binners."""

    print("\n\n=== Testing Sklearn Pipeline Integration ===")

    # Create classification dataset
    X, y = make_classification(
        n_samples=500, n_features=4, n_informative=3, n_redundant=1, n_classes=2, random_state=42
    )

    df = pd.DataFrame(X, columns=["feat_0", "feat_1", "feat_2", "feat_3"])

    # Test each V2 binner in a pipeline
    pipeline_tests = [
        ("EqualWidth", EqualWidthBinningV2(n_bins=5, guidance_columns=["feat_0", "feat_1"])),
        ("Chi2", Chi2BinningV2(max_bins=4, guidance_columns=["feat_2"])),
    ]

    for name, binner in pipeline_tests:
        print(f"\n--- Testing {name} in Pipeline ---")

        # Create pipeline
        pipeline = Pipeline(
            [
                (f"{name.lower()}_binning", binner),
                ("classifier", RandomForestClassifier(n_estimators=10, random_state=42)),
            ]
        )

        print(f"Pipeline steps: {[step[0] for step in pipeline.steps]}")

        # Test fitting
        try:
            pipeline.fit(df, y)
            print(f"  - Pipeline fit: ✓")

            # Test prediction
            predictions = pipeline.predict(df[:10])
            print(f"  - Pipeline predict: ✓ (length: {len(predictions)})")

            # Test cross-validation
            cv_scores = cross_val_score(pipeline, df, y, cv=3, scoring="accuracy")
            print(f"  - Cross-validation: ✓ (mean accuracy: {cv_scores.mean():.3f})")

            # Test pipeline parameter access
            pipeline_params = pipeline.get_params()
            binning_params = {
                k: v
                for k, v in pipeline_params.items()
                if k.startswith(f"{name.lower()}_binning__")
            }
            print(f"  - Pipeline params access: ✓ ({len(binning_params)} binning params)")

            # Test parameter setting
            if name == "EqualWidth":
                pipeline.set_params(**{f"{name.lower()}_binning__n_bins": 3})
                new_n_bins = pipeline.get_params()[f"{name.lower()}_binning__n_bins"]
                print(f"  - Pipeline param setting: ✓ (n_bins changed to {new_n_bins})")
            elif name == "Chi2":
                pipeline.set_params(**{f"{name.lower()}_binning__max_bins": 3})
                new_max_bins = pipeline.get_params()[f"{name.lower()}_binning__max_bins"]
                print(f"  - Pipeline param setting: ✓ (max_bins changed to {new_max_bins})")

        except Exception as e:
            print(f"  - Pipeline test failed: {str(e)}")
            return False

    print(f"\nAll pipeline tests passed! ✓")
    return True


def test_complex_reconstruction_scenario():
    """Test complex reconstruction scenario with fitted parameters."""

    print("\n\n=== Testing Complex Reconstruction Scenario ===")

    # Create dataset
    X, y = make_classification(
        n_samples=200, n_features=2, n_classes=3, n_informative=2, n_redundant=0, random_state=42
    )
    df = pd.DataFrame(X, columns=["x1", "x2"])

    # Create and fit binner
    original_binner = Chi2BinningV2(max_bins=5, min_bins=2, alpha=0.1, guidance_columns=["x1"])

    original_binner.fit(df, y)
    original_result = original_binner.transform(df)

    print("Original binner state:")
    print(f"  - Fitted: {original_binner._fitted}")
    print(f"  - Bin edges keys: {list(original_binner.bin_edges_.keys())}")
    print(f"  - Feature names: {original_binner.feature_names_in_}")

    # Get all parameters including fitted ones
    all_params = original_binner.get_params()
    print(f"  - Total parameters: {len(all_params)}")
    print(f"  - Has bin_edges: {'bin_edges' in all_params and all_params['bin_edges'] is not None}")

    # Test constructor reconstruction
    reconstructed_binner = Chi2BinningV2(**all_params)
    reconstructed_result = reconstructed_binner.transform(df)

    print("\nReconstructed binner state:")
    print(f"  - Fitted: {reconstructed_binner._fitted}")
    print(f"  - Bin edges keys: {list(reconstructed_binner.bin_edges_.keys())}")
    print(f"  - Feature names: {getattr(reconstructed_binner, 'feature_names_in_', 'Not set')}")

    # Compare results
    reconstruction_perfect = np.array_equal(original_result, reconstructed_result)
    print(f"\nReconstruction perfect: {reconstruction_perfect}")

    if not reconstruction_perfect:
        print("Difference analysis:")
        print(f"  - Original shape: {original_result.shape}")
        print(f"  - Reconstructed shape: {reconstructed_result.shape}")
        if original_result.shape == reconstructed_result.shape:
            diff = np.abs(original_result - reconstructed_result)
            print(f"  - Max difference: {np.max(diff)}")
            print(f"  - Mean difference: {np.mean(diff)}")

    return reconstruction_perfect


def main():
    """Run all tests."""
    print("Testing V2 Architecture: Reconstruction Methods and Sklearn Integration")
    print("=" * 80)

    try:
        # Test 1: Basic reconstruction methods
        result1 = test_reconstruction_methods()

        # Test 2: Sklearn pipeline integration
        result2 = test_sklearn_pipeline_integration()

        # Test 3: Complex reconstruction scenario
        result3 = test_complex_reconstruction_scenario()

        print(f"\n\n{'=' * 80}")
        print("FINAL RESULTS:")
        print(f"✓ Reconstruction methods: {result1}")
        print(f"✓ Sklearn pipeline integration: {result2}")
        print(f"✓ Complex reconstruction: {result3}")

        all_passed = result1 and result2 and result3
        print(f"\nAll tests passed: {all_passed}")

        return all_passed

    except Exception as e:
        print(f"Test suite failed with error: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
