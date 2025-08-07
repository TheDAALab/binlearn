#!/usr/bin/env python3
"""
Test script to verify all binning methods work after V2 to non-V2 conversion.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression


def test_all_methods():
    """Test all binning methods for import, instantiation, and basic functionality."""

    # Create test data
    print("Creating test data...")
    X_class, y_class = make_classification(
        n_samples=100, n_features=3, n_classes=3, n_informative=3, n_redundant=0, random_state=42
    )
    X_reg, y_reg = make_regression(n_samples=100, n_features=3, random_state=42)

    # Create positive weights for supervised methods
    weights = np.random.uniform(0.1, 2.0, size=100)  # Positive weights between 0.1 and 2.0

    # Convert to pandas for DataFrame testing
    X_class_df = pd.DataFrame(X_class, columns=[f"feature_{i}" for i in range(3)])
    X_reg_df = pd.DataFrame(X_reg, columns=[f"feature_{i}" for i in range(3)])

    # List all methods to test
    methods_to_test = [
        ("Chi2Binning", {"max_bins": 5}, "supervised", X_class, y_class),
        ("DBSCANBinning", {"eps": 0.5, "min_samples": 5}, "unsupervised", X_reg, None),
        ("EqualFrequencyBinning", {"n_bins": 5}, "unsupervised", X_reg, None),
        ("EqualWidthBinning", {"n_bins": 5}, "unsupervised", X_reg, None),
        (
            "EqualWidthMinimumWeightBinning",
            {"n_bins": 5, "minimum_weight": 0.1},
            "supervised",
            X_class,
            weights,
        ),
        ("GaussianMixtureBinning", {"n_components": 3}, "unsupervised", X_reg, None),
        ("IsotonicBinning", {}, "supervised", X_reg, y_reg),
        ("KMeansBinning", {"n_bins": 5}, "unsupervised", X_reg, None),
        (
            "ManualFlexibleBinning",
            {
                "bin_spec": {
                    0: [(-np.inf, 0), (0, np.inf)],
                    1: [(-np.inf, 0), (0, np.inf)],
                    2: [(-np.inf, 0), (0, np.inf)],
                }
            },
            "flexible",
            X_reg,
            None,
        ),
        (
            "ManualIntervalBinning",
            {"bin_edges": {0: [-2, 0, 2], 1: [-2, 0, 2], 2: [-2, 0, 2]}},
            "manual",
            X_reg,
            None,
        ),
        ("SingletonBinning", {}, "unsupervised", X_reg, None),
        ("TreeBinning", {"tree_params": {"max_depth": 3}}, "supervised", X_class, y_class),
    ]

    results = {}

    for method_name, params, method_type, X_data, y_data in methods_to_test:
        print(f"\n{'='*60}")
        print(f"Testing {method_name}")
        print(f"{'='*60}")

        try:
            # Test 1: Import
            print(f"1. Testing import of {method_name}...")
            import binlearn.methods as methods_module

            method_class = getattr(methods_module, method_name)
            print(f"   ✓ Import successful: {method_class}")

            # Test 2: Instantiation
            print(f"2. Testing instantiation of {method_name}...")
            if method_type == "manual":
                # Manual methods need specific parameters
                binner = method_class(**params)
            else:
                # Other methods can use default parameters or provided ones
                binner = method_class(**params) if params else method_class()
            print(f"   ✓ Instantiation successful: {binner}")

            # Test 3: Fit
            print(f"3. Testing fit of {method_name}...")
            if method_type == "supervised":
                binner.fit(X_data, y_data)
            else:
                binner.fit(X_data)
            print(f"   ✓ Fit successful")

            # Test 4: Transform
            print(f"4. Testing transform of {method_name}...")
            X_binned = binner.transform(X_data)
            print(f"   ✓ Transform successful. Shape: {X_binned.shape}, dtype: {X_binned.dtype}")

            # Test 5: Fit-transform
            print(f"5. Testing fit_transform of {method_name}...")
            if method_type == "manual":
                binner2 = method_class(**params)
            else:
                binner2 = method_class(**params) if params else method_class()

            if method_type == "supervised":
                X_binned2 = binner2.fit_transform(X_data, y_data)
            else:
                X_binned2 = binner2.fit_transform(X_data)
            print(
                f"   ✓ Fit-transform successful. Shape: {X_binned2.shape}, dtype: {X_binned2.dtype}"
            )

            # Test 6: DataFrame compatibility (if not manual with complex specs)
            if method_type != "manual":
                print(f"6. Testing DataFrame compatibility of {method_name}...")
                # Convert to DataFrame for DataFrame compatibility test
                X_df = (
                    X_class_df
                    if method_type == "supervised" and method_name in ["Chi2Binning", "TreeBinning"]
                    else X_reg_df
                )
                y_df = (
                    y_class
                    if method_type == "supervised" and method_name in ["Chi2Binning", "TreeBinning"]
                    else y_reg
                )

                # Special case for EqualWidthMinimumWeightBinning - use positive weights
                if method_name == "EqualWidthMinimumWeightBinning":
                    y_df = weights

                if method_type == "manual":
                    binner3 = method_class(**params)
                else:
                    binner3 = method_class(**params) if params else method_class()

                if method_type == "supervised":
                    X_binned3 = binner3.fit_transform(X_df, y_df)
                else:
                    X_binned3 = binner3.fit_transform(X_df)
                print(f"   ✓ DataFrame compatibility successful. Type: {type(X_binned3)}")

            results[method_name] = "✓ PASSED"
            print(f"\n🎉 {method_name} - ALL TESTS PASSED!")

        except Exception as e:
            results[method_name] = f"✗ FAILED: {str(e)}"
            print(f"\n❌ {method_name} - FAILED: {str(e)}")
            import traceback

            print(f"Full traceback:\n{traceback.format_exc()}")

    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")

    passed_count = 0
    failed_count = 0

    for method_name, result in results.items():
        print(f"{method_name:35} {result}")
        if result.startswith("✓"):
            passed_count += 1
        else:
            failed_count += 1

    print(
        f"\n📊 RESULTS: {passed_count} passed, {failed_count} failed out of {len(results)} methods"
    )

    if failed_count == 0:
        print("🎉 ALL METHODS WORKING PERFECTLY!")
        return True
    else:
        print(f"⚠️  {failed_count} methods need attention")
        return False


if __name__ == "__main__":
    success = test_all_methods()
    exit(0 if success else 1)
