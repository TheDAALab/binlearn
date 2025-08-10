#!/usr/bin/env python3
"""
Test script to verify the refactoring improvements are working.

Tests:
1. validate_random_state utility function
2. allow_fallback parameter functionality
3. Error handling with new ConfigurationError format
"""

import numpy as np
from binlearn.methods import GaussianMixtureBinning, KMeansBinning, DBSCANBinning
from binlearn.utils import validate_random_state, ConfigurationError


def test_validate_random_state():
    """Test the validate_random_state utility function."""
    print("Testing validate_random_state...")

    # Valid cases
    try:
        validate_random_state(42)
        validate_random_state(None)
        validate_random_state(0)
        print("✓ Valid random_state values accepted")
    except Exception as e:
        print(f"✗ Valid random_state values rejected: {e}")
        return False

    # Invalid cases
    test_cases = [
        ("string", "must be an integer or None"),
        (-1, "must be non-negative"),
        (3.14, "must be an integer or None"),
    ]

    for invalid_value, expected_msg in test_cases:
        try:
            validate_random_state(invalid_value)
            print(f"✗ Invalid random_state {invalid_value} was accepted")
            return False
        except ConfigurationError as e:
            if expected_msg in str(e):
                print(f"✓ Invalid random_state {invalid_value} properly rejected")
            else:
                print(f"✗ Wrong error message for {invalid_value}: {e}")
                return False
        except Exception as e:
            print(f"✗ Wrong exception type for {invalid_value}: {type(e).__name__}")
            return False

    return True


def test_fallback_control():
    """Test the allow_fallback parameter functionality."""
    print("\nTesting allow_fallback parameter...")

    # Create problematic data (too few unique values)
    X = np.array([[1, 1, 1, 2, 2]]).T

    # Test GaussianMixtureBinning
    try:
        gmm = GaussianMixtureBinning(n_components=10, allow_fallback=False)
        gmm.fit(X)
        print("✗ GMM should have failed with allow_fallback=False")
        return False
    except ConfigurationError as e:
        if "allow_fallback=True" in str(e):
            print("✓ GMM properly rejected with allow_fallback=False")
        else:
            print(f"✗ GMM error message missing fallback suggestion: {e}")
            return False
    except Exception as e:
        print(f"✗ GMM raised wrong exception type: {type(e).__name__}")
        return False

    # Test with allow_fallback=True (should work)
    try:
        gmm = GaussianMixtureBinning(n_components=3, allow_fallback=True)
        gmm.fit(X)
        print("✓ GMM works with allow_fallback=True")
    except Exception as e:
        print(f"✗ GMM failed even with allow_fallback=True: {e}")
        return False

    # Test KMeansBinning
    try:
        kmeans = KMeansBinning(n_bins=10, allow_fallback=False)
        kmeans.fit(X)
        print("✗ KMeans should have failed with allow_fallback=False")
        return False
    except ConfigurationError as e:
        if "allow_fallback=True" in str(e):
            print("✓ KMeans properly rejected with allow_fallback=False")
        else:
            print(f"✗ KMeans error message missing fallback suggestion: {e}")
            return False
    except Exception as e:
        print(f"✗ KMeans raised wrong exception type: {type(e).__name__}")
        return False

    # Test DBSCANBinning with insufficient clusters
    try:
        dbscan = DBSCANBinning(eps=0.1, min_bins=10, allow_fallback=False)
        dbscan.fit(X)
        print("✗ DBSCAN should have failed with allow_fallback=False")
        return False
    except ConfigurationError as e:
        if "allow_fallback=True" in str(e):
            print("✓ DBSCAN properly rejected with allow_fallback=False")
        else:
            print(f"✗ DBSCAN error message missing fallback suggestion: {e}")
            return False
    except Exception as e:
        print(f"✗ DBSCAN raised wrong exception type: {type(e).__name__}")
        return False

    return True


def test_error_handling():
    """Test the new ConfigurationError format."""
    print("\nTesting new error handling...")

    # Test invalid random_state in GaussianMixtureBinning
    try:
        gmm = GaussianMixtureBinning(random_state="invalid")
        print("✗ GMM should have failed with invalid random_state")
        return False
    except ConfigurationError as e:
        if "random_state" in str(e) and "integer or None" in str(e):
            print("✓ GMM properly rejects invalid random_state")
        else:
            print(f"✗ GMM error message format unexpected: {e}")
            return False
    except Exception as e:
        print(f"✗ GMM raised wrong exception type: {type(e).__name__}")
        return False

    return True


def main():
    """Run all tests."""
    print("Running refactoring tests...\n")

    all_passed = True

    if not test_validate_random_state():
        all_passed = False

    if not test_fallback_control():
        all_passed = False

    if not test_error_handling():
        all_passed = False

    print(f"\n{'='*50}")
    if all_passed:
        print("🎉 All refactoring tests passed!")
    else:
        print("❌ Some tests failed!")
    print(f"{'='*50}")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
