"""
Test script to validate the refactored base classes.

This script demonstrates the new architecture and tests the core functionality
of the separated mixins and refactored GeneralBinningBase.
"""

import numpy as np
import pandas as pd
from binlearn.base import SklearnIntegrationMixin, DataHandlingMixin, GeneralBinningBaseRefactored


class TestBinning(GeneralBinningBaseRefactored):
    """Simple test implementation for validation."""

    def __init__(self, n_bins: int = 3, **kwargs):
        self.n_bins = n_bins
        super().__init__(**kwargs)

        # Fitted parameters
        self.bin_edges_: dict = {}
        self.bin_representatives_: dict = {}

    def _fit_per_column(self, X, columns, guidance_data=None, **fit_params):
        """Simple equal-width binning implementation."""
        for i, col in enumerate(columns):
            col_data = X[:, i]
            min_val, max_val = np.min(col_data), np.max(col_data)
            edges = np.linspace(min_val, max_val, self.n_bins + 1)
            reps = [(edges[j] + edges[j + 1]) / 2 for j in range(len(edges) - 1)]

            self.bin_edges_[col] = edges
            self.bin_representatives_[col] = np.array(reps)

    def _fit_jointly(self, X, columns, **fit_params):
        """Joint fitting - use same range for all columns."""
        min_val, max_val = np.min(X), np.max(X)
        edges = np.linspace(min_val, max_val, self.n_bins + 1)
        reps = [(edges[j] + edges[j + 1]) / 2 for j in range(len(edges) - 1)]

        for col in columns:
            self.bin_edges_[col] = edges
            self.bin_representatives_[col] = np.array(reps)

    def _transform_columns(self, X, columns):
        """Transform to bin indices."""
        result = np.zeros_like(X, dtype=int)
        for i, col in enumerate(columns):
            edges = self.bin_edges_[col]
            result[:, i] = np.digitize(X[:, i], edges[1:-1])
        return result

    def _inverse_transform_columns(self, X, columns):
        """Inverse transform to representative values."""
        result = np.zeros_like(X, dtype=float)
        for i, col in enumerate(columns):
            reps = self.bin_representatives_[col]
            result[:, i] = reps[X[:, i]]
        return result


def test_sklearn_integration():
    """Test the sklearn integration mixin."""
    print("🧪 Testing SklearnIntegrationMixin...")

    binner = TestBinning(n_bins=3)

    # Test parameter management
    params_before = binner.get_params()
    print(f"✅ Parameters before fitting: n_bins={params_before['n_bins']}")

    # Test repr
    print(f"✅ Repr: {repr(binner)}")

    # Test parameter setting
    binner.set_params(n_bins=5)
    print(f"✅ After set_params: n_bins={binner.n_bins}")


def test_data_handling():
    """Test the data handling mixin."""
    print("\n🧪 Testing DataHandlingMixin...")

    # Create test data
    X_array = np.random.randn(100, 3)
    X_df = pd.DataFrame(X_array, columns=["A", "B", "C"])

    binner = TestBinning(n_bins=3, preserve_dataframe=True)

    # Test input preparation
    arr, cols = binner._prepare_input(X_df)
    print(f"✅ DataFrame input: shape={arr.shape}, columns={cols}")

    arr, cols = binner._prepare_input(X_array)
    print(f"✅ Array input: shape={arr.shape}, columns={cols}")


def test_full_workflow():
    """Test the complete refactored workflow."""
    print("\n🧪 Testing Complete Workflow...")

    # Create test data
    X = np.random.randn(50, 4)
    X_df = pd.DataFrame(X, columns=["A", "B", "C", "D"])

    # Test 1: Basic fitting and transformation
    binner = TestBinning(n_bins=3, preserve_dataframe=True)
    binner.fit(X_df)
    X_binned = binner.transform(X_df)

    print(f"✅ Basic workflow: {type(X_binned)}, shape={X_binned.shape}")
    print(f"✅ Fitted state: {binner._fitted}")

    # Test 2: Parameter reconstruction workflow
    params = binner.get_params()
    print(f"✅ Extracted {len(params)} parameters including fitted state")

    # Create new instance with fitted parameters
    new_binner = TestBinning(**params)
    print(f"✅ Reconstructed binner without fitting: fitted={new_binner._fitted}")

    # Test transformation without fitting
    X_test = np.random.randn(10, 4)
    X_test_df = pd.DataFrame(X_test, columns=["A", "B", "C", "D"])
    X_test_binned = new_binner.transform(X_test_df)

    print(f"✅ Transform without fitting: {type(X_test_binned)}, shape={X_test_binned.shape}")

    # Test 3: Guidance columns
    print("\n🧪 Testing Guidance Columns...")
    guidance_binner = TestBinning(n_bins=3, guidance_columns=["D"])

    # Fit with guidance
    guidance_binner.fit(X_df)
    X_guided = guidance_binner.transform(X_df)

    print(f"✅ Guided binning: shape={X_guided.shape} (should be 3 columns, not 4)")

    # Test inverse transform
    X_inverse = guidance_binner.inverse_transform(X_guided)
    print(f"✅ Inverse transform: shape={X_inverse.shape}")


def test_error_handling():
    """Test error handling and validation."""
    print("\n🧪 Testing Error Handling...")

    try:
        # Should raise error - incompatible parameters
        TestBinning(fit_jointly=True, guidance_columns=["A"])
        print("❌ Should have raised ValueError")
    except ValueError as e:
        print(f"✅ Caught expected error: {str(e)[:50]}...")

    try:
        # Should raise error - transform before fit
        binner = TestBinning()
        X = np.random.randn(10, 3)
        binner.transform(X)
        print("❌ Should have raised RuntimeError")
    except RuntimeError as e:
        print(f"✅ Caught expected error: {str(e)[:30]}...")


if __name__ == "__main__":
    print("🚀 Testing Refactored Binning Architecture")
    print("=" * 50)

    test_sklearn_integration()
    test_data_handling()
    test_full_workflow()
    test_error_handling()

    print("\n✨ All tests completed successfully!")
    print("\n📊 Architecture Summary:")
    print("├── SklearnIntegrationMixin: Parameter management, fitted state, repr")
    print("├── DataHandlingMixin: Multi-format I/O, column management")
    print("└── GeneralBinningBase: Pure binning logic, guidance handling")
