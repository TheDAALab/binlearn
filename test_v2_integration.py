#!/usr/bin/env python3
"""
Test script to verify the V2 architecture integration works properly within the package.

This test demonstrates the key capabilities of the new V2 architecture:
1. Clean separation of concerns with three focused mixins
2. Multi-format data handling (numpy, pandas)
3. Comprehensive parameter management with fitted parameter reconstruction
4. Sklearn compatibility and proper state management

The V2 architecture provides:
- SklearnIntegrationMixinV2: Enhanced parameter management with fitted parameter reconstruction
- DataHandlingMixinV2: Multi-format I/O with feature name handling
- GeneralBinningBaseV2: Clean binning orchestration focused on binning logic
"""

import numpy as np
import pandas as pd
from binlearn.base import DataHandlingMixinV2, SklearnIntegrationMixinV2, GeneralBinningBaseV2


# Create a simple concrete implementation for testing
class TestBinnerV2(GeneralBinningBaseV2):
    """Simple test binner that creates equal-width bins."""

    def __init__(self, n_bins=5, random_state=None, bin_edges=None):
        # Store the parameters before calling super
        self.n_bins = n_bins
        self.random_state = random_state
        self.bin_edges = bin_edges  # Store as parameter for sklearn compatibility

        # Initialize parent - this now handles the entire constructor chain
        super().__init__(random_state=random_state)

        # Initialize storage for fitted parameters
        self.bin_edges_ = bin_edges or {}  # Store bin edges per column

        # If bin_edges were provided in constructor, mark as fitted
        if bin_edges:
            self._fitted = True

    def _get_potential_fitted_params(self) -> set[str]:
        """Define which parameters can be fitted parameters for this binner."""
        return {"bin_edges"}

    def _fit_per_column_independently(self, X, columns, guidance_data=None, **fit_params):
        """Fit binning parameters independently for each column."""
        print(f"Fitting {len(columns)} columns independently")

        for i, col in enumerate(columns):
            column_data = X[:, i] if X.ndim > 1 else X
            min_val, max_val = column_data.min(), column_data.max()
            bin_edges = np.linspace(min_val, max_val, self.n_bins + 1)
            self.bin_edges_[col] = bin_edges
            print(f"Column {col}: created {self.n_bins} bins with edges: {bin_edges}")

    def _fit_jointly_across_columns(self, X, columns, **fit_params):
        """Fit binning parameters jointly across all columns."""
        print(f"Fitting {len(columns)} columns jointly")

        # For simplicity, use the same logic but could be different
        for i, col in enumerate(columns):
            column_data = X[:, i] if X.ndim > 1 else X
            min_val, max_val = column_data.min(), column_data.max()
            bin_edges = np.linspace(min_val, max_val, self.n_bins + 1)
            self.bin_edges_[col] = bin_edges
            print(f"Column {col}: created {self.n_bins} bins with edges: {bin_edges}")

    def _transform_columns_to_bins(self, X, columns):
        """Transform columns to bin indices."""
        print(f"Transforming {len(columns)} columns to bins")

        if X.ndim == 1 and len(columns) == 1:
            # Single column case
            col = columns[0]
            bin_edges = self.bin_edges_[col]
            binned = np.digitize(X, bin_edges) - 1
            return np.clip(binned, 0, self.n_bins - 1)

        # Multiple columns case
        result = np.zeros_like(X)
        for i, col in enumerate(columns):
            bin_edges = self.bin_edges_[col]
            binned = np.digitize(X[:, i], bin_edges) - 1
            result[:, i] = np.clip(binned, 0, self.n_bins - 1)

        return result

    def _inverse_transform_bins_to_values(self, X, columns):
        """Inverse transform from bin indices to representative values."""
        print(f"Inverse transforming {len(columns)} columns from bins")

        if X.ndim == 1 and len(columns) == 1:
            # Single column case
            col = columns[0]
            bin_edges = self.bin_edges_[col]
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            return bin_centers[X.astype(int)]

        # Multiple columns case
        result = np.zeros_like(X, dtype=float)
        for i, col in enumerate(columns):
            bin_edges = self.bin_edges_[col]
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            result[:, i] = bin_centers[X[:, i].astype(int)]

        return result


def test_v2_architecture():
    """Test the V2 architecture with various data formats."""
    print("=" * 60)
    print("Testing V2 Architecture Integration")
    print("=" * 60)

    # Create test data
    np.random.seed(42)
    data = np.random.normal(100, 15, 1000)

    # Test with numpy array
    print("\n1. Testing with numpy array:")
    print("-" * 30)
    binner = TestBinnerV2(n_bins=3, random_state=42)

    # Fit the binner (reshape to 2D for proper handling)
    data_2d = data.reshape(-1, 1)
    binner.fit(data_2d)

    # Test parameter access
    print(f"Parameters after fitting: {binner.get_params()}")

    # Transform data
    binned = binner.transform(data_2d[:10])
    print(f"Original values: {data[:10]}")
    print(f"Binned values: {binned.flatten()}")

    # Inverse transform
    inverse = binner.inverse_transform(binned)
    print(f"Inverse values: {inverse.flatten()}")

    # Test with pandas DataFrame
    print("\n2. Testing with pandas DataFrame:")
    print("-" * 30)
    df = pd.DataFrame({"score": data, "id": range(len(data))})

    binner2 = TestBinnerV2(n_bins=4, random_state=123)
    binner2.fit(df[["score"]])

    binned_df = binner2.transform(df[["score"]].head())
    print(f"Original DataFrame:\n{df[['score']].head()}")
    print(f"Binned DataFrame:\n{binned_df}")

    # Test parameter reconstruction
    print("\n3. Testing parameter reconstruction:")
    print("-" * 30)
    params = binner.get_params()
    print(f"Extracted parameters: {params}")

    # Create new instance and set parameters
    new_binner = TestBinnerV2(**params)

    # Test that it can transform without fitting (using reconstructed state)
    test_data = np.array([[95], [105], [115]])
    result = new_binner.transform(test_data)
    print(f"Transform with reconstructed parameters: {test_data.flatten()} -> {result.flatten()}")

    print("\n4. Testing string representation:")
    print("-" * 30)
    print(f"Binner representation:\n{repr(binner)}")

    print("\n" + "=" * 60)
    print("V2 Architecture Integration Test Completed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_v2_architecture()
