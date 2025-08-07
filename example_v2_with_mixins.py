#!/usr/bin/env python3
"""
Example of creating clean binning methods using the V2 architecture with utility mixins.

This demonstrates how the V2 architecture + utility mixins make it much easier
to create concrete binning implementations with minimal boilerplate code.
"""

import numpy as np
import pandas as pd
from binlearn.base import GeneralBinningBaseV2, EdgeBasedBinningMixin, BinningUtilsMixin


class EqualWidthBinningV2(EdgeBasedBinningMixin, BinningUtilsMixin, GeneralBinningBaseV2):
    """V2 implementation of equal-width binning with utility mixins.

    This demonstrates how the V2 architecture + utility mixins drastically
    reduce the implementation complexity for concrete binning methods.
    """

    def __init__(self, n_bins=5, bin_edges=None, **kwargs):
        super().__init__(**kwargs)
        self.n_bins = n_bins
        self.bin_edges = bin_edges  # Parameter for reconstruction

        # Initialize fitted parameter storage
        self.bin_edges_ = bin_edges or {}

        # Mark as fitted if bin_edges were provided
        if bin_edges:
            self._fitted = True

    def _fit_per_column_independently(self, X, columns, guidance_data=None, **fit_params):
        """Fit equal-width bins independently for each column."""
        self._validate_n_bins(X.shape[1] if X.ndim > 1 else 1)

        for i, col in enumerate(columns):
            column_data = X[:, i] if X.ndim > 1 else X

            # Handle constant columns
            min_val, max_val = column_data.min(), column_data.max()
            if min_val == max_val:
                bin_edges, _ = self._handle_constant_column(column_data, col)
            else:
                # Create equal-width bins
                bin_edges = np.linspace(min_val, max_val, self.n_bins + 1)

            self.bin_edges_[col] = bin_edges

    def _fit_jointly_across_columns(self, X, columns, **fit_params):
        """Fit parameters jointly (same as independent for equal-width)."""
        self._fit_per_column_independently(X, columns, **fit_params)

    # Note: _transform_columns_to_bins and _inverse_transform_bins_to_values
    # are inherited from EdgeBasedBinningMixin - no need to implement!


def demonstrate_v2_with_mixins():
    """Demonstrate the V2 architecture with utility mixins."""
    print("=" * 70)
    print("V2 Architecture + Utility Mixins Demonstration")
    print("=" * 70)

    # Create test data
    np.random.seed(42)
    data = np.random.normal(100, 15, 1000)
    df = pd.DataFrame({"score": data, "id": range(len(data))})

    # Create binner with minimal implementation
    print("\n1. Simple binning with utility mixins:")
    print("-" * 40)

    binner = EqualWidthBinningV2(n_bins=5, preserve_dataframe=True, random_state=42)
    binner.fit(df[["score"]])

    # Test functionality
    original_data = df[["score"]].head()
    binned_data = binner.transform(original_data)
    inverse_data = binner.inverse_transform(binned_data)

    print(f"Original:\n{original_data}")
    print(f"Binned:\n{binned_data}")
    print(f"Inverse:\n{inverse_data}")

    print("\n2. Parameter reconstruction:")
    print("-" * 40)

    # Get parameters and reconstruct
    params = binner.get_params()
    print(f"Parameters: {list(params.keys())}")

    # Reconstruct via constructor
    reconstructed = EqualWidthBinningV2(**params)
    reconstructed_result = reconstructed.transform(original_data)

    print(f"Original result:\n{binned_data}")
    print(f"Reconstructed result:\n{reconstructed_result}")

    # Compare values properly (handle DataFrame vs numpy array)
    original_values = binned_data.values if hasattr(binned_data, "values") else binned_data
    reconstructed_values = (
        reconstructed_result.values
        if hasattr(reconstructed_result, "values")
        else reconstructed_result
    )

    print(f"Results match: {np.allclose(original_values, reconstructed_values)}")

    print("\n3. Key benefits of V2 + mixins:")
    print("-" * 40)
    print("✅ Minimal implementation required (just fit methods)")
    print("✅ Automatic transform/inverse transform via mixins")
    print("✅ Complete sklearn compatibility via SklearnIntegrationMixin")
    print("✅ Multi-format I/O via DataHandlingMixin")
    print("✅ Parameter reconstruction workflows")
    print("✅ Built-in validation and error handling")
    print("✅ Clean separation of concerns")

    print("\n" + "=" * 70)
    print("V2 Architecture + Mixins = Maximum Power, Minimal Code!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_v2_with_mixins()
