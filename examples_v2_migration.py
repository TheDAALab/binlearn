#!/usr/bin/env python3
"""
Example of how existing binning methods can be migrated to the new V2 architecture.

This example shows how to convert a traditional binning method to use the
cleaner V2 base classes with their improved separation of concerns.
"""

import numpy as np
import pandas as pd
from binlearn.base import GeneralBinningBaseV2
from sklearn.cluster import KMeans


class EqualWidthBinningV2(GeneralBinningBaseV2):
    """V2 implementation of equal-width binning with cleaner architecture.

    This demonstrates how the V2 architecture simplifies binning implementations
    by handling sklearn integration and data format management automatically.
    """

    def __init__(self, n_bins=5, random_state=None, **kwargs):
        super().__init__(random_state=random_state, **kwargs)
        self.n_bins = n_bins

        # Initialize storage for fitted parameters
        self.bin_edges_ = {}
        self.bin_centers_ = {}

    def _fit_per_column_independently(self, X, columns, guidance_data=None, **fit_params):
        """Fit equal-width bins independently for each column."""
        for i, col in enumerate(columns):
            column_data = X[:, i] if X.ndim > 1 else X

            # Create equal-width bins
            min_val, max_val = column_data.min(), column_data.max()
            if min_val == max_val:
                # Handle constant column
                bin_edges = np.array([min_val - 0.5, max_val + 0.5])
                n_bins_actual = 1
            else:
                bin_edges = np.linspace(min_val, max_val, self.n_bins + 1)
                n_bins_actual = self.n_bins

            # Store fitted parameters
            self.bin_edges_[col] = bin_edges
            self.bin_centers_[col] = (bin_edges[:-1] + bin_edges[1:]) / 2

            print(f"Column {col}: fitted {n_bins_actual} equal-width bins")

    def _fit_jointly_across_columns(self, X, columns, **fit_params):
        """Fit parameters jointly (same as independent for equal-width)."""
        # For equal-width binning, joint fitting is the same as independent
        self._fit_per_column_independently(X, columns, **fit_params)

    def _transform_columns_to_bins(self, X, columns):
        """Transform data to bin indices."""
        if X.ndim == 1 and len(columns) == 1:
            col = columns[0]
            bin_edges = self.bin_edges_[col]
            binned = np.digitize(X, bin_edges) - 1
            return np.clip(binned, 0, len(bin_edges) - 2)

        result = np.zeros_like(X, dtype=int)
        for i, col in enumerate(columns):
            bin_edges = self.bin_edges_[col]
            binned = np.digitize(X[:, i], bin_edges) - 1
            result[:, i] = np.clip(binned, 0, len(bin_edges) - 2)

        return result

    def _inverse_transform_bins_to_values(self, X, columns):
        """Inverse transform using bin centers."""
        if X.ndim == 1 and len(columns) == 1:
            col = columns[0]
            bin_centers = self.bin_centers_[col]
            return bin_centers[X]

        result = np.zeros_like(X, dtype=float)
        for i, col in enumerate(columns):
            bin_centers = self.bin_centers_[col]
            result[:, i] = bin_centers[X[:, i]]

        return result


class KMeansBinningV2(GeneralBinningBaseV2):
    """V2 implementation of KMeans-based binning.

    Demonstrates how more complex binning methods benefit from the V2 architecture's
    clean parameter management and fitted state reconstruction.
    """

    def __init__(self, n_bins=5, random_state=None, **kwargs):
        super().__init__(random_state=random_state, **kwargs)
        self.n_bins = n_bins

        # Initialize storage for fitted parameters
        self.cluster_centers_ = {}
        self.bin_edges_ = {}

    def _fit_per_column_independently(self, X, columns, guidance_data=None, **fit_params):
        """Fit KMeans clustering independently for each column."""
        for i, col in enumerate(columns):
            column_data = X[:, i] if X.ndim > 1 else X

            # Fit KMeans clustering
            kmeans = KMeans(n_clusters=self.n_bins, random_state=self.random_state, n_init=10)
            kmeans.fit(column_data.reshape(-1, 1))

            # Extract cluster centers and create bin boundaries
            centers = kmeans.cluster_centers_.flatten()
            centers = np.sort(centers)

            # Create bin edges as midpoints between centers
            if len(centers) > 1:
                bin_edges = np.concatenate(
                    [
                        [centers[0] - (centers[1] - centers[0]) / 2],  # First edge
                        (centers[:-1] + centers[1:]) / 2,  # Middle edges
                        [centers[-1] + (centers[-1] - centers[-2]) / 2],  # Last edge
                    ]
                )
            else:
                # Single cluster case
                center = centers[0]
                bin_edges = np.array([center - 1, center + 1])

            # Store fitted parameters
            self.cluster_centers_[col] = centers
            self.bin_edges_[col] = bin_edges

            print(f"Column {col}: fitted {len(centers)} KMeans-based bins")

    def _fit_jointly_across_columns(self, X, columns, **fit_params):
        """Fit parameters jointly (same as independent for this example)."""
        self._fit_per_column_independently(X, columns, **fit_params)

    def _transform_columns_to_bins(self, X, columns):
        """Transform data to bin indices."""
        if X.ndim == 1 and len(columns) == 1:
            col = columns[0]
            bin_edges = self.bin_edges_[col]
            binned = np.digitize(X, bin_edges) - 1
            return np.clip(binned, 0, len(bin_edges) - 2)

        result = np.zeros_like(X, dtype=int)
        for i, col in enumerate(columns):
            bin_edges = self.bin_edges_[col]
            binned = np.digitize(X[:, i], bin_edges) - 1
            result[:, i] = np.clip(binned, 0, len(bin_edges) - 2)

        return result

    def _inverse_transform_bins_to_values(self, X, columns):
        """Inverse transform using cluster centers."""
        if X.ndim == 1 and len(columns) == 1:
            col = columns[0]
            centers = self.cluster_centers_[col]
            return centers[X]

        result = np.zeros_like(X, dtype=float)
        for i, col in enumerate(columns):
            centers = self.cluster_centers_[col]
            result[:, i] = centers[X[:, i]]

        return result


def demonstrate_v2_migration():
    """Demonstrate the V2 architecture with improved binning methods."""
    print("=" * 70)
    print("V2 Architecture Migration Demonstration")
    print("=" * 70)

    # Create test data
    np.random.seed(42)
    data = np.random.normal(100, 15, 1000)
    df = pd.DataFrame({"score": data, "category": np.random.choice(["A", "B", "C"], 1000)})

    print("\n1. Equal-Width Binning V2:")
    print("-" * 30)
    ew_binner = EqualWidthBinningV2(n_bins=4, random_state=42)
    ew_binner.fit(df[["score"]])

    # Transform and show results
    binned = ew_binner.transform(df[["score"]].head())
    inverse = ew_binner.inverse_transform(binned)

    print(f"Original: {df['score'].head().values}")
    print(f"Binned: {binned.flatten()}")
    print(f"Inverse: {inverse.flatten()}")

    # Test parameter reconstruction
    params = ew_binner.get_params()
    new_binner = EqualWidthBinningV2()
    new_binner.set_params(**params)

    print(f"Reconstructed transform: {new_binner.transform(df[['score']].head(1)).flatten()}")

    print("\n2. KMeans Binning V2:")
    print("-" * 30)
    kmeans_binner = KMeansBinningV2(n_bins=3, random_state=123)
    kmeans_binner.fit(df[["score"]])

    # Transform and show results
    binned = kmeans_binner.transform(df[["score"]].head())
    inverse = kmeans_binner.inverse_transform(binned)

    print(f"Original: {df['score'].head().values}")
    print(f"Binned: {binned.flatten()}")
    print(f"Inverse: {inverse.flatten()}")

    print("\n3. Multi-format compatibility:")
    print("-" * 30)

    # Test with numpy array
    numpy_result = ew_binner.transform(data[:5].reshape(-1, 1))
    print(f"NumPy input -> {numpy_result.flatten()}")

    # Test with pandas (preserves format if preserve_dataframe=True)
    ew_binner_df = EqualWidthBinningV2(n_bins=4, preserve_dataframe=True)
    ew_binner_df.set_params(**ew_binner.get_params())  # Copy fitted parameters

    df_result = ew_binner_df.transform(df[["score"]].head())
    print(f"DataFrame input -> shape {df_result.shape}, type {type(df_result)}")

    print(f"\n4. String representation:")
    print("-" * 30)
    print(f"EqualWidth: {repr(ew_binner)}")
    print(f"KMeans: {repr(kmeans_binner)}")

    print("\n" + "=" * 70)
    print("V2 Architecture Migration Demonstration Complete!")
    print("Key Benefits:")
    print("- Clean separation of concerns")
    print("- Automatic parameter reconstruction")
    print("- Multi-format data handling")
    print("- Enhanced sklearn compatibility")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_v2_migration()
