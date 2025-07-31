#!/usr/bin/env python3
"""Example script demonstrating EqualFrequencyBinning functionality."""

import numpy as np
from binning.methods import EqualFrequencyBinning, EqualWidthBinning

# Create test data with different distributions
np.random.seed(42)
# Normal distribution
normal_data = np.random.normal(50, 15, 1000)
# Exponential distribution (right-skewed)
exp_data = np.random.exponential(2, 1000)
# Combine into a 2D array
X = np.column_stack([normal_data, exp_data])

print("=== EqualFrequencyBinning vs EqualWidthBinning Comparison ===")
print(f"Data shape: {X.shape}")
print(
    f"Column 1 (normal): min={X[:, 0].min():.2f}, max={X[:, 0].max():.2f}, mean={X[:, 0].mean():.2f}"
)
print(
    f"Column 2 (exponential): min={X[:, 1].min():.2f}, max={X[:, 1].max():.2f}, mean={X[:, 1].mean():.2f}"
)

# Test Equal Frequency Binning
print("\n--- Equal Frequency Binning ---")
efb = EqualFrequencyBinning(n_bins=5)
X_freq_binned = efb.fit_transform(X)

print("Bin edges (column 1):", [f"{x:.2f}" for x in efb._bin_edges[0]])
print("Bin edges (column 2):", [f"{x:.2f}" for x in efb._bin_edges[1]])

# Count observations in each bin for equal frequency
for col in range(2):
    print(f"\nColumn {col+1} bin counts (Equal Frequency):")
    for bin_idx in range(5):
        count = np.sum(X_freq_binned[:, col] == bin_idx)
        print(f"  Bin {bin_idx}: {count} observations")

# Test Equal Width Binning for comparison
print("\n--- Equal Width Binning ---")
ewb = EqualWidthBinning(n_bins=5)
X_width_binned = ewb.fit_transform(X)

print("Bin edges (column 1):", [f"{x:.2f}" for x in ewb._bin_edges[0]])
print("Bin edges (column 2):", [f"{x:.2f}" for x in ewb._bin_edges[1]])

# Count observations in each bin for equal width
for col in range(2):
    print(f"\nColumn {col+1} bin counts (Equal Width):")
    for bin_idx in range(5):
        count = np.sum(X_width_binned[:, col] == bin_idx)
        print(f"  Bin {bin_idx}: {count} observations")

# Test with quantile range to exclude outliers
print("\n--- Equal Frequency with Quantile Range (10%-90%) ---")
efb_quantile = EqualFrequencyBinning(n_bins=4, quantile_range=(0.1, 0.9))
X_quantile_binned = efb_quantile.fit_transform(X)

print("Bin edges (column 1):", [f"{x:.2f}" for x in efb_quantile._bin_edges[0]])
print("Bin edges (column 2):", [f"{x:.2f}" for x in efb_quantile._bin_edges[1]])

# Count observations in each bin (including overflow bins)
for col in range(2):
    print(f"\nColumn {col+1} bin counts (Quantile Range 10%-90%):")
    unique_bins, counts = np.unique(X_quantile_binned[:, col], return_counts=True)
    for bin_idx, count in zip(unique_bins, counts):
        if bin_idx < 0:
            print(f"  Underflow bin: {count} observations")
        elif bin_idx >= 4:
            print(f"  Overflow bin: {count} observations")
        else:
            print(f"  Bin {bin_idx}: {count} observations")
