#!/usr/bin/env python3
"""Quick comparison test of EqualFrequencyBinning vs EqualWidthBinning."""

import numpy as np
from binning.methods import EqualFrequencyBinning, EqualWidthBinning

# Create skewed test data
np.random.seed(42)
skewed_data = np.random.exponential(1, 1000).reshape(-1, 1)

print("=== Skewed Data Distribution Test ===")
print(f"Data min: {skewed_data.min():.3f}, max: {skewed_data.max():.3f}")
print(f"Data mean: {skewed_data.mean():.3f}, std: {skewed_data.std():.3f}")

# Equal Width Binning
ewb = EqualWidthBinning(n_bins=5)
X_width = ewb.fit_transform(skewed_data)

# Equal Frequency Binning
efb = EqualFrequencyBinning(n_bins=5)
X_freq = efb.fit_transform(skewed_data)

print("\n--- Equal Width Binning ---")
print("Bin edges:", [f"{x:.3f}" for x in ewb._bin_edges[0]])
width_counts = [np.sum(X_width == i) for i in range(5)]
print("Bin counts:", width_counts)
print("Bin count std:", np.std(width_counts))

print("\n--- Equal Frequency Binning ---")
print("Bin edges:", [f"{x:.3f}" for x in efb._bin_edges[0]])
freq_counts = [np.sum(X_freq == i) for i in range(5)]
print("Bin counts:", freq_counts)
print("Bin count std:", np.std(freq_counts))

print(
    f"\nEqual frequency binning should have much lower std deviation in counts: {np.std(freq_counts):.1f} vs {np.std(width_counts):.1f}"
)

# Test that equal frequency binning actually produces equal frequencies
expected_count = len(skewed_data) // 5  # 200
max_deviation = max(abs(count - expected_count) for count in freq_counts)
print(f"Max deviation from expected count ({expected_count}): {max_deviation}")
assert (
    max_deviation <= 1
), f"Equal frequency binning should produce nearly equal counts, got deviation {max_deviation}"

print("\n✅ All tests passed! EqualFrequencyBinning works correctly.")
