Basic Binning Examples
======================

This section demonstrates the fundamental concepts of data binning using binlearn.

Simple Equal-Width Binning
---------------------------

The most straightforward binning approach divides data into equal-width intervals:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from binlearn.methods import EqualWidthBinning
   
   # Generate sample data
   np.random.seed(42)
   data = np.random.normal(50, 15, 1000).reshape(-1, 1)
   
   # Create and apply binning
   binner = EqualWidthBinning(n_bins=5)
   data_binned = binner.fit_transform(data)
   
   # Examine results
   print(f"Original data range: [{data.min():.2f}, {data.max():.2f}]")
   print(f"Bin edges: {binner.bin_edges_[0]}")
   print(f"Unique bin values: {np.sort(np.unique(data_binned))}")
   
   # Plot histogram
   plt.figure(figsize=(12, 4))
   plt.subplot(1, 2, 1)
   plt.hist(data, bins=30, alpha=0.7, edgecolor='black')
   plt.title('Original Data Distribution')
   plt.xlabel('Value')
   plt.ylabel('Frequency')
   
   plt.subplot(1, 2, 2)
   plt.hist(data_binned, bins=5, alpha=0.7, edgecolor='black')
   plt.title('Binned Data Distribution')
   plt.xlabel('Bin')
   plt.ylabel('Frequency')
   plt.tight_layout()
   plt.show()

Equal-Frequency Binning
-----------------------

Creates bins with approximately equal numbers of observations:

.. code-block:: python

   from binlearn.methods import EqualFrequencyBinning
   
   # Use the same data
   freq_binner = EqualFrequencyBinning(n_bins=5)
   data_freq_binned = freq_binner.fit_transform(data)
   
   # Compare bin populations
   unique_vals, counts = np.unique(data_freq_binned, return_counts=True)
   
   print("Equal-Frequency Binning Results:")
   for val, count in zip(unique_vals, counts):
       print(f"Bin {val}: {count} samples")
   
   print(f"\\nBin edges: {freq_binner.bin_edges_[0]}")

Working with Multiple Features
------------------------------

Binning multiple columns simultaneously:

.. code-block:: python

   # Multi-dimensional data
   X = np.random.rand(500, 3) * 100
   feature_names = ['feature_1', 'feature_2', 'feature_3']
   
   # Apply binning to all features
   multi_binner = EqualWidthBinning(n_bins=4)
   X_binned = multi_binner.fit_transform(X)
   
   print("Multi-feature binning:")
   print(f"Original shape: {X.shape}")
   print(f"Binned shape: {X_binned.shape}")
   
   # Examine bin edges for each feature
   for i, name in enumerate(feature_names):
       print(f"{name} bin edges: {multi_binner.bin_edges_[i]}")

Handling Edge Cases
-------------------

binlearn gracefully handles common edge cases:

.. code-block:: python

   # Data with missing values
   data_with_nan = np.array([1, 2, np.nan, 4, 5, np.nan, 7, 8]).reshape(-1, 1)
   
   nan_binner = EqualWidthBinning(n_bins=3)
   binned_with_nan = nan_binner.fit_transform(data_with_nan)
   
   print("Handling NaN values:")
   print(f"Original: {data_with_nan.flatten()}")
   print(f"Binned: {binned_with_nan.flatten()}")
   print("NaN values are preserved in the output")
   
   # Constant data (all same values)
   constant_data = np.full((100, 1), 42.0)
   
   try:
       constant_binner = EqualWidthBinning(n_bins=5)
       constant_binned = constant_binner.fit_transform(constant_data)
       print(f"\\nConstant data binning:")
       print(f"All values mapped to bin: {np.unique(constant_binned)}")
   except Exception as e:
       print(f"Error with constant data: {e}")

Inspecting Binning Results
--------------------------

Use built-in tools to analyze your binning results:

.. code-block:: python

   from binlearn.utils.inspection import inspect_bins
   
   # Apply binning
   inspector_binner = EqualWidthBinning(n_bins=4)
   inspected_data = np.random.exponential(2, 1000).reshape(-1, 1)
   inspected_binned = inspector_binner.fit_transform(inspected_data)
   
   # Get detailed statistics
   bin_stats = inspect_bins(inspected_data, inspected_binned, 
                           inspector_binner.bin_edges_)
   
   print("Bin Statistics:")
   print(bin_stats)

This covers the fundamental binning concepts. Next, explore method-specific examples to learn about advanced features and use cases.
