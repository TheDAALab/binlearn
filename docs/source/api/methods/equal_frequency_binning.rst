EqualFrequencyBinning
=====================

.. currentmodule:: binlearn.methods

.. autoclass:: EqualFrequencyBinning
   :members:
   :inherited-members:
   :show-inheritance:

Examples
--------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from binlearn.methods import EqualFrequencyBinning
   
   # Create skewed sample data
   np.random.seed(42)
   X = np.random.exponential(2, (1000, 3))  # Exponentially distributed data
   
   # Create equal-frequency binner
   binner = EqualFrequencyBinning(n_bins=4)
   X_binned = binner.fit_transform(X)
   
   print(f"Original shape: {X.shape}")
   print(f"Binned shape: {X_binned.shape}")
   
   # Check bin counts (should be approximately equal)
   unique, counts = np.unique(X_binned[:, 0], return_counts=True)
   print(f"Bin counts for feature 0: {counts}")

Comparison with Equal-Width
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   from binlearn.methods import EqualWidthBinning, EqualFrequencyBinning
   
   # Highly skewed data
   X_skewed = np.random.lognormal(0, 2, (5000, 1))
   
   # Equal-width binning
   ew_binner = EqualWidthBinning(n_bins=5)
   X_ew_binned = ew_binner.fit_transform(X_skewed)
   
   # Equal-frequency binning
   ef_binner = EqualFrequencyBinning(n_bins=5)
   X_ef_binned = ef_binner.fit_transform(X_skewed)
   
   # Compare bin distributions
   ew_counts = np.bincount(X_ew_binned.astype(int).flatten())
   ef_counts = np.bincount(X_ef_binned.astype(int).flatten())
   
   print("Equal-width bin counts:", ew_counts)
   print("Equal-frequency bin counts:", ef_counts)
   
   # Visualize the difference
   fig, axes = plt.subplots(1, 3, figsize=(15, 5))
   
   axes[0].hist(X_skewed, bins=50, alpha=0.7)
   axes[0].set_title('Original Data Distribution')
   
   axes[1].hist(X_ew_binned, bins=5, alpha=0.7, color='red')
   axes[1].set_title('Equal-Width Binning')
   
   axes[2].hist(X_ef_binned, bins=5, alpha=0.7, color='green')
   axes[2].set_title('Equal-Frequency Binning')
   
   plt.tight_layout()
   plt.show()

With DataFrame
~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   
   # Create DataFrame with different distributions
   df = pd.DataFrame({
       'normal': np.random.normal(50, 15, 1000),
       'exponential': np.random.exponential(5, 1000),
       'uniform': np.random.uniform(0, 100, 1000)
   })
   
   # Apply equal-frequency binning
   binner = EqualFrequencyBinning(n_bins=4, preserve_dataframe=True)
   df_binned = binner.fit_transform(df)
   
   # Check that each bin has approximately equal frequency
   for column in df.columns:
       counts = df_binned[column].value_counts().sort_index()
       print(f"\n{column} bin counts:")
       print(counts)
