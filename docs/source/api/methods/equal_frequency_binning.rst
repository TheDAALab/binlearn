Equal Frequency Binning
======================

.. automodule:: binlearn.methods._equal_frequency_binning
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Overview
--------

Equal frequency binning (also called quantile binning) divides the data into bins containing approximately the same number of observations. This method is particularly useful when you want balanced bin populations.

Key Features
~~~~~~~~~~~~

* **Balanced Populations**: Each bin contains roughly the same number of samples
* **Quantile-Based**: Uses data quantiles to determine bin boundaries
* **Robust to Outliers**: Less sensitive to extreme values than equal-width binning
* **sklearn Compatible**: Full integration with sklearn pipelines

Class Documentation
-------------------

.. autoclass:: binlearn.methods.EqualFrequencyBinning
   :members:
   :inherited-members:
   :show-inheritance:
   :special-members: __init__

Examples
--------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from binlearn.methods import EqualFrequencyBinning
   
   # Create sample data with skewed distribution
   X = np.random.exponential(2, (100, 1))
   
   # Apply equal frequency binning
   binner = EqualFrequencyBinning(n_bins=4)
   X_binned = binner.fit_transform(X)
   
   print(f"Bin edges: {binner.bin_edges_}")
   
   # Check bin populations
   unique, counts = np.unique(X_binned, return_counts=True)
   print(f"Bin counts: {counts}")  # Should be roughly equal

Handling Duplicates
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from binlearn.methods import EqualFrequencyBinning
   
   # Data with many duplicate values
   X = np.array([[1], [1], [1], [1], [2], [2], [3], [4], [5], [5]])
   
   binner = EqualFrequencyBinning(n_bins=3, duplicates='drop')
   X_binned = binner.fit_transform(X)
   
   print(f"Unique bin edges: {binner.bin_edges_}")

With DataFrames
~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   from binlearn.methods import EqualFrequencyBinning
   
   # Create DataFrame
   df = pd.DataFrame({
       'skewed_feature': np.random.exponential(1, 100),
       'normal_feature': np.random.normal(0, 1, 100)
   })
   
   # Apply binning
   binner = EqualFrequencyBinning(n_bins=5, preserve_dataframe=True)
   df_binned = binner.fit_transform(df)
   
   # Verify balanced populations
   for col in df_binned.columns:
       counts = df_binned[col].value_counts().sort_index()
       print(f"{col} bin counts: {counts.values}")

Quantile Binning
~~~~~~~~~~~~~~~~

.. code-block:: python

   from binlearn.methods import EqualFrequencyBinning
   
   # Create data with known quantiles
   X = np.random.rand(1000, 1) * 100
   
   # Create quartiles (4 bins)
   binner = EqualFrequencyBinning(n_bins=4)
   X_binned = binner.fit_transform(X)
   
   # Each bin should contain ~25% of data
   print(f"Bin edges correspond to: {np.percentile(X, [0, 25, 50, 75, 100])}")
   print(f"Actual bin edges: {binner.bin_edges_[0]}")

Pipeline Integration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.linear_model import LogisticRegression
   from sklearn.preprocessing import StandardScaler
   from binlearn.methods import EqualFrequencyBinning
   
   # Pipeline with preprocessing
   pipe = Pipeline([
       ('binning', EqualFrequencyBinning(n_bins=10)),
       ('scaling', StandardScaler()),
       ('classifier', LogisticRegression())
   ])
   
   pipe.fit(X_train, y_train)
   predictions = pipe.predict(X_test)

Advanced Usage
~~~~~~~~~~~~~~

.. code-block:: python

   from binlearn.methods import EqualFrequencyBinning
   
   # Multi-column binning with different strategies
   data = pd.DataFrame({
       'continuous': np.random.exponential(2, 100),
       'discrete': np.random.poisson(3, 100),
       'categorical': np.random.choice(['A', 'B', 'C'], 100)
   })
   
   # Only bin continuous columns
   binner = EqualFrequencyBinning(n_bins=5)
   continuous_data = data[['continuous', 'discrete']]
   binned_data = binner.fit_transform(continuous_data)
   
   print(f"Original range: {continuous_data['continuous'].min():.2f} - {continuous_data['continuous'].max():.2f}")
   print(f"Bin boundaries: {[f'{edge:.2f}' for edge in binner.bin_edges_[0]]}")

When to Use Equal Frequency Binning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Use Equal Frequency Binning when:**

* You need balanced bin populations for statistical analysis
* Working with skewed or heavy-tailed distributions
* Downstream algorithms benefit from uniform sample sizes per bin
* You want to create percentile-based features

**Consider alternatives when:**

* Bin boundaries need to be interpretable (use Equal Width)
* You have domain-specific requirements (use Manual Binning)
* Data contains many tied values (may need special handling)

See Also
--------

* :class:`binlearn.methods.EqualWidthBinning` - For equal-width intervals
* :class:`binlearn.methods.KMeansBinning` - For clustering-based binning
* :class:`binlearn.methods.SupervisedBinning` - For target-aware binning
