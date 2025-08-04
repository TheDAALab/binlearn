EqualWidthBinning
=================

.. currentmodule:: binlearn.methods

.. autoclass:: EqualWidthBinning
   :members:
   :inherited-members:
   :show-inheritance:

Examples
--------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from binlearn.methods import EqualWidthBinning
   
   # Create sample data
   X = np.random.rand(1000, 3)
   
   # Create and fit binner
   binner = EqualWidthBinning(n_bins=5)
   X_binned = binner.fit_transform(X)
   
   print(f"Original shape: {X.shape}")
   print(f"Binned shape: {X_binned.shape}")
   print(f"Bin edges: {binner.bin_edges_}")

With Custom Range
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Specify custom range for binning
   binner = EqualWidthBinning(
       n_bins=4,
       bin_range=(0, 10)  # Force range from 0 to 10
   )
   
   X_binned = binner.fit_transform(X)

With DataFrame Preservation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   
   # Create DataFrame
   df = pd.DataFrame({
       'feature1': np.random.normal(0, 1, 100),
       'feature2': np.random.exponential(2, 100)
   })
   
   # Preserve DataFrame format
   binner = EqualWidthBinning(n_bins=3, preserve_dataframe=True)
   df_binned = binner.fit_transform(df)
   
   print(type(df_binned))  # pandas.DataFrame
   print(df_binned.columns.tolist())  # ['feature1', 'feature2']
