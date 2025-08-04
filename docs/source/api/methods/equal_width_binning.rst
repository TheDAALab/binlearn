Equal Width Binning
====================

Overview
--------

Equal width binning divides the feature range into bins of equal width. This is one of the most straightforward binning methods, creating intervals of consistent size across the feature's range.

Usage
-----

.. code-block:: python

   import numpy as np
   from binlearn.methods import EqualWidthBinning
   
   # Create sample data
   X = np.random.rand(100, 1) * 100
   
   # Apply equal width binning
   binner = EqualWidthBinning(n_bins=5)
   X_binned = binner.fit_transform(X)
   
   print(f"Bin edges: {binner.bin_edges_}")

Key Features
~~~~~~~~~~~~

* **Consistent Intervals**: All bins have the same width
* **Simple Implementation**: Easy to understand and interpret
* **Boundary Control**: Explicit control over bin boundaries
* **sklearn Compatible**: Full integration with sklearn pipelines

Parameters
----------

* ``n_bins`` : int, default=5
    Number of bins to create
* ``preserve_dataframe`` : bool, default=False
    Whether to preserve DataFrame structure in output
* ``feature_names`` : list, optional
    Custom names for output features

See Also
--------

* :doc:`equal_frequency_binning` - For balanced bin populations
* :doc:`singleton_binning` - For categorical data encoding
