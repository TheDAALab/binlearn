SingletonBinning
================

Overview
--------

SingletonBinning creates binary indicator features for unique values in categorical or discrete data. This method is particularly useful for encoding categorical variables as binary features, similar to one-hot encoding but with additional flexibility and controls.

🆕 **New Feature**: SingletonBinning replaces the previous OneHotBinning with improved functionality and clearer naming.

Usage
-----

.. code-block:: python

   import numpy as np
   from binlearn.methods import SingletonBinning
   
   # Create categorical data
   X = np.array([['A'], ['B'], ['A'], ['C'], ['B']])
   
   # Apply singleton binning
   binner = SingletonBinning()
   X_binned = binner.fit_transform(X)
   
   print(f"Original shape: {X.shape}")
   print(f"Binned shape: {X_binned.shape}")
   print(f"Categories: {binner.categories_}")

Key Features
~~~~~~~~~~~~

* **Binary Indicators**: Creates one binary feature per unique value
* **Sparse Output**: Optional sparse matrix output for memory efficiency
* **Category Limits**: Control over maximum number of categories to encode
* **Missing Value Handling**: Configurable treatment of missing values
* **sklearn Compatible**: Full integration with sklearn pipelines

Parameters
----------

* ``max_categories`` : int, optional
    Maximum number of categories to encode
* ``sparse`` : bool, default=False
    Whether to return sparse matrix output
* ``handle_unknown`` : {'error', 'ignore'}, default='error'
    How to handle unknown categories during transform
* ``preserve_dataframe`` : bool, default=False
    Whether to preserve DataFrame structure in output

Examples
--------

Categorical Data Encoding
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Encode categorical variables
   categories = np.array([['red'], ['blue'], ['red'], ['green']])
   binner = SingletonBinning()
   encoded = binner.fit_transform(categories)
   # Result: 3 binary columns (one per unique color)

High Cardinality Data
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Limit categories for high cardinality data
   binner = SingletonBinning(max_categories=10)
   encoded = binner.fit_transform(high_cardinality_data)

See Also
--------

* :doc:`manual_binning` - For mixed interval and singleton binning
* :doc:`equal_width_binning` - For continuous data binning
