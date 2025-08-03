Quick Start
===========

This guide will help you get started with the binlearn library quickly.

Installation
------------

Install the binlearn library using pip:

.. code-block:: bash

   pip install binning

Basic Example
-------------

Here's a simple example showing how to use equal-width binning:

.. code-block:: python

   import numpy as np
   from binlearn.methods import EqualWidthBinning
   
   # Create sample data
   X = np.array([[1.2, 10.5], 
                 [2.1, 15.3], 
                 [3.7, 8.9], 
                 [4.2, 12.1], 
                 [5.8, 9.7]])
   
   # Create and apply binning
   binner = EqualWidthBinning(n_bins=3)
   X_binned = binner.fit_transform(X)
   
   print("Original data:")
   print(X)
   print("\nBinned data:")
   print(X_binned)
   print("\nBin edges:")
   print(binner.bin_edges_)

Available Binning Methods
-------------------------

The framework provides several binning methods for different use cases:

**Interval-based Methods:**

* :class:`~binlearn.methods.EqualWidthBinning` - Equal-width intervals
* :class:`~binlearn.methods.EqualFrequencyBinning` - Equal-frequency (quantile) bins  
* :class:`~binlearn.methods.KMeansBinning` - K-means clustering-based bins
* :class:`~binlearn.methods.EqualWidthMinimumWeightBinning` - Weight-constrained equal-width bins

**Flexible Methods:**

* :class:`~binlearn.methods.ManualFlexibleBinning` - Custom bin definitions with mixed types
* :class:`~binlearn.methods.ManualIntervalBinning` - Custom interval bins

**Categorical Methods:**

* :class:`~binlearn.methods.OneHotBinning` - One-hot encoding for categorical data

**Supervised Methods:**

* :class:`~binlearn.methods.SupervisedBinning` - Decision tree-based supervised binning

Weight-Constrained Binning
--------------------------

The framework supports advanced binning methods like weight-constrained binning:

.. code-block:: python

   from binlearn.methods import EqualWidthMinimumWeightBinning
   
   # Sample data with associated weights
   X = np.array([[1, 10], [2, 11], [3, 12], [4, 13], [5, 14], [6, 15]])
   weights = np.array([0.5, 0.8, 2.1, 1.9, 0.7, 1.2])  # Importance weights
   
   # Create binner with minimum weight constraint
   binner = EqualWidthMinimumWeightBinning(
       n_bins=4, 
       minimum_weight=1.5  # Each bin must contain at least 1.5 total weight
   )
   
   # Fit and transform with guidance data
   X_binned = binner.fit_transform(X, guidance_data=weights)
   
   print(f"Number of final bins: {len(binner._bin_edges[0]) - 1}")

DataFrame Support
-----------------

The framework works seamlessly with pandas DataFrames and preserves column names:

.. code-block:: python

   import pandas as pd
   from binlearn.methods import EqualFrequencyBinning
   
   # Create DataFrame
   df = pd.DataFrame({
       'age': np.random.normal(35, 10, 100),
       'income': np.random.exponential(50000, 100),
       'score': np.random.uniform(0, 100, 100)
   })
   
   # Apply binning with DataFrame preservation  
   binner = EqualFrequencyBinning(n_bins=5, preserve_dataframe=True)
   df_binned = binner.fit_transform(df)
   
   print("Original DataFrame:")
   print(df.head())
   print("\nBinned DataFrame:")
   print(df_binned.head())
   print("\nBin edges for 'age' column:")
   print(binner.bin_edges_['age'])

Selective Column Binning
------------------------

You can bin specific columns while leaving others unchanged:

.. code-block:: python

   from binlearn.methods import EqualWidthBinning
   
   # Bin only specific columns
   binner = EqualWidthBinning(n_bins=3, columns=['age', 'income'])
   df_selective = binner.fit_transform(df)
   
   print("Only 'age' and 'income' columns were binned")
   print(df_selective.head())

Sklearn Integration
-------------------

Use binning transformers in sklearn pipelines with full compatibility:

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split
   from binlearn.methods import EqualWidthBinning
   
   # Create sample classification data
   from sklearn.datasets import make_classification
   X, y = make_classification(n_samples=1000, n_features=4, n_classes=2, 
                             random_state=42)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
   
   # Create pipeline with binning
   pipeline = Pipeline([
       ('binning', EqualWidthBinning(n_bins=5)),
       ('scaling', StandardScaler()),
       ('classifier', RandomForestClassifier(random_state=42))
   ])
   
   # Fit and evaluate
   pipeline.fit(X_train, y_train)
   accuracy = pipeline.score(X_test, y_test)
   print(f"Pipeline accuracy: {accuracy:.3f}")

Supervised Binning Example
--------------------------

Use supervised binning for better predictive performance:

.. code-block:: python

   from binlearn.methods import SupervisedBinning
   
   # Create supervised binner
   sup_binner = SupervisedBinning(
       n_bins=4,
       task_type='classification',
       tree_params={'max_depth': 3, 'min_samples_leaf': 10}
   )
   
   # Fit with target variable
   X_supervised = sup_binner.fit_transform(X_train, guidance_data=y_train)
   
   print("Supervised binning considers target variable for optimal bin boundaries")
   print(f"Bin edges: {sup_binner.bin_edges_}")

Key Concepts
------------

Binning Methods
~~~~~~~~~~~~~~~

The framework provides several binning strategies:

* **EqualWidthBinning**: Creates bins of equal width across the data range
* **EqualFrequencyBinning**: Creates bins with approximately equal number of samples
* **EqualWidthMinimumWeightBinning**: Equal-width bins with weight constraints
* **SupervisedBinning**: Uses target variable to optimize bin boundaries
* **ManualBinning**: Allows custom specification of bin boundaries

Configuration Options
~~~~~~~~~~~~~~~~~~~~~

All binning methods support common configuration options:

* ``n_bins``: Number of bins to create
* ``clip``: Whether to clip out-of-range values
* ``preserve_dataframe``: Whether to return DataFrames for DataFrame inputs
* ``fit_jointly``: Whether to use the same binning parameters across all columns

Next Steps
----------

* Read the :doc:`user_guide` for detailed explanations
* Check out :doc:`tutorials/basic_binning` for comprehensive tutorials
* Browse :doc:`examples/equal_width_binning` for specific use cases
* Explore the :doc:`api/index` for complete API documentation

Common Patterns
---------------

Here are some common usage patterns:

**Preprocessing for Machine Learning**

.. code-block:: python

   # Reduce dimensionality while preserving information
   from binlearn.methods import EqualFrequencyBinning
   
   binner = EqualFrequencyBinning(n_bins=10)
   X_preprocessed = binner.fit_transform(X_continuous)

**Feature Engineering**

.. code-block:: python

   # Create categorical features from continuous ones
   from binlearn.methods import SupervisedBinning
   
   binner = SupervisedBinning(n_bins=5)
   X_categorical = binner.fit_transform(X, y)

**Data Analysis**

.. code-block:: python

   # Discretize for easier analysis and visualization
   from binlearn.methods import EqualWidthBinning
   
   binner = EqualWidthBinning(n_bins=7)
   data_binned = binner.fit_transform(continuous_data)
