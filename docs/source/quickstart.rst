Quick Start Guide
=================

This guide will get you up and running with binlearn in just a few minutes.

Basic Binning Example
----------------------

Let's start with a simple example using equal-width binning:

.. code-block:: python

   import numpy as np
   import pandas as pd
   from binlearn import EqualWidthBinning
   
   # Create sample data
   np.random.seed(42)
   data = pd.DataFrame({
       'age': np.random.normal(35, 10, 1000),
       'income': np.random.lognormal(10, 0.5, 1000),
       'score': np.random.uniform(0, 100, 1000)
   })
   
   # Create and fit the binner
   binner = EqualWidthBinning(n_bins=5, preserve_dataframe=True)
   data_binned = binner.fit_transform(data)
   
   print(f"Original shape: {data.shape}")
   print(f"Binned shape: {data_binned.shape}")
   print(f"Bin edges for age: {binner.bin_edges_['age']}")

This will output something like:

.. code-block:: text

   Original shape: (1000, 3)
   Binned shape: (1000, 3)
   Bin edges for age: [  6.74  17.88  29.02  40.16  51.30  62.44]

Working with Different Data Types
----------------------------------

binlearn supports various data formats:

NumPy Arrays
~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from binlearn import EqualWidthBinning
   
   # NumPy array
   X = np.random.rand(100, 3)
   binner = EqualWidthBinning(n_bins=4)
   X_binned = binner.fit_transform(X)
   
   print(f"Original: {X.shape}, Binned: {X_binned.shape}")

Pandas DataFrames
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   from binlearn import EqualFrequencyBinning
   
   # Pandas DataFrame with preserved column names
   df = pd.DataFrame({
       'feature1': np.random.normal(0, 1, 100),
       'feature2': np.random.exponential(2, 100)
   })
   
   binner = EqualFrequencyBinning(n_bins=3, preserve_dataframe=True)
   df_binned = binner.fit_transform(df)
   
   print(df_binned.columns.tolist())  # ['feature1', 'feature2']

Exploring Different Binning Methods
------------------------------------

Equal-Frequency Binning
~~~~~~~~~~~~~~~~~~~~~~~~

Creates bins with approximately equal number of samples:

.. code-block:: python

   from binlearn import EqualFrequencyBinning
   
   # Create skewed data
   X = np.random.exponential(2, (1000, 2))
   
   binner = EqualFrequencyBinning(n_bins=4)
   X_binned = binner.fit_transform(X)
   
   # Check bin counts
   unique, counts = np.unique(X_binned[:, 0], return_counts=True)
   print(f"Bin counts: {counts}")  # Should be approximately equal

K-Means Binning
~~~~~~~~~~~~~~~

Uses K-means clustering to determine bin boundaries:

.. code-block:: python

   from binlearn import KMeansBinning
   
   # Data with natural clusters
   X = np.concatenate([
       np.random.normal(0, 1, (200, 1)),
       np.random.normal(5, 1, (200, 1)),
       np.random.normal(10, 1, (200, 1))
   ])
   
   binner = KMeansBinning(n_bins=3, random_state=42)
   X_binned = binner.fit_transform(X)
   
   print(f"Bin edges: {binner.bin_edges_[0]}")

Supervised Binning
~~~~~~~~~~~~~~~~~~

Uses target information to create optimal bins:

.. code-block:: python

   from binlearn import SupervisedBinning
   from sklearn.datasets import make_classification
   
   # Create classification dataset
   X, y = make_classification(n_samples=1000, n_features=4, n_classes=2, random_state=42)
   
   # Supervised binning considers the target variable
   sup_binner = SupervisedBinning(
       n_bins=4,
       task_type='classification',
       tree_params={'max_depth': 3}
   )
   
   X_binned = sup_binner.fit_transform(X, guidance_data=y)
   print(f"Supervised binning shape: {X_binned.shape}")

Categorical Data with SingletonBinning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Handle categorical data with the new SingletonBinning method:

.. code-block:: python

   from binlearn import SingletonBinning
   
   # Categorical data
   categorical_data = pd.DataFrame({
       'category': ['A', 'B', 'A', 'C', 'B', 'A', 'D'],
       'rating': ['good', 'bad', 'good', 'excellent', 'bad', 'good', 'fair']
   })
   
   singleton_binner = SingletonBinning(preserve_dataframe=True)
   categorical_binned = singleton_binner.fit_transform(categorical_data)
   
   print(f"Original categories: {categorical_data['category'].unique()}")
   print(f"Binned shape: {categorical_binned.shape}")

Scikit-learn Integration
------------------------

binlearn transformers are fully compatible with scikit-learn pipelines:

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import accuracy_score
   from binlearn import EqualWidthBinning
   
   # Sample data
   X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
   # Create pipeline with binning
   pipeline = Pipeline([
       ('binning', EqualWidthBinning(n_bins=5)),
       ('classifier', RandomForestClassifier(random_state=42))
   ])
   
   # Train and evaluate
   pipeline.fit(X_train, y_train)
   y_pred = pipeline.predict(X_test)
   accuracy = accuracy_score(y_test, y_pred)
   
   print(f"Pipeline accuracy: {accuracy:.3f}")

Configuration Management
-------------------------

binlearn provides global configuration for consistent behavior:

.. code-block:: python

   from binlearn import get_config, set_config
   
   # Check current configuration
   current_config = get_config()
   print(f"Current config: {current_config}")
   
   # Set global defaults
   set_config(
       preserve_dataframe=True,
       clip=True,
       fit_jointly=False
   )
   
   # Now all binners will use these defaults
   binner = EqualWidthBinning(n_bins=3)  # Will preserve DataFrames by default

Error Handling and Validation
------------------------------

binlearn provides comprehensive error handling:

.. code-block:: python

   from binlearn import EqualWidthBinning
   from binlearn.utils.errors import ConfigurationError
   
   try:
       # This will raise a ConfigurationError
       binner = EqualWidthBinning(n_bins=0)  # Invalid: n_bins must be positive
   except ConfigurationError as e:
       print(f"Configuration error: {e}")
   
   try:
       # This will raise a ValidationError during fit
       binner = EqualWidthBinning(n_bins=5)
       binner.fit([[1, 2], [3]])  # Invalid: inconsistent array dimensions
   except Exception as e:
       print(f"Validation error: {e}")

Next Steps
----------

Now that you have the basics down, explore:

1. **User Guide**: Detailed explanations of all binning methods
2. **Examples**: Real-world use cases and advanced techniques  
3. **API Reference**: Complete documentation of all classes and functions
4. **Performance Tips**: Optimization strategies for large datasets

For more advanced usage, check out the :doc:`user_guide/index` or browse the :doc:`examples/index`.
