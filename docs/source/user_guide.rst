User Guide
==========

This comprehensive guide covers all aspects of using the binlearn library effectively.

Overview
--------

The binlearn library is a comprehensive library for data binning and discretization. It provides multiple binning strategies, each optimized for different use cases and data characteristics.

Core Concepts
-------------

What is Binning?
~~~~~~~~~~~~~~~~

Binning (also called discretization) is the process of transforming continuous numerical variables into discrete categorical variables by grouping values into intervals or "bins".

Benefits of binning include:

* **Noise reduction**: Smoothing out minor variations in data
* **Memory efficiency**: Reduced storage requirements for discrete values
* **Algorithm compatibility**: Some algorithms work better with categorical data
* **Interpretability**: Easier to understand and visualize discrete ranges
* **Outlier handling**: Extreme values are contained within bins

When to Use Binning
~~~~~~~~~~~~~~~~~~~

Consider binning when you have:

* **Continuous variables** that would benefit from discretization
* **Noisy data** that needs smoothing
* **Algorithms** that perform better with categorical inputs
* **Memory constraints** requiring more compact representations
* **Interpretability requirements** for business stakeholders

Available Binning Methods
--------------------------

EqualWidthBinning
~~~~~~~~~~~~~~~~~

Creates bins of equal width across the data range.

**Pros:**
* Simple and intuitive
* Preserves data distribution shape
* Consistent bin sizes

**Cons:**
* May create empty bins in sparse regions
* Sensitive to outliers

**Best for:** Uniformly distributed data, when preserving distribution shape is important.

.. code-block:: python

   from binlearn.methods import EqualWidthBinning
   
   binner = EqualWidthBinning(n_bins=5, bin_range=(0, 100))
   X_binned = binner.fit_transform(X)

EqualFrequencyBinning
~~~~~~~~~~~~~~~~~~~~~

Creates bins with approximately equal number of samples.

**Pros:**
* Balanced bin populations
* Good for skewed distributions
* Robust to outliers

**Cons:**
* Bin widths can vary dramatically
* May split similar values into different bins

**Best for:** Skewed data, when you need balanced sample sizes per bin.

.. code-block:: python

   from binlearn.methods import EqualFrequencyBinning
   
   binner = EqualFrequencyBinning(n_bins=5)
   X_binned = binner.fit_transform(X)

EqualWidthMinimumWeightBinning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Equal-width binning with minimum weight constraints from guidance data.

**Pros:**
* Combines equal-width simplicity with weight constraints
* Ensures statistical significance in each bin
* Flexible weight-based merging

**Cons:**
* Requires additional guidance data
* More complex than basic methods

**Best for:** Data with varying importance/weights, quality-constrained analyses.

.. code-block:: python

   from binlearn.methods import EqualWidthMinimumWeightBinning
   
   binner = EqualWidthMinimumWeightBinning(n_bins=5, minimum_weight=10.0)
   X_binned = binner.fit_transform(X, guidance_data=weights)

SupervisedBinning
~~~~~~~~~~~~~~~~~

Uses target variable information to optimize bin boundaries.

**Pros:**
* Optimized for predictive performance
* Maximizes information gain
* Target-aware discretization

**Cons:**
* Requires target variable
* Risk of overfitting
* More computationally expensive

**Best for:** Classification tasks, when predictive performance is critical.

.. code-block:: python

   from binlearn.methods import SupervisedBinning
   
   binner = SupervisedBinning(n_bins=5, criterion='entropy')
   X_binned = binner.fit_transform(X, y)

ManualBinning
~~~~~~~~~~~~~

Allows custom specification of bin boundaries.

**Pros:**
* Complete control over bin boundaries
* Domain knowledge integration
* Reproducible across datasets

**Cons:**
* Requires domain expertise
* Manual specification effort
* May not adapt to new data

**Best for:** Domain-specific requirements, regulatory compliance, standardized ranges.

.. code-block:: python

   from binlearn.methods import ManualBinning
   
   custom_bins = {0: [0, 25, 50, 75, 100]}  # Custom boundaries for column 0
   binner = ManualBinning(bin_edges=custom_bins)
   X_binned = binner.fit_transform(X)

Configuration Options
---------------------

Common Parameters
~~~~~~~~~~~~~~~~~

All binning methods share common configuration options:

``n_bins`` : int
    Number of bins to create. Default varies by method.

``clip`` : bool, optional
    Whether to clip out-of-range values to nearest bin edges.

``preserve_dataframe`` : bool, optional
    Whether to return DataFrames when input is DataFrame.

``fit_jointly`` : bool, optional
    Whether to use same parameters across all columns.

Method-Specific Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

Each method has specialized parameters:

**EqualWidthBinning:**
* ``bin_range``: Custom range for binning

**EqualFrequencyBinning:**
* ``duplicates``: How to handle duplicate values

**EqualWidthMinimumWeightBinning:**
* ``minimum_weight``: Minimum weight required per bin

**SupervisedBinning:**
* ``criterion``: Splitting criterion ('entropy', 'gini', etc.)
* ``max_depth``: Maximum tree depth for optimization

Data Types and Integration
---------------------------

NumPy Arrays
~~~~~~~~~~~~

Basic usage with NumPy arrays:

.. code-block:: python

   import numpy as np
   from binlearn.methods import EqualWidthBinning
   
   X = np.random.rand(100, 3) * 100  # 3 features
   binner = EqualWidthBinning(n_bins=5)
   X_binned = binner.fit_transform(X)  # Returns NumPy array

Pandas DataFrames
~~~~~~~~~~~~~~~~~

Seamless integration with pandas:

.. code-block:: python

   import pandas as pd
   from binlearn.methods import EqualFrequencyBinning
   
   df = pd.DataFrame({
       'age': np.random.normal(35, 10, 1000),
       'income': np.random.lognormal(10, 1, 1000),
       'score': np.random.beta(2, 5, 1000) * 100
   })
   
   binner = EqualFrequencyBinning(n_bins=5, preserve_dataframe=True)
   df_binned = binner.fit_transform(df)  # Returns pandas DataFrame

Polars DataFrames
~~~~~~~~~~~~~~~~~

Support for Polars DataFrames:

.. code-block:: python

   import polars as pl
   from binlearn.methods import EqualWidthBinning
   
   df = pl.DataFrame({
       'feature1': np.random.rand(1000),
       'feature2': np.random.rand(1000)
   })
   
   binner = EqualWidthBinning(n_bins=4, preserve_dataframe=True)
   df_binned = binner.fit_transform(df)  # Returns Polars DataFrame

Sklearn Integration
-------------------

Pipeline Usage
~~~~~~~~~~~~~~

Use binning in sklearn pipelines:

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.ensemble import RandomForestClassifier
   from binlearn.methods import EqualWidthBinning
   
   pipeline = Pipeline([
       ('binning', EqualWidthBinning(n_bins=10)),
       ('scaling', StandardScaler()),
       ('classifier', RandomForestClassifier())
   ])
   
   pipeline.fit(X_train, y_train)
   y_pred = pipeline.predict(X_test)

ColumnTransformer Usage
~~~~~~~~~~~~~~~~~~~~~~~

Apply different binning to different columns:

.. code-block:: python

   from sklearn.compose import ColumnTransformer
   from binlearn.methods import EqualWidthBinning, EqualFrequencyBinning
   
   preprocessor = ColumnTransformer([
       ('numerical_equal_width', EqualWidthBinning(n_bins=5), ['age', 'income']),
       ('numerical_equal_freq', EqualFrequencyBinning(n_bins=3), ['score']),
   ], remainder='passthrough')
   
   X_preprocessed = preprocessor.fit_transform(X)

Parameter Passing
~~~~~~~~~~~~~~~~~

Pass parameters through pipelines:

.. code-block:: python

   # For methods requiring guidance data
   pipeline = Pipeline([
       ('binner', EqualWidthMinimumWeightBinning(n_bins=5, minimum_weight=2.0)),
       ('classifier', LogisticRegression())
   ])
   
   # Pass guidance_data through pipeline
   pipeline.fit(X_train, y_train, binner__guidance_data=sample_weights)

Error Handling and Validation
------------------------------

The framework provides comprehensive error handling:

Configuration Errors
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from binlearn.utils.errors import ConfigurationError
   
   try:
       # Invalid configuration
       binner = EqualWidthBinning(n_bins=0)  # Invalid: n_bins must be positive
   except ConfigurationError as e:
       print(f"Configuration error: {e}")

Data Quality Warnings
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from binlearn.utils.errors import DataQualityWarning
   import warnings
   
   # Set up warning handling
   warnings.simplefilter('always', DataQualityWarning)
   
   # This will generate warnings for columns with all NaN values
   X_with_nans = np.array([[1, np.nan], [2, np.nan], [3, np.nan]])
   binner = EqualWidthBinning(n_bins=3)
   X_binned = binner.fit_transform(X_with_nans)

Fitting Errors
~~~~~~~~~~~~~~

.. code-block:: python

   from binlearn.utils.errors import FittingError
   
   try:
       # Insufficient data for binning
       X_insufficient = np.array([[1.0]])  # Only one data point
       binner = EqualWidthBinning(n_bins=3)
       binner.fit(X_insufficient)
   except FittingError as e:
       print(f"Fitting error: {e}")

Best Practices
--------------

Choose the Right Method
~~~~~~~~~~~~~~~~~~~~~~~

* **EqualWidthBinning**: Start with this for most use cases
* **EqualFrequencyBinning**: Use for highly skewed data
* **EqualWidthMinimumWeightBinning**: When you have importance weights
* **SupervisedBinning**: For classification preprocessing
* **ManualBinning**: When you have domain-specific requirements

Validate Your Results
~~~~~~~~~~~~~~~~~~~~~

Always validate binning results:

.. code-block:: python

   # Check bin distributions
   unique_bins, counts = np.unique(X_binned, return_counts=True)
   print("Bin counts:", dict(zip(unique_bins, counts)))
   
   # Verify bin edges make sense
   print("Bin edges:", binner._bin_edges)
   
   # Check for empty bins
   n_empty_bins = len([c for c in counts if c == 0])
   print(f"Empty bins: {n_empty_bins}")

Handle Missing Values
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Check for missing values in results
   from binlearn.utils.constants import MISSING_VALUE
   
   n_missing = np.sum(X_binned == MISSING_VALUE)
   print(f"Missing values in binned data: {n_missing}")

Performance Considerations
--------------------------

For Large Datasets
~~~~~~~~~~~~~~~~~~~

* Use appropriate bin counts (more bins = more computation)
* Consider memory usage with ``preserve_dataframe=False``
* Use ``fit_jointly=False`` for independent column processing

For Real-time Applications
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Pre-fit transformers and save them
* Use ``transform()`` only for new data
* Consider simpler methods (EqualWidthBinning) for speed

Memory Optimization
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # For memory-constrained environments
   binner = EqualWidthBinning(n_bins=5, preserve_dataframe=False)
   X_binned = binner.fit_transform(X)  # Returns NumPy array

Troubleshooting
---------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Empty bins in EqualWidthBinning**
   * Reduce ``n_bins`` or use EqualFrequencyBinning
   * Check for outliers affecting range calculation

**Inconsistent results across runs**
   * Use fixed ``random_state`` where available
   * Ensure consistent data ordering

**Memory errors with large datasets**
   * Reduce ``n_bins``
   * Use ``preserve_dataframe=False``
   * Process data in chunks

**Unexpected bin assignments**
   * Check ``clip`` parameter setting
   * Verify ``bin_range`` if specified
   * Examine actual vs expected bin edges

Next Steps
----------

* Explore :doc:`tutorials/basic_binning` for hands-on examples
* Check :doc:`examples/equal_width_minimum_weight_binning` for advanced techniques  
* Read the :doc:`api/index` for detailed API documentation
* See :doc:`contributing` to contribute to the project
