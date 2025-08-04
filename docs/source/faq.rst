Frequently Asked Questions
==========================

Common questions and answers about using binlearn.

General Questions
-----------------

What is binlearn?
~~~~~~~~~~~~~~~~~

binlearn is a comprehensive Python library for data binning and discretization. It provides multiple binning methods with sklearn compatibility, DataFrame support, and modern Python features like type safety and comprehensive error handling.

Why use binning/discretization?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Binning is useful for:

- **Noise reduction**: Smoothing out small variations in data
- **Model compatibility**: Some algorithms work better with categorical data
- **Interpretability**: Creating meaningful categories from continuous data  
- **Feature engineering**: Creating new categorical features for ML models
- **Data exploration**: Understanding data distribution patterns

How does binlearn compare to other binning libraries?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

binlearn offers several advantages:

- **Modern Python**: Type safety, comprehensive error handling, 100% test coverage
- **Framework integration**: Native pandas/polars support, full sklearn compatibility
- **Multiple methods**: 8 different binning algorithms including supervised binning
- **Flexibility**: Custom bin specifications, guidance columns, configurable behavior
- **Performance**: Optimized implementations with efficient memory usage

Installation and Setup
-----------------------

What Python versions are supported?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

binlearn supports Python 3.10, 3.11, 3.12, and 3.13.

What are the required dependencies?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Core dependencies** (automatically installed):
- NumPy >= 1.21.0
- SciPy >= 1.7.0  
- Scikit-learn >= 1.0.0
- kmeans1d >= 0.3.0

**Optional dependencies**:
- Pandas >= 1.3.0 (for DataFrame support)
- Polars >= 0.15.0 (for Polars DataFrame support)

How do I install optional dependencies?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Install with pandas support
   pip install binlearn[pandas]
   
   # Install with polars support  
   pip install binlearn[polars]
   
   # Install with all optional dependencies
   pip install binlearn[pandas,polars]

I get import errors with pandas/polars. What should I do?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the optional dependencies:

.. code-block:: bash

   pip install pandas polars

The library will work without them, but DataFrame-specific features won't be available.

Usage Questions
---------------

Which binning method should I choose?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Choose based on your data and goals:

- **EqualWidthBinning**: Simple, interpretable, good for uniformly distributed data
- **EqualFrequencyBinning**: Balanced bin sizes, good for skewed data
- **KMeansBinning**: Natural clusters in data, good for multimodal distributions
- **SupervisedBinning**: When you have target variables, optimizes for prediction
- **SingletonBinning**: For categorical data encoding
- **Manual methods**: When you need specific bin boundaries

How do I preserve DataFrame column names?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the ``preserve_dataframe=True`` parameter:

.. code-block:: python

   binner = EqualWidthBinning(n_bins=5, preserve_dataframe=True)
   df_binned = binner.fit_transform(df)
   # df_binned will be a DataFrame with original column names

Can I bin only specific columns?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, select the columns you want to bin:

.. code-block:: python

   # Method 1: Select columns before binning
   selected_data = df[['column1', 'column2']]
   binner.fit_transform(selected_data)
   
   # Method 2: Use sklearn's ColumnTransformer
   from sklearn.compose import ColumnTransformer
   preprocessor = ColumnTransformer([
       ('binning', EqualWidthBinning(n_bins=5), ['column1', 'column2']),
       ('passthrough', 'passthrough', ['column3', 'column4'])
   ])

How do I handle missing values?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

binlearn handles missing values automatically:

- NaN values are preserved and assigned a special bin value
- The binner will warn if there are excessive missing values
- Missing values don't affect bin edge calculations

.. code-block:: python

   # Data with missing values
   import numpy as np
   data_with_nan = np.array([1, 2, np.nan, 4, 5])
   
   binner = EqualWidthBinning(n_bins=3)
   result = binner.fit_transform(data_with_nan.reshape(-1, 1))
   # NaN values are preserved in the result

What happens with outliers?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, outliers are included in the outermost bins. You can control this with the ``clip`` parameter:

.. code-block:: python

   # Include outliers in outermost bins (default)
   binner = EqualWidthBinning(n_bins=5, clip=False)
   
   # Clip outliers to bin edges
   binner = EqualWidthBinning(n_bins=5, clip=True)

How do I get bin boundaries and representatives?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Access the ``bin_edges_`` and ``bin_representatives_`` attributes after fitting:

.. code-block:: python

   binner = EqualWidthBinning(n_bins=5)
   binner.fit(X)
   
   print(f"Bin edges: {binner.bin_edges_}")
   print(f"Bin representatives: {binner.bin_representatives_}")

Can I save and load trained binners?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, use pickle or joblib:

.. code-block:: python

   import pickle
   
   # Save trained binner
   with open('binner.pkl', 'wb') as f:
       pickle.dump(binner, f)
   
   # Load binner
   with open('binner.pkl', 'rb') as f:
       loaded_binner = pickle.load(f)

Advanced Usage
--------------

How do I use binning in sklearn pipelines?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All binlearn transformers are sklearn-compatible:

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.ensemble import RandomForestClassifier
   
   pipeline = Pipeline([
       ('binning', EqualWidthBinning(n_bins=5)),
       ('classifier', RandomForestClassifier())
   ])
   
   pipeline.fit(X_train, y_train)
   predictions = pipeline.predict(X_test)

What is supervised binning and when should I use it?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Supervised binning uses target variable information to create optimal bins for prediction tasks:

.. code-block:: python

   from binlearn import SupervisedBinning
   
   # For classification
   sup_binner = SupervisedBinning(
       n_bins=4,
       task_type='classification'
   )
   X_binned = sup_binner.fit_transform(X, guidance_data=y)

Use it when:
- You have labeled data (classification/regression)
- You want bins optimized for prediction performance
- Traditional binning doesn't capture important patterns

How do I create custom bin boundaries?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ManualIntervalBinning:

.. code-block:: python

   from binlearn import ManualIntervalBinning
   
   # Define custom bin edges
   custom_edges = {
       'feature1': [0, 25, 50, 75, 100],
       'feature2': [-2, -1, 0, 1, 2]
   }
   
   manual_binner = ManualIntervalBinning(
       bin_edges=custom_edges,
       preserve_dataframe=True
   )

Can I mix different binning methods for different columns?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, use sklearn's ColumnTransformer:

.. code-block:: python

   from sklearn.compose import ColumnTransformer
   from binlearn import EqualWidthBinning, SingletonBinning
   
   preprocessor = ColumnTransformer([
       ('numeric', EqualWidthBinning(n_bins=5), ['age', 'income']),
       ('categorical', SingletonBinning(), ['category', 'region'])
   ])

How do I optimize binning performance for large datasets?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Several strategies:

1. **Use appropriate data types**: Float32 instead of float64 if precision allows
2. **Sample for fitting**: Fit on a representative sample, transform the full dataset
3. **Choose efficient methods**: EqualWidthBinning is faster than KMeansBinning
4. **Use chunked processing**: For datasets larger than memory (future feature)

.. code-block:: python

   # Sample-based fitting for large datasets
   sample_size = 10000
   sample_indices = np.random.choice(len(X), sample_size, replace=False)
   X_sample = X[sample_indices]
   
   binner = EqualWidthBinning(n_bins=5)
   binner.fit(X_sample)
   X_binned = binner.transform(X)  # Transform full dataset

Troubleshooting
---------------

I get a ConfigurationError. What does this mean?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ConfigurationError indicates invalid parameters. Common causes:

- ``n_bins <= 0``: Must be positive
- Invalid ``bin_range``: Must be tuple with min < max
- Conflicting parameters: Can't use ``guidance_columns`` with ``fit_jointly=True``

Check the error message for specific guidance.

My binned data has unexpected values. What's wrong?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Common issues:

1. **Out-of-range values**: Check if ``clip=True`` is needed
2. **Missing values**: NaN inputs produce special bin values
3. **Insufficient data**: Very small datasets may not bin as expected
4. **Wrong method**: Consider if your chosen method suits your data distribution

The binner seems slow. How can I speed it up?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Performance tips:

1. **Use EqualWidthBinning** for fastest performance
2. **Reduce data size** if possible (fewer samples or features)
3. **Use appropriate dtypes** (float32 vs float64)
4. **Avoid KMeansBinning** for very large datasets
5. **Consider sampling** for fitting on large datasets

I get different results each time. Why?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some methods have randomness:

- **KMeansBinning**: Uses random initialization, set ``random_state`` for reproducibility
- **SupervisedBinning**: Decision trees have randomness, set ``random_state`` in ``tree_params``

.. code-block:: python

   # Reproducible results
   binner = KMeansBinning(n_bins=5, random_state=42)

Can I contribute new binning methods?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes! We welcome contributions. See the :doc:`contributing` guide for details on:

- Development setup
- Coding standards  
- Testing requirements
- Pull request process

Integration Questions
--------------------

Does binlearn work with Dask?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Not directly, but you can use binlearn with Dask by:

1. Fitting on a representative sample
2. Applying the trained binner to Dask chunks
3. Using map_partitions for transformation

Does binlearn support sparse matrices?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, binlearn supports scipy sparse matrices for memory-efficient processing of high-dimensional sparse data.

Can I use binlearn with Apache Spark?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Not directly, but you can:

1. Convert Spark DataFrames to pandas for binning
2. Use fitted binners in Spark UDFs
3. Apply binning in preprocessing steps before Spark ML

Still Have Questions?
--------------------

If you don't find your answer here:

1. **Check the documentation**: Browse the user guide and API reference
2. **Search GitHub issues**: Someone may have asked the same question
3. **Create an issue**: For bugs or feature requests
4. **Start a discussion**: For general questions or usage help

Visit our `GitHub repository <https://github.com/TheDAALab/binlearn>`_ for more information.
