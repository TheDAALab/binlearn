Overview
========

binlearn is a comprehensive Python library for data binning and discretization, designed to integrate seamlessly with modern data science workflows. This section provides an overview of the library's architecture, design principles, and key concepts.

Design Principles
-----------------

Modern Python Standards
~~~~~~~~~~~~~~~~~~~~~~~~

binlearn is built with modern Python best practices:

- **Type Safety**: 100% mypy compliance with comprehensive type annotations
- **Code Quality**: 100% ruff compliance following modern Python standards  
- **Error Handling**: Comprehensive validation with helpful error messages
- **Documentation**: Extensive docstrings and examples for all components

Framework Integration
~~~~~~~~~~~~~~~~~~~~~

The library is designed to work seamlessly with popular data science frameworks:

- **Scikit-learn**: Full compatibility with sklearn pipelines and transformers
- **Pandas**: Native DataFrame support with column name preservation
- **Polars**: High-performance columnar data support (optional)
- **NumPy**: Efficient numerical array processing

Flexibility and Extensibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

binlearn provides flexible binning approaches:

- **Multiple Methods**: Eight different binning algorithms
- **Configurable Behavior**: Global and per-instance configuration
- **Custom Binning**: Manual specification of bin boundaries and definitions
- **Guided Binning**: Use auxiliary data to inform binning decisions

Architecture Overview
---------------------

Base Classes
~~~~~~~~~~~~

The library is built on a hierarchy of base classes:

.. code-block:: text

   GeneralBinningBase
   ├── IntervalBinningBase
   │   ├── EqualWidthBinning
   │   ├── EqualFrequencyBinning
   │   ├── KMeansBinning
   │   ├── EqualWidthMinimumWeightBinning
   │   └── ManualIntervalBinning
   ├── FlexibleBinningBase
   │   └── ManualFlexibleBinning
   ├── SupervisedBinningBase
   │   └── SupervisedBinning
   └── SingletonBinning

**GeneralBinningBase**
   The root base class providing core functionality like sklearn compatibility, 
   configuration management, and data validation.

**IntervalBinningBase**
   For methods that create interval-based bins (e.g., [0, 5), [5, 10)).

**FlexibleBinningBase**  
   For methods that support mixed interval and singleton bins.

**SupervisedBinningBase**
   For methods that use guidance data (target variables) to inform binning.

Key Concepts
------------

Binning vs. Discretization
~~~~~~~~~~~~~~~~~~~~~~~~~~

While often used interchangeably, binlearn distinguishes between:

- **Binning**: Converting continuous data into discrete intervals or categories
- **Discretization**: The broader process of making continuous data discrete (includes binning)

Types of Bins
~~~~~~~~~~~~~

binlearn supports different types of bins:

**Interval Bins**
   Continuous ranges like [0, 5), [5, 10), [10, 15]

**Singleton Bins**  
   Individual values, typically used for categorical data

**Flexible Bins**
   Mixed interval and singleton bins in the same feature

Data Flow
~~~~~~~~~

The typical binlearn workflow:

1. **Configuration**: Set global or instance-specific parameters
2. **Initialization**: Create a binner with desired parameters  
3. **Fitting**: Learn bin boundaries from training data
4. **Transformation**: Apply learned binning to new data
5. **Analysis**: Examine bin edges, representatives, and distributions

.. code-block:: python

   # Example workflow
   from binlearn import EqualWidthBinning
   import numpy as np
   
   # 1. Configuration (optional)
   from binlearn import set_config
   set_config(preserve_dataframe=True)
   
   # 2. Initialization  
   binner = EqualWidthBinning(n_bins=5)
   
   # 3. Fitting
   X = np.random.rand(1000, 3)
   binner.fit(X)
   
   # 4. Transformation
   X_binned = binner.transform(X)
   
   # 5. Analysis
   print(f"Bin edges: {binner.bin_edges_}")

Core Features
-------------

Sklearn Compatibility
~~~~~~~~~~~~~~~~~~~~~~

All binlearn transformers implement the sklearn transformer interface:

- ``fit(X, y=None)``: Learn binning parameters from data
- ``transform(X)``: Apply learned binning to data  
- ``fit_transform(X, y=None)``: Fit and transform in one step
- ``get_params()``/``set_params()``: Parameter management

Configuration System
~~~~~~~~~~~~~~~~~~~~~

Global configuration allows consistent behavior across all binners:

.. code-block:: python

   from binlearn import get_config, set_config
   
   # View current configuration
   config = get_config()
   
   # Set global defaults
   set_config(
       preserve_dataframe=True,
       clip=True,
       fit_jointly=False
   )

Error Handling
~~~~~~~~~~~~~~

Comprehensive error handling with helpful messages:

- **ConfigurationError**: Invalid parameters or settings
- **ValidationError**: Data validation failures  
- **FittingError**: Issues during the fitting process
- **TransformationError**: Problems during transformation

Type System
~~~~~~~~~~~

Rich type annotations for better development experience:

.. code-block:: python

   from binlearn.utils.types import (
       ArrayLike,           # Input data types
       BinEdgesDict,        # Bin edges per column
       ColumnList,          # List of column identifiers
       FlexibleBinSpec      # Flexible bin specifications
   )

Data Handling
-------------

Input Formats
~~~~~~~~~~~~~

binlearn accepts various input formats:

- **NumPy arrays**: ``np.ndarray`` of any numeric dtype
- **Pandas DataFrames**: With automatic column name preservation  
- **Polars DataFrames**: High-performance columnar data (optional)
- **Scipy sparse matrices**: For memory-efficient sparse data

Output Formats
~~~~~~~~~~~~~~

Output format depends on configuration and input:

- **preserve_dataframe=True**: Returns same format as input when possible
- **preserve_dataframe=False**: Always returns NumPy array
- Column names and indices are preserved for DataFrame inputs

Missing Values
~~~~~~~~~~~~~~

binlearn handles missing values gracefully:

- **NaN values**: Preserved in output (assigned special bin value)
- **Detection**: Automatic detection of various missing value representations
- **Validation**: Warns about excessive missing values

Performance Considerations
--------------------------

Memory Efficiency
~~~~~~~~~~~~~~~~~

- **In-place operations**: Where possible to reduce memory usage
- **Sparse matrix support**: For high-dimensional sparse data
- **Chunked processing**: For datasets larger than memory (planned)

Computational Efficiency  
~~~~~~~~~~~~~~~~~~~~~~~~

- **Vectorized operations**: Using NumPy for fast computation
- **Optimized algorithms**: Efficient implementations of binning methods
- **Caching**: Intermediate results cached when beneficial

Integration Patterns
--------------------

Machine Learning Pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.ensemble import RandomForestClassifier
   from binlearn import EqualWidthBinning
   
   pipeline = Pipeline([
       ('binning', EqualWidthBinning(n_bins=5)),
       ('classifier', RandomForestClassifier())
   ])

Feature Engineering
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.compose import ColumnTransformer
   from binlearn import EqualWidthBinning, SingletonBinning
   
   preprocessor = ColumnTransformer([
       ('numeric', EqualWidthBinning(n_bins=5), ['age', 'income']),
       ('categorical', SingletonBinning(), ['category', 'region'])
   ])

Cross-Validation
~~~~~~~~~~~~~~~~

binlearn transformers work correctly with cross-validation:

.. code-block:: python

   from sklearn.model_selection import cross_val_score
   from sklearn.pipeline import Pipeline
   
   pipeline = Pipeline([
       ('binning', EqualWidthBinning(n_bins=5)),
       ('classifier', RandomForestClassifier())
   ])
   
   scores = cross_val_score(pipeline, X, y, cv=5)

Next Steps
----------

Now that you understand the overall architecture and design, explore:

- :doc:`binning_methods`: Detailed guide to all available binning methods
- :doc:`data_types`: Working with different data formats  
- :doc:`configuration`: Advanced configuration options
- :doc:`best_practices`: Tips for effective binning strategies
