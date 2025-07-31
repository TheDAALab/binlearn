Base Classes API
===============

The base classes provide the foundation for all binning methods in the framework.

.. currentmodule:: binning.base

Overview
--------

The Binning Framework uses a hierarchical class structure to provide consistent interfaces and shared functionality across all binning methods. Understanding these base classes is essential for:

* Using the framework effectively
* Creating custom binning methods
* Understanding the internal architecture

Class Hierarchy
---------------

.. code-block:: text

   BaseBinner (ABC)
   ├── UnsupervisedBinner
   ├── SupervisedBinner  
   └── GuidedBinner

Base Classes
------------

BaseBinner
~~~~~~~~~~

.. autoclass:: BaseBinner
   :members:
   :inherited-members:
   :show-inheritance:

   The abstract base class for all binning methods. Defines the core interface that all binners must implement.

   **Key Responsibilities:**
   
   * Input validation and preprocessing
   * Framework detection (pandas, polars, numpy)
   * Missing value handling
   * Output format consistency
   * Parameter validation

   **Abstract Methods:**
   
   All subclasses must implement:
   
   * ``_fit_column(column_data, column_index, **kwargs)``
   * ``_transform_column(column_data, column_index)``

   **Usage Example:**
   
   .. code-block:: python
   
      # BaseBinner is abstract - use concrete implementations
      from binning.methods import EqualWidthBinning
      
      binner = EqualWidthBinning(n_bins=5)
      X_binned = binner.fit_transform(X)

UnsupervisedBinner  
~~~~~~~~~~~~~~~~~~

.. autoclass:: UnsupervisedBinner  
   :members:
   :inherited-members:
   :show-inheritance:

   Base class for unsupervised binning methods that don't require target variables.

   **Characteristics:**
   
   * Fits on input features only
   * No target variable required
   * Suitable for exploratory data analysis
   * Can be used in unsupervised learning pipelines

   **Subclasses:**
   
   * :class:`~binning.methods.EqualWidthBinning`
   * :class:`~binning.methods.EqualFrequencyBinning`
   * :class:`~binning.methods.ManualBinning`

   **Usage Example:**
   
   .. code-block:: python
   
      from binning.methods import EqualFrequencyBinning
      
      binner = EqualFrequencyBinning(n_bins=4)
      X_binned = binner.fit_transform(X)  # No y required

SupervisedBinner
~~~~~~~~~~~~~~~~

.. autoclass:: SupervisedBinner
   :members:
   :inherited-members:
   :show-inheritance:

   Base class for supervised binning methods that use target variable information.

   **Characteristics:**
   
   * Requires target variable during fitting
   * Optimizes bins for predictive performance
   * Suitable for classification preprocessing
   * Maximizes information gain or other criteria

   **Subclasses:**
   
   * :class:`~binning.methods.SupervisedBinning`

   **Usage Example:**
   
   .. code-block:: python
   
      from binning.methods import SupervisedBinning
      
      binner = SupervisedBinning(n_bins=5, criterion='entropy')
      X_binned = binner.fit_transform(X, y)  # y required

GuidedBinner
~~~~~~~~~~~~

.. autoclass:: GuidedBinner
   :members:
   :inherited-members:
   :show-inheritance:

   Base class for binning methods that use additional guidance data (e.g., weights, importance scores).

   **Characteristics:**
   
   * Uses guidance data to influence binning decisions
   * Guidance data passed via ``guidance_data`` parameter
   * Enables weight-constrained or importance-aware binning
   * Flexible guidance data interpretation

   **Subclasses:**
   
   * :class:`~binning.methods.EqualWidthMinimumWeightBinning`

   **Usage Example:**
   
   .. code-block:: python
   
      from binning.methods import EqualWidthMinimumWeightBinning
      
      binner = EqualWidthMinimumWeightBinning(n_bins=5, minimum_weight=10.0)
      X_binned = binner.fit_transform(X, guidance_data=sample_weights)

Common Parameters
-----------------

All base classes support these common parameters:

``preserve_dataframe`` : bool, default=None
   Whether to return the same data structure as input:
   
   * ``True``: Always return DataFrames if input is DataFrame
   * ``False``: Always return NumPy arrays
   * ``None``: Auto-detect based on input type

``fit_jointly`` : bool, default=True
   Whether to fit all columns with the same parameters:
   
   * ``True``: Use consistent binning across all columns
   * ``False``: Fit each column independently

``clip`` : bool, default=False
   Whether to clip out-of-range values during transformation:
   
   * ``True``: Clip values to nearest bin edges
   * ``False``: Assign out-of-range values to special bins

Common Methods
--------------

All binning methods inherit these core methods:

fit(X, y=None, **kwargs)
~~~~~~~~~~~~~~~~~~~~~~~~

Fit the binning parameters to the data.

**Parameters:**

* **X** : array-like of shape (n_samples, n_features)
  Input data to fit binning parameters on.

* **y** : array-like of shape (n_samples,), optional  
  Target variable (required for supervised methods).

* **kwargs** : dict
  Additional parameters (e.g., ``guidance_data`` for guided methods).

**Returns:**

* **self** : object
  Returns self for method chaining.

transform(X)
~~~~~~~~~~~~

Transform data using fitted binning parameters.

**Parameters:**

* **X** : array-like of shape (n_samples, n_features)
  Input data to transform.

**Returns:**

* **X_binned** : array-like of shape (n_samples, n_features)
  Binned data in same format as input.

fit_transform(X, y=None, **kwargs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fit binning parameters and transform data in one step.

**Parameters:**

* **X** : array-like of shape (n_samples, n_features)
  Input data to fit and transform.

* **y** : array-like of shape (n_samples,), optional
  Target variable (required for supervised methods).

* **kwargs** : dict
  Additional parameters (e.g., ``guidance_data`` for guided methods).

**Returns:**

* **X_binned** : array-like of shape (n_samples, n_features)
  Binned data in same format as input.

get_params(deep=True)
~~~~~~~~~~~~~~~~~~~~~

Get parameters for this estimator (sklearn compatibility).

**Parameters:**

* **deep** : bool, default=True
  If True, return parameters for sub-estimators too.

**Returns:**

* **params** : dict
  Parameter names mapped to their values.

set_params(**params)
~~~~~~~~~~~~~~~~~~~~

Set parameters for this estimator (sklearn compatibility).

**Parameters:**

* **params** : dict
  Estimator parameters to set.

**Returns:**

* **self** : object
  Returns self for method chaining.

Internal Methods
----------------

These methods are used internally and can be overridden in subclasses:

_validate_inputs(X, y=None, **kwargs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Validate and preprocess input data.

**Parameters:**

* **X** : array-like
  Input features to validate.

* **y** : array-like, optional
  Target variable to validate.

* **kwargs** : dict
  Additional data to validate.

**Returns:**

* **X_validated** : array-like
  Validated and preprocessed input features.

* **y_validated** : array-like or None
  Validated target variable (if provided).

* **kwargs_validated** : dict
  Validated additional data.

_fit_column(column_data, column_index, **kwargs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fit binning parameters for a single column. **Must be implemented by subclasses.**

**Parameters:**

* **column_data** : array-like of shape (n_samples,)
  Data for single column.

* **column_index** : int
  Index of the column being fitted.

* **kwargs** : dict
  Additional parameters (e.g., target data, guidance data).

_transform_column(column_data, column_index)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Transform a single column using fitted parameters. **Must be implemented by subclasses.**

**Parameters:**

* **column_data** : array-like of shape (n_samples,)
  Data for single column to transform.

* **column_index** : int
  Index of the column being transformed.

**Returns:**

* **binned_column** : array-like of shape (n_samples,)
  Binned values for the column.

_detect_framework(X)
~~~~~~~~~~~~~~~~~~~~

Detect the data framework (pandas, polars, numpy) of input data.

**Parameters:**

* **X** : array-like
  Input data to analyze.

**Returns:**

* **framework** : str
  Detected framework ('pandas', 'polars', 'numpy').

Extending Base Classes
----------------------

Creating Custom Binning Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create a custom binning method, inherit from the appropriate base class:

.. code-block:: python

   from binning.base import UnsupervisedBinner
   import numpy as np
   
   class CustomBinner(UnsupervisedBinner):
       """Custom binning implementation."""
       
       def __init__(self, custom_param=5, **kwargs):
           super().__init__(**kwargs)
           self.custom_param = custom_param
       
       def _fit_column(self, column_data, column_index, **kwargs):
           """Implement custom fitting logic."""
           # Remove missing values
           valid_data = column_data[~np.isnan(column_data)]
           
           if len(valid_data) == 0:
               # Handle all-missing case
               self._bin_edges[column_index] = np.array([0, 1])
               return
           
           # Custom binning logic here
           edges = np.linspace(valid_data.min(), valid_data.max(), 
                             self.custom_param + 1)
           self._bin_edges[column_index] = edges
       
       def _transform_column(self, column_data, column_index):
           """Implement custom transformation logic."""
           edges = self._bin_edges[column_index]
           
           # Handle missing values
           missing_mask = np.isnan(column_data)
           result = np.full(len(column_data), -1, dtype=int)
           
           if not missing_mask.all():
               valid_data = column_data[~missing_mask]
               binned = np.digitize(valid_data, edges[1:-1])
               result[~missing_mask] = binned
           
           return result

Best Practices
--------------

When extending base classes:

1. **Always call super().__init__()** in your constructor
2. **Handle missing values** appropriately in both fit and transform
3. **Validate parameters** in your constructor  
4. **Store fitted parameters** in instance variables
5. **Use consistent naming** with existing methods
6. **Add comprehensive docstrings** following NumPy style
7. **Include type hints** for better IDE support

Error Handling
--------------

The base classes provide comprehensive error handling:

**ConfigurationError**
   Raised for invalid parameter combinations or settings.

**FittingError**  
   Raised when fitting fails due to data issues.

**DataQualityWarning**
   Warning issued for data quality concerns that don't prevent operation.

Example custom error handling:

.. code-block:: python

   from binning.utils.errors import ConfigurationError, FittingError
   
   def _fit_column(self, column_data, column_index, **kwargs):
       if self.custom_param <= 0:
           raise ConfigurationError("custom_param must be positive")
       
       valid_data = column_data[~np.isnan(column_data)]
       if len(valid_data) < 2:
           raise FittingError(f"Column {column_index} has insufficient data")
       
       # Continue with fitting logic...

Thread Safety
-------------

The base classes are designed to be thread-safe for read operations after fitting. However:

* **Fitting operations** should not be performed concurrently
* **Multiple transforms** can be performed simultaneously
* **Parameters should not be modified** after fitting

For concurrent operations, create separate instances:

.. code-block:: python

   # Safe: separate instances
   binner1 = EqualWidthBinning(n_bins=5)
   binner2 = EqualWidthBinning(n_bins=5)
   
   # Safe: concurrent transforms after fitting
   binner = EqualWidthBinning(n_bins=5)
   binner.fit(X_train)
   
   # These can run concurrently
   result1 = binner.transform(X_test1)
   result2 = binner.transform(X_test2)

Next Steps
----------

* Explore specific binning methods in :doc:`../methods/index`
* Learn about utilities and tools in :doc:`../utils/index`
* See practical examples in :doc:`../examples/index`
* Check the tutorials in :doc:`../tutorials/index`
