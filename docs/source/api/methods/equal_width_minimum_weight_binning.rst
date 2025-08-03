EqualWidthMinimumWeightBinning
==============================

.. currentmodule:: binlearn.methods

.. autoclass:: EqualWidthMinimumWeightBinning
   :members:
   :inherited-members:
   :show-inheritance:

Overview
--------

The :class:`EqualWidthMinimumWeightBinning` class implements equal-width binning with minimum weight constraints. This method creates bins of equal width across the feature range but ensures each bin contains at least a specified minimum total weight from the guidance data.

Key Features
~~~~~~~~~~~~

* **Equal-width intervals**: Creates bins with uniform width across the data range
* **Weight constraints**: Ensures each bin meets minimum weight requirements
* **Bin merging**: Automatically merges adjacent bins that don't meet weight constraints
* **External guidance data**: Uses separate weight data for constraint calculations
* **Sklearn compatibility**: Full integration with scikit-learn ecosystem

Usage Examples
--------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from binlearn.methods import EqualWidthMinimumWeightBinning
   
   # Create sample data
   X = np.array([[1, 10], [2, 11], [3, 12], [4, 13], [5, 14]])
   weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
   
   # Create binner with minimum weight constraint
   binner = EqualWidthMinimumWeightBinning(
       n_bins=3, 
       minimum_weight=1.5
   )
   
   # Fit and transform data
   X_binned = binner.fit_transform(X, guidance_data=weights)
   print(X_binned)

Advanced Usage
~~~~~~~~~~~~~~

.. code-block:: python

   # Custom bin range and weight distribution
   X = np.random.rand(100, 2) * 50
   weights = np.random.exponential(2.0, 100)  # Exponentially distributed weights
   
   binner = EqualWidthMinimumWeightBinning(
       n_bins=10,
       minimum_weight=5.0,
       bin_range=(0, 60),  # Custom range
       clip=True
   )
   
   X_binned = binner.fit_transform(X, guidance_data=weights)
   
   # Check final number of bins after merging
   final_bins = len(binner._bin_edges[0]) - 1
   print(f"Final number of bins: {final_bins}")

Sklearn Integration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.linear_model import LogisticRegression
   
   # Create pipeline with binning
   pipeline = Pipeline([
       ('binner', EqualWidthMinimumWeightBinning(n_bins=5, minimum_weight=2.0)),
       ('scaler', StandardScaler()),
       ('classifier', LogisticRegression())
   ])
   
   # Fit pipeline (note the parameter passing for guidance_data)
   pipeline.fit(X_train, y_train, binner__guidance_data=sample_weights)

Parameters
----------

n_bins : int, default=10
    Initial number of bins to create for each feature. The final number may be 
    lower due to minimum weight constraints and bin merging.

minimum_weight : float, default=1.0
    Minimum total weight required per bin from the guidance data. Bins with 
    insufficient weight will be merged with adjacent bins.

bin_range : tuple of float, optional
    Custom range for binning as (min, max). If None, uses the actual data range.
    Useful for ensuring consistent binning across different datasets.

clip : bool, optional
    Whether to clip out-of-range values to the nearest bin edge. If None, 
    uses global configuration.

preserve_dataframe : bool, optional
    Whether to return DataFrames when input is DataFrame. If None, uses 
    global configuration.

bin_edges : dict, optional
    Pre-specified bin edges for each column. If provided, these edges are 
    used instead of calculating from data.

bin_representatives : dict, optional
    Pre-specified representative values for each bin. Must be provided 
    along with bin_edges.

fit_jointly : bool, optional
    Whether to fit parameters jointly across all columns. For this method,
    joint fitting is not recommended and will fall back to per-column fitting
    with a warning.

guidance_columns : list or str, optional
    Columns providing weights for minimum weight constraint. If None and 
    no external guidance_data is provided, uses the first column not being binned.

Attributes
----------

_bin_edges : dict
    Computed bin edges for each column after fitting.

_bin_representatives : dict
    Computed representative values (bin centers) for each column.

Methods
-------

fit(X, y=None, \\*\\*fit_params)
    Fit the binning transformer to the data.

transform(X)
    Transform data using the fitted binning parameters.

fit_transform(X, y=None, \\*\\*fit_params)
    Fit the transformer and transform the data in one step.

requires_guidance_columns()
    Check if this binning method requires guidance columns.

Notes
-----

This binning method is particularly useful when you have:

* **Imbalanced weight distributions**: Some regions of your data have much higher 
  importance or frequency than others
* **Quality constraints**: Each bin must represent a minimum amount of signal or information
* **Sample size requirements**: Statistical analyses requiring minimum sample sizes per bin

The algorithm works by:

1. Creating initial equal-width bins across the data range
2. Calculating total weight in each bin from the guidance data  
3. Merging adjacent bins that don't meet the minimum weight requirement
4. Ensuring at least one valid bin exists

Weight constraints are enforced using external guidance data passed via the 
`guidance_data` parameter in `fit()` and `fit_transform()` methods.

See Also
--------

EqualWidthBinning : Basic equal-width binning without weight constraints
EqualFrequencyBinning : Equal-frequency binning method
SupervisedBinning : Supervised binning using target variable information

Examples
--------

See the :doc:`/examples/weight_constrained_binning` tutorial for comprehensive 
examples and use cases.
