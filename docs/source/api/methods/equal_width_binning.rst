EqualWidthBinning
=================

.. currentmodule:: binning.methods

.. autoclass:: EqualWidthBinning
   :members:
   :inherited-members:
   :show-inheritance:

**Equal-width binning** divides the range of continuous features into intervals of equal width. This is one of the most intuitive and commonly used binning methods.

Overview
--------

EqualWidthBinning creates bins by dividing the range [min, max] of each feature into n_bins intervals of equal width. Each interval has the same width, but may contain different numbers of samples.

**Formula:**
   bin_width = (max_value - min_value) / n_bins

**Key Characteristics:**

* ✅ **Simple and intuitive** - Easy to understand and interpret
* ✅ **Preserves distribution shape** - Maintains the original data distribution
* ✅ **Consistent bin sizes** - All bins have the same width
* ❌ **Sensitive to outliers** - Extreme values can create many empty bins
* ❌ **Uneven sample distribution** - Some bins may have very few samples

Parameters
----------

n_bins : int, default=5
    Number of bins to create for each feature.

bin_range : tuple of (min, max), optional
    Custom range for binning. If None, uses the data range.

clip : bool, default=False
    Whether to clip out-of-range values to the nearest bin edge.

preserve_dataframe : bool, optional
    Whether to preserve the input DataFrame format. If None, auto-detects.

fit_jointly : bool, default=True
    Whether to use the same binning parameters across all features.

Attributes
----------

_bin_edges : dict
    Fitted bin edges for each column after calling fit().

Examples
--------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import pandas as pd
   from binning.methods import EqualWidthBinning
   
   # Create sample data
   data = pd.DataFrame({
       'age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
       'income': [30000, 45000, 55000, 65000, 75000, 85000, 95000, 105000, 115000, 125000]
   })
   
   # Create equal-width bins
   binner = EqualWidthBinning(n_bins=3)
   binned_data = binner.fit_transform(data)
   
   print("Original data:")
   print(data)
   print("\nBinned data:")
   print(binned_data)
   
   # Check bin edges
   print("\nBin edges:")
   for col_idx, edges in binner._bin_edges.items():
       col_name = data.columns[col_idx]
       print(f"{col_name}: {edges}")

Working with NumPy Arrays
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from binning.methods import EqualWidthBinning
   
   # Create sample data
   X = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]])
   
   # Apply binning
   binner = EqualWidthBinning(n_bins=3)
   X_binned = binner.fit_transform(X)
   
   print("Original:", X)
   print("Binned:", X_binned)

Custom Range Binning
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from binning.methods import EqualWidthBinning
   
   # Focus on a specific age range
   binner = EqualWidthBinning(
       n_bins=4, 
       bin_range=(20, 60),  # Only consider ages 20-60
       clip=True  # Clip outliers to this range
   )
   
   # Apply to age data including outliers
   ages = pd.DataFrame({'age': [15, 25, 35, 45, 55, 75]})  # 15 and 75 are outliers
   binned_ages = binner.fit_transform(ages)
   
   print("Ages with custom range binning:")
   print(pd.concat([ages, binned_ages], axis=1))

Sklearn Pipeline Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.ensemble import RandomForestClassifier
   from binning.methods import EqualWidthBinning
   
   # Create a preprocessing pipeline
   pipeline = Pipeline([
       ('binning', EqualWidthBinning(n_bins=5)),
       ('scaling', StandardScaler()),
       ('classifier', RandomForestClassifier())
   ])
   
   # Use in machine learning workflow
   pipeline.fit(X_train, y_train)
   predictions = pipeline.predict(X_test)

Handling Different Data Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from binning.methods import EqualWidthBinning
   
   # Mixed data types
   mixed_data = pd.DataFrame({
       'int_feature': [1, 2, 3, 4, 5],
       'float_feature': [1.1, 2.2, 3.3, 4.4, 5.5],
       'large_numbers': [1000000, 2000000, 3000000, 4000000, 5000000]
   })
   
   binner = EqualWidthBinning(n_bins=3, preserve_dataframe=True)
   result = binner.fit_transform(mixed_data)
   
   print("Mixed data types handled:")
   print(result.dtypes)

When to Use EqualWidthBinning
-----------------------------

**Best For:**

* **Uniformly distributed data** - Works well when data is evenly spread
* **Interpretability** - Easy to explain bin boundaries to stakeholders
* **Visualization** - Creates visually appealing histograms
* **Domain knowledge** - When equal intervals make business sense

**Examples:**

* **Age groups**: 20-30, 30-40, 40-50, etc.
* **Temperature ranges**: 0-10°C, 10-20°C, 20-30°C
* **Price tiers**: $0-100, $100-200, $200-300

**Avoid When:**

* **Highly skewed data** - May create many empty bins
* **Data with outliers** - Outliers can dominate the range
* **Need balanced samples** - Use EqualFrequencyBinning instead

Performance Considerations
--------------------------

**Memory Usage:**
   - Low memory footprint for bin edges storage
   - Linear with number of features

**Computational Complexity:**
   - O(n) for fitting (single pass through data)
   - O(n) for transformation (binary search for each value)

**Scalability:**
   - Efficient for large datasets
   - Parallel processing friendly

Comparison with Other Methods
-----------------------------

.. list-table:: Method Comparison
   :header-rows: 1
   :widths: 25 25 25 25

   * - Aspect
     - EqualWidth
     - EqualFrequency
     - Supervised
   * - Bin sizes
     - Equal width
     - Variable width
     - Optimized width
   * - Sample distribution
     - Variable
     - Equal
     - Target-optimized
   * - Outlier handling
     - Poor
     - Good
     - Good
   * - Interpretability
     - Excellent
     - Good
     - Moderate

Common Issues and Solutions
---------------------------

**Issue: Empty bins with sparse data**

.. code-block:: python

   # Problem: Sparse data creates empty bins
   sparse_data = [1, 2, 3, 100, 101, 102]
   
   # Solution 1: Reduce number of bins
   binner = EqualWidthBinning(n_bins=3)  # Instead of n_bins=10
   
   # Solution 2: Use custom range
   binner = EqualWidthBinning(n_bins=5, bin_range=(1, 102), clip=True)

**Issue: Outliers dominating the range**

.. code-block:: python

   # Problem: One outlier affects all bins
   data_with_outlier = [1, 2, 3, 4, 5, 1000]  # 1000 is outlier
   
   # Solution: Clip outliers or use custom range
   binner = EqualWidthBinning(n_bins=5, bin_range=(1, 10), clip=True)

**Issue: Need consistent bins across datasets**

.. code-block:: python

   # Problem: Different datasets have different ranges
   train_data = [1, 2, 3, 4, 5]
   test_data = [0, 1, 2, 3, 4, 5, 6]  # Different range
   
   # Solution: Fit on training data, transform both
   binner = EqualWidthBinning(n_bins=3)
   binner.fit(train_data)
   
   train_binned = binner.transform(train_data)
   test_binned = binner.transform(test_data)  # Uses training bin edges

See Also
--------

* :class:`EqualFrequencyBinning` - For balanced sample sizes per bin
* :class:`SupervisedBinning` - For target-aware binning
* :class:`EqualWidthMinimumWeightBinning` - For weight-constrained equal-width binning
* :class:`ManualIntervalBinning` - For custom bin boundaries

References
----------

* Dougherty, J., Kohavi, R., & Sahami, M. (1995). Supervised and unsupervised discretization of continuous features. In Machine learning proceedings 1995 (pp. 194-202).
* Liu, H., Hussain, F., Tan, C. L., & Dash, M. (2002). Discretization: An enabling technique. Data mining and knowledge discovery, 6(4), 393-423.
