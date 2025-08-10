Chi2Binning
===========

.. currentmodule:: binlearn.methods

.. autoclass:: Chi2Binning
   :members:
   :inherited-members:
   :show-inheritance:

Overview
--------

``Chi2Binning`` is a supervised discretization method that uses the chi-square statistic to find optimal 
split points. The method iteratively merges adjacent intervals to minimize the chi-square statistic, 
creating bins that maximize the association between features and target variables.

This approach is particularly effective for:

* **Classification tasks** where bins need to separate different classes effectively
* **Categorical target variables** with clear class boundaries
* **Feature engineering** for improving downstream classification performance
* **Data preparation** where maintaining class relationships is crucial

Key Features
------------

* **Supervised Learning**: Uses target variable information for optimal binning
* **Statistical Foundation**: Based on chi-square test of independence
* **Iterative Optimization**: Merges intervals to minimize chi-square statistic
* **Classification Focus**: Optimized for categorical target variables
* **Automatic Stopping**: Uses significance levels to determine optimal number of bins
* **Sklearn Compatibility**: Full transformer interface with fit/transform methods
* **DataFrame Support**: Preserves pandas/polars column names and structure

Basic Usage
-----------

.. code-block:: python

   import numpy as np
   import pandas as pd
   from binlearn.methods import Chi2Binning
   from sklearn.datasets import make_classification
   
   # Create sample classification data
   X, y = make_classification(
       n_samples=1000, 
       n_features=3, 
       n_classes=3,
       n_redundant=0,
       random_state=42
   )
   
   # Apply chi-square binning
   binner = Chi2Binning(
       max_bins=10,
       min_bins=3,
       alpha=0.05
   )
   
   # Method 1: Using fit with X and y (sklearn style)
   binner.fit(X, y)
   X_binned = binner.transform(X)
   
   print(f"Original shape: {X.shape}")
   print(f"Binned shape: {X_binned.shape}")
   print(f"Bins for feature 0: {len(binner.bin_edges_[0]) - 1}")

DataFrame Example with Target Column
------------------------------------

.. code-block:: python

   # Create DataFrame with target column
   df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3'])
   df['target'] = y
   
   # Method 2: Using guidance_columns (binlearn style)
   binner = Chi2Binning(
       guidance_columns=['target'],  # Use target column for guidance
       max_bins=8,
       min_bins=2,
       preserve_dataframe=True
   )
   
   # Fit and transform the entire DataFrame
   df_binned = binner.fit_transform(df)
   
   print(f"Bin edges for feature1: {binner.bin_edges_['feature1']}")
   print(f"Bin edges for feature2: {binner.bin_edges_['feature2']}")
   print(f"Target column preserved: {'target' in df_binned.columns}")

Regression Example
------------------

.. code-block:: python

   from sklearn.datasets import make_regression
   
   # Create regression data and discretize target
   X_reg, y_reg = make_regression(n_samples=1000, n_features=2, random_state=42)
   
   # Discretize continuous target for chi-square binning
   y_discrete = pd.cut(y_reg, bins=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
   
   binner = Chi2Binning(
       max_bins=6,
       min_bins=3,
       alpha=0.01  # More stringent significance level
   )
   
   binner.fit(X_reg, y_discrete)
   X_reg_binned = binner.transform(X_reg)
   
   print(f"Regression bins created: {[len(edges)-1 for edges in binner.bin_edges_.values()]}")

Advanced Configuration
----------------------

.. code-block:: python

   # Fine-tuned chi-square binning for specific requirements
   
   # Conservative binning (fewer, more significant bins)
   conservative_binner = Chi2Binning(
       max_bins=15,
       min_bins=2,
       alpha=0.001,        # Very stringent significance level
       initial_bins=20     # Start with more initial bins
   )
   
   # Liberal binning (more bins, less stringent)
   liberal_binner = Chi2Binning(
       max_bins=25,
       min_bins=5,
       alpha=0.1,          # More permissive significance level
       initial_bins=30
   )

Comparison with Other Methods
-----------------------------

.. code-block:: python

   from binlearn.methods import EqualWidthBinning, TreeBinning
   from sklearn.metrics import accuracy_score
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split
   
   # Split data
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
   # Compare different binning methods
   binners = {
       'chi2': Chi2Binning(max_bins=8, alpha=0.05),
       'equal_width': EqualWidthBinning(n_bins=8),
       'supervised': SupervisedBinning(max_depth=3, min_samples_leaf=50)
   }
   
   results = {}
   classifier = RandomForestClassifier(random_state=42, n_estimators=100)
   
   for name, binner in binners.items():
       # Fit binner and transform data
       binner.fit(X_train, y_train)
       X_train_binned = binner.transform(X_train)
       X_test_binned = binner.transform(X_test)
       
       # Train classifier on binned data
       classifier.fit(X_train_binned, y_train)
       y_pred = classifier.predict(X_test_binned)
       
       results[name] = accuracy_score(y_test, y_pred)
       print(f"{name}: {results[name]:.3f} accuracy")

Parameter Tuning
-----------------

.. code-block:: python

   # Grid search for optimal parameters
   from sklearn.model_selection import GridSearchCV
   from sklearn.pipeline import Pipeline
   
   # Create pipeline with chi-square binning
   pipeline = Pipeline([
       ('binning', Chi2Binning()),
       ('classifier', RandomForestClassifier(random_state=42))
   ])
   
   # Parameter grid for binning
   param_grid = {
       'binning__max_bins': [5, 8, 12, 15],
       'binning__alpha': [0.001, 0.01, 0.05, 0.1],
       'binning__initial_bins': [10, 15, 20]
   }
   
   # Grid search
   grid_search = GridSearchCV(
       pipeline, 
       param_grid, 
       cv=5, 
       scoring='accuracy',
       n_jobs=-1
   )
   
   grid_search.fit(X_train, y_train)
   print(f"Best parameters: {grid_search.best_params_}")
   print(f"Best cross-validation score: {grid_search.best_score_:.3f}")

Parameter Guide
---------------

**max_bins** (int, default=10)
    Maximum number of bins to create. The algorithm will never exceed this number:
    
    * Higher values: Allow more granular binning
    * Lower values: Force more aggressive merging
    * Consider your downstream model's capacity

**min_bins** (int, default=2)
    Minimum number of bins to maintain. Prevents over-merging:
    
    * Higher values: Ensure sufficient granularity
    * Lower values: Allow aggressive simplification
    * Should be at least 2 for meaningful binning

**alpha** (float, default=0.05)
    Significance level for chi-square test. Lower values are more stringent:
    
    * Lower values (0.001): More conservative, fewer bins
    * Higher values (0.1): More liberal, more bins
    * Common values: 0.05 (standard), 0.01 (conservative)

**initial_bins** (int, default=10)
    Number of initial equal-width bins before merging:
    
    * Higher values: More potential split points to consider
    * Lower values: Faster computation but less flexibility
    * Should be >= max_bins

Statistical Background
----------------------

The chi-square statistic measures the independence between a feature's bins and the target classes:

.. math::

   \\chi^2 = \\sum_{i=1}^{r} \\sum_{j=1}^{c} \\frac{(O_{ij} - E_{ij})^2}{E_{ij}}

Where:
- :math:`O_{ij}` is the observed frequency in bin i, class j
- :math:`E_{ij}` is the expected frequency under independence
- Lower chi-square values indicate better independence (good for merging)

Handling Edge Cases
-------------------

.. code-block:: python

   # Handling insufficient data
   small_X = X[:50]  # Very small dataset
   small_y = y[:50]
   
   # Use conservative parameters for small datasets
   small_binner = Chi2Binning(
       max_bins=5,      # Fewer bins for small data
       min_bins=2,      # Conservative minimum
       alpha=0.1,       # More permissive for small samples
       initial_bins=8   # Fewer initial bins
   )
   
   small_binned = small_binner.fit_transform(small_X, small_y)

Tips for Best Results
---------------------

1. **Choose initial_bins wisely**: Start with 2-3x your desired max_bins
2. **Adjust alpha based on sample size**: Use smaller alpha for larger datasets
3. **Consider target distribution**: Imbalanced classes may need different alpha values
4. **Validate on holdout data**: Chi-square optimization can overfit to training data

See Also
--------

* :class:`SupervisedBinning` - Decision tree-based supervised binning
* :class:`IsotonicBinning` - Isotonic regression-based supervised binning
* :class:`EqualFrequencyBinning` - Quantile-based unsupervised binning
* :class:`KMeansBinning` - K-means clustering-based binning
