EqualWidthMinimumWeightBinning
================================

.. currentmodule:: binlearn.methods

.. autoclass:: EqualWidthMinimumWeightBinning
   :members:
   :inherited-members:
   :show-inheritance:

Overview
--------

``EqualWidthMinimumWeightBinning`` creates bins of equal width across the range of each feature, 
but adjusts the number of bins to ensure each bin contains at least the specified minimum total 
weight from the guidance column. This method combines the interpretability of equal-width binning 
with weight-based constraints for more balanced bins.

This approach is particularly useful when:

* You want **interpretable equal-width bins** but need weight balance
* You have **sample weights** or **importance scores** that should be considered
* You need to ensure **statistical significance** in each bin
* You want to **prevent empty or sparse bins** in weighted scenarios

Key Features
------------

* **Equal-Width Foundation**: Starts with equal-width intervals for interpretability
* **Weight-Based Adjustment**: Ensures minimum weight per bin from guidance column
* **Automatic Merging**: Intelligently merges bins that don't meet weight requirements
* **Flexible Weight Sources**: Supports any numeric column as weight guidance
* **Range Control**: Optional custom range specification for binning
* **Robust Validation**: Comprehensive error handling and data validation
* **Sklearn Compatibility**: Full transformer interface with fit/transform methods
* **DataFrame Support**: Preserves pandas/polars column names and structure

Basic Usage
-----------

.. code-block:: python

   import numpy as np
   import pandas as pd
   from binlearn.methods import EqualWidthMinimumWeightBinning
   
   # Create sample data with weights
   np.random.seed(42)
   X = np.random.uniform(0, 100, 1000).reshape(-1, 1)
   weights = np.random.exponential(2, 1000)  # Some samples more important
   
   # Apply equal-width binning with minimum weight constraint
   binner = EqualWidthMinimumWeightBinning(
       n_bins=10,
       minimum_weight=50.0  # Each bin must have at least 50 total weight
   )
   
   X_binned = binner.fit_transform(X, guidance_data=weights)
   
   print(f"Original shape: {X.shape}")
   print(f"Binned shape: {X_binned.shape}")
   print(f"Final number of bins: {len(binner.bin_edges_[0]) - 1}")
   print(f"Bin edges: {binner.bin_edges_[0]}")

DataFrame Example with Weight Column
------------------------------------

.. code-block:: python

   # Create DataFrame with features and weight column
   df = pd.DataFrame({
       'income': np.random.lognormal(10, 1, 2000),
       'age': np.random.uniform(18, 80, 2000),
       'transaction_amount': np.random.exponential(100, 2000),
       'sample_weight': np.random.gamma(2, 2, 2000)  # Weights for each sample
   })
   
   # Bin income using transaction_amount as weights
   income_binner = EqualWidthMinimumWeightBinning(
       guidance_columns=['sample_weight'],
       n_bins=8,
       minimum_weight=20.0,
       preserve_dataframe=True
   )
   
   df_binned = income_binner.fit_transform(df)
   
   print(f"Income bins created: {len(income_binner.bin_edges_['income']) - 1}")
   print(f"Income bin edges: {income_binner.bin_edges_['income']}")

Survey Data Example
-------------------

.. code-block:: python

   # Example with survey data where response weights matter
   survey_df = pd.DataFrame({
       'satisfaction_score': np.random.uniform(1, 10, 1500),
       'response_time': np.random.lognormal(3, 1, 1500),
       'respondent_weight': np.random.choice([0.5, 1.0, 1.5, 2.0], 1500)  # Survey weights
   })
   
   # Bin satisfaction ensuring each bin has sufficient weighted responses
   satisfaction_binner = EqualWidthMinimumWeightBinning(
       guidance_columns=['respondent_weight'],
       n_bins=5,
       minimum_weight=50.0,  # At least 50 weighted responses per bin
       bin_range=(1, 10),    # Fixed range for satisfaction scores
       preserve_dataframe=True
   )
   
   survey_binned = satisfaction_binner.fit_transform(survey_df)
   
   # Verify weights per bin
   for i, (start, end) in enumerate(zip(satisfaction_binner.bin_edges_['satisfaction_score'][:-1],
                                       satisfaction_binner.bin_edges_['satisfaction_score'][1:])):
       mask = (survey_df['satisfaction_score'] >= start) & (survey_df['satisfaction_score'] < end)
       total_weight = survey_df.loc[mask, 'respondent_weight'].sum()
       print(f"Bin {i} [{start:.1f}, {end:.1f}): {total_weight:.1f} total weight")

Advanced Configuration
----------------------

.. code-block:: python

   # Fine-tuned binning for different scenarios
   
   # Conservative binning (higher weight requirements)
   conservative_binner = EqualWidthMinimumWeightBinning(
       n_bins=6,
       minimum_weight=100.0,    # High weight requirement
       bin_range=(0, 1000),     # Fixed range
       clip=True                # Clip outliers
   )
   
   # Adaptive binning (lower weight requirements, more bins)
   adaptive_binner = EqualWidthMinimumWeightBinning(
       n_bins="sqrt",           # Dynamic based on sample size
       minimum_weight=10.0,     # Lower weight requirement
       clip=False               # Preserve outliers
   )

Financial Risk Example
----------------------

.. code-block:: python

   # Financial data where exposure amounts act as weights
   financial_df = pd.DataFrame({
       'credit_score': np.random.normal(700, 100, 10000),
       'loan_amount': np.random.lognormal(10, 1, 10000),
       'exposure': np.random.exponential(50000, 10000)  # Dollar exposure per loan
   })
   
   # Bin credit scores ensuring each bin has sufficient exposure
   credit_binner = EqualWidthMinimumWeightBinning(
       guidance_columns=['exposure'],
       n_bins=10,
       minimum_weight=500000,  # At least $500K exposure per bin
       bin_range=(300, 850),   # Standard credit score range
       preserve_dataframe=True
   )
   
   financial_binned = credit_binner.fit_transform(financial_df)
   
   # Analyze exposure distribution across bins
   bin_stats = []
   for i, (start, end) in enumerate(zip(credit_binner.bin_edges_['credit_score'][:-1],
                                       credit_binner.bin_edges_['credit_score'][1:])):
       mask = (financial_df['credit_score'] >= start) & (financial_df['credit_score'] < end)
       total_exposure = financial_df.loc[mask, 'exposure'].sum()
       avg_score = financial_df.loc[mask, 'credit_score'].mean()
       count = mask.sum()
       
       bin_stats.append({
           'bin': f"[{start:.0f}, {end:.0f})",
           'avg_score': avg_score,
           'count': count,
           'total_exposure': total_exposure,
           'avg_exposure': total_exposure / count if count > 0 else 0
       })
   
   for stats in bin_stats:
       print(f"Bin {stats['bin']}: {stats['count']} loans, "
             f"${stats['total_exposure']:,.0f} exposure, "
             f"avg score {stats['avg_score']:.0f}")

Comparison with Standard Equal-Width
------------------------------------

.. code-block:: python

   from binlearn.methods import EqualWidthBinning
   
   # Compare standard equal-width vs. minimum weight equal-width
   X_sample = np.random.exponential(2, 500).reshape(-1, 1)  # Skewed data
   weights = np.random.exponential(1, 500)
   
   # Standard equal-width binning
   standard_binner = EqualWidthBinning(n_bins=8)
   X_standard = standard_binner.fit_transform(X_sample)
   
   # Equal-width with minimum weight
   weighted_binner = EqualWidthMinimumWeightBinning(
       n_bins=8,
       minimum_weight=15.0
   )
   X_weighted = weighted_binner.fit_transform(X_sample, guidance_data=weights)
   
   print("Standard equal-width:")
   print(f"  Bins created: {len(standard_binner.bin_edges_[0]) - 1}")
   print(f"  Bin edges: {standard_binner.bin_edges_[0]}")
   
   print("\\nWeight-constrained equal-width:")
   print(f"  Bins created: {len(weighted_binner.bin_edges_[0]) - 1}")
   print(f"  Bin edges: {weighted_binner.bin_edges_[0]}")
   
   # Analyze weight distribution in standard bins
   print("\\nWeight distribution in standard bins:")
   for i, (start, end) in enumerate(zip(standard_binner.bin_edges_[0][:-1],
                                       standard_binner.bin_edges_[0][1:])):
       mask = (X_sample.flatten() >= start) & (X_sample.flatten() < end)
       total_weight = weights[mask].sum()
       print(f"  Bin {i}: {total_weight:.1f} total weight")

Scikit-learn Pipeline Integration
---------------------------------

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split
   from sklearn.utils.class_weight import compute_sample_weight
   
   # Create classification data with sample weights
   from sklearn.datasets import make_classification
   X, y = make_classification(n_samples=2000, n_features=4, n_classes=2, random_state=42)
   
   # Compute sample weights (e.g., for imbalanced data)
   sample_weights = compute_sample_weight('balanced', y)
   
   # Create pipeline with weight-aware binning
   pipeline = Pipeline([
       ('binning', EqualWidthMinimumWeightBinning(
           n_bins=6,
           minimum_weight=20.0
       )),
       ('classifier', RandomForestClassifier(random_state=42))
   ])
   
   # Split data
   X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
       X, y, sample_weights, test_size=0.2, random_state=42
   )
   
   # Fit pipeline with weights
   pipeline.fit(X_train, y_train, 
                binning__guidance_data=weights_train)  # Pass weights to binning step
   
   accuracy = pipeline.score(X_test, y_test)
   print(f"Pipeline accuracy: {accuracy:.3f}")

Parameter Guide
---------------

**n_bins** (int or str, default=10)
    Initial number of equal-width bins to create:
    
    * int: Direct specification (e.g., 10)
    * "sqrt": Square root of number of samples
    * "log": Natural logarithm of number of samples
    * Actual bins may be fewer due to weight constraints

**minimum_weight** (float, default=1.0)
    Minimum total weight required per bin:
    
    * Higher values: Fewer, more stable bins
    * Lower values: More bins, potentially less stable
    * Should reflect your statistical significance requirements

**bin_range** (tuple, optional)
    Custom range for binning as (min, max):
    
    * None: Uses data min/max
    * Fixed range: Ensures consistent bins across datasets
    * Useful for scores with known ranges (e.g., 0-100)

**guidance_columns** (list, optional)
    Columns providing weights for bin constraints:
    
    * Should contain positive numeric values
    * Can be sample weights, importance scores, etc.
    * Used only for weight calculation, not bin placement

Handling Edge Cases
-------------------

.. code-block:: python

   # Insufficient total weight scenario
   sparse_X = np.random.uniform(0, 100, 50).reshape(-1, 1)
   sparse_weights = np.ones(50) * 0.1  # Very low weights
   
   # Algorithm will create fewer bins to meet weight requirements
   sparse_binner = EqualWidthMinimumWeightBinning(
       n_bins=10,
       minimum_weight=2.0  # May be too high for this data
   )
   
   try:
       sparse_binned = sparse_binner.fit_transform(sparse_X, guidance_data=sparse_weights)
       print(f"Created {len(sparse_binner.bin_edges_[0]) - 1} bins (requested 10)")
   except Exception as e:
       print(f"Error: {e}")
       # Reduce minimum_weight or increase data

Tips for Best Results
---------------------

1. **Set appropriate minimum_weight**: Balance statistical significance with granularity
2. **Consider your weight distribution**: Check weight statistics before setting constraints
3. **Use fixed ranges when appropriate**: Ensures consistent binning across datasets
4. **Validate weight requirements**: Ensure total weight can support desired number of bins
5. **Monitor bin merging**: Check if many bins are being merged due to weight constraints

See Also
--------

* :class:`EqualWidthBinning` - Standard equal-width binning without weight constraints
* :class:`EqualFrequencyBinning` - Quantile-based binning for balanced sample counts
* :class:`SupervisedBinning` - Decision tree-based supervised binning
* :class:`KMeansBinning` - K-means clustering-based binning
