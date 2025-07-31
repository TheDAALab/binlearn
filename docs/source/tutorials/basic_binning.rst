Basic Binning Tutorial
=====================

This tutorial provides a hands-on introduction to the Binning Framework. You'll learn how to use different binning methods with real examples.

Getting Started
---------------

First, let's import the necessary libraries and generate some sample data:

.. code-block:: python

   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   from binning.methods import (
       EqualWidthBinning, 
       EqualFrequencyBinning,
       EqualWidthMinimumWeightBinning,
       SupervisedBinning
   )
   
   # Set random seed for reproducibility
   np.random.seed(42)
   
   # Generate sample data
   n_samples = 1000
   age = np.random.normal(35, 10, n_samples)
   income = np.random.lognormal(10, 0.8, n_samples)
   score = np.random.beta(2, 5, n_samples) * 100
   
   # Create DataFrame
   df = pd.DataFrame({
       'age': age,
       'income': income, 
       'score': score
   })
   
   print(f"Dataset shape: {df.shape}")
   print(df.describe())

Equal Width Binning
-------------------

Let's start with the simplest method - equal width binning:

.. code-block:: python

   # Initialize the binner
   ew_binner = EqualWidthBinning(n_bins=5, preserve_dataframe=True)
   
   # Fit and transform the data
   df_ew_binned = ew_binner.fit_transform(df)
   
   print("Original vs Binned Data:")
   print(f"Age range: {df['age'].min():.1f} - {df['age'].max():.1f}")
   print(f"Age bins: {sorted(df_ew_binned['age'].unique())}")
   
   # Examine bin edges
   print(f"Age bin edges: {ew_binner._bin_edges[0]}")

Visualizing Results
~~~~~~~~~~~~~~~~~~~

Let's visualize the binning results:

.. code-block:: python

   fig, axes = plt.subplots(2, 3, figsize=(15, 10))
   
   columns = ['age', 'income', 'score']
   
   for i, col in enumerate(columns):
       # Original distribution
       axes[0, i].hist(df[col], bins=20, alpha=0.7, color='skyblue')
       axes[0, i].set_title(f'Original {col.title()}')
       axes[0, i].set_ylabel('Frequency')
       
       # Binned distribution
       binned_values = df_ew_binned[col]
       bin_counts = binned_values.value_counts().sort_index()
       axes[1, i].bar(bin_counts.index, bin_counts.values, alpha=0.7, color='lightcoral')
       axes[1, i].set_title(f'Binned {col.title()} (Equal Width)')
       axes[1, i].set_xlabel('Bin')
       axes[1, i].set_ylabel('Count')
   
   plt.tight_layout()
   plt.show()

Equal Frequency Binning
-----------------------

Now let's try equal frequency binning, which creates bins with approximately equal sample counts:

.. code-block:: python

   # Initialize equal frequency binner
   ef_binner = EqualFrequencyBinning(n_bins=5, preserve_dataframe=True)
   
   # Fit and transform
   df_ef_binned = ef_binner.fit_transform(df)
   
   # Compare bin populations
   for col in columns:
       print(f"\n{col.title()} bin populations:")
       ew_counts = df_ew_binned[col].value_counts().sort_index()
       ef_counts = df_ef_binned[col].value_counts().sort_index()
       
       comparison_df = pd.DataFrame({
           'Equal Width': ew_counts,
           'Equal Frequency': ef_counts
       })
       print(comparison_df)

Working with Skewed Data
~~~~~~~~~~~~~~~~~~~~~~~~

Equal frequency binning is particularly useful for skewed data like income:

.. code-block:: python

   # Focus on the highly skewed income variable
   income_data = df[['income']]
   
   # Apply both methods
   ew_income = EqualWidthBinning(n_bins=5).fit_transform(income_data)
   ef_income = EqualFrequencyBinning(n_bins=5).fit_transform(income_data)
   
   # Create comparison plot
   fig, axes = plt.subplots(1, 3, figsize=(15, 5))
   
   # Original data
   axes[0].hist(income_data['income'], bins=50, alpha=0.7)
   axes[0].set_title('Original Income Distribution')
   axes[0].set_xlabel('Income')
   axes[0].set_ylabel('Frequency')
   
   # Equal width binning
   ew_counts = pd.Series(ew_income[:, 0]).value_counts().sort_index()
   axes[1].bar(ew_counts.index, ew_counts.values)
   axes[1].set_title('Equal Width Binning')
   axes[1].set_xlabel('Bin')
   axes[1].set_ylabel('Count')
   
   # Equal frequency binning  
   ef_counts = pd.Series(ef_income[:, 0]).value_counts().sort_index()
   axes[2].bar(ef_counts.index, ef_counts.values)
   axes[2].set_title('Equal Frequency Binning')
   axes[2].set_xlabel('Bin')
   axes[2].set_ylabel('Count')
   
   plt.tight_layout()
   plt.show()

Weight-Constrained Binning
---------------------------

The EqualWidthMinimumWeightBinning method allows you to ensure each bin meets minimum weight requirements:

.. code-block:: python

   # Create sample weights (e.g., importance or reliability scores)
   sample_weights = np.random.exponential(2.0, n_samples)
   
   # Apply weight-constrained binning
   ewmw_binner = EqualWidthMinimumWeightBinning(
       n_bins=8, 
       minimum_weight=100.0,  # Minimum total weight per bin
       preserve_dataframe=True
   )
   
   df_ewmw_binned = ewmw_binner.fit_transform(df, guidance_data=sample_weights)
   
   # Analyze the results
   print("Weight-constrained binning results:")
   for col in columns:
       print(f"\n{col.title()}:")
       
       # Calculate actual weights per bin
       binned_col = df_ewmw_binned[col]
       bin_weights = {}
       for bin_id in sorted(binned_col.unique()):
           mask = binned_col == bin_id
           total_weight = sample_weights[mask].sum()
           count = mask.sum()
           bin_weights[bin_id] = {'count': count, 'weight': total_weight}
       
       weight_df = pd.DataFrame(bin_weights).T
       print(weight_df)

Supervised Binning
------------------

For classification tasks, supervised binning optimizes bin boundaries based on the target variable:

.. code-block:: python

   # Create a binary target variable based on score
   y = (df['score'] > df['score'].median()).astype(int)
   
   # Apply supervised binning
   sup_binner = SupervisedBinning(n_bins=4, criterion='entropy')
   df_sup_binned = sup_binner.fit_transform(df, y)
   
   # Compare supervised vs unsupervised binning for the score variable
   score_data = df[['score']]
   
   # Get different binning results
   ew_score = EqualWidthBinning(n_bins=4).fit_transform(score_data)
   sup_score = sup_binner.transform(score_data)
   
   # Calculate information gain for each method
   from sklearn.metrics import mutual_info_score
   
   ew_info = mutual_info_score(y, ew_score[:, 0])
   sup_info = mutual_info_score(y, sup_score[:, 0])
   
   print(f"Information Gain - Equal Width: {ew_info:.4f}")
   print(f"Information Gain - Supervised: {sup_info:.4f}")
   print(f"Improvement: {(sup_info - ew_info) / ew_info * 100:.1f}%")

Working with Missing Values
---------------------------

The framework handles missing values gracefully:

.. code-block:: python

   # Introduce some missing values
   df_with_missing = df.copy()
   missing_mask = np.random.random(n_samples) < 0.1  # 10% missing
   df_with_missing.loc[missing_mask, 'age'] = np.nan
   
   # Apply binning
   binner = EqualWidthBinning(n_bins=5, preserve_dataframe=True)
   df_binned_missing = binner.fit_transform(df_with_missing)
   
   # Check how missing values are handled
   from binning.utils.constants import MISSING_VALUE
   
   print(f"Original missing values: {df_with_missing['age'].isna().sum()}")
   print(f"Missing values in binned data: {(df_binned_missing['age'] == MISSING_VALUE).sum()}")

Handling Edge Cases
-------------------

Constant Values
~~~~~~~~~~~~~~~

.. code-block:: python

   # Create data with constant values
   constant_data = pd.DataFrame({
       'constant': [5.0] * 100,
       'variable': np.random.rand(100)
   })
   
   binner = EqualWidthBinning(n_bins=3, preserve_dataframe=True)
   
   # This will generate warnings for constant columns
   import warnings
   with warnings.catch_warnings(record=True) as w:
       warnings.simplefilter("always")
       binned_constant = binner.fit_transform(constant_data)
       
       if w:
           print(f"Warning: {w[0].message}")
   
   print("Constant column binned values:", binned_constant['constant'].unique())

Outliers
~~~~~~~~

.. code-block:: python

   # Add extreme outliers
   outlier_data = df.copy()
   outlier_data.loc[0, 'age'] = 200  # Extreme outlier
   outlier_data.loc[1, 'age'] = -50  # Negative outlier
   
   # Compare with and without clipping
   binner_no_clip = EqualWidthBinning(n_bins=5, clip=False)
   binner_with_clip = EqualWidthBinning(n_bins=5, clip=True)
   
   age_no_clip = binner_no_clip.fit_transform(outlier_data[['age']])
   age_with_clip = binner_with_clip.fit_transform(outlier_data[['age']])
   
   print("Without clipping - unique bins:", np.unique(age_no_clip))
   print("With clipping - unique bins:", np.unique(age_with_clip))

Pipeline Integration
--------------------

Use binning in sklearn pipelines:

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import cross_val_score
   
   # Create a pipeline with binning
   pipeline = Pipeline([
       ('binning', EqualFrequencyBinning(n_bins=5)),
       ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
   ])
   
   # Evaluate pipeline performance
   X = df[['age', 'income', 'score']].values
   scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
   
   print(f"Pipeline Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
   
   # Compare with pipeline without binning
   pipeline_no_binning = Pipeline([
       ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
   ])
   
   scores_no_binning = cross_val_score(pipeline_no_binning, X, y, cv=5, scoring='accuracy')
   print(f"No Binning Accuracy: {scores_no_binning.mean():.3f} (+/- {scores_no_binning.std() * 2:.3f})")

Advanced Configuration
----------------------

Custom Bin Ranges
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Specify custom ranges for binning
   custom_binner = EqualWidthBinning(
       n_bins=5, 
       bin_range=(20, 60),  # Focus on ages 20-60
       clip=True  # Clip outliers to this range
   )
   
   age_custom = custom_binner.fit_transform(df[['age']])
   print("Custom range bin edges:", custom_binner._bin_edges[0])

Independent vs Joint Fitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Fit each column independently
   independent_binner = EqualWidthBinning(n_bins=3, fit_jointly=False)
   df_independent = independent_binner.fit_transform(df)
   
   # Fit all columns with same parameters
   joint_binner = EqualWidthBinning(n_bins=3, fit_jointly=True)
   df_joint = joint_binner.fit_transform(df)
   
   print("Independent fitting - unique bins per column:")
   for col in columns:
       print(f"{col}: {sorted(df_independent[col].unique())}")
   
   print("\nJoint fitting - bins should be similar:")
   for col in columns:
       print(f"{col}: {sorted(df_joint[col].unique())}")

Summary
-------

In this tutorial, you learned:

1. **Basic binning concepts** and when to use different methods
2. **Equal width binning** for uniform bin sizes
3. **Equal frequency binning** for balanced bin populations
4. **Weight-constrained binning** for importance-weighted data
5. **Supervised binning** for classification preprocessing
6. **Handling missing values and edge cases**
7. **Pipeline integration** with sklearn
8. **Advanced configuration options**

Next Steps
----------

* Try the :doc:`advanced_binning` tutorial for more sophisticated techniques
* Explore :doc:`../examples/equal_width_minimum_weight_binning` for detailed examples
* Check the :doc:`../api/index` for complete API documentation
* Practice with your own datasets using different binning methods
