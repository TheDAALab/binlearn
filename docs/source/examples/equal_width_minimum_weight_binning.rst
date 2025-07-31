EqualWidthMinimumWeightBinning Examples
======================================

This guide provides comprehensive examples of using the EqualWidthMinimumWeightBinning method for weight-constrained binning scenarios.

Overview
--------

EqualWidthMinimumWeightBinning creates equal-width bins while ensuring each bin meets minimum weight requirements from guidance data. When bins don't meet the weight threshold, adjacent bins are merged to satisfy the constraint.

Basic Usage
-----------

Simple Weight-Constrained Binning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import pandas as pd
   from binning.methods import EqualWidthMinimumWeightBinning
   
   # Generate sample data
   np.random.seed(42)
   n_samples = 1000
   
   # Features to bin
   data = pd.DataFrame({
       'age': np.random.normal(35, 10, n_samples),
       'income': np.random.lognormal(10, 0.5, n_samples),
       'score': np.random.beta(2, 5, n_samples) * 100
   })
   
   # Sample weights (importance or reliability scores)
   sample_weights = np.random.exponential(2.0, n_samples)
   
   # Create and apply the binner
   binner = EqualWidthMinimumWeightBinning(
       n_bins=6, 
       minimum_weight=50.0,
       preserve_dataframe=True
   )
   
   # Fit and transform with guidance data
   binned_data = binner.fit_transform(data, guidance_data=sample_weights)
   
   print("Basic binning results:")
   print(f"Original shape: {data.shape}")
   print(f"Binned shape: {binned_data.shape}")
   print(f"Unique bins per column:")
   for col in data.columns:
       unique_bins = len(binned_data[col].unique())
       print(f"  {col}: {unique_bins} bins")

Analyzing Weight Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Analyze how weights are distributed across bins
   def analyze_bin_weights(original_data, binned_data, weights, column):
       """Analyze weight distribution in bins for a specific column."""
       print(f"\nWeight analysis for {column}:")
       print("-" * 40)
       
       binned_col = binned_data[column]
       original_col = original_data[column]
       
       for bin_id in sorted(binned_col.unique()):
           mask = binned_col == bin_id
           bin_weight = weights[mask].sum()
           bin_count = mask.sum()
           bin_range = (original_col[mask].min(), original_col[mask].max())
           
           print(f"Bin {bin_id}:")
           print(f"  Range: [{bin_range[0]:.2f}, {bin_range[1]:.2f}]")
           print(f"  Count: {bin_count}")
           print(f"  Total Weight: {bin_weight:.2f}")
           print(f"  Avg Weight per Sample: {bin_weight/bin_count:.2f}")
   
   # Analyze each column
   for col in data.columns:
       analyze_bin_weights(data, binned_data, sample_weights, col)

Advanced Configuration
----------------------

Custom Weight Thresholds
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Test different minimum weight thresholds
   weight_thresholds = [10.0, 25.0, 50.0, 100.0]
   
   results = {}
   
   for min_weight in weight_thresholds:
       binner = EqualWidthMinimumWeightBinning(
           n_bins=8,  # Start with more bins
           minimum_weight=min_weight,
           preserve_dataframe=True
       )
       
       binned = binner.fit_transform(data, guidance_data=sample_weights)
       
       # Count actual bins for each column
       bin_counts = {}
       for col in data.columns:
           bin_counts[col] = len(binned[col].unique())
       
       results[min_weight] = bin_counts
   
   # Display results
   print("Effect of minimum weight threshold:")
   print("-" * 50)
   print(f"{'Min Weight':<12} {'Age Bins':<10} {'Income Bins':<12} {'Score Bins'}")
   print("-" * 50)
   
   for min_weight, counts in results.items():
       print(f"{min_weight:<12} {counts['age']:<10} {counts['income']:<12} {counts['score']}")

Different Weight Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Test with different weight distributions
   weight_types = {
       'uniform': np.random.uniform(1, 5, n_samples),
       'exponential': np.random.exponential(2, n_samples), 
       'normal': np.abs(np.random.normal(3, 1, n_samples)),
       'heavy_tailed': np.random.pareto(1, n_samples) + 1
   }
   
   print("Binning with different weight distributions:")
   print("=" * 60)
   
   for weight_name, weights in weight_types.items():
       print(f"\n{weight_name.title()} Weights:")
       print(f"Mean: {weights.mean():.2f}, Std: {weights.std():.2f}")
       
       binner = EqualWidthMinimumWeightBinning(
           n_bins=6,
           minimum_weight=weights.mean() * 20,  # Adaptive threshold
           preserve_dataframe=True
       )
       
       binned = binner.fit_transform(data, guidance_data=weights)
       
       # Analyze results
       for col in ['age']:  # Focus on one column for brevity
           unique_bins = len(binned[col].unique())
           print(f"  {col}: {unique_bins} bins (from {6} requested)")

Real-World Applications
-----------------------

Quality Control in Manufacturing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Manufacturing scenario: bin products by dimensions with quality weights
   
   # Generate manufacturing data
   np.random.seed(123)
   n_products = 2000
   
   manufacturing_data = pd.DataFrame({
       'length': np.random.normal(100, 5, n_products),      # mm
       'width': np.random.normal(50, 3, n_products),        # mm  
       'thickness': np.random.normal(10, 1, n_products),    # mm
       'weight': np.random.normal(500, 50, n_products)      # grams
   })
   
   # Quality scores as weights (higher = better quality)
   quality_scores = np.random.beta(5, 2, n_products) * 10  # 0-10 scale
   
   # Apply quality-weighted binning
   quality_binner = EqualWidthMinimumWeightBinning(
       n_bins=5,
       minimum_weight=15.0,  # Minimum quality per bin
       preserve_dataframe=True
   )
   
   quality_binned = quality_binner.fit_transform(
       manufacturing_data, 
       guidance_data=quality_scores
   )
   
   print("Manufacturing Quality Control Binning:")
   print("-" * 45)
   
   for dimension in manufacturing_data.columns:
       # Calculate quality statistics per bin
       binned_col = quality_binned[dimension]
       original_col = manufacturing_data[dimension]
       
       print(f"\n{dimension.title()} Dimension:")
       for bin_id in sorted(binned_col.unique()):
           mask = binned_col == bin_id
           bin_quality = quality_scores[mask].sum()
           bin_count = mask.sum()
           avg_quality = quality_scores[mask].mean()
           dimension_range = (original_col[mask].min(), original_col[mask].max())
           
           print(f"  Bin {bin_id}: Range [{dimension_range[0]:.1f}, {dimension_range[1]:.1f}]")
           print(f"    Products: {bin_count}, Total Quality: {bin_quality:.1f}")
           print(f"    Avg Quality: {avg_quality:.2f}")

Customer Segmentation with Importance Weighting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Customer segmentation with revenue-based weights
   
   # Generate customer data
   np.random.seed(456)
   n_customers = 1500
   
   customer_data = pd.DataFrame({
       'age': np.random.normal(40, 15, n_customers),
       'annual_spend': np.random.lognormal(8, 1, n_customers),
       'loyalty_score': np.random.beta(3, 2, n_customers) * 100,
       'engagement_index': np.random.gamma(2, 2, n_customers)
   })
   
   # Revenue weights (some customers are more valuable)
   revenue_weights = np.random.lognormal(6, 1.5, n_customers)
   
   # Segment customers with revenue-weighted binning
   segmentation_binner = EqualWidthMinimumWeightBinning(
       n_bins=4,  # Create 4 segments
       minimum_weight=np.percentile(revenue_weights, 60),  # 60th percentile threshold
       preserve_dataframe=True
   )
   
   customer_segments = segmentation_binner.fit_transform(
       customer_data,
       guidance_data=revenue_weights
   )
   
   print("Customer Segmentation Results:")
   print("=" * 40)
   
   # Analyze segments
   for metric in customer_data.columns:
       print(f"\n{metric.replace('_', ' ').title()} Segments:")
       binned_metric = customer_segments[metric]
       original_metric = customer_data[metric]
       
       segment_stats = []
       for seg_id in sorted(binned_metric.unique()):
           mask = binned_metric == seg_id
           stats = {
               'segment': seg_id,
               'customers': mask.sum(),
               'total_revenue': revenue_weights[mask].sum(),
               'avg_revenue': revenue_weights[mask].mean(),
               'metric_range': f"[{original_metric[mask].min():.1f}, {original_metric[mask].max():.1f}]"
           }
           segment_stats.append(stats)
       
       # Display as table
       for stats in segment_stats:
           print(f"  Segment {stats['segment']}: {stats['customers']} customers")
           print(f"    Revenue: Total ${stats['total_revenue']:,.0f}, Avg ${stats['avg_revenue']:,.0f}")
           print(f"    {metric.title()} Range: {stats['metric_range']}")

Error Handling and Edge Cases
------------------------------

Insufficient Weight Scenarios
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Test behavior when total weight is insufficient
   
   # Small dataset with low weights
   small_data = pd.DataFrame({
       'feature': np.random.normal(0, 1, 100)
   })
   
   low_weights = np.random.uniform(0.1, 0.5, 100)  # Very low weights
   
   try:
       # This might result in fewer bins than requested
       binner = EqualWidthMinimumWeightBinning(
           n_bins=10,  # Request many bins
           minimum_weight=20.0,  # High threshold
           preserve_dataframe=True
       )
       
       result = binner.fit_transform(small_data, guidance_data=low_weights)
       
       actual_bins = len(result['feature'].unique())
       print(f"Requested bins: 10, Actual bins: {actual_bins}")
       
       # Analyze what happened
       for bin_id in sorted(result['feature'].unique()):
           mask = result['feature'] == bin_id
           bin_weight = low_weights[mask].sum()
           print(f"Bin {bin_id}: weight = {bin_weight:.2f}")
           
   except Exception as e:
       print(f"Error occurred: {e}")

Handling Missing Values
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from binning.utils.constants import MISSING_VALUE
   
   # Data with missing values
   data_with_missing = pd.DataFrame({
       'feature1': np.concatenate([np.random.normal(0, 1, 80), [np.nan] * 20]),
       'feature2': np.concatenate([np.random.normal(5, 2, 90), [np.nan] * 10])
   })
   
   # Weights for all samples (including those with missing features)
   all_weights = np.random.exponential(1, 100)
   
   binner = EqualWidthMinimumWeightBinning(
       n_bins=4,
       minimum_weight=5.0,
       preserve_dataframe=True
   )
   
   result_with_missing = binner.fit_transform(
       data_with_missing, 
       guidance_data=all_weights
   )
   
   print("Handling missing values:")
   for col in data_with_missing.columns:
       n_missing_original = data_with_missing[col].isna().sum()
       n_missing_binned = (result_with_missing[col] == MISSING_VALUE).sum()
       
       print(f"{col}:")
       print(f"  Original missing: {n_missing_original}")
       print(f"  Binned missing: {n_missing_binned}")
       print(f"  Unique bins: {len(result_with_missing[col].unique())}")

Performance Considerations
--------------------------

Large Dataset Handling
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Simulate performance with larger datasets
   import time
   
   dataset_sizes = [1000, 5000, 10000, 25000]
   
   print("Performance Analysis:")
   print("-" * 40)
   print(f"{'Size':<8} {'Time (s)':<10} {'Bins':<6} {'Memory (MB)':<12}")
   print("-" * 40)
   
   for size in dataset_sizes:
       # Generate data
       large_data = pd.DataFrame({
           'feature1': np.random.normal(0, 1, size),
           'feature2': np.random.exponential(2, size),
           'feature3': np.random.beta(2, 5, size) * 100
       })
       
       large_weights = np.random.gamma(2, 2, size)
       
       # Time the operation
       start_time = time.time()
       
       binner = EqualWidthMinimumWeightBinning(
           n_bins=6,
           minimum_weight=np.mean(large_weights) * 10,
           preserve_dataframe=False  # Save memory
       )
       
       result = binner.fit_transform(large_data, guidance_data=large_weights)
       
       end_time = time.time()
       
       # Calculate metrics
       elapsed = end_time - start_time
       n_bins = len(np.unique(result[:, 0]))  # Check first column
       memory_mb = result.nbytes / (1024 * 1024)
       
       print(f"{size:<8} {elapsed:<10.3f} {n_bins:<6} {memory_mb:<12.2f}")

Integration Patterns
--------------------

Scikit-learn Pipeline Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import accuracy_score
   
   # Create classification dataset
   np.random.seed(789)
   n_samples = 2000
   
   X = pd.DataFrame({
       'feature1': np.random.normal(0, 1, n_samples),
       'feature2': np.random.exponential(2, n_samples),
       'feature3': np.random.beta(2, 5, n_samples) * 100
   })
   
   # Create target variable
   y = ((X['feature1'] > 0) & (X['feature2'] > 2)).astype(int)
   
   # Feature importance weights (could be from domain knowledge)
   feature_weights = np.random.gamma(3, 1, n_samples)
   
   # Split data
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.3, random_state=42
   )
   
   weights_train = feature_weights[:len(X_train)]  # Corresponding weights
   
   # Create pipeline with weight-constrained binning
   pipeline = Pipeline([
       ('binning', EqualWidthMinimumWeightBinning(
           n_bins=5, 
           minimum_weight=np.mean(feature_weights) * 15
       )),
       ('scaling', StandardScaler()),
       ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
   ])
   
   # Fit pipeline (note: guidance_data passed to binning step)
   pipeline.fit(X_train, y_train, binning__guidance_data=weights_train)
   
   # Predict and evaluate
   y_pred = pipeline.predict(X_test)
   accuracy = accuracy_score(y_test, y_pred)
   
   print(f"\nPipeline Results:")
   print(f"Accuracy: {accuracy:.3f}")
   
   # Check binning results
   binning_step = pipeline.named_steps['binning']
   print(f"Actual bins created: {[len(np.unique(X_train.iloc[:, i])) for i in range(X_train.shape[1])]}")

Custom Transformer Class
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.base import BaseEstimator, TransformerMixin
   
   class WeightConstrainedBinningTransformer(BaseEstimator, TransformerMixin):
       """Custom sklearn-compatible transformer for weight-constrained binning."""
       
       def __init__(self, n_bins=5, minimum_weight=10.0):
           self.n_bins = n_bins
           self.minimum_weight = minimum_weight
           self.binner_ = None
       
       def fit(self, X, y=None, sample_weight=None):
           """Fit the binning transformer."""
           self.binner_ = EqualWidthMinimumWeightBinning(
               n_bins=self.n_bins,
               minimum_weight=self.minimum_weight,
               preserve_dataframe=False
           )
           
           if sample_weight is not None:
               self.binner_.fit(X, guidance_data=sample_weight)
           else:
               # Use uniform weights if none provided
               uniform_weights = np.ones(len(X))
               self.binner_.fit(X, guidance_data=uniform_weights)
           
           return self
       
       def transform(self, X):
           """Transform using fitted binning parameters."""
           if self.binner_ is None:
               raise ValueError("Transformer not fitted yet.")
           
           return self.binner_.transform(X)
       
       def fit_transform(self, X, y=None, sample_weight=None):
           """Fit and transform in one step."""
           return self.fit(X, y, sample_weight).transform(X)
   
   # Use the custom transformer
   custom_transformer = WeightConstrainedBinningTransformer(
       n_bins=4, 
       minimum_weight=20.0
   )
   
   # Test with sample weights
   sample_X = np.random.rand(500, 3)
   sample_weights = np.random.exponential(2, 500)
   
   transformed = custom_transformer.fit_transform(sample_X, sample_weight=sample_weights)
   
   print("Custom transformer results:")
   print(f"Input shape: {sample_X.shape}")
   print(f"Output shape: {transformed.shape}")
   print(f"Unique bins per feature: {[len(np.unique(transformed[:, i])) for i in range(3)]}")

Best Practices Summary
----------------------

1. **Weight Selection**: Choose guidance weights that represent true importance or reliability
2. **Threshold Setting**: Set minimum_weight based on statistical significance requirements  
3. **Bin Count**: Start with more bins than needed; let merging optimize the final count
4. **Validation**: Always verify that merged bins still make domain sense
5. **Performance**: Use preserve_dataframe=False for large datasets to save memory
6. **Integration**: Leverage sklearn pipelines for complex preprocessing workflows

Common Pitfalls
---------------

1. **Over-constraining**: Setting minimum_weight too high can result in very few bins
2. **Under-weighting**: Very low weights may not provide meaningful constraints
3. **Ignoring merging**: Merged bins may span ranges that don't make business sense
4. **Memory issues**: Large datasets with preserve_dataframe=True can consume excessive memory

Next Steps
----------

* Experiment with different weight distributions for your use case
* Integrate with your existing ML pipelines
* Consider ensemble approaches combining multiple binning strategies  
* Explore domain-specific weight calculation methods
