Advanced Binning Techniques
==========================

This tutorial covers advanced usage patterns and sophisticated binning techniques for complex scenarios.

Custom Binning Strategies
--------------------------

Creating Custom Binners
~~~~~~~~~~~~~~~~~~~~~~~~

You can create custom binning strategies by inheriting from the base classes:

.. code-block:: python

   import numpy as np
   from binning.base import BaseBinner
   from binning.utils.constants import MISSING_VALUE
   
   class PercentileBinning(BaseBinner):
       """Custom binner using specific percentiles."""
       
       def __init__(self, percentiles=None, **kwargs):
           super().__init__(**kwargs)
           self.percentiles = percentiles or [20, 40, 60, 80]
       
       def _fit_column(self, column_data, column_index, **kwargs):
           """Fit binning parameters for a single column."""
           # Remove missing values for fitting
           valid_data = column_data[~np.isnan(column_data)]
           
           if len(valid_data) == 0:
               # Handle all-missing case
               self._bin_edges[column_index] = np.array([0, 1])
               return
           
           # Calculate percentile-based edges
           edges = [valid_data.min()]
           for p in self.percentiles:
               edges.append(np.percentile(valid_data, p))
           edges.append(valid_data.max())
           
           # Remove duplicates and sort
           edges = sorted(list(set(edges)))
           self._bin_edges[column_index] = np.array(edges)
       
       def _transform_column(self, column_data, column_index):
           """Transform a single column using fitted parameters."""
           edges = self._bin_edges[column_index]
           
           # Handle missing values
           missing_mask = np.isnan(column_data)
           result = np.full(len(column_data), MISSING_VALUE, dtype=int)
           
           if not missing_mask.all():
               # Bin non-missing values
               valid_data = column_data[~missing_mask]
               binned = np.digitize(valid_data, edges[1:-1])
               result[~missing_mask] = binned
           
           return result
   
   # Use the custom binner
   import pandas as pd
   np.random.seed(42)
   
   data = pd.DataFrame({
       'feature1': np.random.exponential(2, 1000),
       'feature2': np.random.normal(50, 15, 1000)
   })
   
   custom_binner = PercentileBinning(percentiles=[10, 30, 70, 90])
   binned_data = custom_binner.fit_transform(data)
   
   print("Custom percentile binning results:")
   print(pd.DataFrame(binned_data, columns=data.columns))

Multi-Stage Binning
~~~~~~~~~~~~~~~~~~~~

Combine multiple binning strategies for complex preprocessing:

.. code-block:: python

   from binning.methods import EqualWidthBinning, EqualFrequencyBinning
   from sklearn.preprocessing import StandardScaler
   
   class MultiStageBinner:
       """Apply different binning strategies in sequence."""
       
       def __init__(self):
           self.stage1 = EqualWidthBinning(n_bins=10)  # Coarse binning
           self.stage2 = EqualFrequencyBinning(n_bins=5)  # Refined binning
           self.scaler = StandardScaler()
       
       def fit_transform(self, X, y=None):
           # Stage 1: Coarse equal-width binning
           X_stage1 = self.stage1.fit_transform(X)
           
           # Stage 2: Scale the binned values
           X_scaled = self.scaler.fit_transform(X_stage1.astype(float))
           
           # Stage 3: Final equal-frequency binning
           X_final = self.stage2.fit_transform(X_scaled)
           
           return X_final
   
   multi_binner = MultiStageBinner()
   multi_result = multi_binner.fit_transform(data)
   
   print("Multi-stage binning results:")
   for i, col in enumerate(data.columns):
       unique_vals = np.unique(multi_result[:, i])
       print(f"{col}: {len(unique_vals)} unique bins")

Adaptive Binning
-----------------

Dynamic Bin Count Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Automatically determine optimal bin count based on data characteristics:

.. code-block:: python

   from binning.methods import EqualWidthBinning
   from scipy.stats import entropy
   
   class AdaptiveBinner:
       """Automatically select optimal number of bins."""
       
       def __init__(self, min_bins=3, max_bins=20, method='entropy'):
           self.min_bins = min_bins
           self.max_bins = max_bins
           self.method = method
           self.optimal_bins_ = {}
           self.binners_ = {}
       
       def _score_binning(self, data, n_bins):
           """Score a binning configuration."""
           binner = EqualWidthBinning(n_bins=n_bins)
           binned = binner.fit_transform(data.reshape(-1, 1))
           
           if self.method == 'entropy':
               # Higher entropy = better distribution
               _, counts = np.unique(binned, return_counts=True)
               return entropy(counts)
           elif self.method == 'variance':
               # Lower variance in bin sizes = better
               _, counts = np.unique(binned, return_counts=True)
               return -np.var(counts)
           else:
               raise ValueError(f"Unknown scoring method: {self.method}")
       
       def fit_transform(self, X):
           X = np.asarray(X)
           if X.ndim == 1:
               X = X.reshape(-1, 1)
           
           result = np.zeros_like(X, dtype=int)
           
           for col_idx in range(X.shape[1]):
               column_data = X[:, col_idx]
               valid_data = column_data[~np.isnan(column_data)]
               
               if len(valid_data) < self.min_bins:
                   # Not enough data, use minimum bins
                   optimal_bins = self.min_bins
               else:
                   # Find optimal number of bins
                   scores = []
                   bin_counts = range(self.min_bins, 
                                    min(self.max_bins + 1, len(valid_data) // 2))
                   
                   for n_bins in bin_counts:
                       score = self._score_binning(valid_data, n_bins)
                       scores.append((score, n_bins))
                   
                   # Select best scoring configuration
                   optimal_bins = max(scores, key=lambda x: x[0])[1]
               
               # Apply optimal binning
               self.optimal_bins_[col_idx] = optimal_bins
               binner = EqualWidthBinning(n_bins=optimal_bins)
               self.binners_[col_idx] = binner
               
               result[:, col_idx] = binner.fit_transform(
                   column_data.reshape(-1, 1)
               ).ravel()
           
           return result
   
   # Test adaptive binning
   adaptive_binner = AdaptiveBinner(method='entropy')
   adaptive_result = adaptive_binner.fit_transform(data)
   
   print("Adaptive binning results:")
   for i, col in enumerate(data.columns):
       optimal_bins = adaptive_binner.optimal_bins_[i]
       unique_bins = len(np.unique(adaptive_result[:, i]))
       print(f"{col}: optimal_bins={optimal_bins}, actual_bins={unique_bins}")

Handling Complex Data Types
----------------------------

Mixed Data Types
~~~~~~~~~~~~~~~~

Handle datasets with mixed numeric and categorical variables:

.. code-block:: python

   from binning.methods import EqualFrequencyBinning
   from sklearn.preprocessing import LabelEncoder
   
   # Create mixed data
   mixed_data = pd.DataFrame({
       'numeric1': np.random.normal(0, 1, 500),
       'numeric2': np.random.exponential(2, 500),
       'categorical': np.random.choice(['A', 'B', 'C', 'D'], 500),
       'ordinal': np.random.choice(['low', 'medium', 'high'], 500)
   })
   
   class MixedDataBinner:
       """Handle mixed numeric and categorical data."""
       
       def __init__(self, numeric_bins=5):
           self.numeric_bins = numeric_bins
           self.numeric_binner = EqualFrequencyBinning(n_bins=numeric_bins)
           self.label_encoders = {}
           self.numeric_columns = []
           self.categorical_columns = []
       
       def fit_transform(self, X):
           if isinstance(X, pd.DataFrame):
               # Identify column types
               for col in X.columns:
                   if X[col].dtype in ['int64', 'float64']:
                       self.numeric_columns.append(col)
                   else:
                       self.categorical_columns.append(col)
               
               result = X.copy()
               
               # Bin numeric columns
               if self.numeric_columns:
                   numeric_data = X[self.numeric_columns]
                   numeric_binned = self.numeric_binner.fit_transform(numeric_data)
                   
                   for i, col in enumerate(self.numeric_columns):
                       result[col] = numeric_binned[:, i]
               
               # Encode categorical columns
               for col in self.categorical_columns:
                   le = LabelEncoder()
                   result[col] = le.fit_transform(X[col].astype(str))
                   self.label_encoders[col] = le
               
               return result
           else:
               raise ValueError("Input must be a pandas DataFrame for mixed data")
   
   mixed_binner = MixedDataBinner(numeric_bins=4)
   mixed_result = mixed_binner.fit_transform(mixed_data)
   
   print("Mixed data binning results:")
   print(mixed_result.head())
   print(f"Numeric columns: {mixed_binner.numeric_columns}")
   print(f"Categorical columns: {mixed_binner.categorical_columns}")

Time Series Binning
~~~~~~~~~~~~~~~~~~~~

Special considerations for temporal data:

.. code-block:: python

   import pandas as pd
   from datetime import datetime, timedelta
   
   # Create time series data
   dates = pd.date_range('2020-01-01', periods=1000, freq='D')
   ts_data = pd.DataFrame({
       'date': dates,
       'value': np.random.normal(100, 15, 1000) + np.sin(np.arange(1000) * 2 * np.pi / 365) * 10,
       'trend': np.arange(1000) * 0.1 + np.random.normal(0, 5, 1000)
   })
   
   class TimeSeriesBinner:
       """Bin time series data with temporal awareness."""
       
       def __init__(self, time_bins=4, value_bins=5):
           self.time_bins = time_bins
           self.value_bins = value_bins
       
       def fit_transform(self, df, time_col='date', value_cols=None):
           if value_cols is None:
               value_cols = [col for col in df.columns if col != time_col]
           
           result = df.copy()
           
           # Time-based binning (seasonal)
           if time_col in df.columns:
               dates = pd.to_datetime(df[time_col])
               day_of_year = dates.dt.dayofyear
               
               # Create seasonal bins
               season_edges = np.linspace(1, 366, self.time_bins + 1)
               time_bins = np.digitize(day_of_year, season_edges[1:-1])
               result[f'{time_col}_bin'] = time_bins
           
           # Value binning with trend awareness
           for col in value_cols:
               if col in df.columns:
                   # Detrend the data for better binning
                   x = np.arange(len(df))
                   trend = np.polyfit(x, df[col], 1)
                   detrended = df[col] - np.polyval(trend, x)
                   
                   # Bin the detrended values
                   binner = EqualFrequencyBinning(n_bins=self.value_bins)
                   binned = binner.fit_transform(detrended.values.reshape(-1, 1))
                   result[f'{col}_bin'] = binned.ravel()
           
           return result
   
   ts_binner = TimeSeriesBinner(time_bins=4, value_bins=5)
   ts_result = ts_binner.fit_transform(ts_data, time_col='date', 
                                      value_cols=['value', 'trend'])
   
   print("Time series binning results:")
   print(ts_result[['date', 'value', 'value_bin', 'date_bin']].head(10))

Optimization and Performance
----------------------------

Memory-Efficient Binning
~~~~~~~~~~~~~~~~~~~~~~~~~

For large datasets, optimize memory usage:

.. code-block:: python

   class MemoryEfficientBinner:
       """Memory-efficient binning for large datasets."""
       
       def __init__(self, n_bins=5, chunk_size=10000):
           self.n_bins = n_bins
           self.chunk_size = chunk_size
           self.bin_edges_ = {}
       
       def fit(self, X):
           """Fit using sample of data to determine bin edges."""
           if isinstance(X, pd.DataFrame):
               X_array = X.values
           else:
               X_array = np.asarray(X)
           
           # Use sample for fitting to save memory
           n_samples = min(50000, len(X_array))
           sample_indices = np.random.choice(len(X_array), n_samples, replace=False)
           sample_data = X_array[sample_indices]
           
           # Fit on sample
           for col_idx in range(sample_data.shape[1]):
               column_data = sample_data[:, col_idx]
               valid_data = column_data[~np.isnan(column_data)]
               
               if len(valid_data) > 0:
                   edges = np.percentile(valid_data, 
                                       np.linspace(0, 100, self.n_bins + 1))
                   self.bin_edges_[col_idx] = edges
           
           return self
       
       def transform(self, X):
           """Transform data in chunks to manage memory."""
           if isinstance(X, pd.DataFrame):
               X_array = X.values
           else:
               X_array = np.asarray(X)
           
           result = np.zeros_like(X_array, dtype=int)
           
           # Process in chunks
           for start_idx in range(0, len(X_array), self.chunk_size):
               end_idx = min(start_idx + self.chunk_size, len(X_array))
               chunk = X_array[start_idx:end_idx]
               
               for col_idx in range(chunk.shape[1]):
                   if col_idx in self.bin_edges_:
                       edges = self.bin_edges_[col_idx]
                       column_data = chunk[:, col_idx]
                       
                       # Handle missing values
                       missing_mask = np.isnan(column_data)
                       chunk_result = np.full(len(column_data), -1, dtype=int)
                       
                       if not missing_mask.all():
                           valid_data = column_data[~missing_mask]
                           binned = np.digitize(valid_data, edges[1:-1])
                           chunk_result[~missing_mask] = binned
                       
                       result[start_idx:end_idx, col_idx] = chunk_result
           
           return result
       
       def fit_transform(self, X):
           return self.fit(X).transform(X)
   
   # Test memory-efficient binning
   large_data = np.random.rand(100000, 5)  # Large dataset
   
   mem_binner = MemoryEfficientBinner(n_bins=10, chunk_size=5000)
   mem_result = mem_binner.fit_transform(large_data)
   
   print(f"Memory-efficient binning on {large_data.shape} data:")
   print(f"Result shape: {mem_result.shape}")
   print(f"Unique bins per column: {[len(np.unique(mem_result[:, i])) for i in range(5)]}")

Parallel Binning
~~~~~~~~~~~~~~~~

Use parallel processing for faster binning:

.. code-block:: python

   from concurrent.futures import ProcessPoolExecutor
   import functools
   
   def _bin_column(args):
       """Helper function for parallel binning."""
       column_data, n_bins, col_idx = args
       binner = EqualWidthBinning(n_bins=n_bins)
       return col_idx, binner.fit_transform(column_data.reshape(-1, 1)).ravel()
   
   class ParallelBinner:
       """Parallel binning using multiple processes."""
       
       def __init__(self, n_bins=5, n_jobs=None):
           self.n_bins = n_bins
           self.n_jobs = n_jobs
       
       def fit_transform(self, X):
           if isinstance(X, pd.DataFrame):
               X_array = X.values
           else:
               X_array = np.asarray(X)
           
           if X_array.ndim == 1:
               X_array = X_array.reshape(-1, 1)
           
           # Prepare arguments for parallel processing
           args_list = []
           for col_idx in range(X_array.shape[1]):
               args_list.append((X_array[:, col_idx], self.n_bins, col_idx))
           
           # Process columns in parallel
           with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
               results = list(executor.map(_bin_column, args_list))
           
           # Combine results
           result = np.zeros_like(X_array, dtype=int)
           for col_idx, binned_column in results:
               result[:, col_idx] = binned_column
           
           return result
   
   # Test parallel binning (use small data for demo)
   test_data = np.random.rand(1000, 4)
   
   parallel_binner = ParallelBinner(n_bins=6, n_jobs=2)
   parallel_result = parallel_binner.fit_transform(test_data)
   
   print(f"Parallel binning result shape: {parallel_result.shape}")
   print(f"Bins per column: {[len(np.unique(parallel_result[:, i])) for i in range(4)]}")

Advanced Validation and Quality Control
----------------------------------------

Bin Quality Metrics
~~~~~~~~~~~~~~~~~~~~

Evaluate the quality of your binning results:

.. code-block:: python

   class BinningQualityEvaluator:
       """Evaluate binning quality with various metrics."""
       
       @staticmethod
       def bin_balance_score(binned_data):
           """Measure how balanced the bins are."""
           _, counts = np.unique(binned_data, return_counts=True)
           # Perfect balance = 1.0, poor balance approaches 0
           ideal_count = len(binned_data) / len(counts)
           balance = 1.0 - np.std(counts) / ideal_count
           return max(0, balance)
       
       @staticmethod
       def bin_separation_score(original_data, binned_data):
           """Measure how well bins separate the original values."""
           bin_means = []
           for bin_val in np.unique(binned_data):
               mask = binned_data == bin_val
               if np.any(mask):
                   bin_means.append(np.mean(original_data[mask]))
           
           if len(bin_means) <= 1:
               return 0.0
           
           # Higher separation = better binning
           between_var = np.var(bin_means)
           within_var = np.mean([
               np.var(original_data[binned_data == bin_val]) 
               for bin_val in np.unique(binned_data)
               if np.sum(binned_data == bin_val) > 1
           ])
           
           if within_var == 0:
               return float('inf')
           
           return between_var / within_var
       
       @staticmethod
       def evaluate_binning(original_data, binned_data):
           """Comprehensive binning evaluation."""
           results = {}
           
           results['n_bins'] = len(np.unique(binned_data))
           results['balance_score'] = BinningQualityEvaluator.bin_balance_score(binned_data)
           results['separation_score'] = BinningQualityEvaluator.bin_separation_score(
               original_data, binned_data)
           
           # Bin statistics
           _, counts = np.unique(binned_data, return_counts=True)
           results['min_bin_size'] = np.min(counts)
           results['max_bin_size'] = np.max(counts)
           results['bin_size_std'] = np.std(counts)
           
           return results
   
   # Evaluate different binning methods
   evaluator = BinningQualityEvaluator()
   
   test_column = data['income'].values
   
   # Test different methods
   methods = {
       'Equal Width': EqualWidthBinning(n_bins=5),
       'Equal Frequency': EqualFrequencyBinning(n_bins=5),
       'Custom Percentile': PercentileBinning(percentiles=[20, 40, 60, 80])
   }
   
   print("Binning Quality Evaluation:")
   print("-" * 60)
   
   for method_name, binner in methods.items():
       binned = binner.fit_transform(test_column.reshape(-1, 1)).ravel()
       quality = evaluator.evaluate_binning(test_column, binned)
       
       print(f"\n{method_name}:")
       for metric, value in quality.items():
           if isinstance(value, float):
               print(f"  {metric}: {value:.4f}")
           else:
               print(f"  {metric}: {value}")

Cross-Validation for Binning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Validate binning stability across different data splits:

.. code-block:: python

   from sklearn.model_selection import KFold
   
   class BinningCrossValidator:
       """Cross-validate binning stability."""
       
       def __init__(self, binner, cv=5):
           self.binner = binner
           self.cv = cv
       
       def validate_stability(self, X, y=None):
           """Check if binning is stable across CV folds."""
           kfold = KFold(n_splits=self.cv, shuffle=True, random_state=42)
           
           fold_edges = []
           fold_scores = []
           
           for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
               X_train = X[train_idx]
               X_val = X[val_idx]
               
               # Fit on training fold
               binner_fold = type(self.binner)(**self.binner.get_params())
               binner_fold.fit(X_train)
               
               # Transform validation fold
               X_val_binned = binner_fold.transform(X_val)
               
               # Store bin edges for stability check
               if hasattr(binner_fold, '_bin_edges'):
                   fold_edges.append(binner_fold._bin_edges)
               
               # Evaluate quality on validation fold
               if X.shape[1] == 1:  # Single column for simplicity
                   quality = BinningQualityEvaluator.evaluate_binning(
                       X_val[:, 0], X_val_binned[:, 0])
                   fold_scores.append(quality)
           
           # Analyze stability
           stability_results = {
               'mean_balance_score': np.mean([s['balance_score'] for s in fold_scores]),
               'std_balance_score': np.std([s['balance_score'] for s in fold_scores]),
               'mean_separation_score': np.mean([s['separation_score'] for s in fold_scores]),
               'std_separation_score': np.std([s['separation_score'] for s in fold_scores]),
           }
           
           return stability_results, fold_scores
   
   # Test binning stability
   single_feature = data[['income']].values
   
   cv_validator = BinningCrossValidator(EqualWidthBinning(n_bins=5), cv=5)
   stability, fold_details = cv_validator.validate_stability(single_feature)
   
   print("Binning Stability Analysis:")
   print("-" * 40)
   for metric, value in stability.items():
       print(f"{metric}: {value:.4f}")

Summary
-------

This advanced tutorial covered:

1. **Custom binning strategies** for specialized requirements
2. **Multi-stage and adaptive binning** for complex preprocessing
3. **Mixed data type handling** for real-world datasets
4. **Time series binning** with temporal awareness
5. **Memory-efficient and parallel processing** for large datasets
6. **Quality evaluation and cross-validation** for robust binning

These techniques enable you to handle sophisticated binning scenarios and ensure high-quality results in production environments.

Next Steps
----------

* Experiment with custom binning strategies for your domain
* Implement quality metrics for your specific use case
* Explore integration with other preprocessing techniques
* Consider ensemble binning approaches for critical applications
