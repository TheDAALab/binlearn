Performance Tips
================

Optimize binlearn for better performance with large datasets and complex workflows.

General Performance Guidelines
------------------------------

Choose the Right Binning Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Different binning methods have different performance characteristics:

**Fastest to Slowest:**

1. **EqualWidthBinning** - O(n) complexity, minimal computation
2. **EqualFrequencyBinning** - O(n log n) due to sorting  
3. **ManualIntervalBinning** - O(n), but with validation overhead
4. **EqualWidthMinimumWeightBinning** - O(n), with weight constraints
5. **SingletonBinning** - O(n), with categorical encoding overhead
6. **KMeansBinning** - O(n × k × iterations), iterative clustering
7. **SupervisedBinning** - O(n log n), decision tree overhead

.. code-block:: python

   import numpy as np
   import time
   from binlearn import EqualWidthBinning, KMeansBinning, SupervisedBinning

   # Large dataset for benchmarking
   n_samples = 100000
   n_features = 20
   X = np.random.rand(n_samples, n_features)
   y = np.random.randint(0, 2, n_samples)

   # Benchmark different methods
   methods = [
       ("EqualWidthBinning", EqualWidthBinning(n_bins=5)),
       ("KMeansBinning", KMeansBinning(n_bins=5, random_state=42)),
       ("SupervisedBinning", SupervisedBinning(n_bins=5, task_type='classification'))
   ]

   for name, binner in methods:
       start_time = time.time()
       if "Supervised" in name:
           binner.fit_transform(X, guidance_data=y)
       else:
           binner.fit_transform(X)
       elapsed = time.time() - start_time
       print(f"{name}: {elapsed:.2f}s")

Memory Optimization
-------------------

Use Appropriate Data Types
~~~~~~~~~~~~~~~~~~~~~~~~~~

Choose data types based on your precision needs:

.. code-block:: python

   import numpy as np
   from binlearn import EqualWidthBinning

   # Original data (float64 - 8 bytes per value)
   X_float64 = np.random.rand(1000000, 10)
   print(f"Float64 memory: {X_float64.nbytes / 1024**2:.1f} MB")

   # Reduced precision (float32 - 4 bytes per value) 
   X_float32 = X_float64.astype(np.float32)
   print(f"Float32 memory: {X_float32.nbytes / 1024**2:.1f} MB")

   # binlearn works with both
   binner = EqualWidthBinning(n_bins=5)
   
   # Both will produce similar results
   result_64 = binner.fit_transform(X_float64)
   result_32 = binner.fit_transform(X_float32)
   
   print(f"Max difference: {np.max(np.abs(result_64 - result_32))}")

Process Large Datasets Efficiently
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For datasets that don't fit in memory:

.. code-block:: python

   import numpy as np
   from binlearn import EqualWidthBinning

   def fit_on_sample_transform_in_chunks(X, binner, sample_ratio=0.1, chunk_size=10000):
       """Efficient processing for large datasets."""
       n_samples = X.shape[0]
       
       # Fit on a representative sample
       sample_size = int(n_samples * sample_ratio)
       sample_indices = np.random.choice(n_samples, sample_size, replace=False)
       X_sample = X[sample_indices]
       
       print(f"Fitting on sample of {sample_size} rows...")
       binner.fit(X_sample)
       
       # Transform in chunks
       print(f"Transforming {n_samples} rows in chunks of {chunk_size}...")
       results = []
       
       for i in range(0, n_samples, chunk_size):
           end_idx = min(i + chunk_size, n_samples)
           chunk = X[i:end_idx]
           chunk_result = binner.transform(chunk)
           results.append(chunk_result)
           
           if (i // chunk_size + 1) % 10 == 0:
               print(f"Processed {end_idx}/{n_samples} rows")
       
       return np.vstack(results)

   # Example usage
   X_large = np.random.rand(500000, 50)
   binner = EqualWidthBinning(n_bins=5)
   
   X_binned = fit_on_sample_transform_in_chunks(
       X_large, binner, sample_ratio=0.05, chunk_size=50000
   )

DataFrame Performance
---------------------

Optimize Pandas Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   import numpy as np
   from binlearn import EqualWidthBinning

   # Create large DataFrame
   n_rows = 1000000
   df = pd.DataFrame({
       f'feature_{i}': np.random.rand(n_rows) 
       for i in range(20)
   })

   # Performance tip 1: Select only columns you need
   columns_to_bin = ['feature_0', 'feature_1', 'feature_2']
   df_subset = df[columns_to_bin]

   # Performance tip 2: Use preserve_dataframe=False for large datasets
   # if you don't need DataFrame output
   binner_fast = EqualWidthBinning(n_bins=5, preserve_dataframe=False)
   result_array = binner_fast.fit_transform(df_subset)  # Returns numpy array

   # Performance tip 3: Use preserve_dataframe=True only when needed
   binner_df = EqualWidthBinning(n_bins=5, preserve_dataframe=True)
   result_df = binner_df.fit_transform(df_subset)  # Returns DataFrame

Consider Polars for Large DataFrames
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For very large datasets, consider using Polars:

.. code-block:: python

   try:
       import polars as pl
       from binlearn import EqualWidthBinning

       # Convert pandas DataFrame to Polars (more memory efficient)
       df_polars = pl.from_pandas(df)
       
       # binlearn supports Polars DataFrames
       binner = EqualWidthBinning(n_bins=5, preserve_dataframe=True)
       
       # Convert to pandas for binning (if needed), then back to Polars
       df_pandas = df_polars.to_pandas()
       result_pandas = binner.fit_transform(df_pandas)
       result_polars = pl.from_pandas(result_pandas)
       
       print(f"Polars result shape: {result_polars.shape}")
       
   except ImportError:
       print("Polars not available")

Pipeline Performance
--------------------

Optimize Sklearn Pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.compose import ColumnTransformer
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split
   from binlearn import EqualWidthBinning, SingletonBinning
   import numpy as np
   import pandas as pd

   # Create mixed dataset
   n_samples = 50000
   df = pd.DataFrame({
       'numeric1': np.random.normal(0, 1, n_samples),
       'numeric2': np.random.exponential(2, n_samples),
       'categorical1': np.random.choice(['A', 'B', 'C'], n_samples),
       'categorical2': np.random.choice(['X', 'Y', 'Z'], n_samples),
       'target': np.random.randint(0, 2, n_samples)
   })

   X = df.drop('target', axis=1)
   y = df['target']

   # Performance tip 1: Use ColumnTransformer for different column types
   preprocessor = ColumnTransformer([
       ('numeric', EqualWidthBinning(n_bins=5), ['numeric1', 'numeric2']),
       ('categorical', SingletonBinning(), ['categorical1', 'categorical2'])
   ], remainder='drop')

   # Performance tip 2: Choose efficient estimators
   pipeline = Pipeline([
       ('preprocessing', preprocessor),
       ('classifier', RandomForestClassifier(
           n_estimators=100,  # Reasonable number
           max_depth=10,      # Limit depth
           n_jobs=-1,         # Use all cores
           random_state=42
       ))
   ])

   # Performance tip 3: Use appropriate train/test split
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42, stratify=y
   )

   pipeline.fit(X_train, y_train)
   score = pipeline.score(X_test, y_test)
   print(f"Pipeline accuracy: {score:.3f}")

Configuration Optimization
--------------------------

Global Configuration for Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from binlearn import set_config, get_config

   # Check current configuration
   current_config = get_config()
   print("Current config:", current_config)

   # Optimize for performance
   set_config(
       preserve_dataframe=False,  # Faster for large datasets
       fit_jointly=False,         # More memory efficient
       clip=False                 # Skip outlier clipping if not needed
   )

   # All new binners will use these optimized defaults
   from binlearn import EqualWidthBinning
   
   # This binner will use the optimized configuration
   fast_binner = EqualWidthBinning(n_bins=5)

Parallel Processing
-------------------

Leverage Multiple Cores
~~~~~~~~~~~~~~~~~~~~~~~

While binlearn doesn't directly support parallelization, you can parallelize at different levels:

.. code-block:: python

   import numpy as np
   from joblib import Parallel, delayed
   from binlearn import EqualWidthBinning
   import time

   def bin_feature_subset(X_subset, n_bins=5):
       """Bin a subset of features."""
       binner = EqualWidthBinning(n_bins=n_bins)
       return binner.fit_transform(X_subset)

   # Large dataset with many features
   X = np.random.rand(10000, 100)

   # Method 1: Sequential processing
   start_time = time.time()
   binner = EqualWidthBinning(n_bins=5)
   X_binned_sequential = binner.fit_transform(X)
   sequential_time = time.time() - start_time
   print(f"Sequential time: {sequential_time:.2f}s")

   # Method 2: Parallel processing by feature groups
   start_time = time.time()
   n_jobs = 4
   features_per_job = X.shape[1] // n_jobs
   
   feature_groups = [
       X[:, i:i+features_per_job] 
       for i in range(0, X.shape[1], features_per_job)
   ]

   # Process feature groups in parallel
   results = Parallel(n_jobs=n_jobs)(
       delayed(bin_feature_subset)(group) 
       for group in feature_groups
   )
   
   X_binned_parallel = np.hstack(results)
   parallel_time = time.time() - start_time
   print(f"Parallel time: {parallel_time:.2f}s")
   print(f"Speedup: {sequential_time/parallel_time:.2f}x")

Caching and Preprocessing
-------------------------

Cache Expensive Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pickle
   import os
   from binlearn import EqualWidthBinning
   import numpy as np

   def cached_binning(X, cache_file, n_bins=5, force_recompute=False):
       """Cache fitted binner to avoid recomputation."""
       
       if os.path.exists(cache_file) and not force_recompute:
           print("Loading cached binner...")
           with open(cache_file, 'rb') as f:
               binner = pickle.load(f)
       else:
           print("Computing and caching binner...")
           binner = EqualWidthBinning(n_bins=n_bins)
           binner.fit(X)
           
           with open(cache_file, 'wb') as f:
               pickle.dump(binner, f)
       
       return binner

   # Example usage
   X_train = np.random.rand(100000, 20)
   
   # First run: computes and caches
   binner = cached_binning(X_train, 'binner_cache.pkl')
   X_binned = binner.transform(X_train)
   
   # Subsequent runs: loads from cache
   binner = cached_binning(X_train, 'binner_cache.pkl')

Benchmarking and Profiling
--------------------------

Profile Your Code
~~~~~~~~~~~~~~~~

.. code-block:: python

   import cProfile
   import pstats
   from binlearn import EqualWidthBinning
   import numpy as np

   def benchmark_binning():
       """Function to profile."""
       X = np.random.rand(100000, 50)
       binner = EqualWidthBinning(n_bins=10)
       return binner.fit_transform(X)

   # Profile the function
   profiler = cProfile.Profile()
   profiler.enable()
   result = benchmark_binning()
   profiler.disable()

   # Analyze results
   stats = pstats.Stats(profiler)
   stats.sort_stats('cumulative')
   stats.print_stats(10)  # Top 10 functions by cumulative time

Memory Profiling
~~~~~~~~~~~~~~~~

.. code-block:: python

   import tracemalloc
   import numpy as np
   from binlearn import EqualWidthBinning

   def memory_benchmark():
       """Monitor memory usage during binning."""
       tracemalloc.start()
       
       # Create large dataset
       X = np.random.rand(500000, 20)
       snapshot1 = tracemalloc.take_snapshot()
       
       # Perform binning
       binner = EqualWidthBinning(n_bins=5)
       X_binned = binner.fit_transform(X)
       snapshot2 = tracemalloc.take_snapshot()
       
       # Calculate memory difference
       top_stats = snapshot2.compare_to(snapshot1, 'lineno')
       
       print("Top memory allocations:")
       for stat in top_stats[:5]:
           print(stat)
       
       tracemalloc.stop()
       return X_binned

   result = memory_benchmark()

Best Practices Summary
---------------------

**For Large Datasets:**
1. Use ``EqualWidthBinning`` for fastest performance
2. Set ``preserve_dataframe=False`` if DataFrame output not needed
3. Consider sample-based fitting for very large datasets
4. Use appropriate data types (float32 vs float64)

**For Complex Pipelines:**
1. Use ``ColumnTransformer`` for different column types
2. Cache fitted binners when possible
3. Choose efficient downstream estimators
4. Leverage parallel processing where applicable

**For Memory Efficiency:**
1. Process data in chunks if necessary
2. Use sparse matrices for high-dimensional sparse data
3. Consider Polars for very large DataFrames
4. Monitor memory usage with profiling tools

**Configuration:**
1. Set global configuration for consistent performance
2. Use ``fit_jointly=False`` for better memory efficiency
3. Disable ``clip`` if outlier handling not needed

Following these guidelines will help you get optimal performance from binlearn in your specific use case.
