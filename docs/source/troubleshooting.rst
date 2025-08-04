Troubleshooting
===============

Common issues and their solutions when using binlearn.

Installation Issues
-------------------

Package Not Found
~~~~~~~~~~~~~~~~~~

**Problem**: ``pip install binlearn`` fails with "No matching distribution found"

**Solutions**:

1. **Check Python version**: Ensure you're using Python 3.10+

   .. code-block:: bash

      python --version

2. **Update pip**: Make sure you have the latest pip version

   .. code-block:: bash

      pip install --upgrade pip

3. **Use specific index**: Try installing from PyPI directly

   .. code-block:: bash

      pip install --index-url https://pypi.org/simple/ binlearn

Dependency Conflicts
~~~~~~~~~~~~~~~~~~~~~

**Problem**: Conflicting dependency versions during installation

**Solutions**:

1. **Create fresh environment**: Use a virtual environment

   .. code-block:: bash

      python -m venv fresh_env
      source fresh_env/bin/activate  # Windows: fresh_env\Scripts\activate
      pip install binlearn

2. **Use conda**: Try conda for better dependency resolution

   .. code-block:: bash

      conda create -n binlearn_env python=3.11
      conda activate binlearn_env
      pip install binlearn

3. **Check requirements**: Manually install core dependencies first

   .. code-block:: bash

      pip install numpy>=1.21.0 scipy>=1.7.0 scikit-learn>=1.0.0
      pip install binlearn

Import Errors
~~~~~~~~~~~~~

**Problem**: ``ImportError`` or ``ModuleNotFoundError`` when importing binlearn

**Solutions**:

1. **Verify installation**: Check if binlearn is properly installed

   .. code-block:: python

      import sys
      print(sys.path)
      
      import pkg_resources
      print(pkg_resources.get_distribution("binlearn"))

2. **Check environment**: Ensure you're in the correct environment

   .. code-block:: bash

      which python
      pip list | grep binlearn

3. **Reinstall**: Clean reinstall of the package

   .. code-block:: bash

      pip uninstall binlearn
      pip install binlearn

Usage Issues
------------

ConfigurationError
~~~~~~~~~~~~~~~~~~

**Problem**: ``ConfigurationError`` when creating binners

**Common causes and solutions**:

.. code-block:: python

   from binlearn import EqualWidthBinning
   from binlearn.utils.errors import ConfigurationError

   # Problem: Invalid n_bins
   try:
       binner = EqualWidthBinning(n_bins=0)  # n_bins must be > 0
   except ConfigurationError as e:
       print(f"Fix: Use positive n_bins: {e}")
       binner = EqualWidthBinning(n_bins=5)  # ✓ Correct

   # Problem: Invalid bin_range
   try:
       binner = EqualWidthBinning(bin_range=(10, 5))  # min > max
   except ConfigurationError as e:
       print(f"Fix: Ensure min < max: {e}")
       binner = EqualWidthBinning(bin_range=(5, 10))  # ✓ Correct

   # Problem: Conflicting parameters
   try:
       binner = EqualWidthBinning(fit_jointly=True, guidance_columns=['col1'])
   except ConfigurationError as e:
       print(f"Fix: Can't use both parameters: {e}")
       binner = EqualWidthBinning(fit_jointly=True)  # ✓ Correct

ValidationError
~~~~~~~~~~~~~~~

**Problem**: ``ValidationError`` during fitting or transformation

**Common causes and solutions**:

.. code-block:: python

   import numpy as np
   from binlearn import EqualWidthBinning
   from binlearn.utils.errors import ValidationError

   # Problem: Inconsistent array dimensions
   try:
       X = [[1, 2, 3], [4, 5]]  # Inconsistent lengths
       binner = EqualWidthBinning()
       binner.fit(X)
   except ValidationError as e:
       print(f"Fix: Use consistent array dimensions: {e}")
       X = np.array([[1, 2, 3], [4, 5, 6]])  # ✓ Correct
       binner.fit(X)

   # Problem: All NaN column
   try:
       X = np.array([[np.nan, 1], [np.nan, 2], [np.nan, 3]])
       binner = EqualWidthBinning()
       binner.fit(X)
   except ValidationError as e:
       print(f"Fix: Ensure columns have valid data: {e}")
       X = np.array([[1, 1], [2, 2], [3, 3]])  # ✓ Correct
       binner.fit(X)

   # Problem: Insufficient data for binning
   try:
       X = np.array([[1], [1], [1]])  # All same values
       binner = EqualWidthBinning(n_bins=5)
       binner.fit(X)
   except ValidationError as e:
       print(f"Warning: {e}")
       # Consider reducing n_bins or using different method

TransformationError
~~~~~~~~~~~~~~~~~~~

**Problem**: ``TransformationError`` during data transformation

**Solutions**:

.. code-block:: python

   from binlearn.utils.errors import TransformationError

   # Problem: Transform before fit
   try:
       binner = EqualWidthBinning()
       X = np.random.rand(100, 3)
       binner.transform(X)  # Not fitted yet
   except TransformationError as e:
       print(f"Fix: Fit before transform: {e}")
       binner.fit(X)
       X_binned = binner.transform(X)  # ✓ Correct

   # Problem: Different number of features
   try:
       X_train = np.random.rand(100, 3)
       X_test = np.random.rand(50, 2)  # Different feature count
       binner = EqualWidthBinning()
       binner.fit(X_train)
       binner.transform(X_test)
   except TransformationError as e:
       print(f"Fix: Ensure same feature count: {e}")
       X_test = np.random.rand(50, 3)  # ✓ Correct
       binner.transform(X_test)

Data Format Issues
------------------

DataFrame Column Names Lost
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Output loses DataFrame column names

**Solution**: Use ``preserve_dataframe=True``

.. code-block:: python

   import pandas as pd
   from binlearn import EqualWidthBinning

   df = pd.DataFrame({'age': [25, 30, 35], 'income': [50000, 60000, 70000]})

   # Problem: Returns numpy array
   binner = EqualWidthBinning()
   result = binner.fit_transform(df)
   print(type(result))  # <class 'numpy.ndarray'>

   # Solution: Preserve DataFrame format
   binner = EqualWidthBinning(preserve_dataframe=True)
   result = binner.fit_transform(df)
   print(type(result))  # <class 'pandas.core.frame.DataFrame'>
   print(result.columns.tolist())  # ['age', 'income']

Unexpected Binning Results
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Binned values are not what you expected

**Debugging steps**:

.. code-block:: python

   import numpy as np
   from binlearn import EqualWidthBinning

   # Create test data
   X = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]])

   binner = EqualWidthBinning(n_bins=3)
   binner.fit(X)

   # Examine bin edges
   print("Bin edges:", binner.bin_edges_)
   
   # Check bin representatives
   print("Bin representatives:", binner.bin_representatives_)
   
   # Transform and examine results
   X_binned = binner.transform(X)
   print("Original data:\n", X)
   print("Binned data:\n", X_binned)
   
   # Check unique values per feature
   for i in range(X.shape[1]):
       unique_bins = np.unique(X_binned[:, i])
       print(f"Feature {i} bins: {unique_bins}")

Missing Values Handling
~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Unexpected behavior with missing values

**Understanding missing value handling**:

.. code-block:: python

   import numpy as np
   import pandas as pd
   from binlearn import EqualWidthBinning

   # Data with missing values
   df = pd.DataFrame({
       'feature1': [1, 2, np.nan, 4, 5],
       'feature2': [10, np.nan, 30, 40, 50]
   })

   binner = EqualWidthBinning(n_bins=3, preserve_dataframe=True)
   df_binned = binner.fit_transform(df)

   print("Original data:")
   print(df)
   print("\nBinned data:")
   print(df_binned)
   print("\nBin edges (calculated ignoring NaN):")
   print(binner.bin_edges_)

   # Check how missing values are handled
   print("\nMissing values are preserved:")
   print(f"NaN in original: {df.isna().sum().sum()}")
   print(f"NaN in binned: {df_binned.isna().sum().sum()}")

Performance Issues
------------------

Slow Binning Performance
~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Binning takes too long for your dataset

**Performance optimization strategies**:

.. code-block:: python

   import numpy as np
   import time
   from binlearn import EqualWidthBinning, KMeansBinning

   # Large dataset
   n_samples = 1000000
   n_features = 100
   X = np.random.rand(n_samples, n_features)

   # Strategy 1: Use faster binning methods
   start_time = time.time()
   fast_binner = EqualWidthBinning(n_bins=5)
   X_binned = fast_binner.fit_transform(X)
   print(f"EqualWidthBinning: {time.time() - start_time:.2f}s")

   # KMeansBinning is slower for large datasets
   # start_time = time.time()
   # slow_binner = KMeansBinning(n_bins=5)
   # X_binned = slow_binner.fit_transform(X)
   # print(f"KMeansBinning: {time.time() - start_time:.2f}s")

   # Strategy 2: Sample for fitting, transform full dataset
   sample_size = 10000
   sample_indices = np.random.choice(n_samples, sample_size, replace=False)
   X_sample = X[sample_indices]

   start_time = time.time()
   sample_binner = KMeansBinning(n_bins=5, random_state=42)
   sample_binner.fit(X_sample)
   X_binned = sample_binner.transform(X)
   print(f"Sample-based fitting: {time.time() - start_time:.2f}s")

Memory Issues
~~~~~~~~~~~~~

**Problem**: Running out of memory with large datasets

**Memory optimization strategies**:

.. code-block:: python

   import numpy as np
   from binlearn import EqualWidthBinning

   # Strategy 1: Use appropriate data types
   # Float32 uses half the memory of float64
   X = np.random.rand(1000000, 50).astype(np.float32)

   binner = EqualWidthBinning(n_bins=5)
   X_binned = binner.fit_transform(X)

   print(f"Original data type: {X.dtype}")
   print(f"Memory usage: {X.nbytes / 1024**2:.1f} MB")

   # Strategy 2: Process in chunks (for very large datasets)
   def chunk_transform(binner, X, chunk_size=10000):
       """Transform large array in chunks."""
       n_samples = X.shape[0]
       results = []
       
       for i in range(0, n_samples, chunk_size):
           end_idx = min(i + chunk_size, n_samples)
           chunk = X[i:end_idx]
           chunk_binned = binner.transform(chunk)
           results.append(chunk_binned)
       
       return np.vstack(results)

   # Fit on sample, transform in chunks
   sample_size = 10000
   sample_indices = np.random.choice(len(X), sample_size, replace=False)
   binner.fit(X[sample_indices])
   
   X_binned = chunk_transform(binner, X, chunk_size=50000)

Integration Issues
------------------

Sklearn Pipeline Issues
~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Issues using binlearn in sklearn pipelines

**Common solutions**:

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import cross_val_score
   from sklearn.datasets import make_classification
   from binlearn import EqualWidthBinning

   X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

   # Problem: Pipeline parameter naming
   pipeline = Pipeline([
       ('binning', EqualWidthBinning(n_bins=5)),
       ('classifier', RandomForestClassifier(random_state=42))
   ])

   # Solution: Access parameters correctly
   pipeline.set_params(binning__n_bins=3)  # Note double underscore
   pipeline.set_params(classifier__n_estimators=100)

   # Cross-validation should work correctly
   scores = cross_val_score(pipeline, X, y, cv=3)
   print(f"CV scores: {scores}")

Pandas/Polars Integration
~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Issues with DataFrame libraries

**Solutions**:

.. code-block:: python

   import pandas as pd
   import numpy as np
   from binlearn import EqualWidthBinning

   # Problem: Mixed data types in DataFrame
   df = pd.DataFrame({
       'numeric': [1.0, 2.0, 3.0],
       'string': ['a', 'b', 'c'],
       'boolean': [True, False, True]
   })

   # Solution: Select only numeric columns for binning
   numeric_cols = df.select_dtypes(include=[np.number]).columns
   df_numeric = df[numeric_cols]

   binner = EqualWidthBinning(preserve_dataframe=True)
   df_binned = binner.fit_transform(df_numeric)

   # Combine with original non-numeric columns if needed
   df_final = pd.concat([df_binned, df[['string', 'boolean']]], axis=1)

Getting Help
------------

If you're still experiencing issues:

1. **Check the FAQ**: :doc:`faq` covers many common questions

2. **Search GitHub Issues**: Someone may have encountered the same problem
   
   Visit: https://github.com/TheDAALab/binlearn/issues

3. **Create a Minimal Example**: Prepare a small, reproducible example

   .. code-block:: python

      import numpy as np
      from binlearn import EqualWidthBinning

      # Minimal example that demonstrates the issue
      X = np.array([[1, 2], [3, 4]])
      binner = EqualWidthBinning(n_bins=2)
      
      # What you tried
      result = binner.fit_transform(X)
      
      # What you expected vs what you got
      print(f"Expected: ..., Got: {result}")

4. **File an Issue**: Create a new GitHub issue with:
   - Clear description of the problem
   - Minimal reproducible example
   - Your environment details (Python version, OS, package versions)
   - Full error message and stack trace

5. **Check Documentation**: Review the :doc:`user_guide/index` for detailed usage information

Environment Information
-----------------------

When reporting issues, include this information:

.. code-block:: python

   import sys
   import numpy as np
   import sklearn
   import binlearn

   print(f"Python version: {sys.version}")
   print(f"NumPy version: {np.__version__}")
   print(f"Scikit-learn version: {sklearn.__version__}")
   print(f"binlearn version: {binlearn.__version__}")
   print(f"Operating system: {sys.platform}")

   # Check optional dependencies
   try:
       import pandas as pd
       print(f"Pandas version: {pd.__version__}")
   except ImportError:
       print("Pandas: Not installed")

   try:
       import polars as pl
       print(f"Polars version: {pl.__version__}")
   except ImportError:
       print("Polars: Not installed")
