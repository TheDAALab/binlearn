GaussianMixtureBinning
======================

.. currentmodule:: binlearn.methods

.. autoclass:: GaussianMixtureBinning
   :members:
   :inherited-members:
   :show-inheritance:

Overview
--------

``GaussianMixtureBinning`` creates bins based on Gaussian Mixture Model (GMM) clustering of each feature. 
The bin edges are determined by finding the decision boundaries between adjacent mixture components, which 
naturally groups similar values together based on probabilistic clustering.

This approach is particularly useful when:

* Your data has natural Gaussian-like clusters
* You want bins that adapt to the probabilistic structure of the data distribution  
* You need clustering that captures overlapping distributions
* You want to model uncertainty in cluster assignments

Key Features
------------

* **Probabilistic Clustering**: Uses Gaussian mixture models for sophisticated clustering
* **Decision Boundaries**: Creates bin edges at optimal decision boundaries between components
* **Overlap Handling**: Naturally handles overlapping distributions
* **Reproducible Results**: Supports random state for consistent clustering
* **Sklearn Compatibility**: Full transformer interface with fit/transform methods
* **DataFrame Support**: Preserves pandas/polars column names and structure

Basic Usage
-----------

.. code-block:: python

   import numpy as np
   import pandas as pd
   from binlearn.methods import GaussianMixtureBinning
   
   # Create sample data with natural clusters
   np.random.seed(42)
   cluster1 = np.random.normal(10, 2, 300)
   cluster2 = np.random.normal(25, 3, 200) 
   cluster3 = np.random.normal(40, 1.5, 100)
   data = np.concatenate([cluster1, cluster2, cluster3])
   
   # Apply Gaussian mixture binning
   binner = GaussianMixtureBinning(n_components=3, random_state=42)
   data_binned = binner.fit_transform(data.reshape(-1, 1))
   
   print(f"Bin edges: {binner.bin_edges_[0]}")
   print(f"Original data shape: {data.shape}")
   print(f"Binned data shape: {data_binned.shape}")

DataFrame Example
-----------------

.. code-block:: python

   # DataFrame usage with multiple features
   df = pd.DataFrame({
       'feature1': np.concatenate([
           np.random.normal(10, 2, 200),
           np.random.normal(30, 3, 200)
       ]),
       'feature2': np.concatenate([
           np.random.normal(5, 1, 200),
           np.random.normal(15, 2, 200)
       ])
   })
   
   binner = GaussianMixtureBinning(
       n_components=2, 
       random_state=42,
       preserve_dataframe=True
   )
   df_binned = binner.fit_transform(df)
   
   print(f"Bin edges for feature1: {binner.bin_edges_['feature1']}")
   print(f"Bin edges for feature2: {binner.bin_edges_['feature2']}")

Advanced Configuration
----------------------

.. code-block:: python

   # Advanced configuration with GMM parameters
   binner = GaussianMixtureBinning(
       n_components=4,
       covariance_type='full',  # Full covariance matrices
       max_iter=200,           # Maximum EM iterations
       random_state=42,
       tol=1e-3,              # Convergence tolerance
       clip=True              # Clip out-of-range values
   )
   
   data_binned = binner.fit_transform(data)

Scikit-learn Pipeline Integration
---------------------------------

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split
   
   # Create a pipeline with Gaussian mixture binning
   pipeline = Pipeline([
       ('binning', GaussianMixtureBinning(n_components=3, random_state=42)),
       ('classifier', RandomForestClassifier(random_state=42))
   ])
   
   # Use in ML workflow
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
   pipeline.fit(X_train, y_train)
   accuracy = pipeline.score(X_test, y_test)

Parameter Guide
---------------

**n_components** (int, default=5)
    Number of mixture components (bins) to create. Higher values create more granular bins.

**covariance_type** (str, default='full')
    Type of covariance parameters:
    * 'full': Each component has its own general covariance matrix
    * 'tied': All components share the same general covariance matrix  
    * 'diag': Each component has its own diagonal covariance matrix
    * 'spherical': Each component has its own single variance

**random_state** (int, optional)
    Random seed for reproducible clustering results.

**max_iter** (int, default=100)
    Maximum number of EM algorithm iterations.

**tol** (float, default=1e-3)
    Tolerance for convergence of the EM algorithm.

See Also
--------

* :class:`KMeansBinning` - K-means clustering-based binning
* :class:`DBSCANBinning` - Density-based clustering binning  
* :class:`EqualFrequencyBinning` - Quantile-based binning
* :class:`TreeBinning` - Decision tree-based supervised binning
