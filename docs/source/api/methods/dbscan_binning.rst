DBSCANBinning
=============

.. currentmodule:: binlearn.methods

.. autoclass:: DBSCANBinning
   :members:
   :inherited-members:
   :show-inheritance:

Overview
--------

``DBSCANBinning`` creates bins based on DBSCAN (Density-Based Spatial Clustering of Applications with Noise) 
clustering of each feature. The bin edges are determined by the natural cluster boundaries identified by DBSCAN, 
which groups densely connected values together while handling outliers as noise.

This approach is particularly useful when:

* Your data has natural density-based clusters
* You want to identify and handle outliers automatically
* You need clustering that adapts to arbitrary cluster shapes
* You want bins that reflect the local density structure of your data

Key Features
------------

* **Density-Based Clustering**: Uses DBSCAN for robust density-based clustering
* **Outlier Detection**: Automatically identifies and handles outliers as noise
* **Arbitrary Shapes**: Can find clusters of any shape (not just spherical)
* **Parameter Control**: Fine-tune clustering with eps and min_samples parameters
* **Fallback Strategy**: Uses equal-width binning when insufficient clusters are found
* **Sklearn Compatibility**: Full transformer interface with fit/transform methods
* **DataFrame Support**: Preserves pandas/polars column names and structure

Basic Usage
-----------

.. code-block:: python

   import numpy as np
   import pandas as pd
   from binlearn.methods import DBSCANBinning
   
   # Create sample data with clusters and outliers
   np.random.seed(42)
   cluster1 = np.random.normal(10, 1, 100)
   cluster2 = np.random.normal(25, 1.5, 80)
   outliers = np.random.uniform(0, 40, 20)  # Scattered outliers
   data = np.concatenate([cluster1, cluster2, outliers])
   
   # Apply DBSCAN binning
   binner = DBSCANBinning(eps=2.0, min_samples=5)
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
           np.random.normal(10, 2, 150),
           np.random.normal(30, 2, 150),
           np.random.uniform(0, 40, 30)  # outliers
       ]),
       'feature2': np.concatenate([
           np.random.normal(5, 1, 150),
           np.random.normal(15, 1, 150),
           np.random.uniform(0, 20, 30)  # outliers
       ])
   })
   
   binner = DBSCANBinning(
       eps=3.0,
       min_samples=10,
       min_bins=2,
       preserve_dataframe=True
   )
   df_binned = binner.fit_transform(df)
   
   print(f"Bin edges for feature1: {binner.bin_edges_['feature1']}")
   print(f"Bin edges for feature2: {binner.bin_edges_['feature2']}")

Advanced Configuration
----------------------

.. code-block:: python

   # Fine-tuned DBSCAN parameters for different data characteristics
   
   # For dense, well-separated clusters
   dense_binner = DBSCANBinning(
       eps=0.5,           # Small neighborhood
       min_samples=10,    # Require more points for core samples
       min_bins=3         # Minimum number of bins
   )
   
   # For sparse data with loose clusters  
   sparse_binner = DBSCANBinning(
       eps=5.0,           # Larger neighborhood
       min_samples=3,     # Fewer points needed for core samples
       min_bins=2,        # Accept fewer bins
       clip=True          # Clip outliers to bin edges
   )

Parameter Tuning Example
------------------------

.. code-block:: python

   # Visualize different parameter effects
   import matplotlib.pyplot as plt
   
   # Test different eps values
   eps_values = [0.5, 1.0, 2.0, 4.0]
   
   fig, axes = plt.subplots(2, 2, figsize=(12, 8))
   axes = axes.flatten()
   
   for i, eps in enumerate(eps_values):
       binner = DBSCANBinning(eps=eps, min_samples=5)
       data_binned = binner.fit_transform(data.reshape(-1, 1))
       
       axes[i].hist(data_binned.flatten(), bins=20, alpha=0.7)
       axes[i].set_title(f'eps={eps}, bins={len(binner.bin_edges_[0])-1}')
   
   plt.tight_layout()
   plt.show()

Scikit-learn Pipeline Integration
---------------------------------

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split
   
   # Create a pipeline with DBSCAN binning
   pipeline = Pipeline([
       ('binning', DBSCANBinning(eps=2.0, min_samples=5)),
       ('classifier', RandomForestClassifier(random_state=42))
   ])
   
   # Use in ML workflow
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
   pipeline.fit(X_train, y_train)
   accuracy = pipeline.score(X_test, y_test)

Parameter Guide
---------------

**eps** (float, default=0.1)
    The maximum distance between two samples for one to be considered in the neighborhood of the other.
    This is the most important DBSCAN parameter:
    
    * Small values: More restrictive clustering, more bins
    * Large values: More permissive clustering, fewer bins
    * Rule of thumb: Start with the standard deviation of your data

**min_samples** (int, default=5)
    The number of samples in a neighborhood for a point to be considered as a core point:
    
    * Higher values: More restrictive clustering, fewer but denser clusters
    * Lower values: More permissive clustering, more clusters but potentially noisier
    * Rule of thumb: Use 2 * dimensions for 2D data, or at least 3

**min_bins** (int, default=2)
    Minimum number of bins to create. If DBSCAN produces fewer clusters than this,
    equal-width binning is used as a fallback strategy.

Handling Edge Cases
-------------------

.. code-block:: python

   # When DBSCAN finds insufficient clusters
   sparse_data = np.random.uniform(0, 100, 50).reshape(-1, 1)
   
   binner = DBSCANBinning(
       eps=1.0,
       min_samples=5,
       min_bins=3  # Fallback to equal-width if < 3 clusters found
   )
   
   # Will use equal-width binning as fallback for sparse data
   data_binned = binner.fit_transform(sparse_data)
   print(f"Used fallback strategy: {len(binner.bin_edges_[0]) - 1} bins created")

Tips for Parameter Selection
----------------------------

1. **Start with data exploration**:
   
   .. code-block:: python
   
      # Analyze data distribution first
      print(f"Data std: {np.std(data)}")
      print(f"Data range: {np.max(data) - np.min(data)}")
      
      # Start with eps ≈ std(data)
      suggested_eps = np.std(data)

2. **Use elbow method for eps**:
   
   .. code-block:: python
   
      from sklearn.neighbors import NearestNeighbors
      
      # Find optimal eps using k-distance plot
      neighbors = NearestNeighbors(n_neighbors=5)
      neighbors_fit = neighbors.fit(data.reshape(-1, 1))
      distances, indices = neighbors_fit.kneighbors(data.reshape(-1, 1))
      
      # Plot sorted distances to find "elbow"
      distances = np.sort(distances[:, 4], axis=0)
      plt.plot(distances)
      plt.ylabel("4th Nearest Neighbor Distance")
      plt.xlabel("Data Points sorted by distance")

See Also
--------

* :class:`KMeansBinning` - K-means clustering-based binning
* :class:`GaussianMixtureBinning` - Gaussian mixture model binning  
* :class:`EqualFrequencyBinning` - Quantile-based binning
* :class:`SupervisedBinning` - Decision tree-based supervised binning
