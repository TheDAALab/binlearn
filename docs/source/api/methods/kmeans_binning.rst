KMeansBinning
=============

.. currentmodule:: binlearn.methods

.. autoclass:: KMeansBinning
   :members:
   :inherited-members:
   :show-inheritance:

Examples
--------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from binlearn.methods import KMeansBinning
   
   # Create data with natural clusters
   np.random.seed(42)
   cluster1 = np.random.normal(0, 1, (300, 2))
   cluster2 = np.random.normal(5, 1, (300, 2))  
   cluster3 = np.random.normal(10, 1, (300, 2))
   
   X = np.vstack([cluster1, cluster2, cluster3])
   
   # Apply K-means binning
   kmeans_binner = KMeansBinning(n_bins=3, random_state=42)
   X_binned = kmeans_binner.fit_transform(X)
   
   print(f"Original shape: {X.shape}")
   print(f"Binned shape: {X_binned.shape}")
   print(f"Unique bins: {np.unique(X_binned)}")

Visualizing Clusters
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   
   # Visualize original data and binning results
   fig, axes = plt.subplots(1, 2, figsize=(12, 5))
   
   # Original data
   axes[0].scatter(X[:, 0], X[:, 1], alpha=0.6)
   axes[0].set_title('Original Data')
   axes[0].set_xlabel('Feature 1')
   axes[0].set_ylabel('Feature 2')
   
   # Binned data (colored by bin)
   scatter = axes[1].scatter(X[:, 0], X[:, 1], c=X_binned[:, 0], 
                           cmap='viridis', alpha=0.6)
   axes[1].set_title('K-Means Binning Results')
   axes[1].set_xlabel('Feature 1')
   axes[1].set_ylabel('Feature 2')
   plt.colorbar(scatter, ax=axes[1])
   
   plt.tight_layout()
   plt.show()

Comparing with Other Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from binlearn.methods import EqualWidthBinning, EqualFrequencyBinning
   
   # Create multimodal data
   X_multimodal = np.concatenate([
       np.random.normal(-3, 0.5, (200, 1)),
       np.random.normal(0, 0.3, (200, 1)), 
       np.random.normal(3, 0.8, (200, 1))
   ])
   
   # Compare different binning methods
   methods = {
       'Equal-Width': EqualWidthBinning(n_bins=3),
       'Equal-Frequency': EqualFrequencyBinning(n_bins=3),
       'K-Means': KMeansBinning(n_bins=3, random_state=42)
   }
   
   fig, axes = plt.subplots(2, 2, figsize=(12, 10))
   axes = axes.ravel()
   
   # Original data
   axes[0].hist(X_multimodal, bins=50, alpha=0.7, density=True)
   axes[0].set_title('Original Data Distribution')
   
   # Binning results
   for i, (name, binner) in enumerate(methods.items(), 1):
       X_binned = binner.fit_transform(X_multimodal)
       axes[i].hist(X_binned, bins=3, alpha=0.7, density=True)
       axes[i].set_title(f'{name} Binning')
       
       # Show bin edges
       if hasattr(binner, 'bin_edges_'):
           edges = binner.bin_edges_[0]
           for edge in edges[1:-1]:  # Skip first and last
               axes[i].axvline(edge, color='red', linestyle='--', alpha=0.7)
   
   plt.tight_layout()
   plt.show()

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Custom K-means parameters
   advanced_binner = KMeansBinning(
       n_bins=4,
       kmeans_params={
           'max_iter': 300,
           'tol': 1e-4,
           'n_init': 10
       },
       random_state=42
   )
   
   X_binned = advanced_binner.fit_transform(X)
   
   # Access the underlying K-means model
   print("K-means inertia:", advanced_binner.kmeans_models_[0].inertia_)
   print("Number of iterations:", advanced_binner.kmeans_models_[0].n_iter_)

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time
   
   # K-means binning can be slower for large datasets
   X_large = np.random.rand(100000, 10)
   
   # Time different methods
   methods = [
       ('EqualWidthBinning', EqualWidthBinning(n_bins=5)),
       ('KMeansBinning', KMeansBinning(n_bins=5, random_state=42))
   ]
   
   for name, binner in methods:
       start_time = time.time()
       binner.fit_transform(X_large)
       elapsed = time.time() - start_time
       print(f"{name}: {elapsed:.2f}s")
   
   # For large datasets, consider fitting on a sample
   sample_size = 10000
   sample_indices = np.random.choice(len(X_large), sample_size, replace=False)
   X_sample = X_large[sample_indices]
   
   start_time = time.time()
   fast_binner = KMeansBinning(n_bins=5, random_state=42)
   fast_binner.fit(X_sample)  # Fit on sample
   X_binned = fast_binner.transform(X_large)  # Transform full dataset
   sample_time = time.time() - start_time
   
   print(f"Sample-based KMeans: {sample_time:.2f}s")
