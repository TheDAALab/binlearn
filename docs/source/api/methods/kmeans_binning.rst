KMeansBinning
=============

.. currentmodule:: binning.methods

.. autoclass:: KMeansBinning
   :members:
   :inherited-members:
   :show-inheritance:

**K-means binning** creates bins by clustering data points and using cluster centroids to define bin boundaries. This method finds natural groupings in the data distribution.

Overview
--------

KMeansBinning applies K-means clustering to each feature independently and uses the resulting cluster centroids to create bin boundaries. This approach creates bins that reflect the natural density patterns in the data.

**Key Characteristics:**

* ✅ **Data-driven boundaries** - Adapts to data distribution patterns
* ✅ **Density-aware** - Creates more bins in dense regions
* ✅ **Robust to outliers** - Less sensitive than equal-width methods
* ✅ **Consistent clusters** - Reproducible with random_state
* ❌ **Computational cost** - More expensive than simple methods
* ❌ **Parameter sensitivity** - Results depend on initialization

Algorithm Details
-----------------

The algorithm works by:

1. **Applying K-means clustering** to each feature independently
2. **Sorting cluster centroids** to create ordered bin boundaries
3. **Creating bins** using midpoints between adjacent centroids
4. **Handling edge cases** for data points outside centroid range

**K-means Objective:**
   minimize Σᵢ Σⱼ ||xᵢⱼ - cⱼ||²

Where:
- xᵢⱼ are data points in cluster j
- cⱼ are cluster centroids

Parameters
----------

n_bins : int, default=5
    Number of clusters/bins to create per feature.

init : str, default='k-means++'
    Initialization method for K-means:
    - 'k-means++': Smart initialization (recommended)
    - 'random': Random initialization

max_iter : int, default=300
    Maximum number of K-means iterations.

n_init : int, default=10
    Number of K-means runs with different initializations.

random_state : int, optional
    Random state for reproducible results.

preserve_dataframe : bool, optional
    Whether to preserve input DataFrame format.

fit_jointly : bool, default=False
    Whether to consider feature interactions (not typically used).

Attributes
----------

_bin_edges : dict
    Fitted bin edges for each column after calling fit().

_cluster_centers : dict
    K-means cluster centers for each feature (internal use).

Examples
--------

Basic Usage Example
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import pandas as pd
   from binning.methods import KMeansBinning
   
   # Create sample data with natural clusters
   np.random.seed(42)
   data = np.concatenate([
       np.random.normal(0, 0.5, 300),    # Cluster around 0
       np.random.normal(3, 0.8, 200),    # Cluster around 3
       np.random.normal(-2, 0.3, 150)    # Cluster around -2
   ])
   
   df = pd.DataFrame({'feature': data})
   
   # Apply K-means binning
   binner = KMeansBinning(n_bins=4, random_state=42)
   df_binned = binner.fit_transform(df)
   
   print("Original data statistics:")
   print(df.describe())
   print("\nBinned data distribution:")
   print(df_binned['feature'].value_counts().sort_index())
   
   # Show bin edges
   print("\nBin edges:")
   print(binner._bin_edges[0])

Comparison with Equal-Width Binning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   from binning.methods import KMeansBinning, EqualWidthBinning
   
   # Create multi-modal data
   np.random.seed(42)
   data = np.concatenate([
       np.random.normal(-3, 0.5, 200),
       np.random.normal(0, 0.3, 500),
       np.random.normal(4, 0.8, 300)
   ])
   
   df = pd.DataFrame({'feature': data})
   
   # Apply different binning methods
   kmeans_binner = KMeansBinning(n_bins=6, random_state=42)
   equal_width_binner = EqualWidthBinning(n_bins=6)
   
   df_kmeans = kmeans_binner.fit_transform(df)
   df_equal_width = equal_width_binner.fit_transform(df)
   
   # Compare bin edges
   print("K-means bin edges:")
   print(kmeans_binner._bin_edges[0])
   print("\nEqual-width bin edges:")
   print(equal_width_binner._bin_edges[0])
   
   # Plot distributions
   fig, axes = plt.subplots(1, 3, figsize=(15, 5))
   
   # Original data
   axes[0].hist(df['feature'], bins=30, alpha=0.7, color='blue')
   axes[0].set_title('Original Data Distribution')
   axes[0].set_xlabel('Value')
   axes[0].set_ylabel('Frequency')
   
   # K-means binning
   bin_counts_kmeans = df_kmeans['feature'].value_counts().sort_index()
   axes[1].bar(range(len(bin_counts_kmeans)), bin_counts_kmeans.values, 
               alpha=0.7, color='green')
   axes[1].set_title('K-means Binning')
   axes[1].set_xlabel('Bin')
   axes[1].set_ylabel('Count')
   
   # Equal-width binning
   bin_counts_equal = df_equal_width['feature'].value_counts().sort_index()
   axes[2].bar(range(len(bin_counts_equal)), bin_counts_equal.values, 
               alpha=0.7, color='red')
   axes[2].set_title('Equal-Width Binning')
   axes[2].set_xlabel('Bin')
   axes[2].set_ylabel('Count')
   
   plt.tight_layout()
   plt.show()

Multi-Feature Dataset Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.datasets import make_blobs
   from binning.methods import KMeansBinning
   
   # Create multi-feature dataset with natural clusters
   X, y = make_blobs(n_samples=1000, centers=4, n_features=3, 
                    random_state=42, cluster_std=1.5)
   
   df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3'])
   
   # Apply K-means binning
   binner = KMeansBinning(n_bins=5, random_state=42)
   df_binned = binner.fit_transform(df)
   
   # Analyze bin distributions for each feature
   for col in df.columns:
       print(f"\n{col} bin distribution:")
       print(df_binned[col].value_counts().sort_index())
       
       # Show cluster centers
       col_idx = df.columns.get_loc(col)
       centers = binner._cluster_centers[col_idx]
       print(f"Cluster centers: {sorted(centers)}")

Handling Skewed Data
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from scipy import stats
   
   # Create heavily skewed data
   np.random.seed(42)
   skewed_data = stats.expon.rvs(scale=2, size=1000)  # Exponential distribution
   
   df = pd.DataFrame({'skewed_feature': skewed_data})
   
   # Compare different binning approaches
   kmeans_binner = KMeansBinning(n_bins=6, random_state=42)
   equal_width_binner = EqualWidthBinning(n_bins=6)
   equal_freq_binner = EqualFrequencyBinning(n_bins=6)
   
   df_kmeans = kmeans_binner.fit_transform(df)
   df_equal_width = equal_width_binner.fit_transform(df)
   df_equal_freq = equal_freq_binner.fit_transform(df)
   
   # Show how each method handles the skewness
   print("Bin distributions for skewed data:")
   print("\nK-means binning:")
   print(df_kmeans['skewed_feature'].value_counts().sort_index())
   print("\nEqual-width binning:")
   print(df_equal_width['skewed_feature'].value_counts().sort_index())
   print("\nEqual-frequency binning:")
   print(df_equal_freq['skewed_feature'].value_counts().sort_index())

Parameter Sensitivity Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Test different K-means parameters
   np.random.seed(42)
   data = np.concatenate([
       np.random.normal(0, 1, 300),
       np.random.normal(5, 1.5, 200)
   ])
   
   df = pd.DataFrame({'feature': data})
   
   # Test different initialization methods
   init_methods = ['k-means++', 'random']
   n_init_values = [1, 5, 10]
   
   results = {}
   
   for init_method in init_methods:
       for n_init in n_init_values:
           binner = KMeansBinning(
               n_bins=4, 
               init=init_method, 
               n_init=n_init,
               random_state=42
           )
           
           df_binned = binner.fit_transform(df)
           
           # Calculate within-cluster sum of squares as quality metric
           bin_edges = binner._bin_edges[0]
           bin_labels = df_binned['feature'].values
           
           # Simple quality metric: standard deviation within bins
           quality = 0
           for bin_id in range(len(bin_edges) - 1):
               bin_data = df[bin_labels == bin_id]['feature']
               if len(bin_data) > 0:
                   quality += bin_data.std()
           
           results[f"{init_method}_n_init_{n_init}"] = {
               'quality': quality,
               'bin_edges': bin_edges,
               'distribution': df_binned['feature'].value_counts().sort_index()
           }
   
   # Show results
   for config, result in results.items():
       print(f"\n{config}:")
       print(f"Quality score: {result['quality']:.3f}")
       print(f"Bin distribution: {result['distribution'].tolist()}")

Pipeline Integration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import cross_val_score
   from sklearn.datasets import make_classification
   
   # Create classification dataset
   X, y = make_classification(n_samples=1000, n_features=4, n_informative=3,
                            n_redundant=1, random_state=42)
   
   df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(4)])
   
   # Create pipeline with K-means binning
   pipeline = Pipeline([
       ('binning', KMeansBinning(n_bins=5, random_state=42)),
       ('scaling', StandardScaler()),
       ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
   ])
   
   # Evaluate pipeline
   scores = cross_val_score(pipeline, df, y, cv=5, scoring='accuracy')
   
   print(f"Pipeline accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
   
   # Compare with other binning methods
   pipelines = {
       'kmeans': KMeansBinning(n_bins=5, random_state=42),
       'equal_width': EqualWidthBinning(n_bins=5),
       'equal_freq': EqualFrequencyBinning(n_bins=5)
   }
   
   for name, binner in pipelines.items():
       pipeline = Pipeline([
           ('binning', binner),
           ('scaling', StandardScaler()),
           ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
       ])
       
       scores = cross_val_score(pipeline, df, y, cv=5, scoring='accuracy')
       print(f"{name}: {scores.mean():.3f} ± {scores.std():.3f}")

When to Use KMeansBinning
-------------------------

**Best For:**

* **Multi-modal data** - Natural clusters in the distribution
* **Density-based binning** - Want more bins where data is dense
* **Robust preprocessing** - Less sensitive to outliers than equal-width
* **Exploratory analysis** - Understanding natural data groupings

**Examples:**

* **Customer segmentation**: Natural groups in spending/behavior
* **Sensor data**: Natural operating modes or states
* **Financial data**: Market regimes or risk levels
* **Image processing**: Natural color or intensity clusters

**Avoid When:**

* **Small datasets** - K-means may not converge well
* **Uniform distributions** - Equal-width would be simpler
* **Very high dimensions** - Curse of dimensionality
* **Need exact sample balance** - Use equal-frequency instead

Performance Considerations
--------------------------

**Computational Complexity:**
   - O(n × k × i × f) where n=samples, k=clusters, i=iterations, f=features
   - More expensive than simple binning methods

**Memory Usage:**
   - Stores cluster centers and bin edges
   - Linear with number of features and bins

**Scalability:**
   - Good for moderate-sized datasets
   - May be slow on very large datasets
   - Parallelizable across features

Hyperparameter Tuning Guidelines
---------------------------------

**n_bins**: Start with 3-7, based on expected clusters
   - Too few: May miss important modes
   - Too many: May create spurious clusters

**init**: Use 'k-means++' for better initialization
   - 'k-means++': More stable, better clusters
   - 'random': Faster but less reliable

**n_init**: Use 10 for important analyses
   - Higher values: More stable results
   - Lower values: Faster computation

**max_iter**: Usually 300 is sufficient
   - Increase if convergence warnings appear

Common Issues and Solutions
---------------------------

**Issue: Empty or very small bins**

.. code-block:: python

   # Problem: Some bins have very few points
   
   # Solution: Reduce number of bins or check data distribution
   binner = KMeansBinning(n_bins=3, random_state=42)  # Reduce from 5
   
   # Or check for outliers
   Q1 = df['feature'].quantile(0.25)
   Q3 = df['feature'].quantile(0.75)
   IQR = Q3 - Q1
   lower_bound = Q1 - 1.5 * IQR
   upper_bound = Q3 + 1.5 * IQR
   
   # Filter outliers before binning
   df_filtered = df[(df['feature'] >= lower_bound) & 
                   (df['feature'] <= upper_bound)]

**Issue: Non-convergence warnings**

.. code-block:: python

   # Problem: K-means doesn't converge
   
   # Solution: Increase max_iter or change initialization
   binner = KMeansBinning(
       n_bins=5,
       max_iter=500,  # Increase from default
       n_init=15,     # More initializations
       random_state=42
   )

**Issue: Inconsistent results across runs**

.. code-block:: python

   # Problem: Different results each time
   
   # Solution: Always set random_state
   binner = KMeansBinning(n_bins=5, random_state=42)
   
   # Or increase n_init for more stable results
   binner = KMeansBinning(n_bins=5, n_init=20, random_state=42)

Comparison with Other Methods
-----------------------------

.. list-table:: Binning Method Comparison
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Method
     - Data Adaptation
     - Computational Cost
     - Outlier Sensitivity
     - Best Use Case
   * - KMeansBinning
     - High
     - Medium-High
     - Low
     - Multi-modal data
   * - EqualWidthBinning
     - None
     - Low
     - High
     - Uniform distributions
   * - EqualFrequencyBinning
     - Medium
     - Low
     - Medium
     - Balanced samples
   * - SupervisedBinning
     - High
     - Medium
     - Low
     - Classification tasks

Advanced Usage Patterns
-----------------------

**Custom Distance Metrics:**

.. code-block:: python

   from sklearn.cluster import KMeans
   
   class CustomKMeansBinning(KMeansBinning):
       def __init__(self, metric='euclidean', **kwargs):
           super().__init__(**kwargs)
           self.metric = metric
           
       def _fit_column(self, column_data, column_index, **kwargs):
           # Custom K-means with specific parameters
           kmeans = KMeans(
               n_clusters=self.n_bins,
               init=self.init,
               n_init=self.n_init,
               max_iter=self.max_iter,
               random_state=self.random_state
           )
           
           # Reshape for sklearn
           X_col = column_data.reshape(-1, 1)
           kmeans.fit(X_col)
           
           # Get cluster centers and sort them
           centers = sorted(kmeans.cluster_centers_.flatten())
           self._cluster_centers[column_index] = np.array(centers)
           
           # Create bin edges from centers
           if len(centers) == 1:
               min_val, max_val = column_data.min(), column_data.max()
               edges = [min_val, max_val]
           else:
               edges = [column_data.min()]
               for i in range(len(centers) - 1):
                   midpoint = (centers[i] + centers[i + 1]) / 2
                   edges.append(midpoint)
               edges.append(column_data.max())
           
           self._bin_edges[column_index] = np.array(edges)

See Also
--------

* :class:`EqualWidthBinning` - For simple equal-width binning
* :class:`EqualFrequencyBinning` - For balanced sample sizes
* :class:`SupervisedBinning` - For target-aware binning
* :doc:`../../examples/kmeans_binning` - Comprehensive examples

References
----------

* MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. In Proceedings of the fifth Berkeley symposium on mathematical statistics and probability (Vol. 1, No. 14, pp. 281-297).
* Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. In Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms (pp. 1027-1035).
* Lloyd, S. (1982). Least squares quantization in PCM. IEEE transactions on information theory, 28(2), 129-137.
