SupervisedBinning
=================

.. currentmodule:: binlearn.methods

.. autoclass:: SupervisedBinning
   :members:
   :inherited-members:
   :show-inheritance:

**Supervised binning** creates bins by considering the target variable to optimize predictive performance. This method finds bin boundaries that maximize information gain or minimize impurity with respect to the target.

Overview
--------

SupervisedBinning uses decision tree-based algorithms to find optimal split points that best separate different target classes or minimize regression error. Unlike unsupervised methods, it takes into account the relationship between features and the target variable.

**Key Characteristics:**

* ✅ **Target-aware** - Optimizes bins for predictive performance
* ✅ **Information-theoretic** - Maximizes information gain (classification) or minimizes variance (regression)
* ✅ **Adaptive boundaries** - Bin edges adapt to data patterns and target relationships
* ✅ **Feature selection** - Can identify irrelevant features through split quality
* ✅ **Flexible** - Supports both classification and regression tasks
* ❌ **Requires target** - Cannot be used for unsupervised tasks
* ❌ **Overfitting risk** - May overfit to training data without proper regularization

Algorithm Details
-----------------

The algorithm works by:

1. **Building decision trees** for each feature independently
2. **Finding split points** that maximize information gain (classification) or minimize MSE (regression)
3. **Extracting bin boundaries** from tree splits
4. **Limiting complexity** through tree parameters to control number of bins

**Information Gain Formula (Classification):**
   IG(S, A) = H(S) - Σ(|Sv|/|S| × H(Sv))

**MSE Reduction Formula (Regression):**
   MSE_reduction = MSE(S) - Σ(|Sv|/|S| × MSE(Sv))

Where:
- H(S) is entropy of the target variable
- MSE(S) is mean squared error of the target variable  
- Sv are subsets created by splitting on attribute A

Parameters
----------

n_bins : int, default=5
    Maximum number of bins to create per feature.

task_type : {'classification', 'regression'}, default='classification'
    Type of supervised learning task.

tree_params : dict | None, default=None
    Additional parameters for the decision tree. Common parameters include:
    
    * 'max_depth': Maximum depth of the tree
    * 'min_samples_split': Minimum samples required to split a node
    * 'min_samples_leaf': Minimum samples required at a leaf node
    * 'random_state': Random state for reproducibility

columns : list[str | int] | None, default=None
    Specific columns to bin. If None, bins all columns.

guidance_columns : list[str | int] | None, default=None
    Columns to exclude from binlearn (used as guidance only).

preserve_dataframe : bool | None, default=None
    Whether to preserve the input DataFrame format. If None, auto-detects.
    Splitting criterion for decision trees:
    - 'entropy': Information gain based on entropy
    - 'gini': Gini impurity reduction

max_depth : int, optional
    Maximum depth of decision trees. If None, limited by n_bins.

min_samples_split : int, default=2
    Minimum samples required to split an internal node.

min_samples_leaf : int, default=1  
    Minimum samples required at a leaf node.

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

_trees : dict
    Decision trees fitted for each feature (internal use).

Examples
--------

Basic Classification Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import pandas as pd
   from binlearn.methods import SupervisedBinning
   from sklearn.datasets import make_classification
   
   # Create sample classification data
   X, y = make_classification(n_samples=1000, n_features=3, n_redundant=0, 
                            n_informative=3, random_state=42)
   
   # Convert to DataFrame for easier interpretation
   df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3'])
   
   # Apply supervised binning
   binner = SupervisedBinning(n_bins=4, criterion='entropy')
   df_binned = binner.fit_transform(df, y)
   
   print("Original features:")
   print(df.head())
   print("\nBinned features:")
   print(df_binned.head())
   
   # Analyze information gain per feature
   from sklearn.metrics import mutual_info_score
   
   for col in df.columns:
       original_info = mutual_info_score(y, df[col])
       binned_info = mutual_info_score(y, df_binned[col])
       improvement = (binned_info - original_info) / original_info * 100
       print(f"{col}: {improvement:.1f}% information gain improvement")

Binary Classification with Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from binlearn.methods import SupervisedBinning, EqualWidthBinning
   from sklearn.model_selection import cross_val_score
   from sklearn.ensemble import RandomForestClassifier
   
   # Create sample data
   np.random.seed(42)
   n_samples = 2000
   
   # Create features with different relationships to target
   feature1 = np.random.normal(0, 1, n_samples)  # Weak relationship
   feature2 = np.random.exponential(2, n_samples)  # Strong relationship
   feature3 = np.random.uniform(-5, 5, n_samples)  # Medium relationship
   
   # Create target with specific relationships
   y = ((feature1 > 0) & (feature2 > 2) | (feature3 > 2)).astype(int)
   
   X = pd.DataFrame({
       'feature1': feature1,
       'feature2': feature2, 
       'feature3': feature3
   })
   
   # Compare supervised vs unsupervised binning
   supervised_binner = SupervisedBinning(n_bins=5, criterion='entropy')
   unsupervised_binner = EqualWidthBinning(n_bins=5)
   
   # Apply binning methods
   X_supervised = supervised_binner.fit_transform(X, y)
   X_unsupervised = unsupervised_binner.fit_transform(X)
   
   # Evaluate with cross-validation
   rf = RandomForestClassifier(n_estimators=100, random_state=42)
   
   # Original features
   scores_original = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
   
   # Supervised binning
   scores_supervised = cross_val_score(rf, X_supervised, y, cv=5, scoring='accuracy')
   
   # Unsupervised binning
   scores_unsupervised = cross_val_score(rf, X_unsupervised, y, cv=5, scoring='accuracy')
   
   print("Cross-validation accuracy:")
   print(f"Original features: {scores_original.mean():.3f} ± {scores_original.std():.3f}")
   print(f"Supervised binning: {scores_supervised.mean():.3f} ± {scores_supervised.std():.3f}")
   print(f"Unsupervised binning: {scores_unsupervised.mean():.3f} ± {scores_unsupervised.std():.3f}")

Multi-class Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.datasets import make_classification
   from binlearn.methods import SupervisedBinning
   
   # Create multi-class dataset
   X, y = make_classification(n_samples=1500, n_features=4, n_classes=3,
                            n_informative=3, n_redundant=1, random_state=42)
   
   df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(4)])
   
   # Apply supervised binning with different criteria
   entropy_binner = SupervisedBinning(n_bins=4, criterion='entropy')
   gini_binner = SupervisedBinning(n_bins=4, criterion='gini')
   
   df_entropy = entropy_binner.fit_transform(df, y)
   df_gini = gini_binner.fit_transform(df, y)
   
   # Compare bin edges
   print("Entropy-based bin edges:")
   for col_idx, edges in entropy_binner._bin_edges.items():
       print(f"Feature {col_idx}: {edges}")
   
   print("\nGini-based bin edges:")
   for col_idx, edges in gini_binner._bin_edges.items():
       print(f"Feature {col_idx}: {edges}")

Pipeline Integration for Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.linear_model import LogisticRegression
   from sklearn.model_selection import GridSearchCV
   
   # Create preprocessing pipeline with hyperparameter tuning
   pipeline = Pipeline([
       ('binning', SupervisedBinning()),
       ('scaling', StandardScaler()),
       ('classifier', LogisticRegression(random_state=42))
   ])
   
   # Define parameter grid
   param_grid = {
       'binning__n_bins': [3, 4, 5, 6],
       'binning__criterion': ['entropy', 'gini'],
       'binning__min_samples_split': [2, 5, 10],
       'classifier__C': [0.1, 1, 10]
   }
   
   # Perform grid search
   grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
   grid_search.fit(X_train, y_train)
   
   print("Best parameters:")
   print(grid_search.best_params_)
   print(f"Best accuracy: {grid_search.best_score_:.3f}")

Feature Analysis and Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from binlearn.methods import SupervisedBinning
   from sklearn.metrics import mutual_info_score
   
   # Create dataset with some irrelevant features
   np.random.seed(42)
   n_samples = 1000
   
   # Relevant features
   relevant1 = np.random.normal(0, 1, n_samples)
   relevant2 = np.random.exponential(1, n_samples)
   
   # Irrelevant features (noise)
   noise1 = np.random.uniform(-1, 1, n_samples)
   noise2 = np.random.normal(0, 0.1, n_samples)
   
   # Target depends only on relevant features
   y = ((relevant1 > 0) & (relevant2 > 1)).astype(int)
   
   X = pd.DataFrame({
       'relevant1': relevant1,
       'relevant2': relevant2,
       'noise1': noise1,
       'noise2': noise2
   })
   
   # Apply supervised binning
   binner = SupervisedBinning(n_bins=5, criterion='entropy')
   X_binned = binner.fit_transform(X, y)
   
   # Calculate mutual information for feature ranking
   feature_scores = {}
   for col in X.columns:
       original_score = mutual_info_score(y, pd.cut(X[col], bins=5, labels=False))
       binned_score = mutual_info_score(y, X_binned[col])
       feature_scores[col] = {
           'original': original_score,
           'binned': binned_score,
           'improvement': binned_score - original_score
       }
   
   print("Feature importance analysis:")
   for feature, scores in sorted(feature_scores.items(), 
                               key=lambda x: x[1]['binned'], reverse=True):
       print(f"{feature}:")
       print(f"  Original MI: {scores['original']:.4f}")
       print(f"  Binned MI: {scores['binned']:.4f}")
       print(f"  Improvement: {scores['improvement']:.4f}")

When to Use SupervisedBinning
-----------------------------

**Best For:**

* **Classification tasks** - Optimizes for target variable relationships
* **Feature preprocessing** - Creates more predictive features
* **Noisy data** - Can find signal in noisy continuous features
* **Interpretable models** - Creates meaningful categorical features

**Examples:**

* **Medical diagnosis**: Finding optimal thresholds for biomarkers
* **Credit scoring**: Optimal age, income, and debt ratio bins
* **Marketing**: Customer segmentation based on purchase behavior
* **Quality control**: Optimal measurement thresholds

**Avoid When:**

* **Unsupervised tasks** - No target variable available
* **Regression with continuous targets** - Designed for classification
* **Very small datasets** - Risk of overfitting
* **High-dimensional data** - May not scale well

Performance Considerations
--------------------------

**Computational Complexity:**
   - O(n × log(n) × f) where n=samples, f=features
   - Decision tree fitting dominates computation

**Memory Usage:**
   - Stores decision trees for each feature
   - Linear with number of features and tree depth

**Scalability:**
   - Efficient for moderate-sized datasets
   - May be slow on very large datasets
   - Parallelizable across features

Hyperparameter Tuning Guidelines
---------------------------------

**n_bins**: Start with 3-5, increase if needed
   - Too few: May miss important patterns
   - Too many: Risk of overfitting

**criterion**: 
   - 'entropy': Better for balanced classes
   - 'gini': Faster computation, similar results

**min_samples_split**: 
   - Small datasets: Use default (2)
   - Large datasets: Increase to 10-50 to prevent overfitting

**min_samples_leaf**:
   - Increase to prevent overfitting
   - Should be at least 1% of total samples

Common Issues and Solutions
---------------------------

**Issue: Overfitting to training data**

.. code-block:: python

   # Problem: Too many bins, splits on noise
   
   # Solution: Increase regularization parameters
   binner = SupervisedBinning(
       n_bins=4,  # Reduce from higher number
       min_samples_split=20,  # Increase from default
       min_samples_leaf=10    # Increase from default
   )

**Issue: Imbalanced classes affecting splits**

.. code-block:: python

   from sklearn.utils.class_weight import compute_sample_weight
   
   # Problem: Imbalanced target classes
   
   # Solution: Use sample weights (requires custom implementation)
   # or balance the dataset before binning
   from sklearn.utils import resample
   
   # Balance the dataset
   X_balanced, y_balanced = resample(X, y, 
                                   random_state=42, 
                                   stratify=y)

**Issue: Features with no predictive power**

.. code-block:: python

   # Problem: Some features create identical bins
   
   # Solution: Check bin boundaries and remove uninformative features
   binner = SupervisedBinning(n_bins=5)
   X_binned = binner.fit_transform(X, y)
   
   # Check which features have meaningful splits
   for col_idx, edges in binner._bin_edges.items():
       if len(edges) <= 2:  # Only min and max
           print(f"Feature {col_idx} has no meaningful splits")

Comparison with Other Methods
-----------------------------

.. list-table:: Binning Method Comparison
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Method
     - Target Aware
     - Bin Balance
     - Interpretability  
     - Best Use Case
   * - SupervisedBinning
     - Yes
     - Variable
     - Good
     - Classification
   * - EqualWidthBinning
     - No
     - Variable
     - Excellent
     - EDA, Visualization
   * - EqualFrequencyBinning
     - No
     - Balanced
     - Good
     - Skewed Data
   * - WeightConstrained
     - Partial
     - Weighted
     - Good
     - Importance Weighting

Advanced Usage Patterns
-----------------------

**Custom Scoring Functions:**

.. code-block:: python

   from sklearn.tree import DecisionTreeClassifier
   
   class CustomSupervisedBinning(SupervisedBinning):
       def __init__(self, **kwargs):
           super().__init__(**kwargs)
           
       def _fit_column(self, column_data, column_index, y=None, **kwargs):
           # Custom tree with specific parameters
           tree = DecisionTreeClassifier(
               max_leaf_nodes=self.n_bins,
               criterion=self.criterion,
               min_samples_split=self.min_samples_split,
               min_samples_leaf=self.min_samples_leaf,
               random_state=self.random_state
           )
           
           # Reshape for sklearn
           X_col = column_data.reshape(-1, 1)
           tree.fit(X_col, y)
           
           # Extract split points
           split_points = self._extract_splits(tree, column_data)
           
           # Create bin edges
           min_val, max_val = column_data.min(), column_data.max()
           edges = sorted([min_val] + split_points + [max_val])
           edges = list(dict.fromkeys(edges))  # Remove duplicates
           
           self._bin_edges[column_index] = np.array(edges)

See Also
--------

* :class:`EqualWidthBinning` - For simple equal-width binning
* :class:`EqualFrequencyBinning` - For balanced sample sizes
* :class:`EqualWidthMinimumWeightBinning` - For weight-constrained binning
* :doc:`../../examples/supervised_binning` - Comprehensive examples

References
----------

* Fayyad, U., & Irani, K. (1993). Multi-interval discretization of continuous-valued attributes for classification learning. In Proceedings of the 13th international joint conference on artificial intelligence (pp. 1022-1027).
* Dougherty, J., Kohavi, R., & Sahami, M. (1995). Supervised and unsupervised discretization of continuous features. In Machine learning proceedings 1995 (pp. 194-202).
* Quinlan, J. R. (1986). Induction of decision trees. Machine learning, 1(1), 81-106.

Supervised binning transformer that uses target variable information to optimize bin boundaries.
