TreeBinning
===========

.. currentmodule:: binlearn.methods

.. autoclass:: TreeBinning
   :members:
   :inherited-members:
   :show-inheritance:

Overview
--------

``TreeBinning`` creates bins using decision tree-based supervised discretization. The method uses 
decision trees to find optimal split points that maximize predictive performance for classification or 
regression tasks. This approach creates bins that are optimized for a specific target variable.

Key Features
------------

* **Decision Tree Foundation**: Uses sklearn's DecisionTreeClassifier/Regressor for optimal splits
* **Dual Interface**: Supports both sklearn-style (X, y) and binlearn-style (guidance_columns) usage
* **Task Flexibility**: Handles both classification and regression targets
* **Configurable Trees**: Full control over decision tree parameters
* **Automatic Extraction**: Extracts bin edges from tree structure automatically
* **Sklearn Compatibility**: Full transformer interface with fit/transform methods
* **DataFrame Support**: Preserves pandas/polars column names and structure

Examples
--------

Basic Classification (sklearn-style)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from sklearn.datasets import make_classification
   from binlearn.methods import TreeBinning
   
   # Create classification dataset
   X, y = make_classification(n_samples=1000, n_features=4, n_classes=2, random_state=42)
   
   # Method 1: sklearn-style interface
   sup_binner = TreeBinning(
       task_type='classification',
       tree_params={'max_depth': 3, 'min_samples_leaf': 20}
   )
   
   # Fit with X and y separately
   sup_binner.fit(X, y)
   X_binned = sup_binner.transform(X)
   
   print(f"Original shape: {X.shape}")
   print(f"Binned shape: {X_binned.shape}")
   print(f"Bin edges for feature 0: {sup_binner.bin_edges_[0]}")

Basic Classification (binlearn-style)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Method 2: binlearn-style with guidance_columns
   # Combine features and target into single dataset
   X_with_target = np.column_stack([X, y])
   
   sup_binner2 = TreeBinning(
       guidance_columns=[4],  # Use column 4 (target) as guidance
       task_type='classification',
       tree_params={'max_depth': 3, 'min_samples_leaf': 20}
   )
   
   X_binned2 = sup_binner2.fit_transform(X_with_target)
   
   # Both methods produce identical results
   print(f"Methods produce same results: {np.array_equal(X_binned, X_binned2)}")

DataFrame Example
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   
   # Create DataFrame with features and target
   df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3', 'feature4'])
   df['target'] = y
   
   # Sklearn-style: fit with separate X and y
   binner1 = TreeBinning(
       task_type='classification',
       preserve_dataframe=True
   )
   binner1.fit(df[['feature1', 'feature2', 'feature3', 'feature4']], df['target'])
   df_binned1 = binner1.transform(df[['feature1', 'feature2', 'feature3', 'feature4']])
   
   # Binlearn-style: use guidance_columns
   binner2 = TreeBinning(
       guidance_columns=['target'],
       task_type='classification',
       preserve_dataframe=True
   )
   df_binned2 = binner2.fit_transform(df)
   
   print("Feature binning with DataFrame:")
   print(df_binned2.head())

Regression Task
~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.datasets import make_regression
   
   # Create regression dataset
   X_reg, y_reg = make_regression(n_samples=1000, n_features=3, noise=0.1, random_state=42)
   
   # Supervised binning for regression
   reg_binner = TreeBinning(
       task_type='regression',
       tree_params={
           'max_depth': 4,
           'min_samples_leaf': 50,
           'random_state': 42
       }
   )
   
   # Fit and transform
   reg_binner.fit(X_reg, y_reg)
   X_reg_binned = reg_binner.transform(X_reg)
   
   print(f"Regression binning:")
   print(f"  Original features: {X_reg.shape[1]}")
   print(f"  Bins per feature: {[len(edges)-1 for edges in reg_binner.bin_edges_.values()]}")

Multi-class Classification
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Multi-class classification example
   X_multi, y_multi = make_classification(
       n_samples=1500, 
       n_features=5, 
       n_classes=3,
       n_informative=4,
       random_state=42
   )
   
   multi_binner = TreeBinning(
       task_type='classification',
       tree_params={
           'max_depth': 5,
           'min_samples_split': 100,
           'min_samples_leaf': 30,
           'random_state': 42
       }
   )
   
   multi_binner.fit(X_multi, y_multi)
   X_multi_binned = multi_binner.transform(X_multi)
   
   # Analyze bins created for each class
   print(f"Multi-class binning results:")
   for i, edges in multi_binner.bin_edges_.items():
       print(f"  Feature {i}: {len(edges)-1} bins, edges: {edges}")

Advanced Tree Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Fine-tuned decision tree parameters
   advanced_binner = SupervisedBinning(
       task_type='classification',
       tree_params={
           'max_depth': 6,              # Deeper trees for more bins
           'min_samples_split': 200,    # Require more samples for splits
           'min_samples_leaf': 100,     # Larger leaf nodes
           'max_features': 'sqrt',      # Feature sampling
           'random_state': 42,          # Reproducibility
           'class_weight': 'balanced'   # Handle imbalanced classes
       }
   )
   
   advanced_binner.fit(X, y)
   X_advanced_binned = advanced_binner.transform(X)
   
   print("Advanced configuration results:")
   for feature_idx, edges in advanced_binner.bin_edges_.items():
       print(f"  Feature {feature_idx}: {len(edges)-1} bins")

Comparison with Unsupervised Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from binlearn.methods import EqualWidthBinning, EqualFrequencyBinning
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import accuracy_score
   
   # Split data for comparison
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
   # Compare different binning methods
   binners = {
       'supervised': SupervisedBinning(task_type='classification'),
       'equal_width': EqualWidthBinning(n_bins=5),
       'equal_frequency': EqualFrequencyBinning(n_bins=5)
   }
   
   results = {}
   classifier = RandomForestClassifier(random_state=42, n_estimators=100)
   
   for name, binner in binners.items():
       # Fit binner
       if name == 'supervised':
           binner.fit(X_train, y_train)
       else:
           binner.fit(X_train)
       
       # Transform data
       X_train_binned = binner.transform(X_train)
       X_test_binned = binner.transform(X_test)
       
       # Train classifier
       classifier.fit(X_train_binned, y_train)
       y_pred = classifier.predict(X_test_binned)
       
       results[name] = accuracy_score(y_test, y_pred)
       print(f"{name}: {results[name]:.3f} accuracy")

Scikit-learn Pipeline Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.model_selection import GridSearchCV
   
   # Create pipeline with supervised binning
   pipeline = Pipeline([
       ('binning', SupervisedBinning(task_type='classification')),
       ('classifier', RandomForestClassifier(random_state=42))
   ])
   
   # Parameter grid for both binning and classification
   param_grid = {
       'binning__tree_params': [
           {'max_depth': 3, 'min_samples_leaf': 20},
           {'max_depth': 4, 'min_samples_leaf': 30},
           {'max_depth': 5, 'min_samples_leaf': 50}
       ],
       'classifier__n_estimators': [50, 100, 200]
   }
   
   # Grid search
   grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
   grid_search.fit(X_train, y_train)
   
   print(f"Best parameters: {grid_search.best_params_}")
   print(f"Best cross-validation score: {grid_search.best_score_:.3f}")

Parameter Guide
---------------

**task_type** (str, required)
    Type of supervised learning task:
    
    * 'classification': For discrete target variables
    * 'regression': For continuous target variables

**tree_params** (dict, optional)
    Parameters passed to sklearn's DecisionTree:
    
    * **max_depth**: Maximum depth of decision tree
    * **min_samples_split**: Minimum samples required to split
    * **min_samples_leaf**: Minimum samples in leaf nodes
    * **random_state**: Random seed for reproducibility
    * **class_weight**: Class weighting for imbalanced data (classification only)

**guidance_columns** (list, optional)
    Columns to use as targets (binlearn-style interface):
    
    * Alternative to passing y separately
    * Useful when target is part of input DataFrame
    * Can specify multiple guidance columns

Usage Patterns
--------------

**When to Use Supervised Binning:**

1. **Target-Optimized Features**: When bins should maximize predictive performance
2. **Feature Engineering**: Creating informative bins for downstream models
3. **Interpretable Models**: When bin boundaries need to be explainable
4. **Imbalanced Data**: Tree parameters can handle class imbalance

**Best Practices:**

1. **Validation**: Always validate on holdout data to avoid overfitting
2. **Tree Tuning**: Adjust tree parameters based on your data size and complexity
3. **Task Type**: Ensure correct task_type for your target variable
4. **Feature Selection**: Consider feature importance when interpreting results

See Also
--------

* :class:`Chi2Binning` - Chi-square statistic-based supervised binning
* :class:`IsotonicBinning` - Isotonic regression-based supervised binning
* :class:`EqualWidthBinning` - Unsupervised equal-width binning
* :class:`EqualFrequencyBinning` - Unsupervised quantile-based binning
