IsotonicBinning
===============

.. currentmodule:: binlearn.methods

.. autoclass:: IsotonicBinning
   :members:
   :inherited-members:
   :show-inheritance:

Overview
--------

``IsotonicBinning`` creates bins using isotonic regression to find optimal cut points that preserve 
monotonic relationships between features and targets. The transformer fits an isotonic (non-decreasing) 
function to the data and identifies significant changes in this function to determine bin boundaries.

This method is particularly effective when:

* There's a **known monotonic relationship** between feature and target
* You want bins that **respect monotonic ordering**
* Traditional tree-based methods might create non-monotonic splits
* You need **interpretable bins** that maintain logical ordering

Key Features
------------

* **Monotonicity Preservation**: Ensures bins respect monotonic relationships
* **Isotonic Regression**: Uses sklearn's IsotonicRegression for optimal fitting
* **Automatic Cut Points**: Identifies significant changes in isotonic function
* **Flexible Direction**: Supports both increasing and decreasing monotonicity
* **Sample Size Control**: Ensures minimum samples per bin for statistical validity
* **Supervised Learning**: Uses target variable information for optimal binning
* **Sklearn Compatibility**: Full transformer interface with fit/transform methods
* **DataFrame Support**: Preserves pandas/polars column names and structure

Basic Usage
-----------

.. code-block:: python

   import numpy as np
   import pandas as pd
   from binlearn.methods import IsotonicBinning
   
   # Create sample data with monotonic relationship
   np.random.seed(42)
   X = np.random.uniform(0, 10, 500).reshape(-1, 1)
   y = 2 * X.flatten() + np.random.normal(0, 1, 500)  # Linear + noise
   
   # Apply isotonic binning
   binner = IsotonicBinning(
       max_bins=6,
       min_samples_per_bin=20,
       increasing=True
   )
   
   # Fit using X and y (sklearn style)
   binner.fit(X, y)
   X_binned = binner.transform(X)
   
   print(f"Original shape: {X.shape}")
   print(f"Binned shape: {X_binned.shape}")
   print(f"Bin edges: {binner.bin_edges_[0]}")

Classification Example
----------------------

.. code-block:: python

   from sklearn.datasets import make_classification
   
   # Create classification data with monotonic relationship
   X, y = make_classification(
       n_samples=1000,
       n_features=1,
       n_redundant=0,
       n_clusters_per_class=1,
       random_state=42
   )
   
   # Sort by feature to create monotonic relationship
   sort_idx = np.argsort(X.flatten())
   X_sorted = X[sort_idx]
   y_sorted = y[sort_idx]
   
   binner = IsotonicBinning(
       max_bins=8,
       min_samples_per_bin=30,
       increasing=True,
       min_change_threshold=0.05
   )
   
   binner.fit(X_sorted, y_sorted)
   X_binned = binner.transform(X_sorted)
   
   print(f"Created {len(binner.bin_edges_[0]) - 1} bins")
   print(f"Bin edges: {binner.bin_edges_[0]}")

DataFrame Example with Guidance Columns
----------------------------------------

.. code-block:: python

   # Create DataFrame with target column
   df = pd.DataFrame({
       'age': np.random.uniform(18, 80, 1000),
       'income': np.random.uniform(20000, 150000, 1000),
       'credit_score': np.random.uniform(300, 850, 1000)
   })
   
   # Create monotonic target: risk increases with age, decreases with income/credit
   df['default_risk'] = (
       0.3 * (df['age'] - 18) / 62 +  # Age increases risk
       -0.4 * (df['income'] - 20000) / 130000 +  # Income decreases risk
       -0.3 * (df['credit_score'] - 300) / 550 +  # Credit decreases risk
       np.random.normal(0, 0.1, 1000)
   )
   
   # Bin each feature with appropriate monotonicity
   age_binner = IsotonicBinning(
       guidance_columns=['default_risk'],
       max_bins=5,
       increasing=True,  # Risk increases with age
       preserve_dataframe=True
   )
   
   income_binner = IsotonicBinning(
       guidance_columns=['default_risk'],
       max_bins=6,
       increasing=False,  # Risk decreases with income
       preserve_dataframe=True
   )
   
   # Apply binning
   df_age_binned = age_binner.fit_transform(df[['age', 'default_risk']])
   df_income_binned = income_binner.fit_transform(df[['income', 'default_risk']])

Advanced Configuration
----------------------

.. code-block:: python

   # Fine-tuned isotonic binning for specific requirements
   
   # Conservative binning (fewer bins, stricter requirements)
   conservative_binner = IsotonicBinning(
       max_bins=5,
       min_samples_per_bin=50,     # Larger bins for stability
       min_change_threshold=0.1,   # Require larger changes
       increasing=True,
       y_min=0.0,                  # Bound target values
       y_max=1.0
   )
   
   # Granular binning (more bins, sensitive to changes)
   granular_binner = IsotonicBinning(
       max_bins=15,
       min_samples_per_bin=10,     # Smaller bins allowed
       min_change_threshold=0.01,  # Sensitive to small changes
       increasing=True
   )

Parameter Guide
---------------

**max_bins** (int, default=10)
    Maximum number of bins to create. Actual number may be smaller:
    
    * Higher values: Allow more granular binning
    * Lower values: Force simpler, broader bins
    * Consider your model's complexity needs

**min_samples_per_bin** (int, default=5)
    Minimum samples required per bin for statistical validity:
    
    * Higher values: More stable bins, fewer total bins
    * Lower values: More granular binning, potentially less stable
    * Rule of thumb: At least 30 for regression, 10+ per class for classification

**increasing** (bool, default=True)
    Direction of monotonic relationship:
    
    * True: Higher feature values → higher target values
    * False: Higher feature values → lower target values
    * Must match your domain knowledge

**min_change_threshold** (float, default=0.01)
    Minimum relative change in isotonic function to create new bin:
    
    * Smaller values: More sensitive, create more bins
    * Larger values: Less sensitive, create fewer bins
    * Typical range: 0.005 to 0.1

**y_min, y_max** (float, optional)
    Bounds for target values in isotonic regression:
    
    * Helps constrain the isotonic function
    * Useful for normalized targets or known ranges
    * If None, uses data min/max

Scikit-learn Pipeline Integration
---------------------------------

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.ensemble import RandomForestRegressor
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import mean_squared_error
   
   # Create pipeline with isotonic binning
   pipeline = Pipeline([
       ('binning', IsotonicBinning(max_bins=6, increasing=True)),
       ('regressor', RandomForestRegressor(random_state=42))
   ])
   
   # Use in ML workflow
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
   pipeline.fit(X_train, y_train)
   y_pred = pipeline.predict(X_test)
   
   mse = mean_squared_error(y_test, y_pred)
   print(f"Pipeline MSE: {mse:.4f}")

Tips for Best Results
---------------------

1. **Validate monotonicity first**: Use correlation analysis to confirm monotonic relationships
2. **Choose appropriate direction**: Set `increasing` parameter based on domain knowledge
3. **Balance bin size and count**: Larger `min_samples_per_bin` gives more stable bins
4. **Consider target distribution**: Normalize targets if they have extreme ranges
5. **Validate on holdout data**: Check that monotonic relationship holds on test data

See Also
--------

* :class:`TreeBinning` - Decision tree-based supervised binning
* :class:`Chi2Binning` - Chi-square statistic-based supervised binning
* :class:`EqualFrequencyBinning` - Quantile-based unsupervised binning
* :class:`KMeansBinning` - K-means clustering-based binning

Overview
--------

``IsotonicBinning`` creates bins using isotonic regression to find optimal cut points that preserve 
monotonic relationships between features and targets. The transformer fits an isotonic (non-decreasing) 
function to the data and identifies significant changes in this function to determine bin boundaries.

This approach is particularly effective when:

* **Monotonic relationships exist** between features and targets
* **Ordinal consistency** is important in your binning
* **Regulatory requirements** mandate monotonic scoring models
* **Risk scoring** applications where higher values should consistently indicate higher risk

Key Features
------------

* **Monotonicity Preservation**: Ensures bins respect monotonic ordering relationships
* **Regression-Based**: Uses isotonic regression for optimal cut point identification  
* **Automatic Boundary Detection**: Identifies significant changes in fitted isotonic function
* **Flexible Direction**: Supports both increasing and decreasing monotonicity
* **Sample Control**: Ensures minimum samples per bin for statistical reliability
* **Sklearn Compatibility**: Full transformer interface with fit/transform methods
* **DataFrame Support**: Preserves pandas/polars column names and structure

Basic Usage
-----------

.. code-block:: python

   import numpy as np
   import pandas as pd
   from binlearn.methods import IsotonicBinning
   
   # Create data with monotonic relationship
   np.random.seed(42)
   X = np.random.rand(1000, 2)
   
   # Create target with monotonic relationship to first feature
   y = 2 * X[:, 0] + 0.5 * np.random.randn(1000)
   
   # Apply isotonic binning
   binner = IsotonicBinning(
       max_bins=8,
       min_samples_per_bin=50,
       increasing=True
   )
   
   # Method 1: Using fit with X and y (sklearn style)
   binner.fit(X, y)
   X_binned = binner.transform(X)
   
   print(f"Original shape: {X.shape}")
   print(f"Binned shape: {X_binned.shape}")
   print(f"Bins for feature 0: {len(binner.bin_edges_[0]) - 1}")

DataFrame Example with Target Column
------------------------------------

.. code-block:: python

   # Create DataFrame with monotonic relationships
   df = pd.DataFrame({
       'age': np.random.uniform(18, 80, 1000),
       'income': np.random.uniform(20000, 150000, 1000),
       'experience': np.random.uniform(0, 40, 1000)
   })
   
   # Create target with monotonic relationships
   df['default_risk'] = (
       0.01 * df['age'] + 
       -0.00001 * df['income'] + 
       -0.005 * df['experience'] + 
       0.2 * np.random.randn(1000)
   )
   
   # Method 2: Using guidance_columns (binlearn style)
   binner = IsotonicBinning(
       guidance_columns=['default_risk'],
       max_bins=6,
       min_samples_per_bin=100,
       preserve_dataframe=True
   )
   
   df_binned = binner.fit_transform(df)
   
   print(f"Bin edges for age: {binner.bin_edges_['age']}")
   print(f"Bin edges for income: {binner.bin_edges_['income']}")

Decreasing Monotonicity Example
-------------------------------

.. code-block:: python

   # Example where higher feature values should lead to lower target values
   X_credit = np.random.uniform(0, 100, 500).reshape(-1, 1)  # Credit score
   y_default = 1 / (1 + np.exp(0.1 * (X_credit.flatten() - 50)))  # Lower default prob for higher scores
   
   # Use decreasing monotonicity
   binner = IsotonicBinning(
       max_bins=5,
       min_samples_per_bin=50,
       increasing=False,  # Higher credit score = lower default probability
       min_change_threshold=0.05
   )
   
   binner.fit(X_credit, y_default)
   X_credit_binned = binner.transform(X_credit)
   
   # Verify monotonicity: bin representatives should decrease
   print("Bin representatives:", binner.bin_representatives_[0])
   print("Monotonically decreasing:", 
         all(binner.bin_representatives_[0][i] >= binner.bin_representatives_[0][i+1] 
             for i in range(len(binner.bin_representatives_[0])-1)))

Advanced Configuration
----------------------

.. code-block:: python

   # Fine-tuned isotonic binning for different scenarios
   
   # High-precision binning (more sensitive to changes)
   precise_binner = IsotonicBinning(
       max_bins=12,
       min_samples_per_bin=30,
       min_change_threshold=0.005,  # More sensitive to changes
       increasing=True,
       y_min=0.0,                   # Explicit bounds
       y_max=1.0
   )
   
   # Robust binning (less sensitive, larger bins)
   robust_binner = IsotonicBinning(
       max_bins=6,
       min_samples_per_bin=100,     # Larger bins for stability
       min_change_threshold=0.1,    # Less sensitive to changes
       increasing=True
   )

Classification Example
----------------------

.. code-block:: python

   from sklearn.datasets import make_classification
   from sklearn.preprocessing import LabelEncoder
   
   # Create classification data
   X_class, y_class = make_classification(
       n_samples=1000, 
       n_features=3, 
       n_classes=3,
       n_redundant=0,
       random_state=42
   )
   
   # Isotonic binning works with classification by treating classes ordinally
   binner = IsotonicBinning(
       max_bins=7,
       min_samples_per_bin=50,
       increasing=True
   )
   
   binner.fit(X_class, y_class)
   X_class_binned = binner.transform(X_class)
   
   print(f"Classification bins: {[len(edges)-1 for edges in binner.bin_edges_.values()]}")

Risk Scoring Pipeline
---------------------

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.linear_model import LogisticRegression
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import roc_auc_score
   
   # Create a risk scoring pipeline with monotonic binning
   risk_pipeline = Pipeline([
       ('isotonic_binning', IsotonicBinning(
           max_bins=8,
           min_samples_per_bin=50,
           increasing=True
       )),
       ('logistic_regression', LogisticRegression(random_state=42))
   ])
   
   # Use in credit risk modeling
   X_train, X_test, y_train, y_test = train_test_split(
       X, y > np.median(y), test_size=0.2, random_state=42
   )
   
   risk_pipeline.fit(X_train, y_train)
   y_proba = risk_pipeline.predict_proba(X_test)[:, 1]
   auc_score = roc_auc_score(y_test, y_proba)
   
   print(f"AUC with isotonic binning: {auc_score:.3f}")

Parameter Guide
---------------

**max_bins** (int, default=10)
    Maximum number of bins to create per feature:
    
    * Higher values: More granular risk segments
    * Lower values: Broader, more stable risk categories
    * Consider regulatory requirements and model interpretability

**min_samples_per_bin** (int, default=5)
    Minimum samples required per bin for statistical reliability:
    
    * Higher values: More stable bins, better statistical power
    * Lower values: More granular binning, potential instability
    * Rule of thumb: At least 30-50 for reliable estimates

**increasing** (bool, default=True)
    Direction of monotonicity to enforce:
    
    * True: Higher feature values → higher target values
    * False: Higher feature values → lower target values
    * Choose based on domain knowledge and expected relationships

**min_change_threshold** (float, default=0.01)
    Minimum relative change in fitted values to create new bin:
    
    * Lower values: More sensitive, more bins
    * Higher values: Less sensitive, fewer bins
    * Typical range: 0.005 (sensitive) to 0.1 (robust)

**y_min, y_max** (float, optional)
    Bounds for target values in isotonic regression:
    
    * Explicit bounds can improve numerical stability
    * Useful for probability targets: y_min=0.0, y_max=1.0
    * Auto-detected from data if not specified

Monotonicity Validation
-----------------------

.. code-block:: python

   # Function to validate monotonicity in binned results
   def validate_monotonicity(binner, feature_idx=0, increasing=True):
       """Validate that bin representatives follow monotonic order."""
       reps = binner.bin_representatives_[feature_idx]
       
       if increasing:
           is_monotonic = all(reps[i] <= reps[i+1] for i in range(len(reps)-1))
           direction = "increasing"
       else:
           is_monotonic = all(reps[i] >= reps[i+1] for i in range(len(reps)-1))
           direction = "decreasing"
       
       print(f"Monotonicity ({direction}): {is_monotonic}")
       print(f"Representatives: {reps}")
       return is_monotonic
   
   # Validate our binning results
   validate_monotonicity(binner, feature_idx=0, increasing=True)

Tips for Best Results
---------------------

1. **Verify monotonic relationships exist** in your data before applying
2. **Choose appropriate min_samples_per_bin** based on your sample size
3. **Adjust min_change_threshold** based on noise level in your data
4. **Consider regulatory constraints** for financial/medical applications
5. **Validate results** on holdout data to avoid overfitting

Common Use Cases
----------------

* **Credit Risk Scoring**: Age, income, debt-to-income ratio
* **Medical Risk Assessment**: Biomarkers, age, symptom severity  
* **Marketing Response**: Customer value, engagement metrics
* **Predictive Maintenance**: Usage hours, temperature readings
* **Quality Control**: Process parameters, environmental conditions

See Also
--------

* :class:`TreeBinning` - Decision tree-based supervised binning
* :class:`Chi2Binning` - Chi-square statistic-based supervised binning
* :class:`EqualFrequencyBinning` - Quantile-based unsupervised binning
* :class:`KMeansBinning` - K-means clustering-based binning
