OneHotBinning
=============

.. currentmodule:: binning.methods

.. autoclass:: OneHotBinning
   :members:
   :inherited-members:
   :show-inheritance:

**One-hot binning** creates separate bins for each unique value in the data, essentially performing categorical encoding on continuous or discrete numerical features.

Overview
--------

OneHotBinning transforms each unique value in numerical features into its own bin. This is particularly useful for discrete numerical features or when you want to treat each unique value as a separate category.

**Key Characteristics:**

* ✅ **Value preservation** - Each unique value gets its own bin
* ✅ **No information loss** - Preserves all distinct values
* ✅ **Simple interpretation** - Direct mapping from values to bins
* ✅ **Handles discrete data** - Perfect for integer/categorical-like features
* ❌ **Dimension explosion** - Can create many bins with unique data
* ❌ **Memory intensive** - May require significant memory for sparse data

Algorithm Details
-----------------

The algorithm works by:

1. **Identifying unique values** in each feature
2. **Creating one bin per unique value**
3. **Mapping each data point** to its corresponding bin
4. **Handling limits** via max_unique_values parameter

**Mapping Function:**
   f(x) = bin_id where bin_id corresponds to unique value x

Parameters
----------

max_unique_values : int, default=50
    Maximum number of unique values allowed per feature.
    If exceeded, raises an error to prevent dimension explosion.

bin_spec : dict, optional
    Dictionary specifying custom bin specifications for specific columns.
    Format: {column_index: list_of_unique_values}

preserve_dataframe : bool, optional
    Whether to preserve input DataFrame format.

fit_jointly : bool, default=False
    Whether to consider feature interactions (not typically used).

Attributes
----------

_bin_edges : dict
    Fitted bin mappings for each column after calling fit().
    For one-hot binning, this contains the unique values found.

_unique_values : dict
    Dictionary mapping column indices to their unique values (internal use).

Examples
--------

Basic Usage with Discrete Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import pandas as pd
   from binning.methods import OneHotBinning
   
   # Create sample discrete data
   np.random.seed(42)
   ratings = np.random.choice([1, 2, 3, 4, 5], size=1000, 
                             p=[0.1, 0.2, 0.4, 0.2, 0.1])
   categories = np.random.choice([10, 20, 30, 40], size=1000)
   
   df = pd.DataFrame({
       'rating': ratings,
       'category': categories
   })
   
   # Apply one-hot binning
   binner = OneHotBinning(max_unique_values=10)
   df_binned = binner.fit_transform(df)
   
   print("Original data:")
   print(df.head(10))
   print("\nBinned data:")
   print(df_binned.head(10))
   
   # Show unique values mapped to bins
   for col_idx, unique_vals in binner._unique_values.items():
       col_name = df.columns[col_idx]
       print(f"\n{col_name} unique values: {sorted(unique_vals)}")
       print(f"Number of bins created: {len(unique_vals)}")

Handling Limited Unique Values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create data with controlled unique values
   np.random.seed(42)
   
   # Discrete score data (common in surveys, ratings)
   survey_scores = np.random.choice(range(1, 8), size=500)  # 1-7 scale
   
   # Product categories (limited set)
   product_ids = np.random.choice([101, 102, 103, 104, 105], size=500)
   
   df = pd.DataFrame({
       'survey_score': survey_scores,
       'product_id': product_ids
   })
   
   # Apply one-hot binning
   binner = OneHotBinning(max_unique_values=20)
   df_binned = binner.fit_transform(df)
   
   # Analyze the binning results
   print("Survey Score Distribution:")
   print("Original:", df['survey_score'].value_counts().sort_index())
   print("Binned:", df_binned['survey_score'].value_counts().sort_index())
   
   print("\nProduct ID Distribution:")  
   print("Original:", df['product_id'].value_counts().sort_index())
   print("Binned:", df_binned['product_id'].value_counts().sort_index())

Comparison with Other Binning Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from binning.methods import OneHotBinning, EqualWidthBinning, EqualFrequencyBinning
   
   # Create mixed data: some discrete, some continuous
   np.random.seed(42)
   discrete_feature = np.random.choice([1, 2, 3, 4, 5], size=1000)
   continuous_feature = np.random.normal(10, 2, 1000)
   
   df = pd.DataFrame({
       'discrete': discrete_feature,
       'continuous': continuous_feature
   })
   
   # Apply different binning methods
   onehot_binner = OneHotBinning(max_unique_values=10)
   equal_width_binner = EqualWidthBinning(n_bins=5)
   equal_freq_binner = EqualFrequencyBinning(n_bins=5)
   
   # Bin only the discrete feature for fair comparison
   df_discrete = df[['discrete']]
   
   df_onehot = onehot_binner.fit_transform(df_discrete)
   df_equal_width = equal_width_binner.fit_transform(df_discrete)
   df_equal_freq = equal_freq_binner.fit_transform(df_discrete)
   
   print("Discrete feature binning comparison:")
   print("\nOne-hot binning (preserves all values):")
   print(df_onehot['discrete'].value_counts().sort_index())
   print("\nEqual-width binning (5 bins):")
   print(df_equal_width['discrete'].value_counts().sort_index())
   print("\nEqual-frequency binning (5 bins):")
   print(df_equal_freq['discrete'].value_counts().sort_index())

Custom Bin Specifications
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create data where you want to control which values get their own bins
   np.random.seed(42)
   
   # Customer segments with some rare categories
   segments = np.concatenate([
       np.random.choice([1, 2, 3], size=800, p=[0.4, 0.4, 0.2]),  # Common
       np.random.choice([4, 5], size=150, p=[0.7, 0.3]),          # Uncommon  
       np.random.choice([6, 7, 8], size=50, p=[0.6, 0.3, 0.1])    # Rare
   ])
   
   df = pd.DataFrame({'segment': segments})
   
   # Default one-hot binning (all values get bins)
   default_binner = OneHotBinning(max_unique_values=10)
   df_default = default_binner.fit_transform(df)
   
   print("Default one-hot binning:")
   print(df_default['segment'].value_counts().sort_index())
   
   # Custom bin specification (group rare categories)
   # Note: This shows the concept, actual implementation may vary
   custom_spec = {0: [1, 2, 3, 4]}  # Only create bins for values 1,2,3,4
   
   try:
       custom_binner = OneHotBinning(max_unique_values=10, bin_spec=custom_spec)
       df_custom = custom_binner.fit_transform(df)
       print("\nCustom bin specification:")
       print(df_custom['segment'].value_counts().sort_index())
   except Exception as e:
       print(f"\nCustom specification not supported in this way: {e}")
       print("Use manual preprocessing for rare category grouping")

Handling Large Cardinality
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Simulate high-cardinality categorical data
   np.random.seed(42)
   
   # User IDs (high cardinality)
   user_ids = np.random.choice(range(1, 1000), size=2000, replace=True)
   
   # Timestamp hours (limited cardinality)
   hours = np.random.choice(range(0, 24), size=2000)
   
   df = pd.DataFrame({
       'user_id': user_ids,
       'hour': hours
   })
   
   print(f"User ID unique values: {df['user_id'].nunique()}")
   print(f"Hour unique values: {df['hour'].nunique()}")
   
   # Try one-hot binning with different limits
   try:
       # This should work for hours
       hour_binner = OneHotBinning(max_unique_values=25)
       df_hours = hour_binner.fit_transform(df[['hour']])
       print(f"\nHour binning successful: {df_hours['hour'].nunique()} bins created")
       
       # This should fail for user_ids (too many unique values)
       user_binner = OneHotBinning(max_unique_values=50)
       df_users = user_binner.fit_transform(df[['user_id']])
       print(f"User binning successful: {df_users['user_id'].nunique()} bins created")
       
   except Exception as e:
       print(f"\nError with high cardinality: {e}")
       print("Consider preprocessing to reduce cardinality or use different binning method")

Pipeline Integration for Categorical Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.compose import ColumnTransformer
   from sklearn.preprocessing import StandardScaler
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split, cross_val_score
   
   # Create mixed dataset with categorical-like and continuous features
   np.random.seed(42)
   n_samples = 1000
   
   # Categorical-like features (perfect for one-hot binning)
   education_level = np.random.choice([1, 2, 3, 4, 5], size=n_samples)
   region = np.random.choice([10, 20, 30, 40], size=n_samples)
   
   # Continuous features
   income = np.random.lognormal(10, 0.5, n_samples)
   age = np.random.normal(40, 15, n_samples)
   
   # Target based on complex interactions
   y = ((education_level >= 3) & (region == 10) | (income > 30000)).astype(int)
   
   df = pd.DataFrame({
       'education': education_level,
       'region': region,
       'income': income,
       'age': age
   })
   
   # Create preprocessing pipeline
   categorical_features = ['education', 'region']
   continuous_features = ['income', 'age']
   
   preprocessor = ColumnTransformer(
       transformers=[
           ('cat', OneHotBinning(max_unique_values=10), categorical_features),
           ('num', EqualWidthBinning(n_bins=5), continuous_features)
       ])
   
   # Full pipeline
   pipeline = Pipeline([
       ('preprocessor', preprocessor),
       ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
   ])
   
   # Evaluate
   scores = cross_val_score(pipeline, df, y, cv=5, scoring='accuracy')
   print(f"Pipeline with one-hot binning: {scores.mean():.3f} ± {scores.std():.3f}")

Feature Engineering with One-Hot Binning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Demonstrate one-hot binning for feature engineering
   np.random.seed(42)
   
   # E-commerce data example
   purchase_data = pd.DataFrame({
       'day_of_week': np.random.choice(range(1, 8), size=1000),  # 1=Monday, 7=Sunday
       'payment_method': np.random.choice([1, 2, 3], size=1000),  # Credit, Debit, PayPal
       'customer_segment': np.random.choice([1, 2, 3, 4], size=1000),  # Bronze, Silver, Gold, Platinum
       'purchase_amount': np.random.lognormal(4, 1, 1000)  # Continuous
   })
   
   # Apply one-hot binning to categorical-like features
   categorical_cols = ['day_of_week', 'payment_method', 'customer_segment']
   
   binner = OneHotBinning(max_unique_values=15)
   categorical_data = purchase_data[categorical_cols]
   categorical_binned = binner.fit_transform(categorical_data)
   
   # Combine with original continuous features
   result_data = categorical_binned.copy()
   result_data['purchase_amount'] = purchase_data['purchase_amount']
   
   print("Feature engineering results:")
   print("\nOriginal categorical distributions:")
   for col in categorical_cols:
       print(f"{col}: {sorted(purchase_data[col].unique())}")
   
   print("\nBinned categorical distributions:")
   for col in categorical_cols:
       print(f"{col}: {sorted(categorical_binned[col].unique())}")
   
   # Show that binning preserved the categorical nature
   for col in categorical_cols:
       original_unique = set(purchase_data[col].unique())
       binned_unique = set(categorical_binned[col].unique())
       print(f"\n{col} - Values preserved: {original_unique == binned_unique}")

When to Use OneHotBinning
-------------------------

**Best For:**

* **Discrete numerical data** - Integer codes, ratings, IDs
* **Low cardinality features** - Limited number of unique values
* **Categorical-like numbers** - Numerical codes representing categories
* **Preserving exact values** - When each unique value is meaningful

**Examples:**

* **Survey responses**: Likert scales (1-5, 1-7)
* **Product categories**: Numerical category codes
* **Geographic codes**: ZIP codes, area codes (if limited)
* **Rating systems**: Star ratings, quality scores
* **Status codes**: Order status, user types

**Avoid When:**

* **High cardinality data** - Too many unique values
* **Continuous data** - True continuous numerical features
* **Memory constraints** - Limited memory for sparse representations  
* **Need dimensionality reduction** - Want fewer features, not more

Performance Considerations
--------------------------

**Computational Complexity:**
   - O(n × f) where n=samples, f=features
   - Very fast, just value mapping

**Memory Usage:**
   - Stores unique values for each feature
   - Can become large with high cardinality

**Scalability:**
   - Excellent for low cardinality features
   - Poor for high cardinality features
   - Linear scaling with data size

Parameter Guidelines
--------------------

**max_unique_values**: Set based on memory and interpretability
   - Small datasets: 10-50
   - Large datasets: 5-20
   - Consider downstream model capacity

**bin_spec**: Use for custom value grouping
   - Group rare categories together
   - Maintain important distinctions
   - Balance granularity vs. sparsity

Common Issues and Solutions
---------------------------

**Issue: Too many unique values**

.. code-block:: python

   # Problem: Feature has hundreds of unique values
   
   # Solution 1: Increase max_unique_values (if manageable)
   binner = OneHotBinning(max_unique_values=100)
   
   # Solution 2: Preprocess to reduce cardinality
   def group_rare_values(series, min_count=10):
       value_counts = series.value_counts()
       common_values = value_counts[value_counts >= min_count].index
       return series.where(series.isin(common_values), other=-1)  # Group rare as -1
   
   df['processed_feature'] = group_rare_values(df['high_cardinality_feature'])

**Issue: Memory usage too high**

.. code-block:: python

   # Problem: One-hot binning creates too many features
   
   # Solution: Use sparse representation or different binning method
   from sklearn.preprocessing import OneHotEncoder
   
   # Alternative: Use sklearn's OneHotEncoder with sparse output
   encoder = OneHotEncoder(sparse=True, handle_unknown='ignore')
   sparse_encoded = encoder.fit_transform(df[['categorical_feature']])

**Issue: New unseen values in transform**

.. code-block:: python

   # Problem: Transform data has values not seen during fit
   
   # Solution: The binning framework should handle this
   # But check the implementation for unknown value handling
   
   # Manual solution: Preprocess to handle unknowns
   def handle_unknown_values(train_series, test_series, unknown_value=-999):
       known_values = set(train_series.unique())
       return test_series.where(test_series.isin(known_values), other=unknown_value)

Comparison with Other Methods
-----------------------------

.. list-table:: Binning Method Comparison
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Method
     - Information Loss
     - Feature Count
     - Memory Usage
     - Best Use Case
   * - OneHotBinning
     - None
     - High (= unique values)
     - High
     - Discrete data
   * - EqualWidthBinning
     - Medium
     - Fixed (n_bins)
     - Low
     - Continuous data
   * - EqualFrequencyBinning
     - Medium
     - Fixed (n_bins)
     - Low
     - Balanced binning
   * - KMeansBinning
     - Medium
     - Fixed (n_bins)  
     - Medium
     - Clustered data

Advanced Usage Patterns
-----------------------

**Custom Value Mapping:**

.. code-block:: python

   class CustomOneHotBinning(OneHotBinning):
       def __init__(self, value_mapping=None, **kwargs):
           super().__init__(**kwargs)
           self.value_mapping = value_mapping or {}
           
       def _transform_column(self, column_data, column_index, **kwargs):
           # Apply custom value mapping before one-hot binning
           if column_index in self.value_mapping:
               mapping = self.value_mapping[column_index]
               column_data = column_data.map(mapping).fillna(column_data)
           
           return super()._transform_column(column_data, column_index, **kwargs)

**Integration with Categorical Encoders:**

.. code-block:: python

   from sklearn.preprocessing import LabelEncoder
   
   # Combine with label encoding for string categories
   def preprocess_mixed_categorical(df, categorical_cols):
       processed_df = df.copy()
       
       for col in categorical_cols:
           if processed_df[col].dtype == 'object':  # String categories
               le = LabelEncoder()
               processed_df[col] = le.fit_transform(processed_df[col])
       
       # Now apply one-hot binning
       binner = OneHotBinning(max_unique_values=20)
       return binner.fit_transform(processed_df[categorical_cols])

See Also
--------

* :class:`EqualWidthBinning` - For continuous data binning
* :class:`EqualFrequencyBinning` - For balanced binning
* :class:`ManualFlexibleBinning` - For custom categorical groupings
* :doc:`../../examples/onehot_binning` - Comprehensive examples

References
----------

* Duda, R. O., Hart, P. E., & Stork, D. G. (2012). Pattern classification. John Wiley & Sons.
* Hastie, T., Tibshirani, R., & Friedman, J. (2009). The elements of statistical learning: data mining, inference, and prediction. Springer Science & Business Media.
* Potharst, R., & Bioch, J. C. (2000). Decision trees for ordinal classification. Intelligent Data Analysis, 4(2), 97-111.
