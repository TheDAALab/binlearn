Manual Binning Methods
======================

Manual Binning
==============

.. currentmodule:: binlearn.methods

This section documents the manual binning methods that allow users to specify custom bin boundaries and groupings.

ManualIntervalBinning
---------------------

.. autoclass:: ManualIntervalBinning
   :members:
   :inherited-members:
   :show-inheritance:

**Manual interval binning** allows users to specify exact bin boundaries for creating intervals with custom breakpoints.

Overview
^^^^^^^^

ManualIntervalBinning provides complete control over bin boundaries by accepting user-defined edge points. This is ideal when domain knowledge suggests specific meaningful breakpoints.

**Key Characteristics:**

* ✅ **Complete control** - User defines all bin boundaries
* ✅ **Domain knowledge** - Incorporates business rules and expertise
* ✅ **Interpretable bins** - Meaningful intervals (e.g., age groups, income brackets)
* ✅ **Consistent binning** - Same boundaries across different datasets
* ❌ **Manual specification** - Requires domain expertise
* ❌ **Static boundaries** - Doesn't adapt to data distribution

Algorithm Details
^^^^^^^^^^^^^^^^^

The algorithm works by:

1. **Accepting user-defined boundaries** for each feature
2. **Creating intervals** between consecutive boundary points
3. **Assigning data points** to appropriate intervals
4. **Handling edge cases** for values outside specified ranges

Parameters
^^^^^^^^^^

bin_edges : dict
    Dictionary mapping column indices to arrays of bin edges.
    Format: {column_index: array_of_edges}

preserve_dataframe : bool, optional
    Whether to preserve input DataFrame format.

fit_jointly : bool, default=False
    Whether to consider feature interactions (not typically used).

Examples
^^^^^^^^

Age Group Binning
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import pandas as pd
   from binlearn.methods import ManualIntervalBinning
   
   # Create sample age data
   np.random.seed(42)
   ages = np.random.normal(35, 15, 1000)
   ages = np.clip(ages, 18, 80)  # Keep realistic ages
   
   df = pd.DataFrame({'age': ages})
   
   # Define meaningful age groups
   age_boundaries = [18, 25, 35, 50, 65, 80]
   
   # Apply manual interval binning
   binner = ManualIntervalBinning(bin_edges={0: age_boundaries})
   df_binned = binner.fit_transform(df)
   
   print("Age group distribution:")
   bin_counts = df_binned['age'].value_counts().sort_index()
   
   # Create meaningful labels for interpretation
   age_labels = ['18-25', '25-35', '35-50', '50-65', '65+']
   for i, count in enumerate(bin_counts):
       print(f"{age_labels[i]}: {count} people ({count/len(df)*100:.1f}%)")

Income Bracket Analysis
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create sample income data (log-normal distribution)
   np.random.seed(42)
   incomes = np.random.lognormal(mean=10.5, sigma=0.8, size=1000)
   
   df = pd.DataFrame({'income': incomes})
   
   # Define income brackets based on tax brackets or business rules
   income_boundaries = [0, 25000, 50000, 75000, 100000, 150000, np.inf]
   
   binner = ManualIntervalBinning(bin_edges={0: income_boundaries})
   df_binned = binner.fit_transform(df)
   
   print("Income bracket analysis:")
   bin_counts = df_binned['income'].value_counts().sort_index()
   
   income_labels = ['<$25K', '$25K-$50K', '$50K-$75K', 
                   '$75K-$100K', '$100K-$150K', '>$150K']
   
   for i, count in enumerate(bin_counts):
       avg_income = df[df_binned['income'] == i]['income'].mean()
       print(f"{income_labels[i]}: {count} people (avg: ${avg_income:,.0f})")

Multi-Feature Custom Binning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create business data with multiple features requiring custom binning
   np.random.seed(42)
   
   # Customer data
   customer_ages = np.random.normal(40, 15, 1000)
   customer_ages = np.clip(customer_ages, 18, 80)
   
   # Purchase amounts (different scale and meaning)
   purchase_amounts = np.random.lognormal(6, 1, 1000)
   
   # Store ratings
   ratings = np.random.beta(4, 1, 1000) * 5  # Skewed toward higher ratings
   
   df = pd.DataFrame({
       'age': customer_ages,
       'purchase_amount': purchase_amounts,
       'rating': ratings
   })
   
   # Define custom boundaries for each feature
   custom_boundaries = {
       0: [18, 30, 45, 60, 80],              # Age groups
       1: [0, 100, 500, 1000, 2000, np.inf], # Purchase tiers
       2: [0, 2, 3, 4, 5]                    # Rating categories
   }
   
   binner = ManualIntervalBinning(bin_edges=custom_boundaries)
   df_binned = binner.fit_transform(df)
   
   # Analyze each feature
   feature_names = ['age', 'purchase_amount', 'rating']
   labels = [
       ['18-30', '30-45', '45-60', '60+'],
       ['<$100', '$100-$500', '$500-$1K', '$1K-$2K', '>$2K'],
       ['Poor (0-2)', 'Fair (2-3)', 'Good (3-4)', 'Excellent (4-5)']
   ]
   
   for i, feature in enumerate(feature_names):
       print(f"\n{feature.title()} Distribution:")
       bin_counts = df_binned[feature].value_counts().sort_index()
       for j, count in enumerate(bin_counts):
           print(f"  {labels[i][j]}: {count} ({count/len(df)*100:.1f}%)")

ManualFlexibleBinning
---------------------

.. autoclass:: ManualFlexibleBinning
   :members:
   :inherited-members:
   :show-inheritance:

**Manual flexible binning** allows users to specify custom groupings of values, particularly useful for categorical data or when specific values should be grouped together.

Overview
^^^^^^^^

ManualFlexibleBinning enables custom value-to-bin mappings where users can group arbitrary values into named bins. This is perfect for categorical data or when business logic requires specific value groupings.

**Key Characteristics:**

* ✅ **Flexible grouping** - Group any values together
* ✅ **Named bins** - Meaningful bin names/labels
* ✅ **Categorical handling** - Perfect for categorical data
* ✅ **Business logic** - Implements complex grouping rules
* ❌ **Manual specification** - Requires complete value mapping
* ❌ **Maintenance** - Must update when new values appear

Algorithm Details
^^^^^^^^^^^^^^^^^

The algorithm works by:

1. **Accepting value-to-bin mappings** from users
2. **Creating lookup tables** for efficient mapping
3. **Transforming values** according to specified groupings
4. **Handling unmapped values** (typically as separate category)

Parameters
^^^^^^^^^^

bin_spec : dict
    Dictionary mapping column indices to bin specifications.
    Format: {column_index: {value: bin_id, ...}} or 
    {column_index: [(bin_id, [values]), ...]}

preserve_dataframe : bool, optional
    Whether to preserve input DataFrame format.

fit_jointly : bool, default=False
    Whether to consider feature interactions (not typically used).

Examples
^^^^^^^^

Product Category Grouping
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import pandas as pd
   from binlearn.methods import ManualFlexibleBinning
   
   # Create sample product data
   np.random.seed(42)
   
   # Product categories (numeric codes)
   product_codes = np.random.choice([101, 102, 103, 201, 202, 301, 302, 303, 401], 
                                  size=1000, 
                                  p=[0.15, 0.1, 0.05, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05])
   
   df = pd.DataFrame({'product_category': product_codes})
   
   # Define business-meaningful groupings
   category_mapping = {
       0: {  # Column 0 (product_category)
           101: 'Electronics', 102: 'Electronics', 103: 'Electronics',
           201: 'Clothing', 202: 'Clothing',
           301: 'Home', 302: 'Home', 303: 'Home',
           401: 'Sports'
       }
   }
   
   binner = ManualFlexibleBinning(bin_spec=category_mapping)
   df_binned = binner.fit_transform(df)
   
   print("Product category grouping:")
   print("Original codes:", sorted(df['product_category'].unique()))
   print("Grouped categories:", sorted(df_binned['product_category'].unique()))
   
   print("\nDistribution by group:")
   print(df_binned['product_category'].value_counts())

Geographic Region Mapping
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create sample geographic data
   np.random.seed(42)
   
   # ZIP codes (first 3 digits representing regions)
   zip_prefixes = np.random.choice([100, 101, 102, 200, 201, 300, 301, 400, 401, 500], 
                                 size=1000)
   
   df = pd.DataFrame({'zip_prefix': zip_prefixes})
   
   # Group ZIP prefixes into regions
   region_mapping = {
       0: {  # Column 0 (zip_prefix)
           100: 'Northeast', 101: 'Northeast', 102: 'Northeast',
           200: 'Southeast', 201: 'Southeast',
           300: 'Midwest', 301: 'Midwest', 
           400: 'Southwest', 401: 'Southwest',
           500: 'West'
       }
   }
   
   binner = ManualFlexibleBinning(bin_spec=region_mapping)
   df_binned = binner.fit_transform(df)
   
   print("Geographic region mapping:")
   region_counts = df_binned['zip_prefix'].value_counts()
   
   for region in ['Northeast', 'Southeast', 'Midwest', 'Southwest', 'West']:
       count = region_counts.get(region, 0)
       print(f"{region}: {count} customers ({count/len(df)*100:.1f}%)")

Customer Segmentation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create customer behavior data
   np.random.seed(42)
   
   # Customer types based on purchase behavior
   customer_scores = np.random.choice(range(1, 101), size=1000)
   purchase_frequency = np.random.choice(['Low', 'Medium', 'High'], size=1000,
                                       p=[0.4, 0.4, 0.2])
   
   df = pd.DataFrame({
       'customer_score': customer_scores,
       'purchase_frequency': purchase_frequency
   })
   
   # Define customer segments based on score ranges and frequency
   score_segments = {
       0: {}  # Will populate based on score ranges
   }
   
   # Create score-based segments
   for score in range(1, 101):
       if score <= 30:
           score_segments[0][score] = 'Bronze'
       elif score <= 60:
           score_segments[0][score] = 'Silver'  
       elif score <= 85:
           score_segments[0][score] = 'Gold'
       else:
           score_segments[0][score] = 'Platinum'
   
   # Frequency mapping
   frequency_mapping = {
       1: {'Low': 'Occasional', 'Medium': 'Regular', 'High': 'Frequent'}
   }
   
   # Combine both mappings
   combined_mapping = {**score_segments, **frequency_mapping}
   
   binner = ManualFlexibleBinning(bin_spec=combined_mapping)
   df_binned = binner.fit_transform(df)
   
   print("Customer segmentation results:")
   print("\nScore-based segments:")
   print(df_binned['customer_score'].value_counts())
   
   print("\nFrequency-based segments:")
   print(df_binned['purchase_frequency'].value_counts())
   
   # Cross-tabulation for insights
   crosstab = pd.crosstab(df_binned['customer_score'], 
                         df_binned['purchase_frequency'])
   print("\nCross-tabulation (Score vs Frequency):")
   print(crosstab)

Advanced Value Grouping
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Demonstrate complex grouping scenarios
   np.random.seed(42)
   
   # Mixed categorical data
   departments = np.random.choice(['HR', 'IT', 'Sales', 'Marketing', 'Finance', 
                                 'Operations', 'R&D', 'Legal'], size=500)
   
   experience_years = np.random.choice(range(0, 31), size=500)
   
   df = pd.DataFrame({
       'department': departments,
       'experience': experience_years
   })
   
   # Group departments by function
   dept_mapping = {
       0: {  # Column 0 (department) - using string values
           'HR': 'Support', 'Legal': 'Support', 'Finance': 'Support',
           'IT': 'Technical', 'R&D': 'Technical', 'Operations': 'Technical',
           'Sales': 'Revenue', 'Marketing': 'Revenue'
       }
   }
   
   # Group experience into career stages
   exp_mapping = {1: {}}  # Column 1 (experience)
   for year in range(31):
       if year < 2:
           exp_mapping[1][year] = 'Entry'
       elif year < 7:
           exp_mapping[1][year] = 'Mid-Level'
       elif year < 15:
           exp_mapping[1][year] = 'Senior'
       else:
           exp_mapping[1][year] = 'Executive'
   
   combined_mapping = {**dept_mapping, **exp_mapping}
   
   # Note: This example shows the concept, but string keys might need 
   # preprocessing with LabelEncoder first
   try:
       binner = ManualFlexibleBinning(bin_spec=combined_mapping)
       df_binned = binner.fit_transform(df)
       
       print("Department functional grouping:")
       print(df_binned['department'].value_counts())
       
       print("\nExperience level grouping:")
       print(df_binned['experience'].value_counts())
       
   except Exception as e:
       print(f"Note: String categorical data may need preprocessing: {e}")
       print("Consider using LabelEncoder first for string categories")

When to Use Manual Binning Methods
----------------------------------

**ManualIntervalBinning - Best For:**

* **Domain-specific breakpoints** - Age groups, income brackets, test scores
* **Regulatory compliance** - Tax brackets, risk categories
* **Business rules** - Customer tiers, service levels
* **Interpretable analysis** - Meaningful intervals for reporting

**ManualFlexibleBinning - Best For:**

* **Categorical grouping** - Product categories, geographic regions
* **Business logic** - Custom grouping rules
* **Data consolidation** - Reducing high cardinality
* **Semantic grouping** - Meaningful category combinations

**Examples:**

* **Healthcare**: Age groups, BMI categories, risk levels
* **Finance**: Income brackets, credit score ranges, risk tiers
* **Retail**: Product categories, customer segments, price ranges
* **Education**: Grade levels, score brackets, performance tiers

**Avoid When:**

* **No domain knowledge** - Automatic methods may be better
* **Exploratory phase** - Data-driven methods help discover patterns
* **Frequently changing data** - Static boundaries may become outdated
* **High-dimensional data** - Manual specification becomes impractical

Performance Considerations
--------------------------

**Computational Complexity:**
   - O(n × f) where n=samples, f=features
   - Very fast, just lookup operations

**Memory Usage:**
   - Stores mapping dictionaries for each feature
   - Minimal memory footprint

**Scalability:**
   - Excellent scalability with data size
   - Constant time per data point
   - Linear with number of features

Common Issues and Solutions
---------------------------

**Issue: Unmapped values in new data**

.. code-block:: python

   # Problem: Transform data contains values not in bin_spec
   
   # Solution: Handle unknown values explicitly
   def create_robust_mapping(known_values, mapping, default_bin='Other'):
       robust_mapping = mapping.copy()
       for value in known_values:
           if value not in robust_mapping:
               robust_mapping[value] = default_bin
       return robust_mapping

**Issue: Inconsistent bin definitions across features**

.. code-block:: python

   # Problem: Different mapping styles cause confusion
   
   # Solution: Use consistent mapping format
   standard_mapping = {
       0: {val: bin_id for val, bin_id in mapping_pairs_0},
       1: {val: bin_id for val, bin_id in mapping_pairs_1}
   }

**Issue: String categorical data**

.. code-block:: python

   # Problem: Framework expects numeric data
   
   # Solution: Preprocess strings to numeric codes
   from sklearn.preprocessing import LabelEncoder
   
   # Encode strings first
   le = LabelEncoder()
   df['category_encoded'] = le.fit_transform(df['category_string'])
   
   # Then apply flexible binning
   binner = ManualFlexibleBinning(bin_spec=numeric_mapping)

Comparison Between Manual Methods
---------------------------------

.. list-table:: Manual Binning Method Comparison
   :header-rows: 1
   :widths: 25 25 25 25

   * - Aspect
     - ManualIntervalBinning
     - ManualFlexibleBinning
     - Best Use
   * - Input Type
     - Continuous/Ordinal
     - Categorical/Discrete
     - Data dependent
   * - Boundary Type
     - Interval edges
     - Value mappings
     - Problem dependent
   * - Complexity
     - Low
     - Medium
     - Simplicity vs flexibility
   * - Maintenance
     - Low
     - High
     - Stability requirements

Advanced Usage Patterns
-----------------------

**Combining Both Methods:**

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.compose import ColumnTransformer
   
   # Different binning for different feature types
   preprocessor = ColumnTransformer(
       transformers=[
           ('interval', ManualIntervalBinning(bin_edges={0: age_edges}), ['age']),
           ('flexible', ManualFlexibleBinning(bin_spec=category_mapping), ['category'])
       ])
   
   pipeline = Pipeline([
       ('binning', preprocessor),
       ('classifier', RandomForestClassifier())
   ])

**Dynamic Bin Creation:**

.. code-block:: python

   def create_quantile_based_manual_bins(data, n_bins=5):
       """Create manual bin edges based on data quantiles"""
       quantiles = np.linspace(0, 1, n_bins + 1)
       edges = np.quantile(data, quantiles)
       return edges
   
   # Use with ManualIntervalBinning for data-driven but fixed boundaries
   dynamic_edges = create_quantile_based_manual_bins(df['feature'])
   binner = ManualIntervalBinning(bin_edges={0: dynamic_edges})

See Also
--------

* :class:`EqualWidthBinning` - For automatic interval binning
* :class:`EqualFrequencyBinning` - For balanced automatic binning
* :class:`OneHotBinning` - For unique value binning
* :doc:`../../examples/manual_binning` - Comprehensive examples

References
----------

* Dougherty, J., Kohavi, R., & Sahami, M. (1995). Supervised and unsupervised discretization of continuous-valued attributes for classification learning.
* Kerber, R. (1992). ChiMerge: Discretization of numeric attributes. In Proceedings of the tenth national conference on Artificial intelligence (pp. 123-128).
* Liu, H., Hussain, F., Tan, C. L., & Dash, M. (2002). Discretization: An enabling technique. Data mining and knowledge discovery, 6(4), 393-423.
