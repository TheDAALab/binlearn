ManualFlexibleBinning
=====================

.. currentmodule:: binlearn.methods

.. autoclass:: ManualFlexibleBinning
   :members:
   :inherited-members:
   :show-inheritance:

Overview
--------

``ManualFlexibleBinning`` creates bins using explicitly provided bin specifications that can include both 
singleton bins (exact numeric value matches) and interval bins (numeric range matches). This transformer 
offers maximum flexibility for complex binning scenarios that combine exact value matching with traditional 
interval binning.

This approach is ideal for:

* **Numeric data requiring both exact and range matching**
* **Complex domain-specific numeric binning rules**
* **Outlier handling** with specific value bins
* **Standardized binning** with both singleton and continuous elements
* **Integration** with external flexible binning specifications

Key Features
------------

* **Mixed Bin Types**: Combines singleton (exact value) and interval (range) bins
* **Complete Control**: User defines all bin specifications explicitly
* **Numeric Focus**: Designed specifically for numeric data and values
* **Flexible Matching**: Supports exact matches and range-based matching
* **Auto-Representatives**: Automatic generation of appropriate representatives
* **Comprehensive Validation**: Thorough validation of bin specifications
* **Sklearn Compatibility**: Full transformer interface with fit/transform methods
* **DataFrame Support**: Preserves pandas/polars column names and structure

Basic Usage
-----------

.. code-block:: python

   import numpy as np
   import pandas as pd
   from binlearn.methods import ManualFlexibleBinning
   
   # Create sample numeric data
   np.random.seed(42)
   data = pd.DataFrame({
       'score': [95, 85, 75, 65, 45, 25, 85, 95, 12, 88],
       'age': [22, 35, 45, 67, 28, 19, 65, 72, 16, 41]
   })
   
   # Define flexible bin specifications
   flexible_specs = {
       'score': [
           95,           # Singleton bin for perfect scores
           85,           # Singleton bin for high achievers  
           (60, 80),     # Interval bin for passing grades
           (0, 60)       # Interval bin for failing grades
       ],
       'age': [
           (0, 18),      # Minors
           (18, 35),     # Young adults
           (35, 65),     # Middle-aged
           65            # Seniors (singleton for retirement age)
       ]
   }
   
   # Apply flexible binning
   binner = ManualFlexibleBinning(
       bin_spec=flexible_specs,
       preserve_dataframe=True
   )
   
   data_binned = binner.fit_transform(data)
   
   print("Original data:")
   print(data.head())
   print("\\nBinned data:")
   print(data_binned.head())
   print("\\nBin specifications used:")
   for col, specs in flexible_specs.items():
       print(f"  {col}: {specs}")

Grade Analysis Example
----------------------

.. code-block:: python

   # Academic grading with special handling for specific scores
   grades_df = pd.DataFrame({
       'midterm_score': np.random.choice([100, 95, 88, 76, 65, 42, 0], 500, 
                                        p=[0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05]),
       'participation': np.random.uniform(0, 100, 500)
   })
   
   # Academic bin specifications
   academic_specs = {
       'midterm_score': [
           100,          # Perfect score (singleton)
           0,            # Zero score (singleton)
           (90, 100),    # A grade range
           (80, 90),     # B grade range  
           (70, 80),     # C grade range
           (60, 70),     # D grade range
           (0, 60)       # F grade range
       ],
       'participation': [
           100,          # Perfect participation (singleton)
           (80, 100),    # High participation
           (60, 80),     # Moderate participation
           (0, 60)       # Low participation
       ]
   }
   
   # Custom representatives for interpretability
   academic_reps = {
       'midterm_score': ['Perfect', 'Zero', 'A', 'B', 'C', 'D', 'F'],
       'participation': ['Perfect', 'High', 'Moderate', 'Low']
   }
   
   academic_binner = ManualFlexibleBinning(
       bin_spec=academic_specs,
       bin_representatives=academic_reps,
       preserve_dataframe=True
   )
   
   grades_binned = academic_binner.fit_transform(grades_df)
   
   # Analyze grade distribution
   print("Midterm Score Distribution:")
   for i, rep in enumerate(academic_reps['midterm_score']):
       count = (grades_binned['midterm_score'] == i).sum()
       percentage = count / len(grades_df) * 100
       print(f"  {rep}: {count} students ({percentage:.1f}%)")

Financial Risk Assessment
-------------------------

.. code-block:: python

   # Financial data with special handling for extreme values
   financial_df = pd.DataFrame({
       'credit_score': np.random.choice([850, 300] + list(range(400, 800, 20)), 1000),
       'debt_ratio': np.random.exponential(0.3, 1000),
       'years_employment': np.random.choice([0] + list(range(1, 31)), 1000)
   })
   
   # Financial risk bin specifications
   risk_specs = {
       'credit_score': [
           850,          # Perfect credit (singleton)
           300,          # Minimum credit (singleton)
           (740, 850),   # Excellent credit
           (670, 740),   # Good credit
           (580, 670),   # Fair credit
           (300, 580)    # Poor credit
       ],
       'debt_ratio': [
           0.0,          # No debt (singleton)
           (0, 0.28),    # Low debt
           (0.28, 0.36), # Moderate debt
           (0.36, 0.5),  # High debt
           (0.5, 2.0)    # Very high debt
       ],
       'years_employment': [
           0,            # Unemployed (singleton)
           (1, 2),       # New employee
           (2, 5),       # Junior employee
           (5, 10),      # Experienced
           (10, 30)      # Senior employee
       ]
   }
   
   risk_labels = {
       'credit_score': ['Perfect', 'Minimum', 'Excellent', 'Good', 'Fair', 'Poor'],
       'debt_ratio': ['No Debt', 'Low', 'Moderate', 'High', 'Very High'],
       'years_employment': ['Unemployed', 'New', 'Junior', 'Experienced', 'Senior']
   }
   
   risk_binner = ManualFlexibleBinning(
       bin_spec=risk_specs,
       bin_representatives=risk_labels,
       preserve_dataframe=True
   )
   
   financial_binned = risk_binner.fit_transform(financial_df)
   
   # Risk profile analysis
   print("Financial Risk Profile Distribution:")
   for feature in ['credit_score', 'debt_ratio', 'years_employment']:
       print(f"\\n{feature.replace('_', ' ').title()}:")
       for i, label in enumerate(risk_labels[feature]):
           count = (financial_binned[feature] == i).sum()
           print(f"  {label}: {count} ({count/len(financial_df)*100:.1f}%)")

Medical Diagnostic Example
--------------------------

.. code-block:: python

   # Medical data with critical values as singletons
   medical_df = pd.DataFrame({
       'temperature': np.random.normal(98.6, 2, 800),
       'heart_rate': np.random.normal(70, 15, 800),
       'blood_sugar': np.random.lognormal(4.5, 0.3, 800)
   })
   
   # Add some extreme values
   medical_df.loc[:10, 'temperature'] = [105, 95, 106, 94]  # Critical temperatures
   medical_df.loc[:10, 'heart_rate'] = [200, 40, 180, 35]   # Critical heart rates
   
   # Medical bin specifications with critical values
   medical_specs = {
       'temperature': [
           105,          # High fever (singleton)
           95,           # Hypothermia (singleton)
           (100.4, 105), # Fever
           (98, 100.4),  # Normal
           (95, 98),     # Low normal
           (90, 95)      # Hypothermic range
       ],
       'heart_rate': [
           200,          # Tachycardia crisis (singleton)
           40,           # Bradycardia crisis (singleton)
           (100, 200),   # Tachycardia
           (60, 100),    # Normal
           (40, 60),     # Bradycardia
           (20, 40)      # Severe bradycardia
       ],
       'blood_sugar': [
           (70, 100),    # Normal
           (100, 126),   # Pre-diabetic
           (126, 300),   # Diabetic
           (0, 70),      # Hypoglycemic
           (300, 500)    # Severe hyperglycemic
       ]
   }
   
   medical_labels = {
       'temperature': ['High Fever', 'Hypothermia', 'Fever', 'Normal', 'Low Normal', 'Hypothermic'],
       'heart_rate': ['Tachy Crisis', 'Brady Crisis', 'Tachycardia', 'Normal', 'Bradycardia', 'Severe Brady'],
       'blood_sugar': ['Normal', 'Pre-diabetic', 'Diabetic', 'Hypoglycemic', 'Severe Hyperglycemic']
   }
   
   medical_binner = ManualFlexibleBinning(
       bin_spec=medical_specs,
       bin_representatives=medical_labels,
       preserve_dataframe=True
   )
   
   medical_binned = medical_binner.fit_transform(medical_df)

Quality Control Example
-----------------------

.. code-block:: python

   # Manufacturing quality control with specification limits
   qc_df = pd.DataFrame({
       'diameter': np.random.normal(10.0, 0.5, 1000),      # Target: 10.0mm
       'hardness': np.random.normal(50, 5, 1000),          # Target: 50 HRC
       'weight': np.random.normal(100, 3, 1000)            # Target: 100g
   })
   
   # Add some out-of-spec values
   qc_df.loc[:5, 'diameter'] = [12.5, 7.5, 11.0, 9.0]   # Out of tolerance
   
   # Quality control specifications
   qc_specs = {
       'diameter': [
           12.5,         # Upper specification limit (singleton)
           7.5,          # Lower specification limit (singleton)
           (9.8, 10.2),  # Within tolerance
           (9.5, 9.8),   # Low acceptable
           (10.2, 10.5), # High acceptable
           (7.5, 9.5),   # Low reject
           (10.5, 12.5)  # High reject
       ],
       'hardness': [
           (45, 55),     # Target range
           (40, 45),     # Low acceptable
           (55, 60),     # High acceptable
           (0, 40),      # Low reject
           (60, 100)     # High reject
       ],
       'weight': [
           (98, 102),    # Target range
           (95, 98),     # Light
           (102, 105),   # Heavy
           (0, 95),      # Too light
           (105, 200)    # Too heavy
       ]
   }
   
   qc_labels = {
       'diameter': ['Upper Limit', 'Lower Limit', 'Target', 'Low OK', 'High OK', 'Low Reject', 'High Reject'],
       'hardness': ['Target', 'Low OK', 'High OK', 'Low Reject', 'High Reject'],
       'weight': ['Target', 'Light', 'Heavy', 'Too Light', 'Too Heavy']
   }
   
   qc_binner = ManualFlexibleBinning(
       bin_spec=qc_specs,
       bin_representatives=qc_labels,
       preserve_dataframe=True
   )
   
   qc_binned = qc_binner.fit_transform(qc_df)
   
   # Quality analysis
   print("Quality Control Analysis:")
   for feature in ['diameter', 'hardness', 'weight']:
       print(f"\\n{feature.title()}:")
       for i, label in enumerate(qc_labels[feature]):
           count = (qc_binned[feature] == i).sum()
           print(f"  {label}: {count} units ({count/len(qc_df)*100:.1f}%)")

Bin Specification Guide
-----------------------

.. code-block:: python

   # Examples of different bin specification formats
   
   specification_examples = {
       # Example 1: Mixed singleton and interval bins
       'feature1': [
           42,           # Singleton: exact match for value 42
           (0, 25),      # Interval: values in range [0, 25)
           (25, 50),     # Interval: values in range [25, 50)
           100           # Singleton: exact match for value 100
       ],
       
       # Example 2: Mostly intervals with key singletons
       'feature2': [
           0,            # Singleton: zero values
           (0, 10),      # Interval: low values
           (10, 90),     # Interval: normal range
           (90, 100),    # Interval: high values
           100           # Singleton: maximum values
       ],
       
       # Example 3: Mostly singletons (discrete-like)
       'feature3': [
           1, 2, 3, 4, 5,     # Individual values
           (6, 10),           # Range for higher values
           (10, float('inf')) # Open upper range
       ]
   }
   
   # Demonstration of matching behavior
   test_data = pd.DataFrame({
       'feature1': [42, 15, 35, 100, 75],
       'feature2': [0, 5, 45, 95, 100],
       'feature3': [1, 3, 7, 15, 25]
   })
   
   demo_binner = ManualFlexibleBinning(
       bin_spec=specification_examples,
       preserve_dataframe=True
   )
   
   result = demo_binner.fit_transform(test_data)
   
   print("Bin matching demonstration:")
   for col in test_data.columns:
       print(f"\\n{col}:")
       print(f"  Original: {test_data[col].tolist()}")
       print(f"  Binned:   {result[col].tolist()}")
       print(f"  Specs:    {specification_examples[col]}")

Scikit-learn Pipeline Integration
---------------------------------

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split
   from sklearn.datasets import make_classification
   
   # Create sample data
   X, y = make_classification(n_samples=1000, n_features=3, n_classes=2, random_state=42)
   
   # Define flexible binning for each feature
   pipeline_specs = {
       0: [
           -2.5,         # Extreme low (singleton)
           (-2, -1),     # Low range
           (-1, 1),      # Medium range
           (1, 2),       # High range
           2.5           # Extreme high (singleton)
       ],
       1: [
           (-3, -1),     # Low
           (-1, 1),      # Medium
           (1, 3),       # High
           3.5           # Extreme (singleton)
       ],
       2: [
           -2.0,         # Extreme low (singleton)
           (-1.5, 0),    # Low-medium
           (0, 1.5),     # Medium-high
           2.0           # Extreme high (singleton)
       ]
   }
   
   # Create pipeline
   pipeline = Pipeline([
       ('binning', ManualFlexibleBinning(bin_spec=pipeline_specs)),
       ('classifier', RandomForestClassifier(random_state=42))
   ])
   
   # Train and evaluate
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   pipeline.fit(X_train, y_train)
   accuracy = pipeline.score(X_test, y_test)
   
   print(f"Pipeline accuracy with flexible binning: {accuracy:.3f}")

Parameter Guide
---------------

**bin_spec** (dict, required)
    Dictionary mapping column identifiers to flexible bin specification lists:
    
    * Keys: Column names (str) or indices (int)
    * Values: Lists containing:
      
      * **Singleton bins**: Numeric values for exact matches
      * **Interval bins**: Tuples (min, max) for range matches
    
    * Order matters: earlier specifications take precedence
    * No overlap validation (user responsibility)

**bin_representatives** (dict, optional)
    Dictionary mapping columns to bin representative values:
    
    * Keys: Must match bin_spec keys  
    * Values: Lists with same length as corresponding bin_spec
    * Can be numeric values or category names/labels
    * If None, auto-generates appropriate representatives

Tips for Best Results
---------------------

1. **Order specifications carefully**: Earlier bins take precedence in matching
2. **Avoid overlapping intervals**: Can lead to ambiguous matches
3. **Use singletons for critical values**: Exact matches for important thresholds
4. **Consider floating point precision**: Use appropriate precision for your data
5. **Test with representative data**: Validate that all expected values match correctly
6. **Document specification logic**: Keep records of binning rationale

Common Patterns
---------------

* **Outlier Isolation**: Use singletons for extreme values, intervals for normal ranges
* **Threshold Systems**: Combine critical value singletons with range intervals
* **Quality Control**: Specification limits as singletons, tolerance ranges as intervals
* **Grade Systems**: Perfect scores as singletons, grade ranges as intervals
* **Medical Diagnostics**: Critical values as singletons, normal ranges as intervals

See Also
--------

* :class:`ManualIntervalBinning` - Manual binning with only interval bins
* :class:`SingletonBinning` - Automatic binning with only singleton bins  
* :class:`EqualWidthBinning` - Automatic equal-width interval binning
* :class:`SupervisedBinning` - Automatic supervised binning with decision trees
