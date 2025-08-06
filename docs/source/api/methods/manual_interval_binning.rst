ManualIntervalBinning
====================

.. currentmodule:: binlearn.methods

.. autoclass:: ManualIntervalBinning
   :members:
   :inherited-members:
   :show-inheritance:

Overview
--------

``ManualIntervalBinning`` creates bins using explicitly provided bin edges, giving users complete 
control over binning boundaries. Unlike automatic binning methods, this transformer never infers 
bin edges from data - they must always be provided by the user.

This approach is ideal for:

* **Standardized binning** across multiple datasets
* **Domain-specific binning** requirements with business rules
* **Reproducible binning** with known boundaries
* **Integration** with external binning specifications
* **Regulatory compliance** where specific bins are mandated

Key Features
------------

* **Complete Control**: User defines all bin boundaries explicitly
* **Consistency**: Same bins across different datasets and time periods
* **Validation**: Comprehensive validation of user-provided bin edges
* **Auto-Representatives**: Automatic generation of bin center representatives
* **Flexible Keys**: Supports both column names and indices as keys
* **Out-of-Range Handling**: Configurable clipping for values outside bin ranges
* **Sklearn Compatibility**: Full transformer interface with fit/transform methods
* **DataFrame Support**: Preserves pandas/polars column names and structure

Basic Usage
-----------

.. code-block:: python

   import numpy as np
   import pandas as pd
   from binlearn.methods import ManualIntervalBinning
   
   # Create sample data
   np.random.seed(42)
   X = np.random.uniform(0, 100, 200).reshape(-1, 2)
   
   # Define custom bin edges for each feature
   custom_edges = {
       0: [0, 20, 40, 60, 80, 100],      # Feature 0: quintiles
       1: [0, 25, 50, 75, 100]           # Feature 1: quartiles
   }
   
   # Apply manual binning
   binner = ManualIntervalBinning(bin_edges=custom_edges)
   X_binned = binner.fit_transform(X)
   
   print(f"Original shape: {X.shape}")
   print(f"Binned shape: {X_binned.shape}")
   print(f"Bin edges for feature 0: {binner.bin_edges_[0]}")
   print(f"Bin edges for feature 1: {binner.bin_edges_[1]}")
   print(f"Representatives for feature 0: {binner.bin_representatives_[0]}")

DataFrame Example with Named Columns
------------------------------------

.. code-block:: python

   # Create DataFrame with named columns
   df = pd.DataFrame({
       'age': np.random.uniform(18, 80, 1000),
       'income': np.random.uniform(20000, 200000, 1000),
       'credit_score': np.random.uniform(300, 850, 1000)
   })
   
   # Define business-relevant bin edges
   business_edges = {
       'age': [18, 25, 35, 50, 65, 80],          # Life stages
       'income': [0, 30000, 60000, 100000, 200000],  # Income brackets
       'credit_score': [300, 580, 670, 740, 850]     # Credit categories
   }
   
   # Optional: Define custom representatives
   representatives = {
       'age': [21, 30, 42, 57, 72],              # Midpoint ages
       'income': [15000, 45000, 80000, 150000],  # Representative incomes
       'credit_score': [440, 625, 705, 795]      # Representative scores
   }
   
   binner = ManualIntervalBinning(
       bin_edges=business_edges,
       bin_representatives=representatives,
       preserve_dataframe=True,
       clip=True  # Clip outliers to bin boundaries
   )
   
   df_binned = binner.fit_transform(df)
   
   print("Age bins:")
   for i, (start, end) in enumerate(zip(business_edges['age'][:-1], business_edges['age'][1:])):
       count = ((df['age'] >= start) & (df['age'] < end)).sum()
       print(f"  Bin {i}: [{start}, {end}) - {count} samples")

Financial Risk Example
----------------------

.. code-block:: python

   # Financial data with regulatory-defined risk categories
   financial_df = pd.DataFrame({
       'debt_to_income': np.random.uniform(0, 1.5, 5000),
       'loan_to_value': np.random.uniform(0.3, 1.2, 5000),
       'fico_score': np.random.uniform(300, 850, 5000)
   })
   
   # Regulatory risk categories (example)
   risk_edges = {
       'debt_to_income': [0, 0.28, 0.36, 0.43, 1.5],     # DTI risk categories
       'loan_to_value': [0, 0.8, 0.9, 0.95, 1.2],        # LTV risk categories
       'fico_score': [300, 580, 620, 680, 740, 850]       # Credit score tiers
   }
   
   # Risk level names as representatives
   risk_representatives = {
       'debt_to_income': ['Low', 'Moderate', 'High', 'Very High'],
       'loan_to_value': ['Conservative', 'Standard', 'Aggressive', 'High Risk'],
       'fico_score': ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
   }
   
   risk_binner = ManualIntervalBinning(
       bin_edges=risk_edges,
       bin_representatives=risk_representatives,
       preserve_dataframe=True,
       clip=True
   )
   
   financial_binned = risk_binner.fit_transform(financial_df)
   
   # Show distribution across risk categories
   for col in ['debt_to_income', 'loan_to_value', 'fico_score']:
       print(f"\\n{col.replace('_', ' ').title()} Distribution:")
       for i, rep in enumerate(risk_representatives[col]):
           mask = financial_binned[col] == i
           count = mask.sum()
           percentage = count / len(financial_df) * 100
           print(f"  {rep}: {count} ({percentage:.1f}%)")

Medical/Clinical Example
------------------------

.. code-block:: python

   # Medical data with clinical thresholds
   medical_df = pd.DataFrame({
       'bmi': np.random.normal(25, 5, 2000),
       'blood_pressure_systolic': np.random.normal(120, 20, 2000),
       'cholesterol': np.random.normal(200, 40, 2000),
       'age': np.random.uniform(18, 90, 2000)
   })
   
   # Clinical classification thresholds
   clinical_edges = {
       'bmi': [0, 18.5, 25, 30, 40],                    # BMI categories
       'blood_pressure_systolic': [0, 120, 130, 140, 180, 300],  # BP stages
       'cholesterol': [0, 200, 240, 300],               # Cholesterol levels
       'age': [0, 18, 40, 65, 100]                      # Age groups
   }
   
   clinical_labels = {
       'bmi': ['Underweight', 'Normal', 'Overweight', 'Obese'],
       'blood_pressure_systolic': ['Normal', 'Elevated', 'Stage 1', 'Stage 2', 'Crisis'],
       'cholesterol': ['Desirable', 'Borderline', 'High'],
       'age': ['Child', 'Adult', 'Middle Age', 'Senior']
   }
   
   clinical_binner = ManualIntervalBinning(
       bin_edges=clinical_edges,
       bin_representatives=clinical_labels,
       preserve_dataframe=True,
       clip=True
   )
   
   medical_binned = clinical_binner.fit_transform(medical_df)
   
   # Clinical summary
   print("Clinical Distribution Summary:")
   for condition in ['bmi', 'blood_pressure_systolic', 'cholesterol']:
       print(f"\\n{condition.replace('_', ' ').title()}:")
       for i, label in enumerate(clinical_labels[condition]):
           count = (medical_binned[condition] == i).sum()
           print(f"  {label}: {count} patients ({count/len(medical_df)*100:.1f}%)")

Cross-Dataset Consistency
-------------------------

.. code-block:: python

   # Ensure consistent binning across training and test sets
   
   # Training data
   train_data = pd.DataFrame({
       'feature1': np.random.normal(50, 15, 1000),
       'feature2': np.random.exponential(2, 1000)
   })
   
   # Test data (different distribution)
   test_data = pd.DataFrame({
       'feature1': np.random.normal(45, 20, 500),  # Different mean/std
       'feature2': np.random.exponential(3, 500)   # Different scale
   })
   
   # Fixed bin edges ensure consistency
   standard_edges = {
       'feature1': [0, 25, 40, 55, 70, 100],
       'feature2': [0, 1, 3, 6, 10, 20]
   }
   
   binner = ManualIntervalBinning(
       bin_edges=standard_edges,
       preserve_dataframe=True,
       clip=True
   )
   
   # Same binning applied to both datasets
   train_binned = binner.fit_transform(train_data)
   test_binned = binner.transform(test_data)  # No fitting needed
   
   print("Training data distribution:")
   print(train_binned['feature1'].value_counts().sort_index())
   print("\\nTest data distribution:")
   print(test_binned['feature1'].value_counts().sort_index())

Advanced Bin Edge Validation
-----------------------------

.. code-block:: python

   # Example of comprehensive bin edge validation
   
   def validate_custom_edges(edges_dict, data_ranges):
       \"\"\"Validate that bin edges cover expected data ranges.\"\"\"
       for col, edges in edges_dict.items():
           if col in data_ranges:
               data_min, data_max = data_ranges[col]
               edge_min, edge_max = min(edges), max(edges)
               
               if edge_min > data_min:
                   print(f"Warning: {col} edges start at {edge_min}, data starts at {data_min}")
               if edge_max < data_max:
                   print(f"Warning: {col} edges end at {edge_max}, data ends at {data_max}")
               
               # Check for reasonable bin sizes
               bin_widths = np.diff(edges)
               if max(bin_widths) / min(bin_widths) > 10:
                   print(f"Warning: {col} has very uneven bin sizes")
   
   # Usage example
   data_ranges = {
       'age': (df['age'].min(), df['age'].max()),
       'income': (df['income'].min(), df['income'].max())
   }
   
   validate_custom_edges(business_edges, data_ranges)

Scikit-learn Pipeline Integration
---------------------------------

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split
   from sklearn.datasets import make_classification
   
   # Create classification data
   X, y = make_classification(n_samples=2000, n_features=3, n_classes=2, random_state=42)
   
   # Define standardized bin edges for each feature
   pipeline_edges = {
       0: [-3, -1, 0, 1, 3],
       1: [-3, -1.5, 0, 1.5, 3],
       2: [-3, -1, 1, 3]
   }
   
   # Create pipeline with manual binning
   pipeline = Pipeline([
       ('binning', ManualIntervalBinning(
           bin_edges=pipeline_edges,
           clip=True
       )),
       ('classifier', RandomForestClassifier(random_state=42))
   ])
   
   # Train and evaluate
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   pipeline.fit(X_train, y_train)
   accuracy = pipeline.score(X_test, y_test)
   
   print(f"Pipeline accuracy with manual binning: {accuracy:.3f}")

Parameter Guide
---------------

**bin_edges** (dict, required)
    Dictionary mapping column identifiers to bin edge lists:
    
    * Keys: Column names (str) or indices (int)
    * Values: Sorted lists/arrays of bin boundaries
    * Must have at least 2 elements per list
    * Will create len(edges)-1 bins

**bin_representatives** (dict, optional)
    Dictionary mapping columns to bin representative values:
    
    * Keys: Must match bin_edges keys
    * Values: Lists with len(edges)-1 elements
    * If None, uses bin centers as representatives
    * Can be numeric values or category names

**clip** (bool, optional)
    Whether to clip out-of-range values:
    
    * True: Clip to nearest bin edge
    * False: Assign special out-of-range indicators
    * None: Use global configuration default

Edge Case Handling
-------------------

.. code-block:: python

   # Handling data outside bin ranges
   
   # Data with outliers
   outlier_data = pd.DataFrame({
       'normal_feature': np.concatenate([
           np.random.normal(50, 10, 900),  # Normal data
           [5, 95, 120, -10]               # Outliers
       ])
   })
   
   normal_edges = {'normal_feature': [20, 40, 60, 80]}
   
   # With clipping
   clipper = ManualIntervalBinning(bin_edges=normal_edges, clip=True)
   clipped_result = clipper.fit_transform(outlier_data)
   
   # Without clipping (outliers get special values)
   no_clipper = ManualIntervalBinning(bin_edges=normal_edges, clip=False)
   unclipped_result = no_clipper.fit_transform(outlier_data)
   
   print("With clipping - unique values:", np.unique(clipped_result))
   print("Without clipping - unique values:", np.unique(unclipped_result))

Tips for Best Results
---------------------

1. **Validate edge coverage**: Ensure edges cover your expected data range
2. **Consider domain knowledge**: Use meaningful boundaries from your field
3. **Check bin balance**: Avoid bins that are too small or too large
4. **Plan for outliers**: Decide on clipping strategy early
5. **Document edge rationale**: Keep records of why specific edges were chosen
6. **Test across datasets**: Validate that edges work across different data samples

Common Use Cases
----------------

* **Age Groups**: [18, 25, 35, 50, 65, 80] for life stage analysis
* **Income Brackets**: [0, 25000, 50000, 100000, 200000] for economic segments
* **Test Scores**: [0, 60, 70, 80, 90, 100] for grade boundaries
* **Medical Thresholds**: Disease-specific clinical cutoffs
* **Risk Categories**: Regulatory or business-defined risk levels

See Also
--------

* :class:`ManualFlexibleBinning` - Manual binning with mixed interval and singleton bins
* :class:`EqualWidthBinning` - Automatic equal-width interval binning
* :class:`EqualFrequencyBinning` - Automatic quantile-based binning
* :class:`SupervisedBinning` - Automatic supervised binning with decision trees
