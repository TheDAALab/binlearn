🆕 SingletonBinning Examples
=============================

The new **SingletonBinning** method provides advanced categorical data encoding with multiple strategies for different use cases.

Basic Categorical Encoding
---------------------------

Transform categorical data into numerical representations:

.. code-block:: python

   import numpy as np
   from binlearn.methods import SingletonBinning
   
   # Sample categorical data
   categories = ['red', 'blue', 'green', 'red', 'blue', 'green', 'red']
   X = np.array(categories).reshape(-1, 1)
   
   # Basic singleton binning (default: binary/one-hot style)
   singleton = SingletonBinning()
   X_encoded = singleton.fit_transform(X)
   
   print(f"Original categories: {categories}")
   print(f"Encoded shape: {X_encoded.shape}")
   print(f"Discovered categories: {singleton.categories_}")
   print(f"Feature names: {singleton.get_feature_names_out()}")
   print(f"Encoded data:\\n{X_encoded}")

Different Encoding Strategies
-----------------------------

SingletonBinning supports multiple encoding strategies:

.. code-block:: python

   # Sample data with more categories
   data = ['low', 'medium', 'high', 'low', 'high', 'medium', 'high', 'low']
   X = np.array(data).reshape(-1, 1)
   
   print("Original data:", data)
   print()
   
   # Strategy 1: Binary (One-hot encoding)
   binary_encoder = SingletonBinning(strategy='binary')
   X_binary = binary_encoder.fit_transform(X)
   print("Binary encoding (one-hot):")
   print(f"Shape: {X_binary.shape}")
   print(f"Features: {binary_encoder.get_feature_names_out()}")
   print(X_binary)
   print()
   
   # Strategy 2: Ordinal (Label encoding)
   ordinal_encoder = SingletonBinning(strategy='ordinal')
   X_ordinal = ordinal_encoder.fit_transform(X)
   print("Ordinal encoding (label):")
   print(f"Shape: {X_ordinal.shape}")
   print(f"Mapping: {dict(zip(ordinal_encoder.categories_[0], range(len(ordinal_encoder.categories_[0]))))}")
   print(X_ordinal.flatten())
   print()
   
   # Strategy 3: Frequency-based encoding
   frequency_encoder = SingletonBinning(strategy='frequency')
   X_frequency = frequency_encoder.fit_transform(X)
   print("Frequency encoding:")
   print(f"Shape: {X_frequency.shape}")
   print(f"Frequency mapping: {frequency_encoder.category_frequencies_}")
   print(X_frequency.flatten())

Handling Unknown Categories
---------------------------

Control how unknown categories are handled during transformation:

.. code-block:: python

   # Training data
   train_data = ['cat', 'dog', 'bird', 'cat', 'dog']
   X_train = np.array(train_data).reshape(-1, 1)
   
   # Test data with unknown category
   test_data = ['cat', 'fish', 'dog', 'elephant']  # 'fish' and 'elephant' are unknown
   X_test = np.array(test_data).reshape(-1, 1)
   
   # Fit on training data
   encoder = SingletonBinning(handle_unknown='ignore')
   encoder.fit(X_train)
   
   print(f"Known categories: {encoder.categories_[0]}")
   
   # Transform test data
   X_test_encoded = encoder.transform(X_test)
   print(f"Test data: {test_data}")
   print(f"Encoded (unknown ignored): {X_test_encoded.flatten()}")
   
   # Alternative: error on unknown
   strict_encoder = SingletonBinning(handle_unknown='error')
   strict_encoder.fit(X_train)
   
   try:
       strict_encoder.transform(X_test)
   except ValueError as e:
       print(f"Error with unknown categories: {e}")

Multiple Categorical Features
-----------------------------

Handle multiple categorical columns simultaneously:

.. code-block:: python

   # Multiple categorical features
   colors = ['red', 'blue', 'green', 'red', 'blue']
   sizes = ['small', 'large', 'medium', 'large', 'small']
   materials = ['wood', 'plastic', 'metal', 'wood', 'plastic']
   
   X_multi = np.column_stack([colors, sizes, materials])
   
   print("Original multi-feature data:")
   print(X_multi)
   print()
   
   # Apply singleton binning to all features
   multi_encoder = SingletonBinning(strategy='binary')
   X_multi_encoded = multi_encoder.fit_transform(X_multi)
   
   print(f"Encoded shape: {X_multi_encoded.shape}")
   print(f"Feature names: {multi_encoder.get_feature_names_out()}")
   print("Encoded data:")
   print(X_multi_encoded)

Integration with DataFrames
---------------------------

SingletonBinning works seamlessly with pandas DataFrames:

.. code-block:: python

   import pandas as pd
   
   # Create DataFrame with categorical data
   df = pd.DataFrame({
       'color': ['red', 'blue', 'green', 'red', 'blue', 'green'],
       'size': ['S', 'M', 'L', 'M', 'S', 'L'],
       'material': ['cotton', 'silk', 'wool', 'cotton', 'silk', 'wool'],
       'price': [10.5, 25.0, 40.5, 12.0, 28.5, 45.0]  # numerical column
   })
   
   print("Original DataFrame:")
   print(df)
   print()
   
   # Apply to categorical columns only
   categorical_columns = df.select_dtypes(include=['object']).columns
   
   df_encoder = SingletonBinning(strategy='binary')
   df_categorical_encoded = df_encoder.fit_transform(df[categorical_columns])
   
   # Combine with numerical columns
   df_result = pd.concat([
       pd.DataFrame(df_categorical_encoded, 
                   columns=df_encoder.get_feature_names_out()),
       df[['price']].reset_index(drop=True)
   ], axis=1)
   
   print("DataFrame after categorical encoding:")
   print(df_result)

Pipeline Integration
--------------------

Use SingletonBinning in scikit-learn pipelines:

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import accuracy_score
   
   # Sample dataset with categorical features
   np.random.seed(42)
   categories_1 = np.random.choice(['A', 'B', 'C'], 1000)
   categories_2 = np.random.choice(['X', 'Y', 'Z'], 1000)
   
   # Create target based on categories (for demonstration)
   y = ((categories_1 == 'A') & (categories_2 == 'X')).astype(int)
   X = np.column_stack([categories_1, categories_2])
   
   # Split the data
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
   # Create pipeline
   pipeline = Pipeline([
       ('encoding', SingletonBinning(strategy='binary')),
       ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
   ])
   
   # Fit and predict
   pipeline.fit(X_train, y_train)
   y_pred = pipeline.predict(X_test)
   
   accuracy = accuracy_score(y_test, y_pred)
   print(f"Pipeline accuracy with SingletonBinning: {accuracy:.3f}")
   
   # Check feature importance
   feature_names = pipeline.named_steps['encoding'].get_feature_names_out()
   feature_importance = pipeline.named_steps['classifier'].feature_importances_
   
   print("\\nFeature importance:")
   for name, importance in zip(feature_names, feature_importance):
       print(f"{name}: {importance:.3f}")

Advanced Configuration
----------------------

Fine-tune SingletonBinning behavior for specific use cases:

.. code-block:: python

   # Custom configuration for specific needs
   advanced_encoder = SingletonBinning(
       strategy='binary',
       handle_unknown='ignore',
       sparse_output=False,  # Dense output instead of sparse
       dtype=np.float32      # Control output data type
   )
   
   sample_data = ['category_a', 'category_b', 'category_a']
   X_sample = np.array(sample_data).reshape(-1, 1)
   
   X_encoded = advanced_encoder.fit_transform(X_sample)
   print(f"Output dtype: {X_encoded.dtype}")
   print(f"Output type: {type(X_encoded)}")
   print(f"Encoded data:\\n{X_encoded}")

Performance Considerations
--------------------------

SingletonBinning is optimized for performance with large categorical datasets:

.. code-block:: python

   import time
   
   # Generate large categorical dataset
   large_categories = np.random.choice(['cat_1', 'cat_2', 'cat_3', 'cat_4', 'cat_5'], 
                                      size=100000)
   X_large = large_categories.reshape(-1, 1)
   
   # Measure encoding time
   start_time = time.time()
   fast_encoder = SingletonBinning(strategy='ordinal')  # Fastest for large data
   X_large_encoded = fast_encoder.fit_transform(X_large)
   end_time = time.time()
   
   print(f"Encoded {len(X_large):,} samples in {end_time - start_time:.3f} seconds")
   print(f"Memory usage: {X_large_encoded.nbytes / 1024 / 1024:.2f} MB")

SingletonBinning provides a powerful and flexible solution for categorical data encoding, with performance optimizations and extensive customization options for any use case.
