Fitted State Reconstruction
==========================

binlearn provides a powerful fitted state reconstruction system that allows complete restoration of fitted estimators from their parameters. This page explains how ``get_params()`` and ``set_params()`` work together to enable model persistence, pipeline serialization, and state reconstruction, with all parameter dictionaries being JSON serializable.

The Reconstruction Philosophy
----------------------------

binlearn follows the **"complete state reconstruction"** principle:

* ``get_params()`` captures ALL information needed to reconstruct a fitted estimator
* ``set_params(params)`` or ``Constructor(**get_params())`` completely restores the fitted state
* Parameter dictionaries are always JSON serializable for easy persistence
* No separate ``fit()`` call is needed after reconstruction

This enables seamless model persistence, distributed computing, and stateless transformations.

Basic Reconstruction Example
---------------------------

.. code-block:: python

    import numpy as np
    from binlearn import EqualWidthBinning
    import json
    
    # Create and fit original estimator
    X = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]])
    original_binner = EqualWidthBinning(n_bins=3, preserve_dataframe=False)
    original_binner.fit(X)
    
    # Transform some data
    X_test = np.array([[2.5, 25], [4.5, 45]])
    original_result = original_binner.transform(X_test)
    print(original_result)  # [[1 1], [2 2]]
    
    # Method 1: Reconstruction via get_params/set_params
    params = original_binner.get_params()
    reconstructed_binner = EqualWidthBinning()
    reconstructed_binner.set_params(**params)
    
    # Method 2: Reconstruction via constructor  
    params = original_binner.get_params()
    reconstructed_binner2 = EqualWidthBinning(**params)
    
    # Both methods produce identical behavior
    reconstructed_result1 = reconstructed_binner.transform(X_test)
    reconstructed_result2 = reconstructed_binner2.transform(X_test)
    
    np.testing.assert_array_equal(original_result, reconstructed_result1)
    np.testing.assert_array_equal(original_result, reconstructed_result2)
    print("✓ Perfect reconstruction achieved!")

JSON Serialization Support
--------------------------

All parameter dictionaries are guaranteed to be JSON serializable:

.. code-block:: python

    # Get fitted parameters
    fitted_binner = EqualWidthBinning(n_bins=4, clip=True)
    fitted_binner.fit(X)
    params = fitted_binner.get_params()
    
    # Serialize to JSON (always works)
    json_string = json.dumps(params)
    print("✓ JSON serialization successful")
    
    # Deserialize and reconstruct
    loaded_params = json.loads(json_string)
    restored_binner = EqualWidthBinning(**loaded_params)
    
    # Test equivalence
    test_data = np.array([[1.5, 15], [3.5, 35]])
    original_output = fitted_binner.transform(test_data)
    restored_output = restored_binner.transform(test_data)
    
    np.testing.assert_array_equal(original_output, restored_output)
    print("✓ JSON round-trip reconstruction successful!")

What get_params() Returns
-------------------------

The ``get_params()`` method returns a comprehensive dictionary containing:

Constructor Parameters
~~~~~~~~~~~~~~~~~~~~~

All parameters from the ``__init__`` method signature:

.. code-block:: python

    binner = EqualWidthBinning(n_bins=5, clip=True, preserve_dataframe=False)
    params = binner.get_params()
    
    # Constructor parameters
    print(params['n_bins'])              # 5
    print(params['clip'])                # True  
    print(params['preserve_dataframe'])  # False
    print(params['bin_range'])           # None (default value)

Fitted Parameters (When Fitted)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the estimator is fitted, ``get_params()`` includes fitted state:

.. code-block:: python

    binner.fit(X)
    fitted_params = binner.get_params()
    
    # Fitted parameters (without trailing underscores)
    print(fitted_params.keys())
    # Includes: 'bin_edges', 'bin_representatives', etc.
    
    print(fitted_params['bin_edges'])
    # {'0': [1.0, 2.33, 3.67, 5.0], '1': [10.0, 23.33, 36.67, 50.0]}
    
    print(fitted_params['bin_representatives'])  
    # {'0': [1.665, 3.0, 4.335], '1': [16.665, 30.0, 43.335]}

Class Metadata
~~~~~~~~~~~~~

For automatic reconstruction without explicit class references:

.. code-block:: python

    params = binner.get_params()
    print(params['class_'])   # 'EqualWidthBinning'
    print(params['module_'])  # 'binlearn.methods._equal_width_binning'
    
    # This enables dynamic class loading:
    class_name = params['class_']
    module_name = params['module_']
    # Can be used to dynamically import and reconstruct

Parameter Type Conversion
-------------------------

binlearn automatically converts numpy types to Python types for JSON compatibility:

.. code-block:: python

    import numpy as np
    
    # Example of internal type conversion
    raw_edges = {'0': np.array([1.0, 2.0, 3.0])}  # NumPy array
    params = binner.get_params()
    json_compatible_edges = params['bin_edges']     # Python list
    
    print(type(raw_edges['0']))                     # <class 'numpy.ndarray'>
    print(type(json_compatible_edges['0']))         # <class 'list'>
    
    # Both represent the same data
    np.testing.assert_array_equal(raw_edges['0'], json_compatible_edges['0'])

The conversion handles nested structures:

.. code-block:: python

    # Complex nested structure with numpy types
    complex_data = {
        'edges': {
            'feature1': np.array([1.0, 2.0, 3.0]),
            'feature2': np.array([10, 20, 30])
        },
        'representatives': {
            'feature1': [np.float64(1.5), np.float64(2.5)],
            'feature2': [np.int64(15), np.int64(25)]
        }
    }
    
    # After get_params(), everything becomes JSON-serializable Python types
    json.dumps(complex_data)  # This would work after conversion

How set_params() Works
---------------------

The ``set_params()`` method intelligently handles parameter restoration:

Regular Parameters
~~~~~~~~~~~~~~~~~

Constructor parameters are set through sklearn's standard mechanism:

.. code-block:: python

    binner = EqualWidthBinning()
    binner.set_params(
        n_bins=5,
        clip=True,
        preserve_dataframe=False
    )
    # These become: binner.n_bins, binner.clip, binner.preserve_dataframe

Fitted Parameters  
~~~~~~~~~~~~~~~~

Fitted parameters are set as attributes with trailing underscores:

.. code-block:: python

    binner.set_params(
        bin_edges={'0': [1, 2, 3, 4]},           # Becomes: binner.bin_edges_
        bin_representatives={'0': [1.5, 2.5, 3.5]}  # Becomes: binner.bin_representatives_
    )
    
    # After set_params(), the binner is immediately ready for transform()
    print(binner._fitted)  # True - no need to call fit()

Class Metadata Handling
~~~~~~~~~~~~~~~~~~~~~~

Class and module parameters are ignored during reconstruction:

.. code-block:: python

    params_with_metadata = {
        'n_bins': 3,
        'bin_edges': {'0': [1, 2, 3, 4]},
        'class_': 'EqualWidthBinning',      # Ignored
        'module_': 'binlearn.methods...',   # Ignored
    }
    
    binner.set_params(**params_with_metadata)
    # class_ and module_ are silently ignored

Advanced Reconstruction Scenarios
--------------------------------

Pipeline Persistence
~~~~~~~~~~~~~~~~~~~

Fitted pipelines can be completely reconstructed:

.. code-block:: python

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    
    # Create and fit pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('binner', EqualWidthBinning(n_bins=4))
    ])
    pipeline.fit(X)
    
    # Get complete pipeline parameters
    pipeline_params = pipeline.get_params(deep=True)
    
    # Reconstruct pipeline
    new_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('binner', EqualWidthBinning())
    ])
    new_pipeline.set_params(**pipeline_params)
    
    # Test pipeline equivalence
    original_prediction = pipeline.predict(X_test[:5])
    reconstructed_prediction = new_pipeline.predict(X_test[:5])
    np.testing.assert_array_equal(original_prediction, reconstructed_prediction)

Cross-Process Communication
~~~~~~~~~~~~~~~~~~~~~~~~~~

Perfect for distributed computing scenarios:

.. code-block:: python

    # Worker process 1: Train and serialize
    def train_and_serialize(X_train):
        binner = EqualWidthBinning(n_bins=5)
        binner.fit(X_train) 
        params = binner.get_params()
        return json.dumps(params)
    
    # Worker process 2: Deserialize and predict
    def load_and_predict(json_params, X_test):
        params = json.loads(json_params)
        binner = EqualWidthBinning(**params)  # Immediately ready
        return binner.transform(X_test)
    
    # Example usage
    serialized_model = train_and_serialize(X_train)
    predictions = load_and_predict(serialized_model, X_test)

Database Storage
~~~~~~~~~~~~~~~

Store fitted models in databases as JSON:

.. code-block:: python

    import sqlite3
    
    # Store fitted model
    def store_model(model_name, fitted_binner):
        params = fitted_binner.get_params()
        params_json = json.dumps(params)
        
        conn = sqlite3.connect('models.db')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS models 
            (name TEXT PRIMARY KEY, params TEXT)
        ''')
        conn.execute('INSERT OR REPLACE INTO models VALUES (?, ?)', 
                    (model_name, params_json))
        conn.commit()
        conn.close()
    
    # Load fitted model
    def load_model(model_name):
        conn = sqlite3.connect('models.db')
        cursor = conn.execute('SELECT params FROM models WHERE name = ?', (model_name,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            params = json.loads(row[0])
            return EqualWidthBinning(**params)  # Ready to use
        return None

Implementation Details
---------------------

Parameter Extraction Process
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``get_params()`` method uses introspection to extract parameters:

.. code-block:: python

    # Pseudo-code for get_params() logic
    def get_params(self, deep=True):
        # 1. Extract constructor signature
        init_signature = inspect.signature(self.__class__.__init__)
        
        # 2. Get constructor parameter values
        params = {}
        for param_name in init_signature.parameters:
            if param_name != 'self':
                params[param_name] = getattr(self, param_name, default_value)
        
        # 3. Add fitted parameters if fitted
        if self._fitted:
            fitted_params = self._extract_fitted_params()
            params.update(fitted_params)
        
        # 4. Add class metadata
        params['class_'] = self.__class__.__name__
        params['module_'] = self.__class__.__module__
        
        # 5. Convert numpy types to Python types
        return convert_to_python_types(params)

Fitted State Detection
~~~~~~~~~~~~~~~~~~~~~

The system automatically detects fitted state:

.. code-block:: python

    # Each binning class defines fitted attributes
    class EqualWidthBinning(IntervalBinningBase):
        def __init__(self, ...):
            # ...
            self._fitted_attributes = ['bin_edges_', 'bin_representatives_']
        
        @property
        def _fitted(self):
            # Check if any fitted attribute contains data
            for attr_name in self._fitted_attributes:
                attr_value = getattr(self, attr_name, None)
                if attr_value is not None and attr_value:
                    return True
            return False

Best Practices
-------------

1. **Always use get_params() for serialization**:
   
   .. code-block:: python
   
       # Good: Complete state capture
       params = fitted_binner.get_params()
       
       # Bad: Manual parameter extraction (incomplete)
       manual_params = {'n_bins': fitted_binner.n_bins}

2. **Prefer constructor reconstruction for new instances**:
   
   .. code-block:: python
   
       # Good: Clean instantiation
       new_binner = EqualWidthBinning(**params)
       
       # Also good: Explicit reconstruction
       new_binner = EqualWidthBinning()
       new_binner.set_params(**params)

3. **Use JSON serialization for persistence**:
   
   .. code-block:: python
   
       # Store model
       with open('model.json', 'w') as f:
           json.dump(fitted_binner.get_params(), f)
       
       # Load model
       with open('model.json', 'r') as f:
           params = json.load(f)
           restored_binner = EqualWidthBinning(**params)

4. **Test reconstruction in critical applications**:
   
   .. code-block:: python
   
       # Validation function
       def validate_reconstruction(original_estimator, test_data):
           # Get parameters and reconstruct
           params = original_estimator.get_params()
           reconstructed = original_estimator.__class__(**params)
           
           # Test equivalence
           original_output = original_estimator.transform(test_data)
           reconstructed_output = reconstructed.transform(test_data)
           
           np.testing.assert_array_equal(original_output, reconstructed_output)
           return True

Common Use Cases
---------------

* **Model Serving**: Serialize models to JSON for web services
* **Batch Processing**: Store fitted models between batch jobs
* **A/B Testing**: Compare different model configurations
* **Model Versioning**: Track model parameters over time
* **Distributed Computing**: Share fitted models across workers
* **Pipeline Checkpointing**: Save intermediate pipeline states

The fitted state reconstruction system makes binlearn estimators fully stateless and serializable, enabling flexible deployment and integration patterns.
