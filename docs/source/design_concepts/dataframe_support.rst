DataFrame Support and Column Handling
=====================================

One of binlearn's most powerful features is its comprehensive support for different data formats while maintaining consistent behavior and column integrity across operations. This page explains how binlearn handles numpy arrays, pandas DataFrames, and polars DataFrames, the logic behind ``preserve_dataframe``, and the internal column representation system.

Overview of Data Format Support
-------------------------------

binlearn supports three primary data formats:

**NumPy Arrays**
    The foundation format - all internal operations work on numpy arrays. Column identifiers are numeric indices (0, 1, 2, ...).

**Pandas DataFrames** 
    Full support with column name preservation, index handling, and dtype consistency. Requires pandas installation.

**Polars DataFrames**
    High-performance columnar support with column name preservation. Requires polars installation (optional dependency).

The Core Design Principle
-------------------------

binlearn follows a **"format-agnostic processing, format-aware output"** design:

1. **Input**: Accept any supported format
2. **Processing**: Convert to numpy arrays internally for consistent computation  
3. **Output**: Return data in the format specified by ``preserve_dataframe`` setting

This approach ensures computational consistency while providing flexibility in data handling.

The preserve_dataframe Parameter
-------------------------------

The ``preserve_dataframe`` parameter controls output format behavior:

.. code-block:: python

    from binlearn import EqualWidthBinning
    import pandas as pd
    import numpy as np
    
    # Create sample data
    df = pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'feature2': [10.0, 20.0, 30.0, 40.0, 50.0]
    })
    
    # preserve_dataframe=True: Output matches input format
    binner_preserve = EqualWidthBinning(n_bins=3, preserve_dataframe=True)
    result_df = binner_preserve.fit_transform(df)
    print(type(result_df))  # <class 'pandas.core.frame.DataFrame'>
    print(result_df.columns.tolist())  # ['feature1', 'feature2']
    
    # preserve_dataframe=False: Always output numpy arrays  
    binner_array = EqualWidthBinning(n_bins=3, preserve_dataframe=False)
    result_array = binner_array.fit_transform(df)
    print(type(result_array))  # <class 'numpy.ndarray'>

Global Configuration
~~~~~~~~~~~~~~~~~~~~

The ``preserve_dataframe`` setting can be controlled globally:

.. code-block:: python

    import binlearn
    
    # Check current setting
    config = binlearn.get_config()
    print(config.preserve_dataframe)  # Default: False
    
    # Set globally
    binlearn.set_config(preserve_dataframe=True)
    
    # Now all binners will preserve DataFrame format by default
    binner = EqualWidthBinning(n_bins=5)  # preserve_dataframe=True by default

Column Representation and Handling
----------------------------------

binlearn uses a sophisticated column handling system that maintains consistency across different input formats and usage patterns.

Column Identification Priority
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When determining column identifiers, binlearn follows this priority order:

1. **Column names from input data** (DataFrames): ``['feature1', 'feature2']``
2. **Stored original columns** (for fitted estimators): Maintains training consistency
3. **Numeric indices** (arrays): ``[0, 1, 2]``
4. **Generated indices** (fallback): Based on data shape

.. code-block:: python

    # DataFrame input - column names extracted
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    binner = EqualWidthBinning(n_bins=2)
    binner.fit(df)
    # Internal representation uses: ['A', 'B']
    
    # NumPy input - numeric indices generated
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    binner = EqualWidthBinning(n_bins=2) 
    binner.fit(arr)
    # Internal representation uses: [0, 1, 2]

Column Consistency Across Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

binlearn maintains column consistency between fitting and transformation:

.. code-block:: python

    # Training with DataFrame
    train_df = pd.DataFrame({
        'income': [30000, 45000, 60000, 80000, 120000],
        'age': [25, 35, 45, 55, 65]
    })
    
    binner = EqualWidthBinning(n_bins=3, preserve_dataframe=True)
    binner.fit(train_df)
    
    # Transform maintains column structure even with different input
    test_data = np.array([[50000, 40], [90000, 50]])  # NumPy format
    result = binner.transform(test_data)  
    # Result preserves training column structure as DataFrame
    
    print(type(result))  # pandas.DataFrame
    print(result.columns.tolist())  # ['income', 'age']

Internal Column Resolution
~~~~~~~~~~~~~~~~~~~~~~~~~

The column resolution system handles format mismatches gracefully:

.. code-block:: python

    # Train with numeric columns (NumPy)
    X_train = np.array([[1, 10], [2, 20], [3, 30]])
    binner = EqualWidthBinning(n_bins=2)
    binner.fit(X_train)  # Uses columns: [0, 1]
    
    # Transform with named columns (DataFrame)
    X_test = pd.DataFrame({'feature_0': [1.5], 'feature_1': [15]})
    result = binner.transform(X_test)
    # Automatic mapping: 'feature_0' -> 0, 'feature_1' -> 1

Advanced Column Handling
-----------------------

Guidance Column Separation
~~~~~~~~~~~~~~~~~~~~~~~~~~

For supervised binning methods, binlearn automatically separates binning columns from guidance columns:

.. code-block:: python

    from binlearn import EqualWidthMinimumWeightBinning
    
    # Data with features and weights
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50], 
        'sample_weight': [1.0, 2.0, 1.5, 3.0, 2.5]
    })
    
    # Specify which columns are for guidance
    binner = EqualWidthMinimumWeightBinning(
        n_bins=3,
        minimum_weight=2.0,
        guidance_columns='sample_weight',  # This column provides weights
        preserve_dataframe=True
    )
    
    # Fit processes all columns but only bins feature columns
    binner.fit(data)
    
    # Transform only processes and outputs feature columns
    result = binner.transform(data)
    print(result.columns.tolist())  # ['feature1', 'feature2'] - no weight column

Column Key Resolution
~~~~~~~~~~~~~~~~~~~~

binlearn handles different column identifier formats:

.. code-block:: python

    # Bin specifications can use different key formats
    edges_dict = {
        'feature_0': [0, 1, 2, 3],    # String format
        1: [10, 20, 30, 40]           # Integer format  
    }
    
    # Works with both NumPy arrays and DataFrames
    binner = EqualWidthBinning(bin_edges=edges_dict, preserve_dataframe=True)
    
    # Automatic key resolution during transformation
    df_result = binner.transform(pd.DataFrame({
        'feature_0': [0.5, 1.5], 
        'feature_1': [15, 25]
    }))

Implementation Details
---------------------

Data Flow Architecture
~~~~~~~~~~~~~~~~~~~~~

The complete data flow follows this pattern:

.. code-block:: text

    Input Data (Any Format)
           ↓
    prepare_input_with_columns()
           ↓  
    [numpy_array, column_list]
           ↓
    Binning Operations (NumPy)
           ↓
    return_like_input()
           ↓ 
    Output (Format based on preserve_dataframe)

Key Functions
~~~~~~~~~~~~

The core data handling functions are:

**prepare_input_with_columns(X, fitted, original_columns)**
    - Converts any input format to numpy array
    - Extracts or generates column identifiers
    - Maintains column consistency for fitted estimators

**return_like_input(result, original_input, columns, preserve_dataframe)**
    - Formats output to match desired format
    - Preserves column names and structure when requested
    - Handles pandas and polars DataFrame construction

**convert_to_python_types(value)**
    - Recursively converts numpy types to Python types
    - Essential for JSON serialization of fitted parameters
    - Handles nested structures (dicts, lists, arrays)

Memory and Performance Considerations
------------------------------------

Format Conversion Overhead
~~~~~~~~~~~~~~~~~~~~~~~~~

- **DataFrame → NumPy**: Moderate overhead for format conversion
- **NumPy Operations**: Minimal overhead - native computation 
- **NumPy → DataFrame**: Moderate overhead for format reconstruction
- **Column Tracking**: Minimal overhead for metadata management

Optimization Tips
~~~~~~~~~~~~~~~~

.. code-block:: python

    # For performance-critical applications with large DataFrames
    # Option 1: Use preserve_dataframe=False for faster processing
    binner = EqualWidthBinning(n_bins=5, preserve_dataframe=False)
    result = binner.fit_transform(large_df)  # Returns NumPy array
    
    # Option 2: Work with NumPy arrays directly
    arr = large_df.values
    result = binner.fit_transform(arr)  # Avoids DataFrame conversion
    
    # Option 3: Use global setting to avoid repeated parameter specification
    binlearn.set_config(preserve_dataframe=False)

Best Practices
--------------

1. **Consistent Input Formats**: Use the same format for training and prediction when possible
2. **Column Names**: Use meaningful column names in DataFrames for better interpretability  
3. **Global Configuration**: Set preserve_dataframe globally for consistent behavior across projects
4. **Performance**: Consider using NumPy arrays for performance-critical applications
5. **Mixed Formats**: binlearn handles format mismatches, but consistency improves performance

.. code-block:: python

    # Good: Consistent formats
    df_train = pd.DataFrame({'income': [...], 'age': [...]})
    df_test = pd.DataFrame({'income': [...], 'age': [...]})
    
    # Also good: Consistent NumPy usage for performance
    X_train = np.array([[...], [...]])  
    X_test = np.array([[...], [...]])
    
    # Works but less optimal: Mixed formats
    df_train = pd.DataFrame({'income': [...], 'age': [...]})
    X_test = np.array([[...], [...]])  # Format conversion required
