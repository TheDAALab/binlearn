SingletonBinning
================

.. currentmodule:: binlearn.methods

.. autoclass:: SingletonBinning
   :members:
   :inherited-members:
   :show-inheritance:

Examples
--------

Basic Categorical Encoding
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   import numpy as np
   from binlearn.methods import SingletonBinning
   
   # Create categorical data
   categorical_data = pd.DataFrame({
       'category': ['A', 'B', 'A', 'C', 'B', 'A', 'D'],
       'rating': ['good', 'bad', 'good', 'excellent', 'bad', 'good', 'fair']
   })
   
   # Apply singleton binning
   singleton_binner = SingletonBinning(preserve_dataframe=True)
   categorical_binned = singleton_binner.fit_transform(categorical_data)
   
   print(f"Original shape: {categorical_data.shape}")
   print(f"Binned shape: {categorical_binned.shape}")
   print(f"Original categories: {categorical_data['category'].unique()}")

Mixed Data Types
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Mixed categorical and numerical data
   mixed_data = pd.DataFrame({
       'category': ['A', 'B', 'A', 'C', 'B'],
       'numerical': [1.5, 2.3, 1.5, 4.1, 2.3],
       'rating': ['high', 'low', 'high', 'medium', 'low']
   })
   
   # SingletonBinning works on all data types
   binner = SingletonBinning(preserve_dataframe=True)
   mixed_binned = binner.fit_transform(mixed_data)

With Custom Representatives
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Using custom bin representatives
   from binlearn.utils.types import BinRepsDict
   
   custom_reps: BinRepsDict = {
       'category': {0: 'Type_A', 1: 'Type_B', 2: 'Type_C'},
       'rating': {0: 'Poor', 1: 'Good', 2: 'Excellent'}
   }
   
   binner = SingletonBinning(
       bin_representatives=custom_reps,
       preserve_dataframe=True
   )
   
   # Note: You still need to provide bin_edges for custom representatives
   # This is typically used when you have pre-trained binning parameters
