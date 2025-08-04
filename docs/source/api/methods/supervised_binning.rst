Supervised Binning
==================

.. automodule:: binlearn.methods._supervised_binning
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Overview
--------

Supervised Binning uses decision tree algorithms to create bins that are optimized for a target variable. This method creates target-aware bin boundaries that can improve downstream model performance.

Class Documentation
-------------------

.. autoclass:: binlearn.methods.SupervisedBinning
   :members:
   :inherited-members:
   :show-inheritance:
   :special-members: __init__

Examples
--------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from binlearn.methods import SupervisedBinning
   from sklearn.datasets import make_classification
   
   # Create sample data
   X, y = make_classification(n_samples=100, n_features=2, random_state=42)
   
   # Apply supervised binning
   binner = SupervisedBinning(n_bins=5)
   X_binned = binner.fit_transform(X, y)
   
   print(f"Bin edges based on target: {binner.bin_edges_}")

Feature Importance
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get feature importance from the decision tree
   importance = binner.get_feature_importance()
   print(f"Feature importance: {importance}")

Tree Structure
~~~~~~~~~~~~~

.. code-block:: python

   # Examine the decision tree structure
   tree_info = binner.get_tree_structure()
   print(f"Tree structure: {tree_info}")

See Also
--------

* :class:`binlearn.methods.KMeansBinning` - For unsupervised clustering
* :class:`binlearn.methods.EqualFrequencyBinning` - For balanced bins
