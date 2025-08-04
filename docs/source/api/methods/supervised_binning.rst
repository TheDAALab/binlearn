SupervisedBinning
=================

.. currentmodule:: binlearn.methods

.. autoclass:: SupervisedBinning
   :members:
   :inherited-members:
   :show-inheritance:

Examples
--------

Classification Task
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from sklearn.datasets import make_classification
   from binlearn.methods import SupervisedBinning
   
   # Create classification dataset
   X, y = make_classification(n_samples=1000, n_features=4, n_classes=2, random_state=42)
   
   # Create supervised binner
   sup_binner = SupervisedBinning(
       n_bins=4,
       task_type='classification',
       tree_params={'max_depth': 3, 'min_samples_leaf': 20}
   )
   
   # Fit using target variable as guidance
   X_binned = sup_binner.fit_transform(X, guidance_data=y)
   
   print(f"Binned shape: {X_binned.shape}")
   print(f"Bin edges per feature: {[len(edges)-1 for edges in sup_binner.bin_edges_.values()]}")

Regression Task
~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.datasets import make_regression
   
   # Create regression dataset
   X, y = make_regression(n_samples=1000, n_features=4, noise=0.1, random_state=42)
   
   # Create supervised binner for regression
   reg_binner = SupervisedBinning(
       n_bins=5,
       task_type='regression',
       tree_params={'max_depth': 4, 'min_samples_leaf': 50}
   )
   
   X_binned = reg_binner.fit_transform(X, guidance_data=y)

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Advanced tree configuration
   advanced_binner = SupervisedBinning(
       n_bins=6,
       task_type='classification',
       tree_params={
           'max_depth': 5,
           'min_samples_split': 100,
           'min_samples_leaf': 50,
           'random_state': 42
       }
   )
   
   X_binned = advanced_binner.fit_transform(X, guidance_data=y)
