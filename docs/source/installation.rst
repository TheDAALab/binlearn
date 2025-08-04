Installation
============

Requirements
------------

binlearn requires Python 3.8 or later and depends on:

* numpy >= 1.20.0
* pandas >= 1.3.0 (optional, for DataFrame support)
* polars >= 0.18.0 (optional, for Polars DataFrame support)
* scikit-learn >= 1.0.0

Install from PyPI
-----------------

The easiest way to install binlearn is using pip:

.. code-block:: bash

   pip install binlearn

This will install the core package with numpy and scikit-learn dependencies.

Install with Optional Dependencies
----------------------------------

For full DataFrame support:

.. code-block:: bash

   # For pandas support
   pip install "binlearn[pandas]"
   
   # For polars support  
   pip install "binlearn[polars]"
   
   # For complete feature set
   pip install "binlearn[all]"

Development Installation
------------------------

To install binlearn for development:

.. code-block:: bash

   git clone https://github.com/TheDAALab/binlearn.git
   cd binlearn
   pip install -e ".[dev]"

This installs the package in editable mode with all development dependencies including testing and documentation tools.

Verify Installation
-------------------

To verify your installation works correctly:

.. code-block:: python

   import binlearn
   print(f"binlearn version: {binlearn.__version__}")
   
   # Test basic functionality
   from binlearn.methods import EqualWidthBinning
   import numpy as np
   
   X = np.random.rand(10, 1)
   binner = EqualWidthBinning(n_bins=3)
   X_binned = binner.fit_transform(X)
   print("Installation successful!")

If you encounter any issues, please check our `GitHub Issues <https://github.com/TheDAALab/binlearn/issues>`_ page.
