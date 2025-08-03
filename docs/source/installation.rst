Installation
============

Requirements
------------

The binlearn library requires Python 3.8 or later and has the following dependencies:

Required Dependencies
~~~~~~~~~~~~~~~~~~~~~

* ``numpy >= 1.20.0``
* ``scikit-learn >= 1.0.0``

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

* ``pandas >= 1.3.0`` - For DataFrame support
* ``polars >= 0.15.0`` - For Polars DataFrame support

Install from PyPI
-----------------

The easiest way to install the binlearn library is using pip:

.. code-block:: bash

   pip install binning-framework

Install from Source
-------------------

To install the latest development version from source:

.. code-block:: bash

   git clone https://github.com/TheDAALab/binning.git
   cd binning
   pip install -e .

Development Installation
------------------------

For development, install with additional dependencies:

.. code-block:: bash

   git clone https://github.com/TheDAALab/binning.git
   cd binning
   pip install -e ".[dev,test,docs]"

Verify Installation
-------------------

To verify your installation, run:

.. code-block:: python

   import binlearn
   print(binning.__version__)
   
   # Test basic functionality
   from binlearn.methods import EqualWidthBinning
   import numpy as np
   
   X = np.random.rand(10, 2)
   binner = EqualWidthBinning(n_bins=3)
   X_binned = binner.fit_transform(X)
   print("Installation successful!")

Docker Installation
-------------------

A Docker image is also available:

.. code-block:: bash

   docker pull thedaalab/binning-framework:latest

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError: No module named 'binning'**
   Make sure you have installed the package correctly and are using the right Python environment.

**Dependency conflicts**
   Try creating a fresh virtual environment:
   
   .. code-block:: bash
   
      python -m venv binning_env
      source binning_env/bin/activate  # On Windows: binning_env\Scripts\activate
      pip install binning-framework

**Performance issues**
   For better performance with large datasets, consider installing optional dependencies:
   
   .. code-block:: bash
   
      pip install "binning-framework[performance]"
