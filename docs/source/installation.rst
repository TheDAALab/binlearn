Installation
============

Requirements
------------

**Python Versions**: 3.10, 3.11, 3.12, 3.13

**Core Dependencies**:
  * NumPy >= 1.21.0
  * SciPy >= 1.7.0
  * Scikit-learn >= 1.0.0
  * kmeans1d >= 0.3.0

**Optional Dependencies**:
  * Pandas >= 1.3.0 (for DataFrame support)
  * Polars >= 0.15.0 (for Polars DataFrame support)

**Development Dependencies**:
  * pytest >= 6.0 (for testing)
  * ruff >= 0.1.0 (for linting and formatting)
  * mypy >= 1.0.0 (for type checking)

Installing from PyPI
---------------------

The easiest way to install binlearn is using pip:

.. code-block:: bash

   pip install binlearn

Installing with Optional Dependencies
-------------------------------------

To install with pandas support:

.. code-block:: bash

   pip install binlearn[pandas]

To install with polars support:

.. code-block:: bash

   pip install binlearn[polars]

To install with all optional dependencies:

.. code-block:: bash

   pip install binlearn[pandas,polars]

Installing from Source
----------------------

To install the latest development version from GitHub:

.. code-block:: bash

   git clone https://github.com/TheDAALab/binlearn.git
   cd binlearn
   pip install -e .

For development with all dependencies:

.. code-block:: bash

   pip install -e ".[tests,dev,pandas,polars]"

Verifying Installation
----------------------

To verify that binlearn is installed correctly:

.. code-block:: python

   import binlearn
   print(binlearn.__version__)
   
   # Test basic functionality
   from binlearn import EqualWidthBinning
   import numpy as np
   
   X = np.random.rand(100, 3)
   binner = EqualWidthBinning(n_bins=5)
   X_binned = binner.fit_transform(X)
   print(f"Binning successful! Shape: {X_binned.shape}")

Troubleshooting Installation
----------------------------

If you encounter issues during installation:

1. **Python Version**: Ensure you're using Python 3.10 or higher:

   .. code-block:: bash

      python --version

2. **Update pip**: Make sure you have the latest version of pip:

   .. code-block:: bash

      pip install --upgrade pip

3. **Virtual Environment**: Consider using a virtual environment:

   .. code-block:: bash

      python -m venv binlearn_env
      source binlearn_env/bin/activate  # On Windows: binlearn_env\Scripts\activate
      pip install binlearn

4. **Dependency Conflicts**: If you have dependency conflicts, try installing in a fresh environment or use conda:

   .. code-block:: bash

      conda create -n binlearn_env python=3.11
      conda activate binlearn_env
      pip install binlearn

Common Issues
-------------

**ImportError with Optional Dependencies**

If you see import errors related to pandas or polars, install the optional dependencies:

.. code-block:: bash

   pip install pandas polars

**NumPy/SciPy Compilation Issues**

On some systems, you might need to install system dependencies for NumPy/SciPy:

**Ubuntu/Debian:**

.. code-block:: bash

   sudo apt-get install python3-dev libopenblas-dev

**macOS:**

.. code-block:: bash

   brew install openblas

**Windows:**

Use conda for easier dependency management:

.. code-block:: bash

   conda install numpy scipy scikit-learn
