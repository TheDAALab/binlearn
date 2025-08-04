binlearn: Data Binning and Discretization Library
===============================================

.. image:: https://img.shields.io/pypi/v/binlearn
    :alt: PyPI Version
    :target: https://pypi.org/project/binlearn/

.. image:: https://img.shields.io/pypi/pyversions/binlearn
    :alt: Python Versions
    :target: https://pypi.org/project/binlearn/

.. image:: https://img.shields.io/github/actions/workflow/status/TheDAALab/binlearn/build.yml?branch=main
    :alt: Build Status
    :target: https://github.com/TheDAALab/binlearn/actions/workflows/build.yml

.. image:: https://img.shields.io/codecov/c/github/TheDAALab/binlearn
    :alt: Code Coverage
    :target: https://codecov.io/gh/TheDAALab/binlearn

.. image:: https://img.shields.io/github/license/TheDAALab/binlearn
    :alt: License
    :target: https://github.com/TheDAALab/binlearn/blob/main/LICENSE

.. image:: https://img.shields.io/readthedocs/binlearn
    :alt: Documentation Status
    :target: https://binlearn.readthedocs.io/

A modern, type-safe Python library for data binning and discretization with comprehensive error handling, sklearn compatibility, and DataFrame support.

🚀 **Key Features**
---------------------

✨ **Multiple Binning Methods**
  * **EqualWidthBinning** - Equal-width intervals across data range
  * **EqualFrequencyBinning** - Equal-frequency (quantile-based) bins  
  * **KMeansBinning** - K-means clustering-based discretization
  * **EqualWidthMinimumWeightBinning** - Weight-constrained equal-width binning
  * **SupervisedBinning** - Decision tree-based supervised binning for classification and regression
  * **ManualIntervalBinning** - Custom interval boundary specification
  * **ManualFlexibleBinning** - Mixed interval and singleton bin definitions
  * **SingletonBinning** - Clean categorical encoding for discrete values 🆕

🔧 **Framework Integration**
  * **Pandas DataFrames** - Native support with column name preservation
  * **Polars DataFrames** - High-performance columnar data support (optional)
  * **NumPy Arrays** - Efficient numerical array processing
  * **Scikit-learn Pipelines** - Full transformer compatibility

⚡ **Modern Code Quality**
  * **Type Safety** - 100% mypy compliance with comprehensive type annotations
  * **Code Quality** - 100% ruff compliance with modern Python syntax
  * **Error Handling** - Comprehensive validation with helpful error messages and suggestions
  * **Test Coverage** - 100% code coverage with 841 comprehensive tests
  * **Documentation** - Extensive examples and API documentation

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install binlearn

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import pandas as pd
   from binlearn import EqualWidthBinning
   
   # Create sample data
   data = pd.DataFrame({
       'age': np.random.normal(35, 10, 1000),
       'income': np.random.lognormal(10, 0.5, 1000),
       'score': np.random.uniform(0, 100, 1000)
   })
   
   # Equal-width binning with DataFrame preservation
   binner = EqualWidthBinning(n_bins=5, preserve_dataframe=True)
   data_binned = binner.fit_transform(data)
   
   print(f"Original shape: {data.shape}")
   print(f"Binned shape: {data_binned.shape}")
   print(f"Bin edges for age: {binner.bin_edges_['age']}")

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog
   license

.. toctree::
   :maxdepth: 1
   :caption: Additional Information

   faq
   troubleshooting
   performance_tips

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
