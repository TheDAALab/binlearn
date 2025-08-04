binlearn - Binning and Discretization Library
==============================================

.. image:: https://img.shields.io/badge/python-3.8%2B-blue
   :alt: Python Version
   
.. image:: https://img.shields.io/badge/license-MIT-green
   :alt: License

.. image:: https://img.shields.io/badge/code%20quality-ruff-black
   :alt: Code Quality - Ruff

.. image:: https://img.shields.io/badge/type%20checking-mypy-blue
   :alt: Type Checking - MyPy

Welcome to the binlearn documentation! This comprehensive library provides advanced data binning and discretization methods for machine learning and data analysis.

binlearn offers a unified interface for various binning strategies, including equal-width, equal-frequency, supervised, and custom binning methods. All methods are sklearn-compatible and support pandas/polars DataFrames with full type safety.

Quick Start
-----------

Install the package:

.. code-block:: bash

   pip install binlearn

Basic usage:

.. code-block:: python

   import numpy as np
   from binlearn.methods import EqualWidthBinning
   
   # Create sample data
   X = np.random.rand(100, 2) * 100
   
   # Apply binning
   binner = EqualWidthBinning(n_bins=5, preserve_dataframe=True)
   X_binned = binner.fit_transform(X)
   
   print(f"Original shape: {X.shape}")
   print(f"Binned shape: {X_binned.shape}")
   print(f"Bin edges: {binner.bin_edges_}")

Key Features
------------

* **Multiple Binning Methods**: Equal-width, equal-frequency, supervised, manual, and weight-constrained binning
* **Sklearn Compatibility**: Full integration with scikit-learn pipelines and transformers
* **DataFrame Support**: Native support for pandas and polars DataFrames with column preservation
* **Type Safety**: Comprehensive type annotations with 100% mypy compliance
* **Modern Python**: Supports Python 3.8+ with modern type hints and syntax
* **Flexible Configuration**: Extensive customization options for all binning methods
* **Robust Error Handling**: Comprehensive validation and meaningful error messages with suggestions
* **High Performance**: Optimized implementations with efficient memory usage and O(n) complexity
* **Code Quality**: 100% test coverage, ruff-compliant code, and comprehensive CI/CD pipeline

Available Methods
-----------------

**Interval-based Methods:**

* :class:`~binlearn.methods.EqualWidthBinning` - Creates bins of equal width
* :class:`~binlearn.methods.EqualFrequencyBinning` - Creates bins with equal sample counts  
* :class:`~binlearn.methods.KMeansBinning` - K-means clustering-based binning
* :class:`~binlearn.methods.EqualWidthMinimumWeightBinning` - Weight-constrained equal-width bins

**Flexible Methods:**

* :class:`~binlearn.methods.ManualFlexibleBinning` - Custom bin definitions with mixed interval/singleton types
* :class:`~binlearn.methods.ManualIntervalBinning` - Custom interval-based bins

**Categorical Methods:**

* :class:`~binlearn.methods.SingletonBinning` - Singleton encoding for categorical data

**Supervised Methods:**

* :class:`~binlearn.methods.SupervisedBinning` - Decision tree-based supervised binning for classification and regression

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   user_guide
   
.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: Examples
   
   examples/index

.. toctree::
   :maxdepth: 3
   :caption: API Reference
   
   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development
   
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
