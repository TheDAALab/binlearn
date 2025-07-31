Binning Framework Documentation
=================================

.. image:: https://img.shields.io/badge/python-3.8%2B-blue
   :alt: Python Version
   
.. image:: https://img.shields.io/badge/license-MIT-green
   :alt: License

Welcome to the Binning Framework documentation! This comprehensive library provides advanced data binning and discretization methods for machine learning and data analysis.

The Binning Framework offers a unified interface for various binning strategies, including equal-width, equal-frequency, supervised, and custom binning methods. All methods are sklearn-compatible and support pandas/polars DataFrames.

Quick Start
-----------

Install the package:

.. code-block:: bash

   pip install binning-framework

Basic usage:

.. code-block:: python

   import numpy as np
   from binning.methods import EqualWidthBinning
   
   # Create sample data
   X = np.random.rand(100, 2) * 100
   
   # Apply binning
   binner = EqualWidthBinning(n_bins=5)
   X_binned = binner.fit_transform(X)

Key Features
------------

* **Multiple Binning Methods**: Equal-width, equal-frequency, supervised, manual, and weight-constrained binning
* **Sklearn Compatibility**: Full integration with scikit-learn pipelines and transformers
* **DataFrame Support**: Native support for pandas and polars DataFrames
* **Flexible Configuration**: Extensive customization options for all binning methods
* **Robust Error Handling**: Comprehensive validation and meaningful error messages
* **High Performance**: Optimized implementations with efficient memory usage

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
   
   tutorials/basic_binning
   tutorials/advanced_techniques
   tutorials/sklearn_integration
   tutorials/dataframe_usage

.. toctree::
   :maxdepth: 2
   :caption: Examples
   
   examples/equal_width_binning
   examples/equal_frequency_binning
   examples/supervised_binning
   examples/weight_constrained_binning

.. toctree::
   :maxdepth: 3
   :caption: API Reference
   
   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development
   
   contributing
   changelog
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
