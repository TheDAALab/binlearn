.. binning documentation master file, created by
   sphinx-quickstart on Thu Dec 12 15:19:36 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
=====================
binning documentation
=====================

General Purpose Binning Module
================================
This project provides a set of utilities for binning numerical data into different categories or bins. Binning is essential for many data processing tasks, including data transformation, visualization, and modeling, as it allows grouping continuous values into discrete intervals. This module offers various binning strategies and classes, all derived from a base class, each with its particular approach to binning data.

.. toctree::
    :maxdepth: 2
    :caption: Getting started

    00_introduction

.. toctree::
    :maxdepth: 2
    :caption: API

    03_api

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api 
   changelog



Features
========
- **Base Class for Binning:** `BinningBase` serves as the foundation for all binning strategies, providing the essential interface and structure for fit-transform operations.
  
- **Predefined Binning:** Includes `PredefinedBinCentersBinning`, `PredefinedBinRangesBinning`, and `PredefinedDiscreteBinning` for scenarios where bin characteristics are specified beforehand.

- **Data-Driven Binning:** Options for `EqualWidthBinning`, `EqualFrequencyBinning`, and `KMeansClusteringBinning` where bins are determined based on input data distribution.

- **Adaptive Binning:** `AdaptiveBinning` adjusts bin boundaries dynamically based on minimum weight or frequency criteria to maintain relevance and accuracy.

- **Handling of Non-Trivial Cases:** The module provides mechanisms to address edge cases like empty bins, ties, or uneven data distributions.

- **Integration with Machine Learning Pipelines:** Designed to potentially integrate as transformers in scikit-learn pipelines for more seamless use in data processing workflows.



Planned Enhancements
====================

- Improvement of interfaces and structure for ease of use and clarity.
- Potential integration as scikit-learn transformers for more seamless interaction with machine learning models.
- A native implementation of one-dimensional k-means clustering for reliance on well-maintained libraries.
- Better management of ties and empty bins, especially for equal frequency binning.


Dependencies
============

- NumPy
- Pandas
- scikit-learn
- kmeans1d


Usage
=====
Here is a quick example of how the module can be used to apply equal-width binning to a data set:


.. code-block:: Python

   import numpy as np
   from your_module import EqualWidthBinning  
   
   # Sample data
   data = np.array([1.3, 2.4, 3.1, 4.8, 5.9])  
   
   # Initialize binning with the number of bins
   binning = EqualWidthBinning(n_bins=3)  
   
   # Fit to the data
   binning.fit(data)  
   
   # Transform the data into bin indices
   binned_data = binning.transform(data)

   print(binned_data)  
   # Output: [0 0 1 2 2]


Contributing
===============
Contributions to enhance this module further are welcome. Bug reports and improvement suggestions can be filed in the project's issue tracker.

License
=======
This project is licensed under the MIT License - see the LICENSE file for details.


