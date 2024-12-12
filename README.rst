README
======

General Purpose Binning Library
-------------------------------

This module offers a versatile set of binning capabilities often required in data preprocessing and feature engineering tasks. It provides various strategies for organizing numeric data into discrete groups, known as bins. 

Features
--------

- **Predefined Binning**: Customize bins according to predefined centers, ranges, or discrete sets.
- **Equal Width and Frequency Binning**: Simple and efficient binning based on equal widths or frequencies of values.
- **KMeans Clustering Binning**: Utilize k-means clustering to define bins according to detected clusters in the data.
- **Adaptive Binning**: Flexible bin adjustment according to weights or frequency requirements.
- **Inferred Binning**: Automatically infer optimal bin configuration based on input data.

Fancy Dynamic Badges
--------------------

.. image:: https://img.shields.io/badge/python-3.8%2B-green
   :target: https://www.python.org/downloads/

.. image:: https://img.shields.io/badge/license-MIT-blue
   :target: LICENSE

.. image:: https://img.shields.io/github/downloads/TheDAALab/binning/total
   :target: https://github.com/TheDAALab/binning/

.. image:: https://img.shields.io/github/actions/workflow/status/TheDAALab/binning/build.yml?branch=main
   :target: https://github.com/TheDAALab/binning/actions

Installation
------------

To install the package, simply run:

.. code-block:: bash

   pip install daa-adaptive-binning

Usage
-----

.. code-block:: python

   from general_binning import EqualWidthBinning, KMeansClusteringBinning

   # Example data
   data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

   # Equal Width Binning
   ewb = EqualWidthBinning(n_bins=3)
   ewb.fit(data)
   binned_data = ewb.transform(data)

   # KMeans Clustering Binning
   kmb = KMeansClusteringBinning(n_bins=3)
   kmb.fit(data)
   binned_data_kmeans = kmb.transform(data)

Examples
--------

- For working code examples, refer to the `examples` directory in the repository.
- Detailed Jupyter notebooks can be found in the `notebooks` directory.

References
----------

- **Scikit-learn Documentation**: https://scikit-learn.org/stable/documentation.html
- **KMeans1D Library**: https://github.com/esokolov/kmeans1d

Contributions
-------------

We welcome contributions to this repository! Please ensure your code follows the PEP8 style guide and includes appropriate tests.

License
-------

This project is licensed under the MIT License - see the `LICENSE.md` file for details.
