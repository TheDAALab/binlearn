Changelog
=========

All notable changes to binlearn will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

Added
^^^^^
* 🆕 **SingletonBinning**: New method for categorical data encoding
* Support for multiple encoding strategies in SingletonBinning
* Enhanced DataFrame integration with automatic column type detection
* Comprehensive documentation with ReadTheDocs support

Changed
^^^^^^^
* Improved error messages across all binning methods
* Enhanced performance for large datasets
* Better memory efficiency in K-means binning

Fixed
^^^^^
* Fixed edge cases with constant data
* Improved handling of NaN values
* Resolved issues with empty bins in equal-width binning

[0.1.0] - 2024-01-15
--------------------

Added
^^^^^
* Initial release of binlearn
* **EqualWidthBinning**: Divide data into equal-width intervals
* **EqualFrequencyBinning**: Create bins with equal number of observations
* **KMeansBinning**: Clustering-based binning method
* **SupervisedBinning**: Target-aware binning for classification/regression
* **ManualFlexibleBinning**: Custom bin definitions
* **ManualIntervalBinning**: Manual interval specification
* Full sklearn compatibility with BaseEstimator and TransformerMixin
* Comprehensive base classes for extending functionality
* pandas and polars DataFrame support
* Complete type annotations and mypy compatibility
* 100% test coverage
* Extensive documentation and examples

Infrastructure
^^^^^^^^^^^^^^
* Automated testing with pytest
* Code quality enforcement with ruff
* Type checking with mypy
* Continuous integration with GitHub Actions
* ReadTheDocs documentation hosting
