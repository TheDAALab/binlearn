=============================================
binlearn - Binning and Discretization Library
=============================================

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

.. image:: https://img.shields.io/pypi/dm/binlearn
    :alt: Monthly Downloads
    :target: https://pypi.org/project/binlearn/

.. image:: https://img.shields.io/github/stars/TheDAALab/binlearn?style=social
    :alt: GitHub Stars
    :target: https://github.com/TheDAALab/binlearn

.. image:: https://img.shields.io/badge/code%20style-ruff-000000.svg
    :alt: Code Style - Ruff
    :target: https://github.com/astral-sh/ruff

.. image:: https://img.shields.io/badge/typing-mypy-blue
    :alt: Type Checking - MyPy
    :target: https://mypy.readthedocs.io/

A modern, type-safe Python library for data binning and discretization with comprehensive error handling, sklearn compatibility, and DataFrame support. Features our new **SingletonBinning** method for cleaner categorical data encoding!

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
  * **Documentation** - Extensive examples and API documentation with SingletonBinning guide

📦 **Installation**
---------------------

.. code-block:: bash

   pip install binlearn

🔥 **Quick Start**
--------------------

.. code-block:: python

   import numpy as np
   import pandas as pd
   from binlearn import EqualWidthBinning, SupervisedBinning, SingletonBinning
   
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
   
   # NEW: SingletonBinning for categorical features
   categorical_data = pd.DataFrame({
       'category': ['A', 'B', 'A', 'C', 'B', 'A'],
       'rating': [1, 2, 1, 3, 2, 1]
   })
   
   singleton_binner = SingletonBinning(preserve_dataframe=True)
   categorical_binned = singleton_binner.fit_transform(categorical_data)
   print(f"Categorical binning: {categorical_binned.shape}")

🎯 **Supervised Binning Example**
-----------------------------------

.. code-block:: python

   from binlearn import SupervisedBinning
   from sklearn.datasets import make_classification
   from sklearn.model_selection import train_test_split
   
   # Create classification dataset
   X, y = make_classification(n_samples=1000, n_features=4, n_classes=2, random_state=42)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
   # Create supervised binner that considers target variable
   sup_binner = SupervisedBinning(
       n_bins=4,
       task_type='classification',
       tree_params={'max_depth': 3, 'min_samples_leaf': 20}
   )
   
   # Fit using guidance data (target variable)
   X_train_binned = sup_binner.fit_transform(X_train, guidance_data=y_train)
   X_test_binned = sup_binner.transform(X_test)
   
   print(f"Supervised binning created bins optimized for target separation")
   print(f"Bin edges per feature: {[len(edges)-1 for edges in sup_binner.bin_edges_.values()]}")

🛠️ **Scikit-learn Integration**
---------------------------------

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.ensemble import RandomForestClassifier
   from binlearn import EqualFrequencyBinning
   
   # Create ML pipeline with binning preprocessing
   pipeline = Pipeline([
       ('binning', EqualFrequencyBinning(n_bins=5)),
       ('classifier', RandomForestClassifier(random_state=42))
   ])
   
   # Train and evaluate
   pipeline.fit(X_train, y_train)
   accuracy = pipeline.score(X_test, y_test)
   print(f"Pipeline accuracy: {accuracy:.3f}")

📚 **Available Methods**
--------------------------

**Interval-based Methods:**

* ``EqualWidthBinning`` - Creates bins of equal width across the data range
* ``EqualFrequencyBinning`` - Creates bins with approximately equal number of samples  
* ``KMeansBinning`` - Uses K-means clustering to determine bin boundaries
* ``EqualWidthMinimumWeightBinning`` - Equal-width bins with weight constraints

**Flexible Methods:**

* ``ManualIntervalBinning`` - Specify custom interval boundaries
* ``ManualFlexibleBinning`` - Define mixed interval and singleton bins

**Categorical Methods:**

* ``SingletonBinning`` - Clean categorical encoding for discrete values 🆕

**Supervised Methods:**

* ``SupervisedBinning`` - Decision tree-based binning optimized for target variables (classification and regression)

⚙️ **Requirements**
---------------------

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

🧪 **Development Setup**
--------------------------

.. code-block:: bash

   # Clone repository
   git clone https://github.com/TheDAALab/binlearn.git
   cd binlearn
   
   # Install in development mode with all dependencies
   pip install -e ".[tests,dev,pandas,polars]"
   
   # Run all tests
   pytest
   
   # Run code quality checks
   ruff check binlearn/
   mypy binlearn/ --ignore-missing-imports
   
   # Build documentation
   cd docs && make html

🏆 **Code Quality Standards**
-------------------------------

* ✅ **100% Test Coverage** - Comprehensive test suite with 841 tests
* ✅ **100% Type Safety** - Complete mypy compliance with modern type annotations
* ✅ **100% Code Quality** - Full ruff compliance with modern Python standards
* * ✅ **Comprehensive Documentation** - Detailed API docs and examples with SingletonBinning guide
* ✅ **Modern Python** - Uses latest Python features and best practices
* ✅ **Robust Error Handling** - Helpful error messages with actionable suggestions

🤝 **Contributing**
---------------------

We welcome contributions! Here's how to get started:

1. Fork the repository on GitHub
2. Create a feature branch: ``git checkout -b feature/your-feature``
3. Make your changes and add tests
4. Ensure all quality checks pass:
   
   .. code-block:: bash
   
      pytest                                    # Run tests
      ruff check binlearn/                      # Check code quality  
      mypy binlearn/ --ignore-missing-imports   # Check types

5. Submit a pull request

**Areas for Contribution**:
  * 🐛 Bug reports and fixes
  * ✨ New binning algorithms
  * 📚 Documentation improvements
  * 🧪 Additional test cases
  * 🎯 Performance optimizations

🔗 **Links**
--------------

* **GitHub Repository**: https://github.com/TheDAALab/binlearn
* **Issue Tracker**: https://github.com/TheDAALab/binlearn/issues
* **Documentation**: https://binlearn.readthedocs.io/

📄 **License**
----------------

This project is licensed under the MIT License. See the `LICENSE <https://github.com/TheDAALab/binlearn/blob/main/LICENSE>`_ file for details.

---

**Developed by TheDAALab** 

*A modern, type-safe binning framework for Python data science workflows.*

.. image:: https://img.shields.io/badge/Powered%20by-Python-blue.svg
    :alt: Powered by Python
    :target: https://www.python.org/

.. image:: https://img.shields.io/badge/Built%20with-NumPy-orange.svg
    :alt: Built with NumPy
    :target: https://numpy.org/

.. image:: https://img.shields.io/badge/Compatible%20with-Pandas-green.svg
    :alt: Compatible with Pandas
    :target: https://pandas.pydata.org/

.. image:: https://img.shields.io/badge/Integrates%20with-Scikit--learn-red.svg
    :alt: Integrates with Scikit-learn
    :target: https://scikit-learn.org/

.. image:: https://img.shields.io/pypi/status/binlearn
    :alt: Development Status
    :target: https://pypi.org/project/binlearn/

.. image:: https://img.shields.io/github/contributors/TheDAALab/binlearn
    :alt: Contributors
    :target: https://github.com/TheDAALab/binlearn/graphs/contributors
