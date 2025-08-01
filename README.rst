===============================
Binning Framework
===============================

.. image:: https://img.shields.io/pypi/v/binning-framework.svg
    :target: https://pypi.python.org/pypi/binning-framework
    :alt: PyPI Version

.. image:: https://img.shields.io/pypi/pyversions/binning-framework.svg
    :target: https://pypi.python.org/pypi/binning-framework
    :alt: Python Versions

.. image:: https://github.com/TheDAALab/binning/workflows/CI/badge.svg
    :target: https://github.com/TheDAALab/binning/actions
    :alt: CI Status

.. image:: https://codecov.io/gh/TheDAALab/binning/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/TheDAALab/binning
    :alt: Code Coverage

.. image:: https://readthedocs.org/projects/binning-framework/badge/?version=latest
    :target: https://binning-framework.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: Code Style: Black

.. image:: https://img.shields.io/github/license/TheDAALab/binning.svg
    :target: https://github.com/TheDAALab/binning/blob/main/LICENSE
    :alt: License

.. image:: https://img.shields.io/pypi/dm/binning-framework.svg
    :target: https://pypi.python.org/pypi/binning-framework
    :alt: Monthly Downloads

.. image:: https://img.shields.io/github/stars/TheDAALab/binning.svg?style=social&label=Star
    :target: https://github.com/TheDAALab/binning
    :alt: GitHub Stars

A comprehensive, high-performance Python framework for data binning and discretization with advanced algorithms, seamless integration with pandas/polars/numpy, and scikit-learn compatibility.

🚀 **Key Features**
-------------------

✨ **Multiple Binning Algorithms**
  * **EqualWidthBinning** - Uniform bin widths
  * **EqualFrequencyBinning** - Balanced sample distributions  
  * **EqualWidthMinimumWeightBinning** - Weight-constrained equal-width binning
  * **SupervisedBinning** - Target-aware discretization
  * **ManualBinning** - Custom boundary specification

🔧 **Framework Integration**
  * **Pandas DataFrames** - Native support with type preservation
  * **Polars DataFrames** - High-performance columnar operations
  * **NumPy Arrays** - Efficient numerical computing
  * **Scikit-learn Pipelines** - Seamless ML workflow integration

⚡ **Performance & Scalability**
  * Optimized algorithms for large datasets
  * Memory-efficient processing
  * Parallel computation support
  * Lazy evaluation where possible

🎯 **Advanced Capabilities**
  * Weight-constrained binning with guidance data
  * Missing value handling
  * Robust error handling and validation
  * Comprehensive test coverage (100%)
  * Type hints and documentation

📦 **Quick Install**
--------------------

.. code-block:: bash

   pip install binning-framework

🔥 **Quick Start**
------------------

.. code-block:: python

   import numpy as np
   import pandas as pd
   from binning.methods import EqualWidthBinning, EqualFrequencyBinning
   
   # Sample data
   data = pd.DataFrame({
       'age': np.random.normal(35, 10, 1000),
       'income': np.random.lognormal(10, 0.5, 1000),
       'score': np.random.beta(2, 5, 1000) * 100
   })
   
   # Equal-width binning
   ew_binner = EqualWidthBinning(n_bins=5, preserve_dataframe=True)
   data_binned = ew_binner.fit_transform(data)
   
   # Equal-frequency binning for skewed data
   ef_binner = EqualFrequencyBinning(n_bins=4, preserve_dataframe=True)
   data_balanced = ef_binner.fit_transform(data)
   
   print(f"Original shape: {data.shape}")
   print(f"Binned unique values per column:")
   for col in data.columns:
       print(f"  {col}: {len(data_binned[col].unique())} bins")

🎯 **Advanced Example: Weight-Constrained Binning**
---------------------------------------------------

.. code-block:: python

   from binning.methods import EqualWidthMinimumWeightBinning
   
   # Customer data with importance weights
   customers = pd.DataFrame({
       'age': np.random.normal(40, 15, 2000),
       'spend': np.random.lognormal(8, 1, 2000),
       'loyalty': np.random.beta(3, 2, 2000) * 100
   })
   
   # Revenue-based importance weights
   revenue_weights = np.random.lognormal(6, 1.5, 2000)
   
   # Create segments ensuring minimum revenue per bin
   segmenter = EqualWidthMinimumWeightBinning(
       n_bins=6, 
       minimum_weight=1000.0,  # Minimum total revenue per segment
       preserve_dataframe=True
   )
   
   customer_segments = segmenter.fit_transform(
       customers, 
       guidance_data=revenue_weights
   )
   
   # Analyze segments
   for segment_id in sorted(customer_segments['age'].unique()):
       mask = customer_segments['age'] == segment_id
       segment_revenue = revenue_weights[mask].sum()
       segment_size = mask.sum()
       print(f"Segment {segment_id}: {segment_size} customers, "
             f"${segment_revenue:,.0f} revenue")

🛠️ **Scikit-learn Integration**
-------------------------------

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.ensemble import RandomForestClassifier
   from binning.methods import EqualFrequencyBinning
   
   # Create ML pipeline with binning
   pipeline = Pipeline([
       ('binning', EqualFrequencyBinning(n_bins=5)),
       ('classifier', RandomForestClassifier(n_estimators=100))
   ])
   
   # Train and predict
   pipeline.fit(X_train, y_train)
   y_pred = pipeline.predict(X_test)

📚 **Documentation**
--------------------

* 📖 **Full Documentation**: https://binning-framework.readthedocs.io/
* 🎓 **Tutorials**: https://binning-framework.readthedocs.io/en/latest/tutorials/
* 📋 **API Reference**: https://binning-framework.readthedocs.io/en/latest/api/
* 🔍 **Examples**: https://binning-framework.readthedocs.io/en/latest/examples/

🎯 **Use Cases**
----------------

**Data Preprocessing**
  * Feature engineering for machine learning
  * Noise reduction in continuous variables
  * Memory optimization through discretization

**Business Analytics**
  * Customer segmentation with revenue constraints
  * Risk scoring and credit analysis
  * Market research and survey analysis

**Scientific Computing**
  * Experimental data analysis
  * Statistical modeling preparation
  * Quality control in manufacturing

**Financial Applications**
  * Portfolio risk assessment
  * Trading signal generation
  * Regulatory compliance reporting

⚙️ **Supported Environments**
-----------------------------

**Python Versions**: 3.8, 3.9, 3.10, 3.11, 3.12

**Core Dependencies**:
  * NumPy >= 1.20.0
  * Pandas >= 1.3.0
  * Scikit-learn >= 1.0.0

**Optional Dependencies**:
  * Polars >= 0.15.0 (for Polars DataFrame support)
  * Matplotlib >= 3.5.0 (for examples and tutorials)

**Operating Systems**: Linux, macOS, Windows

🧪 **Development & Testing**
----------------------------

.. code-block:: bash

   # Clone repository
   git clone https://github.com/TheDAALab/binning.git
   cd binning
   
   # Install in development mode
   pip install -e .
   pip install -r requirements-dev.txt
   
   # Run tests with coverage
   pytest --cov=binning --cov-report=html
   
   # Build documentation
   cd docs && make html

🏆 **Quality Assurance**
------------------------

* ✅ **100% Test Coverage** - Comprehensive test suite
* ✅ **Type Hints** - Full type annotation support
* ✅ **Code Formatting** - Black and isort for consistent style  
* ✅ **Linting** - Flake8 for code quality
* ✅ **Documentation** - Comprehensive docs with examples
* ✅ **CI/CD** - Automated testing and deployment

🤝 **Contributing**
-------------------

We welcome contributions! Please see our `Contributing Guide <https://binning-framework.readthedocs.io/en/latest/contributing.html>`_ for details.

**Quick Contribution Steps**:

1. Fork the repository
2. Create a feature branch: ``git checkout -b feature/amazing-feature``
3. Make your changes and add tests
4. Ensure tests pass: ``pytest``
5. Submit a pull request

**Types of Contributions Welcome**:
  * 🐛 Bug reports and fixes
  * ✨ New binning algorithms
  * 📚 Documentation improvements
  * 🎯 Performance optimizations
  * 🧪 Additional test cases


🎓 **Research & Citations**
---------------------------

If you use this framework in academic research, please cite:

.. code-block:: bibtex

   @software{binning_framework,
     title={Binning Framework: Advanced Data Discretization for Python},
     author={TheDAALab},
     year={2025},
     url={https://github.com/TheDAALab/binning},
     version={1.0.0}
   }

📊 **Success Stories**
----------------------


🔮 **Roadmap**
--------------

**Upcoming Features**:
  * 🧠 Adaptive binning with automatic parameter selection
  * 🔄 Streaming data support for real-time applications  
  * 🎯 GPU acceleration for large-scale processing
  * 📊 Built-in visualization tools
  * 🌐 Distributed computing support (Dask integration)
  * 🔗 More supervised binning algorithms

**Version 1.1** (Q3 2025):
  * Entropy-based binning
  * Bayesian optimization for parameter tuning
  * Enhanced categorical data support

**Version 1.2** (Q4 2025):
  * Time series binning capabilities
  * Interactive visualization dashboard
  * Cloud deployment templates

📞 **Support & Community**
--------------------------

* 🐛 **Bug Reports**: `GitHub Issues <https://github.com/TheDAALab/binning/issues>`_
* 💬 **Discussions**: `GitHub Discussions <https://github.com/TheDAALab/binning/discussions>`_
* 📧 **Email**: binning-support@thedaalab.org
* 💼 **LinkedIn**: `TheDAALab <https://linkedin.com/company/thedaalab>`_
* 🐦 **Twitter**: `@TheDAALab <https://twitter.com/thedaalab>`_

⭐ **Star History**
-------------------

.. image:: https://api.star-history.com/svg?repos=TheDAALab/binning&type=Date
    :target: https://star-history.com/#TheDAALab/binning&Date

📄 **License**
--------------

This project is licensed under the MIT License - see the `LICENSE <https://github.com/TheDAALab/binning/blob/main/LICENSE>`_ file for details.

**MIT License Summary**:
  * ✅ Commercial use allowed
  * ✅ Modification allowed  
  * ✅ Distribution allowed
  * ✅ Private use allowed
  * ❌ No liability or warranty

---

**Made with ❤️ by TheDAALab**

*Empowering data scientists and researchers with advanced discretization tools for better insights and model performance.*

.. image:: https://img.shields.io/badge/Powered%20by-Python-blue.svg
    :target: https://www.python.org/
    :alt: Powered by Python

.. image:: https://img.shields.io/badge/Built%20with-NumPy-orange.svg
    :target: https://numpy.org/
    :alt: Built with NumPy

.. image:: https://img.shields.io/badge/Compatible%20with-Pandas-green.svg
    :target: https://pandas.pydata.org/
    :alt: Compatible with Pandas

.. image:: https://img.shields.io/badge/Integrates%20with-Scikit--learn-red.svg
    :target: https://scikit-learn.org/
    :alt: Integrates with Scikit-learn
