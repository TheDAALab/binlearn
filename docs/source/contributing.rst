Contributing to binlearn
========================

We welcome contributions to binlearn! This guide will help you get started with contributing code, documentation, or reporting issues.

Getting Started
---------------

1. **Fork the Repository**
   
   Fork the `binlearn repository <https://github.com/TheDAALab/binlearn>`_ on GitHub.

2. **Clone Your Fork**
   
   .. code-block:: bash
   
      git clone https://github.com/yourusername/binlearn.git
      cd binlearn

3. **Set Up Development Environment**
   
   .. code-block:: bash
   
      # Create virtual environment
      python -m venv venv
      source venv/bin/activate  # On Windows: venv\\Scripts\\activate
      
      # Install in development mode
      pip install -e ".[dev]"

4. **Create a Branch**
   
   .. code-block:: bash
   
      git checkout -b feature/your-feature-name

Development Workflow
--------------------

Code Standards
^^^^^^^^^^^^^^

binlearn follows strict code quality standards:

* **Type Hints**: All functions must have complete type annotations
* **Docstrings**: Use NumPy-style docstrings for all public functions
* **Code Style**: We use `ruff` for linting and formatting
* **Testing**: 100% test coverage is required

Run quality checks:

.. code-block:: bash

   # Format code
   python -m ruff format .
   
   # Check linting
   python -m ruff check .
   
   # Type checking
   python -m mypy binlearn/
   
   # Run tests
   python -m pytest tests/ -v

Testing
^^^^^^^

Write comprehensive tests for all new features:

.. code-block:: python

   # tests/methods/test_your_method.py
   import numpy as np
   import pytest
   from binlearn.methods import YourMethod
   
   class TestYourMethod:
       def test_basic_functionality(self):
           X = np.random.rand(100, 2)
           method = YourMethod(n_bins=5)
           X_transformed = method.fit_transform(X)
           
           assert X_transformed.shape == X.shape
           assert len(np.unique(X_transformed[:, 0])) <= 5
       
       def test_edge_cases(self):
           # Test with NaN values
           X_with_nan = np.array([[1], [np.nan], [3]])
           method = YourMethod(n_bins=2)
           result = method.fit_transform(X_with_nan)
           assert np.isnan(result[1, 0])

Run specific tests:

.. code-block:: bash

   # Run all tests
   python -m pytest
   
   # Run specific test file
   python -m pytest tests/methods/test_your_method.py
   
   # Run with coverage
   python -m pytest --cov=binlearn tests/

Documentation
^^^^^^^^^^^^^

Update documentation for any new features:

1. **Docstrings**: Follow NumPy style
   
   .. code-block:: python
   
      def your_function(X: np.ndarray, n_bins: int = 5) -> np.ndarray:
          """Brief description of the function.
          
          Parameters
          ----------
          X : np.ndarray of shape (n_samples, n_features)
              Input data to be binned.
          n_bins : int, default=5
              Number of bins to create.
              
          Returns
          -------
          np.ndarray of shape (n_samples, n_features)
              Binned data.
              
          Examples
          --------
          >>> import numpy as np
          >>> X = np.random.rand(10, 1)
          >>> result = your_function(X, n_bins=3)
          >>> result.shape
          (10, 1)
          """

2. **Update Documentation Files**: Add examples and usage guides

3. **Build Documentation Locally**:
   
   .. code-block:: bash
   
      cd docs/
      python -m sphinx -b html source _build/html

Types of Contributions
----------------------

Bug Reports
^^^^^^^^^^^

When reporting bugs, please include:

* **Description**: Clear description of the issue
* **Reproduction Steps**: Minimal code to reproduce the bug
* **Environment**: Python version, binlearn version, OS
* **Expected vs Actual**: What you expected vs what happened

Feature Requests
^^^^^^^^^^^^^^^^

For new features, please:

* **Describe the Use Case**: Why is this feature needed?
* **Proposed API**: How should the feature work?
* **Examples**: Show expected usage patterns
* **Implementation Ideas**: Any thoughts on implementation

Code Contributions
^^^^^^^^^^^^^^^^^^

**New Binning Methods**

To add a new binning method:

1. Create the method class in `binlearn/methods/`
2. Inherit from appropriate base class
3. Implement required methods: `fit()`, `transform()`, `_validate_parameters()`
4. Add comprehensive tests
5. Update documentation

Example structure:

.. code-block:: python

   from binlearn.base import IntervalBinningBase
   
   class YourBinningMethod(IntervalBinningBase):
       def __init__(self, n_bins: int = 5, **kwargs):
           super().__init__(n_bins=n_bins, **kwargs)
       
       def _validate_parameters(self) -> None:
           if self.n_bins < 1:
               raise ValueError("n_bins must be positive")
       
       def _fit_feature(self, X_feature: np.ndarray) -> np.ndarray:
           # Your binning logic here
           bin_edges = self._compute_bin_edges(X_feature)
           return bin_edges

**Utility Functions**

Add utility functions to appropriate modules in `binlearn/utils/`:

* `data_handling.py`: Data validation and conversion
* `bin_operations.py`: Core binning operations
* `inspection.py`: Analysis and visualization tools

**Base Class Improvements**

Improvements to base classes should:

* Maintain backward compatibility
* Include comprehensive tests
* Update all derived classes if needed

Documentation Contributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Help improve documentation:

* **Fix Typos**: Simple but valuable!
* **Add Examples**: Real-world usage examples
* **Improve Clarity**: Rewrite confusing sections
* **Add Tutorials**: Step-by-step guides

Review Process
--------------

1. **Automated Checks**: All PRs must pass CI checks
   
   * Code formatting (ruff)
   * Type checking (mypy)
   * Tests (pytest)
   * Coverage (100% required)

2. **Manual Review**: Core maintainers review:
   
   * Code quality and design
   * Test coverage and quality
   * Documentation completeness
   * API consistency

3. **Feedback**: Address reviewer feedback promptly

4. **Merge**: Once approved, changes are merged

Release Process
---------------

binlearn follows semantic versioning:

* **Patch** (0.1.1): Bug fixes
* **Minor** (0.2.0): New features, backward compatible
* **Major** (1.0.0): Breaking changes

Code of Conduct
---------------

We are committed to providing a welcoming and inclusive environment. Please:

* **Be Respectful**: Treat all contributors with respect
* **Be Collaborative**: Work together constructively
* **Be Patient**: Remember that everyone is learning
* **Give Credit**: Acknowledge others' contributions

Communication
-------------

* **GitHub Issues**: Bug reports and feature requests
* **GitHub Discussions**: Questions and general discussion
* **Pull Requests**: Code and documentation contributions

Getting Help
------------

If you need help contributing:

1. Check existing issues and documentation
2. Create a GitHub Discussion
3. Reach out to maintainers

We appreciate all contributions, no matter how small! Every bug report, documentation fix, and feature addition makes binlearn better for everyone.

Development Tips
----------------

**Setting Up IDE**

For VS Code, use these settings:

.. code-block:: json

   {
       "python.linting.enabled": true,
       "python.linting.mypyEnabled": true,
       "python.formatting.provider": "ruff",
       "python.linting.ruffEnabled": true
   }

**Common Patterns**

Follow these patterns when adding new features:

.. code-block:: python

   # Always validate inputs
   def your_method(self, X, y=None):
       X = self._validate_data(X)
       
       # Use consistent naming
       self.bin_edges_ = self._compute_bin_edges(X)
       self.n_features_in_ = X.shape[1]
       
       return self

**Performance Guidelines**

* Use NumPy operations when possible
* Avoid Python loops for large data
* Profile performance-critical code
* Consider memory usage for large datasets

Thank you for contributing to binlearn! 🎉
