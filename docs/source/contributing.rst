Contributing
============

We welcome contributions to binlearn! This guide will help you get started with contributing to the project.

Ways to Contribute
------------------

There are many ways you can contribute to binlearn:

🐛 **Bug Reports**
  * Report issues you encounter
  * Provide clear reproduction steps
  * Include system information and error messages

✨ **Feature Requests**
  * Suggest new binning algorithms
  * Request API improvements
  * Propose new integrations

📚 **Documentation**
  * Improve existing documentation
  * Add examples and tutorials
  * Fix typos and clarify explanations

🧪 **Code Contributions**
  * Fix bugs
  * Implement new features
  * Improve performance
  * Add test cases

Getting Started
---------------

Development Setup
~~~~~~~~~~~~~~~~~

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:

   .. code-block:: bash

      git clone https://github.com/YOUR_USERNAME/binlearn.git
      cd binlearn

3. **Set up development environment**:

   .. code-block:: bash

      # Create virtual environment
      python -m venv binlearn_dev
      source binlearn_dev/bin/activate  # On Windows: binlearn_dev\Scripts\activate
      
      # Install in development mode with all dependencies
      pip install -e ".[tests,dev,pandas,polars]"

4. **Verify setup**:

   .. code-block:: bash

      # Run tests
      pytest
      
      # Check code quality
      ruff check binlearn/
      mypy binlearn/ --ignore-missing-imports

Development Workflow
~~~~~~~~~~~~~~~~~~~~

1. **Create a feature branch**:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

2. **Make your changes** following our coding standards
3. **Add tests** for new functionality
4. **Run quality checks**:

   .. code-block:: bash

      # Run all tests
      pytest
      
      # Check code formatting and linting
      ruff check binlearn/
      ruff format binlearn/
      
      # Type checking
      mypy binlearn/ --ignore-missing-imports

5. **Commit your changes**:

   .. code-block:: bash

      git add .
      git commit -m "Add your descriptive commit message"

6. **Push to your fork**:

   .. code-block:: bash

      git push origin feature/your-feature-name

7. **Create a Pull Request** on GitHub

Coding Standards
----------------

Code Quality
~~~~~~~~~~~~

We maintain high code quality standards:

- **100% ruff compliance** - All code must pass ruff linting and formatting
- **100% mypy compliance** - Complete type annotations required
- **100% test coverage** - All new code must include comprehensive tests

Code Style
~~~~~~~~~~

- Follow **PEP 8** style guidelines
- Use **descriptive variable names**
- Write **clear docstrings** for all public functions and classes
- Keep functions **focused and small**
- Use **type hints** for all function signatures

Example of good code style:

.. code-block:: python

   from typing import Optional, Tuple
   import numpy as np
   from ..utils.types import ArrayLike, BinEdges

   def calculate_bin_edges(
       data: ArrayLike,
       n_bins: int,
       bin_range: Optional[Tuple[float, float]] = None
   ) -> BinEdges:
       """Calculate equal-width bin edges for the given data.
       
       Args:
           data: Input data array
           n_bins: Number of bins to create
           bin_range: Optional custom range for binning
           
       Returns:
           Array of bin edges
           
       Raises:
           ValueError: If n_bins is not positive
       """
       if n_bins <= 0:
           raise ValueError(f"n_bins must be positive, got {n_bins}")
           
       # Implementation here...
       return edges

Documentation Standards
~~~~~~~~~~~~~~~~~~~~~~~

- Use **Google-style docstrings**
- Include **clear examples** in docstrings
- Document **all parameters and return values**
- Explain **complex algorithms** with comments
- Update **relevant documentation files**

Testing Guidelines
------------------

Test Structure
~~~~~~~~~~~~~~

Tests are organized in the ``tests/`` directory, mirroring the source structure:

.. code-block:: text

   tests/
   ├── test_config.py
   ├── base/
   │   ├── test_general_binning_base.py
   │   └── ...
   ├── methods/
   │   ├── test_equal_width_binning.py
   │   └── ...
   └── utils/
       └── ...

Writing Tests
~~~~~~~~~~~~~

Follow these guidelines when writing tests:

.. code-block:: python

   import pytest
   import numpy as np
   from binlearn.methods import EqualWidthBinning
   from binlearn.utils.errors import ConfigurationError

   class TestEqualWidthBinning:
       """Test suite for EqualWidthBinning."""
       
       def test_basic_functionality(self):
           """Test basic binning functionality."""
           X = np.random.rand(100, 3)
           binner = EqualWidthBinning(n_bins=5)
           X_binned = binner.fit_transform(X)
           
           assert X_binned.shape == X.shape
           assert len(binner.bin_edges_) == X.shape[1]
       
       def test_invalid_n_bins(self):
           """Test error handling for invalid n_bins."""
           with pytest.raises(ConfigurationError):
               EqualWidthBinning(n_bins=0)
       
       @pytest.mark.parametrize("n_bins", [1, 3, 5, 10])
       def test_different_n_bins(self, n_bins):
           """Test binning with different numbers of bins."""
           X = np.random.rand(50, 2)
           binner = EqualWidthBinning(n_bins=n_bins)
           X_binned = binner.fit_transform(X)
           
           # Check that all values are in expected range
           assert np.all(X_binned >= 0)
           assert np.all(X_binned < n_bins)

Running Tests
~~~~~~~~~~~~~

.. code-block:: bash

   # Run all tests
   pytest
   
   # Run tests with coverage
   pytest --cov=binlearn --cov-report=html
   
   # Run specific test file
   pytest tests/methods/test_equal_width_binning.py
   
   # Run tests matching pattern
   pytest -k "test_basic"

Pull Request Guidelines
-----------------------

Before Submitting
~~~~~~~~~~~~~~~~~

Ensure your pull request meets these requirements:

1. **All tests pass**: ``pytest`` returns no failures
2. **Code quality**: ``ruff check binlearn/`` passes
3. **Type checking**: ``mypy binlearn/ --ignore-missing-imports`` passes
4. **Documentation**: Updated relevant docs and docstrings
5. **Changelog**: Added entry to appropriate section

Pull Request Description
~~~~~~~~~~~~~~~~~~~~~~~~

Include in your PR description:

- **Summary** of changes made
- **Motivation** for the changes
- **Type of change**: bug fix, new feature, documentation, etc.
- **Testing** performed
- **Checklist** of requirements met

Example PR template:

.. code-block:: text

   ## Summary
   
   Adds a new KMeansBinning method that uses K-means clustering to determine optimal bin boundaries.
   
   ## Motivation
   
   Users requested a clustering-based binning method for data with natural groupings.
   
   ## Type of Change
   
   - [x] New feature
   - [ ] Bug fix
   - [ ] Documentation update
   - [ ] Performance improvement
   
   ## Testing
   
   - Added comprehensive test suite with 95% coverage
   - Tested with various data distributions
   - Verified sklearn compatibility
   
   ## Checklist
   
   - [x] All tests pass
   - [x] Code follows style guidelines
   - [x] Self-review completed
   - [x] Documentation updated
   - [x] Changelog entry added

Release Process
---------------

Version Numbering
~~~~~~~~~~~~~~~~~

We use semantic versioning (SemVer):

- **Major** (X.0.0): Breaking changes
- **Minor** (0.X.0): New features, backward compatible
- **Patch** (0.0.X): Bug fixes, backward compatible

Creating Releases
~~~~~~~~~~~~~~~~~

1. Update version in ``binlearn/_version.py``
2. Update ``CHANGELOG.md`` with release notes
3. Create release tag: ``git tag -a v1.2.3 -m "Release v1.2.3"``
4. Push tag: ``git push origin v1.2.3``
5. Create GitHub release from tag

Communication
-------------

Getting Help
~~~~~~~~~~~~

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: For security issues or private matters

Community Guidelines
~~~~~~~~~~~~~~~~~~~~

- Be **respectful** and **inclusive**
- **Help others** learn and contribute
- **Stay on topic** in discussions
- **Follow** the project's code of conduct

Recognition
-----------

Contributors are recognized in:

- **README.md**: Major contributors listed
- **Release notes**: Contributions acknowledged
- **GitHub insights**: Contribution history visible

Thank you for contributing to binlearn! 🚀
