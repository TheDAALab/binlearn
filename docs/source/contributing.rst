Contributing Guide
=================

We welcome contributions to the binlearn library! This guide explains how to contribute effectively.

Getting Started
---------------

Setting Up Development Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Fork and clone the repository:**

   .. code-block:: bash

      git clone https://github.com/yourusername/binning-framework.git
      cd binning-framework

2. **Create a virtual environment:**

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install in development mode:**

   .. code-block:: bash

      pip install -e .
      pip install -r requirements-dev.txt

4. **Install pre-commit hooks:**

   .. code-block:: bash

      pre-commit install

Types of Contributions
----------------------

Code Contributions
~~~~~~~~~~~~~~~~~~

**New Binning Methods**
   Implement new binning algorithms by extending base classes.

**Bug Fixes**
   Fix issues reported in GitHub issues or discovered during testing.

**Performance Improvements**
   Optimize existing code for better performance or memory usage.

**Test Coverage**
   Add tests for untested code paths or edge cases.

Documentation Contributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**API Documentation**
   Improve docstrings and API reference documentation.

**Tutorials and Examples**
   Create new tutorials or improve existing ones.

**User Guide**
   Enhance the user guide with better explanations and examples.

**Translation**
   Help translate documentation to other languages.

Issue Reports and Feature Requests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Bug Reports**
   Report bugs with clear reproduction steps and expected behavior.

**Feature Requests**
   Suggest new features with clear use cases and requirements.

**Documentation Issues**
   Report unclear or incorrect documentation.

Development Guidelines
----------------------

Code Style
~~~~~~~~~~

We follow PEP 8 with some modifications:

* **Line length**: 88 characters (Black formatter)
* **Import organization**: Use isort for consistent import ordering
* **Type hints**: Add type hints for all public APIs
* **Docstrings**: Use NumPy-style docstrings

**Formatting Tools:**

.. code-block:: bash

   # Format code with Black
   black binning/
   
   # Sort imports with isort
   isort binning/
   
   # Lint with flake8
   flake8 binning/

Testing Guidelines
~~~~~~~~~~~~~~~~~~

**Test Coverage**
   All new code must have >= 95% test coverage.

**Test Structure**
   Organize tests in ``tests/`` directory mirroring the source structure.

**Test Naming**
   Use descriptive test names: ``test_method_behavior_condition``

**Running Tests:**

.. code-block:: bash

   # Run all tests
   pytest
   
   # Run with coverage
   pytest --cov=binning --cov-report=html
   
   # Run specific test file
   pytest tests/methods/test_equal_width_binning.py

Documentation Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~

**Docstring Format:**
   Use NumPy-style docstrings for all public APIs.

**Code Examples:**
   Include runnable code examples in docstrings and documentation.

**Type Information:**
   Document parameter and return types clearly.

**Building Documentation:**

.. code-block:: bash

   cd docs
   make html
   # Open build/html/index.html in browser

Creating New Binning Methods
-----------------------------

Step-by-Step Guide
~~~~~~~~~~~~~~~~~~

1. **Choose the appropriate base class:**

   * ``UnsupervisedBinner`` - No target variable needed
   * ``SupervisedBinner`` - Uses target variable
   * ``GuidedBinner`` - Uses additional guidance data

2. **Create the implementation file:**

   .. code-block:: bash

      # Create in binning/methods/
      touch binning/methods/_my_new_binning.py

3. **Implement the class:**

   .. code-block:: python

      from binlearn.base import UnsupervisedBinner
      import numpy as np
      
      class MyNewBinning(UnsupervisedBinner):
          """Brief description of the binning method.
          
          Longer description explaining the algorithm,
          use cases, and behavior.
          
          Parameters
          ----------
          n_bins : int, default=5
              Number of bins to create.
          custom_param : float, default=1.0
              Custom parameter specific to this method.
              
          Examples
          --------
          >>> from binlearn.methods import MyNewBinning
          >>> binner = MyNewBinning(n_bins=4)
          >>> X_binned = binner.fit_transform(X)
          """
          
          def __init__(self, n_bins=5, custom_param=1.0, **kwargs):
              super().__init__(**kwargs)
              self.n_bins = n_bins
              self.custom_param = custom_param
          
          def _fit_column(self, column_data, column_index, **kwargs):
              """Fit binning parameters for a single column."""
              # Implementation here
              pass
          
          def _transform_column(self, column_data, column_index):
              """Transform a single column using fitted parameters."""
              # Implementation here
              pass

4. **Add to package exports:**

   .. code-block:: python

      # In binning/methods/__init__.py
      from ._my_new_binning import MyNewBinning
      
      __all__ = [
          # ... existing exports ...
          'MyNewBinning',
      ]

5. **Create comprehensive tests:**

   .. code-block:: python

      # In tests/methods/test_my_new_binning.py
      import pytest
      import numpy as np
      from binlearn.methods import MyNewBinning
      
      class TestMyNewBinning:
          """Test suite for MyNewBinning."""
          
          def test_basic_functionality(self):
              """Test basic binning functionality."""
              # Test implementation
              pass
          
          def test_edge_cases(self):
              """Test edge cases and error conditions."""
              # Test implementation
              pass

6. **Add API documentation:**

   .. code-block:: rst

      # In docs/source/api/methods/my_new_binning.rst
      MyNewBinning
      ============
      
      .. autoclass:: binning.methods.MyNewBinning
         :members:
         :inherited-members:
         :show-inheritance:

Pull Request Process
--------------------

Preparing Your Pull Request
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Create a feature branch:**

   .. code-block:: bash

      git checkout -b feature/my-new-feature

2. **Make your changes following the guidelines above**

3. **Run the full test suite:**

   .. code-block:: bash

      pytest --cov=binning --cov-report=term-missing

4. **Check code formatting:**

   .. code-block:: bash

      black --check binning/
      isort --check-only binning/
      flake8 binning/

5. **Update documentation if needed**

6. **Commit with clear messages:**

   .. code-block:: bash

      git commit -m "Add MyNewBinning method with custom algorithm"

Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~~~~

**Title Format:**
   Use clear, descriptive titles: "Add support for X" or "Fix bug in Y"

**Description Content:**
   
   * Summarize the changes made
   * Reference any related issues
   * Describe testing performed
   * Note any breaking changes

**Review Process:**
   
   * Ensure all CI checks pass
   * Address reviewer feedback promptly
   * Keep the PR focused on a single feature/fix
   * Update documentation for user-facing changes

Example Pull Request Template
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: markdown

   ## Description
   Brief description of changes made.
   
   ## Type of Change
   - [ ] Bug fix (non-breaking change which fixes an issue)
   - [ ] New feature (non-breaking change which adds functionality)
   - [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
   - [ ] Documentation update
   
   ## Testing
   - [ ] All existing tests pass
   - [ ] New tests added for new functionality
   - [ ] Test coverage maintained/improved
   
   ## Checklist
   - [ ] Code follows project style guidelines
   - [ ] Self-review of code completed
   - [ ] Code is commented, particularly in hard-to-understand areas
   - [ ] Documentation updated where necessary
   - [ ] No new warnings introduced

Reporting Issues
----------------

Bug Report Template
~~~~~~~~~~~~~~~~~~~

When reporting bugs, include:

**Environment Information:**
   * Python version
   * Package version
   * Operating system
   * Other relevant dependencies

**Reproduction Steps:**
   Clear, minimal steps to reproduce the issue

**Expected Behavior:**
   What you expected to happen

**Actual Behavior:**
   What actually happened

**Code Example:**
   Minimal code example that demonstrates the issue

Feature Request Template
~~~~~~~~~~~~~~~~~~~~~~~~

For feature requests, include:

**Use Case:**
   Describe the problem you're trying to solve

**Proposed Solution:**
   Your ideas for how it could be implemented

**Alternatives Considered:**
   Other approaches you've considered

**Additional Context:**
   Any other relevant information

Getting Help
------------

**Documentation:**
   Check existing documentation and examples first

**GitHub Issues:**
   Search existing issues before creating new ones

**Discussions:**
   Use GitHub Discussions for questions and general discussion

**Code Review:**
   All contributions go through code review for quality assurance

Recognition
-----------

Contributors are recognized in:

* ``CONTRIBUTORS.md`` file in the repository
* Release notes for significant contributions
* Documentation credits where appropriate

We appreciate all contributions, whether code, documentation, testing, or community support!

Development Resources
---------------------

**Useful Commands:**

.. code-block:: bash

   # Run tests with coverage
   pytest --cov=binning --cov-report=html
   
   # Build documentation locally
   cd docs && make html
   
   # Format code
   black binning/ && isort binning/
   
   # Run linting
   flake8 binning/
   
   # Check type hints
   mypy binning/

**Configuration Files:**
   
   * ``.pre-commit-config.yaml`` - Pre-commit hooks
   * ``pyproject.toml`` - Project configuration
   * ``setup.cfg`` - Tool configurations
   * ``pytest.ini`` - Test configuration

Thank you for contributing to the binlearn library!
