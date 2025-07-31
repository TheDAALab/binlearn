Tutorials
=========

Learn how to use the Binning Framework effectively with these comprehensive tutorials.

Getting Started
---------------

If you're new to the Binning Framework, start with these foundational tutorials:

.. toctree::
   :maxdepth: 1

   basic_binning
   advanced_binning

Basic Usage
-----------

:doc:`basic_binning`
   Learn the fundamentals of binning with practical examples. This tutorial covers:
   
   * Understanding binning concepts and when to use different methods
   * Working with EqualWidthBinning for uniform bin sizes
   * Using EqualFrequencyBinning for balanced populations
   * Applying EqualWidthMinimumWeightBinning with weight constraints
   * Integrating SupervisedBinning for classification tasks
   * Handling missing values and edge cases
   * Pipeline integration with scikit-learn

Advanced Techniques
-------------------

:doc:`advanced_binning`
   Master sophisticated binning strategies for complex scenarios:
   
   * Creating custom binning methods by extending base classes
   * Multi-stage binning workflows for complex preprocessing
   * Adaptive binning with automatic parameter selection
   * Handling mixed data types (numeric and categorical)
   * Time series binning with temporal awareness
   * Memory-efficient processing for large datasets
   * Parallel binning for performance optimization
   * Quality evaluation and cross-validation techniques

Tutorial Progression
--------------------

The tutorials are designed to build upon each other:

1. **Start with Basic Binning** to understand core concepts
2. **Progress to Advanced Techniques** for specialized scenarios
3. **Apply concepts** to your own datasets and use cases

Code Examples
-------------

All tutorials include complete, runnable code examples that you can:

* Copy and run in your own environment
* Modify for your specific datasets
* Use as templates for your own binning workflows

Prerequisites
-------------

The tutorials assume basic familiarity with:

* Python programming
* NumPy and pandas for data manipulation
* Basic machine learning concepts
* Scikit-learn for pipelines (for advanced topics)

Running the Examples
--------------------

To run the tutorial examples, make sure you have the required dependencies:

.. code-block:: bash

   pip install binning-framework numpy pandas matplotlib scikit-learn

Some advanced examples may require additional packages:

.. code-block:: bash

   pip install scipy polars  # For advanced tutorials

Next Steps
----------

After completing the tutorials, explore:

* :doc:`../examples/equal_width_minimum_weight_binning` for detailed implementation examples
* :doc:`../api/index` for complete API documentation
* :doc:`../user_guide` for comprehensive usage patterns
* Real-world applications with your own datasets
