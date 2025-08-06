Integration Tools
=================

Tools for integrating binning methods with machine learning workflows.

.. currentmodule:: binlearn.tools

.. toctree::
   :maxdepth: 1

   integration

Quick Reference
---------------

.. autosummary::
   :toctree: generated/

   BinningFeatureSelector
   BinningPipeline
   make_binning_scorer

Integration Classes
-------------------

**BinningFeatureSelector**
    Feature selector that uses binning-based importance measures

**BinningPipeline**
    Specialized pipeline for binning workflows with enhanced functionality

**make_binning_scorer**
    Creates scoring functions that account for binning transformations

Purpose and Usage
-----------------

The tools module provides:

* **Pipeline Integration**: Seamless integration with scikit-learn pipelines
* **Feature Selection**: Binning-aware feature selection methods  
* **Performance Evaluation**: Scoring functions optimized for binned data
* **Workflow Utilities**: Tools for common binning workflow patterns

These tools make it easier to incorporate binning into complex machine learning
workflows and evaluate the impact of different binning strategies on model performance.
