Base Classes
============

Foundation classes that provide core functionality for all binning methods.

.. currentmodule:: binlearn.base

.. toctree::
   :maxdepth: 1

   general_binning_base
   interval_binning_base
   flexible_binning_base
   supervised_binning_base

Quick Reference
---------------

.. autosummary::
   :toctree: generated/

   GeneralBinningBase
   IntervalBinningBase
   FlexibleBinningBase
   SupervisedBinningBase

Base Class Hierarchy
--------------------

The binlearn library uses a hierarchical inheritance structure:

1. **GeneralBinningBase** - Root base class for all binning methods
   
   * Provides core transformer interface (fit/transform)
   * Handles DataFrame preservation and column management
   * Implements common validation and error handling

2. **IntervalBinningBase** - Base for interval-based methods
   
   * Extends GeneralBinningBase with interval-specific functionality
   * Handles bin edges, representatives, and clipping
   * Used by: EqualWidthBinning, EqualFrequencyBinning, KMeansBinning, etc.

3. **FlexibleBinningBase** - Base for flexible binning methods
   
   * Extends GeneralBinningBase with mixed bin type support
   * Handles both singleton and interval bins
   * Used by: SingletonBinning, ManualFlexibleBinning

4. **SupervisedBinningBase** - Base for supervised methods
   
   * Extends IntervalBinningBase with target-aware functionality
   * Handles guidance data and supervised learning integration
   * Used by: SupervisedBinning, Chi2Binning, IsotonicBinning

Purpose and Usage
-----------------

These base classes provide:

* **Consistent Interface**: All binning methods share the same API
* **Code Reuse**: Common functionality implemented once
* **Type Safety**: Proper inheritance and method signatures
* **Extensibility**: Easy to add new binning methods
* **Validation**: Comprehensive parameter and data validation

Users typically don't interact with base classes directly, but they provide
the foundation that makes all binning methods work consistently together.
