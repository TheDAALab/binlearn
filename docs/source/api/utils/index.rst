Utilities
=========

Utility functions and classes that support binning operations.

.. currentmodule:: binlearn.utils

.. toctree::
   :maxdepth: 1

   types
   errors
   data_handling
   parameter_conversion
   bin_operations

Quick Reference
---------------

Core utility modules:

**types**
    Type definitions and aliases used throughout binlearn

**errors**
    Custom exception classes with helpful error messages

**data_handling**
    Functions for preparing and processing input data

**parameter_conversion**
    Utilities for validating and converting parameters

**bin_operations**
    Low-level operations for working with bins and edges

Purpose and Usage
-----------------

The utilities module provides:

* **Type Safety**: Comprehensive type definitions for all data structures
* **Error Handling**: Clear, actionable error messages with suggestions
* **Data Processing**: Robust handling of pandas/polars DataFrames and numpy arrays
* **Validation**: Parameter validation with helpful feedback
* **Bin Operations**: Low-level functions for bin manipulation

These utilities are used internally by all binning methods to ensure
consistent behavior and comprehensive error handling.
