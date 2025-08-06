Error Handling
==============

.. currentmodule:: binlearn.utils.errors

Custom Exception Classes
------------------------

binlearn provides detailed error messages with actionable suggestions to help users
quickly identify and resolve issues.

.. autoclass:: BinningError
   :members:
   :show-inheritance:

.. autoclass:: ConfigurationError
   :members:
   :show-inheritance:

.. autoclass:: FittingError
   :members:
   :show-inheritance:

.. autoclass:: InvalidDataError
   :members:
   :show-inheritance:

.. autoclass:: DataQualityWarning
   :members:
   :show-inheritance:

Error Categories
----------------

**BinningError**
    Base exception for all binning-related errors

**ConfigurationError**
    Raised when parameters are invalid or incompatible

**FittingError**
    Raised when the fitting process encounters problems

**InvalidDataError**
    Raised when input data doesn't meet requirements

**DataQualityWarning**
    Warning for data quality issues that don't prevent operation

Error Message Features
---------------------

All binlearn errors include:

* **Clear Description**: What went wrong and why
* **Actionable Suggestions**: Specific steps to fix the problem
* **Context Information**: Relevant data about the error situation
* **Examples**: When helpful, examples of correct usage

Example Error Messages
---------------------

.. code-block:: python

   # ConfigurationError example
   try:
       binner = EqualWidthBinning(n_bins=0)
   except ConfigurationError as e:
       print(e)
       # Output:
       # n_bins must be a positive integer, got 0
       # Suggestions:
       # - Set n_bins to a positive integer (e.g., n_bins=5)
       # - Consider using 'auto' for automatic bin number selection

   # InvalidDataError example  
   try:
       binner.fit(data_with_all_nans)
   except InvalidDataError as e:
       print(e)
       # Output:
       # All values are NaN in column 'feature1'
       # Suggestions:
       # - Remove columns with all NaN values
       # - Impute missing values before binning
       # - Check data loading process

Best Practices
--------------

When handling binlearn errors:

1. **Read the full error message** - Contains specific guidance
2. **Check the suggestions** - Usually provide the exact fix needed
3. **Validate your data first** - Many errors stem from data quality issues
4. **Use appropriate parameters** - Error messages guide parameter selection
5. **Test with small datasets** - Easier to debug issues with smaller data
