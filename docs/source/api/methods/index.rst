Binning Methods
===============

All available binning and discretization methods organized by their base classes.

.. currentmodule:: binlearn.methods

Interval-Based Methods (Unsupervised)
--------------------------------------

These methods create interval bins through unsupervised analysis of the data distribution.

.. toctree::
   :maxdepth: 1

   equal_width_binning
   equal_frequency_binning
   kmeans_binning
   gaussian_mixture_binning
   dbscan_binning
   equal_width_minimum_weight_binning
   manual_interval_binning

Supervised Methods
------------------

These methods use target variable information to create optimal bins for prediction tasks.

.. toctree::
   :maxdepth: 1

   tree_binning
   chi2_binning
   isotonic_binning

Flexible Methods
----------------

These methods allow for custom bin specifications and handle discrete values.

.. toctree::
   :maxdepth: 1

   manual_flexible_binning
   singleton_binning

Quick Reference
---------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   EqualWidthBinning
   EqualFrequencyBinning  
   KMeansBinning
   GaussianMixtureBinning
   DBSCANBinning
   EqualWidthMinimumWeightBinning
   ManualIntervalBinning
   TreeBinning
   Chi2Binning
   IsotonicBinning
   ManualFlexibleBinning
   SingletonBinning
