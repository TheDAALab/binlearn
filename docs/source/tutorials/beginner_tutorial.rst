Beginner Tutorial: Your First Binning Project
==============================================

Welcome to binlearn! This tutorial will walk you through your first data binning project, covering the fundamentals you need to get started.

What is Data Binning?
---------------------

Data binning (also called discretization) is the process of converting continuous numerical data into discrete intervals or categories. This is useful for:

- **Reducing noise** in your data
- **Simplifying complex relationships** 
- **Making data compatible** with algorithms that require categorical input
- **Creating interpretable features** for analysis
- **Improving model performance** in some cases

Let's see this in action!

Setting Up Your Environment
----------------------------

First, let's import the libraries we'll need:

.. code-block:: python

   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   from binlearn import EqualWidthBinning, EqualFrequencyBinning
   
   # Set random seed for reproducibility
   np.random.seed(42)

Creating Sample Data
--------------------

Let's create some sample data that represents customer information:

.. code-block:: python

   # Generate sample customer data
   n_customers = 1000
   
   customer_data = pd.DataFrame({
       'age': np.random.normal(40, 15, n_customers),           # Age in years
       'income': np.random.lognormal(10.5, 0.8, n_customers), # Annual income
       'spending_score': np.random.beta(2, 5, n_customers) * 100, # Spending score 0-100
       'account_balance': np.random.exponential(5000, n_customers)  # Account balance
   })
   
   # Clean up negative ages
   customer_data['age'] = np.maximum(customer_data['age'], 18)
   
   print("Sample data:")
   print(customer_data.head())
   print(f"\nData shape: {customer_data.shape}")
   print(f"\nData types:\n{customer_data.dtypes}")

Let's visualize our data to understand its distribution:

.. code-block:: python

   # Create histograms for each feature
   fig, axes = plt.subplots(2, 2, figsize=(12, 10))
   axes = axes.ravel()
   
   for i, column in enumerate(customer_data.columns):
       axes[i].hist(customer_data[column], bins=30, alpha=0.7, edgecolor='black')
       axes[i].set_title(f'Distribution of {column}')
       axes[i].set_xlabel(column)
       axes[i].set_ylabel('Frequency')
   
   plt.tight_layout()
   plt.show()

Your First Binning: Equal-Width Binning
----------------------------------------

Let's start with the simplest binning method - equal-width binning. This divides the range of each feature into bins of equal width.

.. code-block:: python

   # Create an equal-width binner
   ew_binner = EqualWidthBinning(
       n_bins=5,                    # Create 5 bins for each feature
       preserve_dataframe=True      # Keep the DataFrame format
   )
   
   # Fit the binner to our data and transform it
   customer_data_binned = ew_binner.fit_transform(customer_data)
   
   print("Binned data:")
   print(customer_data_binned.head())
   print(f"\nBinned data shape: {customer_data_binned.shape}")

Understanding the Results
~~~~~~~~~~~~~~~~~~~~~~~~~

Let's examine what the binner learned:

.. code-block:: python

   # Check the bin edges for each feature
   print("Bin edges for each feature:")
   for feature, edges in ew_binner.bin_edges_.items():
       print(f"{feature}: {edges}")
   
   # Look at the range of binned values
   print("\nRange of binned values:")
   for column in customer_data_binned.columns:
       unique_values = sorted(customer_data_binned[column].unique())
       print(f"{column}: {unique_values}")

Visualizing the Binning Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's create a comparison plot to see the effect of binning:

.. code-block:: python

   # Create comparison plots
   fig, axes = plt.subplots(2, 4, figsize=(16, 8))
   
   for i, column in enumerate(customer_data.columns):
       # Original data
       axes[0, i].hist(customer_data[column], bins=30, alpha=0.7, 
                      edgecolor='black', color='blue')
       axes[0, i].set_title(f'Original {column}')
       axes[0, i].set_xlabel(column)
       axes[0, i].set_ylabel('Frequency')
       
       # Binned data
       axes[1, i].hist(customer_data_binned[column], bins=20, alpha=0.7, 
                      edgecolor='black', color='red')
       axes[1, i].set_title(f'Binned {column}')
       axes[1, i].set_xlabel(f'{column} (binned)')
       axes[1, i].set_ylabel('Frequency')
   
   plt.tight_layout()
   plt.show()

Trying Equal-Frequency Binning
-------------------------------

Equal-width binning can sometimes create bins with very different numbers of samples. Let's try equal-frequency binning, which creates bins with approximately equal numbers of samples:

.. code-block:: python

   # Create an equal-frequency binner
   ef_binner = EqualFrequencyBinning(
       n_bins=5,
       preserve_dataframe=True
   )
   
   # Fit and transform
   customer_data_eq_freq = ef_binner.fit_transform(customer_data)
   
   print("Equal-frequency binned data:")
   print(customer_data_eq_freq.head())

Comparing the Methods
~~~~~~~~~~~~~~~~~~~~~

Let's compare how the two methods distribute the samples:

.. code-block:: python

   # Compare bin distributions for the 'income' feature
   print("Sample distribution comparison for 'income' feature:")
   print("\nEqual-Width Binning:")
   ew_counts = customer_data_binned['income'].value_counts().sort_index()
   for bin_val, count in ew_counts.items():
       print(f"  Bin {bin_val}: {count} samples")
   
   print("\nEqual-Frequency Binning:")
   ef_counts = customer_data_eq_freq['income'].value_counts().sort_index()
   for bin_val, count in ef_counts.items():
       print(f"  Bin {bin_val}: {count} samples")

Working with Individual Features
--------------------------------

Sometimes you might want to bin only specific features:

.. code-block:: python

   # Bin only age and income
   selected_data = customer_data[['age', 'income']]
   
   binner = EqualWidthBinning(n_bins=3, preserve_dataframe=True)
   selected_binned = binner.fit_transform(selected_data)
   
   print("Binning only selected features:")
   print(selected_binned.head())

Custom Bin Ranges
------------------

You can also specify custom ranges for binning:

.. code-block:: python

   # Create a binner with custom range for age (18-80 years)
   custom_binner = EqualWidthBinning(
       n_bins=4,
       bin_range=(18, 80),  # Custom range for all features
       preserve_dataframe=True
   )
   
   # Apply only to age column
   age_data = customer_data[['age']]
   age_binned = custom_binner.fit_transform(age_data)
   
   print("Custom range binning for age:")
   print(f"Bin edges: {custom_binner.bin_edges_['age']}")
   print(f"Unique binned values: {sorted(age_binned['age'].unique())}")

Handling Missing Values and Outliers
------------------------------------

binlearn provides robust handling of missing values and outliers:

.. code-block:: python

   # Create data with some outliers and missing values
   noisy_data = customer_data.copy()
   
   # Add some outliers
   noisy_data.loc[0, 'income'] = 1000000  # Very high income
   noisy_data.loc[1, 'age'] = 150         # Impossible age
   
   # Add missing values
   noisy_data.loc[2:4, 'spending_score'] = np.nan
   
   print("Data with noise:")
   print(noisy_data.head())
   
   # Bin the noisy data
   robust_binner = EqualWidthBinning(
       n_bins=5,
       clip=True,  # Clip outliers to bin edges
       preserve_dataframe=True
   )
   
   try:
       noisy_binned = robust_binner.fit_transform(noisy_data)
       print("\nSuccessfully binned noisy data:")
       print(noisy_binned.head())
   except Exception as e:
       print(f"Error handling noisy data: {e}")

Saving and Loading Binners
---------------------------

You can save trained binners for later use:

.. code-block:: python

   import pickle
   
   # Train a binner
   production_binner = EqualWidthBinning(n_bins=5, preserve_dataframe=True)
   production_binner.fit(customer_data)
   
   # Save the trained binner
   with open('customer_binner.pkl', 'wb') as f:
       pickle.dump(production_binner, f)
   
   # Load and use the binner
   with open('customer_binner.pkl', 'rb') as f:
       loaded_binner = pickle.load(f)
   
   # Use the loaded binner on new data
   new_customer = pd.DataFrame({
       'age': [35],
       'income': [50000],
       'spending_score': [65],
       'account_balance': [3000]
   })
   
   new_customer_binned = loaded_binner.transform(new_customer)
   print("Binned new customer data:")
   print(new_customer_binned)

Next Steps
----------

Congratulations! You've completed your first binning project. You've learned how to:

- Create and apply equal-width and equal-frequency binning
- Visualize binning results
- Handle noisy data
- Save and load trained binners
- Work with DataFrames and individual features

**What to explore next:**

1. **Intermediate Tutorial**: Learn about K-means binning and supervised binning
2. **Advanced Tutorial**: Explore manual binning and flexible binning strategies
3. **sklearn Integration**: Use binning in machine learning pipelines
4. **Performance Tips**: Optimize binning for large datasets

**Key Takeaways:**

- Equal-width binning is simple but may create uneven sample distributions
- Equal-frequency binning creates more balanced distributions
- Always visualize your results to understand the impact of binning
- binlearn handles edge cases like missing values and outliers gracefully
- Trained binners can be saved and reused on new data

Ready for more? Check out the :doc:`intermediate_tutorial` to learn about more advanced binning methods!
