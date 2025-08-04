Basic Examples
==============

Simple, practical examples to get you started with binlearn.

Customer Segmentation
---------------------

This example shows how to use binning for customer segmentation analysis.

.. code-block:: python

   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   from binlearn import EqualWidthBinning, EqualFrequencyBinning
   
   # Generate customer data
   np.random.seed(42)
   n_customers = 5000
   
   customers = pd.DataFrame({
       'age': np.random.normal(40, 15, n_customers),
       'income': np.random.lognormal(10.5, 0.8, n_customers),
       'spending_score': np.random.beta(2, 5, n_customers) * 100,
       'years_customer': np.random.exponential(3, n_customers)
   })
   
   # Clean up the data
   customers['age'] = np.clip(customers['age'], 18, 80)
   customers['years_customer'] = np.clip(customers['years_customer'], 0, 20)
   
   print("Customer data summary:")
   print(customers.describe())

Equal-Width Segmentation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create age and income segments using equal-width binning
   segmenter = EqualWidthBinning(n_bins=4, preserve_dataframe=True)
   customer_segments = segmenter.fit_transform(customers[['age', 'income']])
   
   # Add segment labels to original data
   customers['age_segment'] = customer_segments['age']
   customers['income_segment'] = customer_segments['income'] 
   
   # Create combined segment identifier
   customers['segment'] = (customers['age_segment'].astype(str) + '_' + 
                          customers['income_segment'].astype(str))
   
   print("Segment distribution:")
   print(customers['segment'].value_counts())

Equal-Frequency Segmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Compare with equal-frequency binning for more balanced segments
   eq_freq_segmenter = EqualFrequencyBinning(n_bins=4, preserve_dataframe=True)
   customer_eq_freq = eq_freq_segmenter.fit_transform(customers[['age', 'income']])
   
   customers['age_eq_freq'] = customer_eq_freq['age']
   customers['income_eq_freq'] = customer_eq_freq['income']
   customers['segment_eq_freq'] = (customers['age_eq_freq'].astype(str) + '_' + 
                                  customers['income_eq_freq'].astype(str))
   
   print("Equal-frequency segment distribution:")
   print(customers['segment_eq_freq'].value_counts())

Visualization
~~~~~~~~~~~~~

.. code-block:: python

   # Visualize the segmentation
   fig, axes = plt.subplots(1, 2, figsize=(15, 6))
   
   # Equal-width segments
   scatter1 = axes[0].scatter(customers['age'], customers['income'], 
                             c=customers['age_segment'], cmap='viridis', alpha=0.6)
   axes[0].set_xlabel('Age')
   axes[0].set_ylabel('Income')
   axes[0].set_title('Equal-Width Age Segments')
   plt.colorbar(scatter1, ax=axes[0])
   
   # Equal-frequency segments  
   scatter2 = axes[1].scatter(customers['age'], customers['income'],
                             c=customers['age_eq_freq'], cmap='viridis', alpha=0.6)
   axes[1].set_xlabel('Age')
   axes[1].set_ylabel('Income') 
   axes[1].set_title('Equal-Frequency Age Segments')
   plt.colorbar(scatter2, ax=axes[1])
   
   plt.tight_layout()
   plt.show()

Feature Engineering for Time Series
------------------------------------

Using binning to create categorical features from continuous time series data.

.. code-block:: python

   import pandas as pd
   import numpy as np
   from binlearn import EqualWidthBinning, KMeansBinning
   from datetime import datetime, timedelta
   
   # Generate time series data
   np.random.seed(42)
   start_date = datetime(2023, 1, 1)
   n_days = 365
   
   # Simulate temperature data with seasonal patterns
   dates = [start_date + timedelta(days=i) for i in range(n_days)]
   day_of_year = np.array([d.timetuple().tm_yday for d in dates])
   
   # Seasonal temperature pattern
   seasonal_temp = 20 + 15 * np.sin(2 * np.pi * day_of_year / 365)
   noise = np.random.normal(0, 5, n_days)
   temperature = seasonal_temp + noise
   
   # Additional weather features
   humidity = np.random.beta(2, 2, n_days) * 100
   wind_speed = np.random.exponential(10, n_days)
   
   weather_data = pd.DataFrame({
       'date': dates,
       'temperature': temperature,
       'humidity': humidity, 
       'wind_speed': wind_speed
   })
   
   print("Weather data sample:")
   print(weather_data.head())

Temperature Categories
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create temperature categories using equal-width binning
   temp_binner = EqualWidthBinning(n_bins=5, preserve_dataframe=True)
   weather_binned = temp_binner.fit_transform(weather_data[['temperature']])
   
   # Add descriptive labels
   temp_labels = {0: 'Very Cold', 1: 'Cold', 2: 'Mild', 3: 'Warm', 4: 'Hot'}
   weather_data['temp_category'] = weather_binned['temperature'].map(temp_labels)
   
   print("Temperature category distribution:")
   print(weather_data['temp_category'].value_counts())
   
   # Show bin edges
   print(f"Temperature bin edges: {temp_binner.bin_edges_['temperature']}")

Multi-Feature Weather Categories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use K-means binning for natural weather pattern clusters
   weather_features = weather_data[['temperature', 'humidity', 'wind_speed']]
   kmeans_binner = KMeansBinning(n_bins=4, random_state=42, preserve_dataframe=True)
   weather_clusters = kmeans_binner.fit_transform(weather_features)
   
   # Add cluster information
   weather_data['weather_cluster'] = weather_clusters['temperature']  # Use first feature's clusters
   
   # Analyze clusters
   cluster_summary = weather_data.groupby('weather_cluster')[
       ['temperature', 'humidity', 'wind_speed']
   ].mean()
   
   print("Weather cluster characteristics:")
   print(cluster_summary)

Survey Data Analysis
--------------------

Processing and analyzing survey responses with different binning strategies.

.. code-block:: python

   import pandas as pd
   import numpy as np
   from binlearn import EqualFrequencyBinning, SingletonBinning
   
   # Generate survey data
   np.random.seed(42)
   n_responses = 2000
   
   survey_data = pd.DataFrame({
       'satisfaction_score': np.random.choice(range(1, 11), n_responses, 
                                            p=[0.05, 0.05, 0.1, 0.1, 0.15, 0.2, 0.15, 0.1, 0.05, 0.05]),
       'age_group': np.random.choice(['18-25', '26-35', '36-45', '46-55', '56+'], n_responses,
                                   p=[0.2, 0.3, 0.25, 0.15, 0.1]),
       'usage_frequency': np.random.choice(['Daily', 'Weekly', 'Monthly', 'Rarely'], n_responses,
                                         p=[0.3, 0.4, 0.2, 0.1]),
       'income': np.random.lognormal(10.8, 0.6, n_responses)
   })
   
   print("Survey data summary:")
   print(survey_data.head(10))

Satisfaction Score Binning
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Group satisfaction scores into meaningful categories
   satisfaction_binner = EqualWidthBinning(n_bins=3, preserve_dataframe=True)
   survey_binned = satisfaction_binner.fit_transform(survey_data[['satisfaction_score']])
   
   # Map to descriptive labels
   satisfaction_labels = {0: 'Low Satisfaction', 1: 'Medium Satisfaction', 2: 'High Satisfaction'}
   survey_data['satisfaction_level'] = survey_binned['satisfaction_score'].map(satisfaction_labels)
   
   print("Satisfaction level distribution:")
   print(survey_data['satisfaction_level'].value_counts())

Categorical Data Encoding
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use SingletonBinning for categorical variables
   categorical_cols = ['age_group', 'usage_frequency']
   singleton_binner = SingletonBinning(preserve_dataframe=True)
   categorical_encoded = singleton_binner.fit_transform(survey_data[categorical_cols])
   
   print("Encoded categorical data:")
   print(categorical_encoded.head())
   
   # Show the mapping
   print("Age group encoding:")
   age_mapping = dict(zip(survey_data['age_group'].unique(), 
                         categorical_encoded['age_group'].unique()))
   print(age_mapping)

Income Quantile Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create income quartiles using equal-frequency binning
   income_binner = EqualFrequencyBinning(n_bins=4, preserve_dataframe=True)
   income_quartiles = income_binner.fit_transform(survey_data[['income']])
   
   survey_data['income_quartile'] = income_quartiles['income']
   
   # Analyze satisfaction by income quartile
   satisfaction_by_income = survey_data.groupby('income_quartile')['satisfaction_score'].agg([
       'mean', 'std', 'count'
   ]).round(2)
   
   print("Satisfaction by income quartile:")
   print(satisfaction_by_income)

Sales Data Processing
---------------------

Binning sales data for reporting and analysis.

.. code-block:: python

   import pandas as pd
   import numpy as np
   from binlearn import EqualWidthBinning, EqualFrequencyBinning
   
   # Generate sales data
   np.random.seed(42)
   n_sales = 10000
   
   sales_data = pd.DataFrame({
       'transaction_amount': np.random.lognormal(4, 1.5, n_sales),
       'customer_age': np.random.normal(42, 16, n_sales),
       'items_purchased': np.random.poisson(3, n_sales) + 1,
       'discount_applied': np.random.uniform(0, 0.3, n_sales)
   })
   
   # Clean the data
   sales_data['customer_age'] = np.clip(sales_data['customer_age'], 18, 80)
   sales_data['items_purchased'] = np.clip(sales_data['items_purchased'], 1, 20)
   
   print("Sales data summary:")
   print(sales_data.describe())

Transaction Amount Categories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create transaction size categories
   amount_binner = EqualWidthBinning(n_bins=4, preserve_dataframe=True)
   sales_binned = amount_binner.fit_transform(sales_data[['transaction_amount']])
   
   # Add descriptive labels
   amount_labels = {0: 'Small', 1: 'Medium', 2: 'Large', 3: 'Extra Large'}
   sales_data['transaction_size'] = sales_binned['transaction_amount'].map(amount_labels)
   
   print("Transaction size distribution:")
   print(sales_data['transaction_size'].value_counts())
   
   # Show average amounts per category
   size_amounts = sales_data.groupby('transaction_size')['transaction_amount'].agg(['mean', 'min', 'max']).round(2)
   print("\nTransaction size ranges:")
   print(size_amounts)

Customer Segmentation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Segment customers by age and purchasing behavior
   customer_features = sales_data[['customer_age', 'items_purchased']]
   customer_binner = EqualFrequencyBinning(n_bins=3, preserve_dataframe=True)
   customer_segments = customer_binner.fit_transform(customer_features)
   
   sales_data['age_segment'] = customer_segments['customer_age']
   sales_data['purchase_segment'] = customer_segments['items_purchased']
   
   # Create combined customer profile
   sales_data['customer_profile'] = (sales_data['age_segment'].astype(str) + '_' + 
                                   sales_data['purchase_segment'].astype(str))
   
   print("Customer profile distribution:")
   print(sales_data['customer_profile'].value_counts())

Cross-Tabulation Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Analyze transaction sizes by customer profile
   cross_tab = pd.crosstab(sales_data['customer_profile'], 
                          sales_data['transaction_size'], 
                          normalize='index').round(3)
   
   print("Transaction size distribution by customer profile:")
   print(cross_tab)
   
   # Average discount by segment
   discount_analysis = sales_data.groupby(['age_segment', 'transaction_size'])['discount_applied'].mean().round(3)
   print("\nAverage discount by age segment and transaction size:")
   print(discount_analysis.unstack())

Next Steps
----------

These examples demonstrate the versatility of binlearn for various data analysis tasks. For more advanced examples, see:

- :doc:`machine_learning_examples`: Using binning in ML pipelines
- :doc:`advanced_examples`: Complex binning strategies and custom configurations
- :doc:`performance_examples`: Optimizing binning for large datasets
