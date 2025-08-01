# Manual Flexible Binning Examples

This page demonstrates the use of `ManualFlexibleBinning` for creating bins with different structures per feature, allowing maximum flexibility for complex data transformation requirements.

## Basic Usage

### Understanding Manual Flexible Binning

```python
import numpy as np
import matplotlib.pyplot as plt
from binning import ManualFlexibleBinning, ManualIntervalBinning

# Create multi-feature dataset with different characteristics
np.random.seed(42)
n_samples = 2000

# Feature 1: Age (normal distribution, requires age-appropriate bins)
age = np.random.normal(35, 12, n_samples)
age = np.clip(age, 18, 70)

# Feature 2: Income (log-normal distribution, requires income brackets)
income = np.random.lognormal(10.5, 0.7, n_samples)

# Feature 3: Score (uniform distribution, requires percentile-based bins)
score = np.random.uniform(0, 100, n_samples)

# Feature 4: Count (discrete/Poisson, requires count-based bins)
count = np.random.poisson(8, n_samples)

# Define different binning strategies for each feature
binning_configs = {
    0: {  # Age feature - life stage intervals
        'type': 'intervals',
        'intervals': [
            [18, 25],   # Young adults
            [25, 35],   # Early career
            [35, 50],   # Mid career
            [50, 70]    # Senior
        ]
    },
    1: {  # Income feature - income brackets
        'type': 'intervals', 
        'intervals': [
            [0, 30000],
            [30000, 60000],
            [60000, 100000],
            [100000, 150000],
            [150000, 500000]
        ]
    },
    2: {  # Score feature - percentile-based
        'type': 'percentiles',
        'percentiles': [0, 25, 50, 75, 100]
    },
    3: {  # Count feature - specific count ranges
        'type': 'intervals',
        'intervals': [
            [0, 3],     # Low activity
            [3, 6],     # Medium activity  
            [6, 10],    # High activity
            [10, 25]    # Very high activity
        ]
    }
}

# Create the flexible binner
flexible_binner = ManualFlexibleBinning(binning_configs=binning_configs)

# Prepare data
X = np.column_stack([age, income, score, count])
print(f"Dataset shape: {X.shape}")

# Apply flexible binning
X_binned = flexible_binner.fit_transform(X)

print(f"Binned dataset shape: {X_binned.shape}")

# Analyze the results
feature_names = ['Age', 'Income', 'Score', 'Count']

plt.figure(figsize=(16, 12))

for i, feature_name in enumerate(feature_names):
    # Original distribution
    plt.subplot(4, 3, i*3 + 1)
    plt.hist(X[:, i], bins=50, alpha=0.7, color='lightblue', edgecolor='black')
    plt.xlabel(feature_name)
    plt.ylabel('Frequency')
    plt.title(f'Original {feature_name} Distribution')
    
    # Binned distribution
    plt.subplot(4, 3, i*3 + 2)
    unique_bins = np.unique(X_binned[:, i])
    bin_counts = [np.sum(X_binned[:, i] == bin_id) for bin_id in unique_bins]
    
    bars = plt.bar(unique_bins, bin_counts, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.xlabel('Bin')
    plt.ylabel('Count')  
    plt.title(f'{feature_name} Bin Distribution')
    
    # Add count labels
    for bar, count in zip(bars, bin_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(count), ha='center', va='bottom')
    
    # Box plot by bins
    plt.subplot(4, 3, i*3 + 3)
    bin_data = [X[X_binned[:, i] == bin_id, i] for bin_id in unique_bins]
    
    plt.boxplot(bin_data, labels=[f'Bin {bin_id}' for bin_id in unique_bins])
    plt.xlabel('Bin')
    plt.ylabel(feature_name)
    plt.title(f'{feature_name} Values by Bin')

plt.tight_layout()
plt.show()

# Print bin statistics
print("\\nBin Analysis:")
for i, feature_name in enumerate(feature_names):
    print(f"\\n{feature_name} Feature:")
    unique_bins = np.unique(X_binned[:, i])
    
    for bin_id in unique_bins:
        mask = X_binned[:, i] == bin_id
        bin_values = X[mask, i]
        count = len(bin_values)
        mean_val = bin_values.mean()
        min_val = bin_values.min()
        max_val = bin_values.max()
        
        print(f"  Bin {bin_id}: {count} samples, range [{min_val:.1f}, {max_val:.1f}], mean {mean_val:.1f}")
```

### Comparison with Uniform Binning

```python
import numpy as np
from binning import ManualFlexibleBinning, EqualWidthBinning

# Create dataset with mixed feature types
np.random.seed(42)
n_samples = 1000

# Different feature types requiring different binning approaches
features = {
    'price': np.random.lognormal(7, 1, n_samples),  # Highly skewed
    'rating': np.random.beta(8, 2, n_samples) * 5,  # Concentrated at high end
    'age_days': np.random.exponential(200, n_samples),  # Exponential decay
    'category_size': np.random.poisson(15, n_samples)  # Discrete counts
}

X = np.column_stack(list(features.values()))
feature_names = list(features.keys())

# Method 1: Uniform equal-width binning
uniform_binner = EqualWidthBinning(n_bins=4)
X_uniform = uniform_binner.fit_transform(X)

# Method 2: Flexible binning tailored to each feature
flexible_configs = {
    0: {  # Price - log-scale intervals
        'type': 'intervals',
        'intervals': [
            [0, 500],
            [500, 1500], 
            [1500, 4000],
            [4000, 20000]
        ]
    },
    1: {  # Rating - focused on high-rating distinctions
        'type': 'intervals',
        'intervals': [
            [0, 3.0],
            [3.0, 4.0],
            [4.0, 4.5],
            [4.5, 5.0]
        ]
    },
    2: {  # Age_days - exponential-aware intervals
        'type': 'intervals',
        'intervals': [
            [0, 100],    # Very new
            [100, 300],  # New
            [300, 600],  # Moderate
            [600, 2000]  # Old
        ]
    },
    3: {  # Category_size - count-based ranges
        'type': 'intervals',
        'intervals': [
            [0, 8],      # Small
            [8, 15],     # Medium
            [15, 25],    # Large
            [25, 100]    # Very large
        ]
    }
}

flexible_binner = ManualFlexibleBinning(binning_configs=flexible_configs)
X_flexible = flexible_binner.fit_transform(X)

# Compare bin distributions
print("Comparison: Uniform vs Flexible Binning")
print("=" * 50)

for i, feature_name in enumerate(feature_names):
    print(f"\\n{feature_name.upper()} FEATURE:")
    
    # Uniform binning distribution
    uniform_counts = [np.sum(X_uniform[:, i] == j) for j in range(4)]
    uniform_balance = np.std(uniform_counts) / np.mean(uniform_counts)
    
    # Flexible binning distribution  
    flexible_counts = [np.sum(X_flexible[:, i] == j) for j in range(4)]
    flexible_balance = np.std(flexible_counts) / np.mean(flexible_counts)
    
    print(f"  Uniform binning:  {uniform_counts} (balance: {uniform_balance:.3f})")
    print(f"  Flexible binning: {flexible_counts} (balance: {flexible_balance:.3f})")
    
    # Show actual value ranges for flexible binning  
    print(f"  Flexible bin ranges:")
    for bin_id in range(4):
        mask = X_flexible[:, i] == bin_id
        if mask.any():
            bin_values = X[mask, i]
            print(f"    Bin {bin_id}: [{bin_values.min():.1f}, {bin_values.max():.1f}] (n={mask.sum()})")

# Visualize the comparison
plt.figure(figsize=(16, 10))

for i, feature_name in enumerate(feature_names):
    # Original data
    plt.subplot(4, 3, i*3 + 1)
    plt.hist(X[:, i], bins=30, alpha=0.7, color='lightblue', edgecolor='black')
    plt.xlabel(feature_name)
    plt.ylabel('Frequency')
    plt.title(f'{feature_name} Distribution')
    
    # Uniform binning
    plt.subplot(4, 3, i*3 + 2)
    uniform_counts = [np.sum(X_uniform[:, i] == j) for j in range(4)]
    plt.bar(range(4), uniform_counts, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.xlabel('Bin')
    plt.ylabel('Count')
    plt.title(f'{feature_name} - Uniform Binning')
    
    # Flexible binning
    plt.subplot(4, 3, i*3 + 3)
    flexible_counts = [np.sum(X_flexible[:, i] == j) for j in range(4)]
    plt.bar(range(4), flexible_counts, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.xlabel('Bin')
    plt.ylabel('Count')
    plt.title(f'{feature_name} - Flexible Binning')

plt.tight_layout()
plt.show()
```

## Real-world Applications

### E-commerce Customer Segmentation

```python
import numpy as np
import pandas as pd
from binning import ManualFlexibleBinning
import matplotlib.pyplot as plt

# Simulate e-commerce customer dataset
np.random.seed(42)
n_customers = 5000

# Customer characteristics with different distributions
customer_data = {
    'recency_days': np.random.exponential(30, n_customers),  # Days since last purchase
    'frequency': np.random.poisson(12, n_customers),  # Number of purchases
    'monetary_value': np.random.lognormal(6, 1.2, n_customers),  # Total spending
    'avg_order_value': np.random.lognormal(4.5, 0.8, n_customers),  # Average order size
    'customer_age_months': np.random.gamma(2, 12, n_customers),  # Account age
    'support_tickets': np.random.poisson(2, n_customers)  # Support interactions
}

# Create DataFrame
df = pd.DataFrame(customer_data)

print("E-commerce Customer Dataset:")
print(df.describe())

# Define RFM and additional customer segmentation bins
segmentation_configs = {
    0: {  # Recency - recent customers are more valuable
        'type': 'intervals',
        'intervals': [
            [0, 7],      # Very recent (last week)
            [7, 30],     # Recent (last month)  
            [30, 90],    # Moderate (last quarter)
            [90, 365]    # Distant (last year)
        ]
    },
    1: {  # Frequency - purchase behavior tiers
        'type': 'intervals', 
        'intervals': [
            [0, 3],      # Low frequency
            [3, 8],      # Medium frequency
            [8, 15],     # High frequency
            [15, 100]    # Very high frequency
        ]
    },
    2: {  # Monetary Value - spending tiers
        'type': 'intervals',
        'intervals': [
            [0, 200],        # Low spenders
            [200, 500],      # Medium spenders
            [500, 1500],     # High spenders
            [1500, 50000]    # VIP spenders
        ]
    },
    3: {  # Average Order Value - order size categories
        'type': 'intervals',
        'intervals': [
            [0, 50],     # Small orders
            [50, 100],   # Medium orders
            [100, 200],  # Large orders
            [200, 1000]  # Premium orders
        ]
    },
    4: {  # Customer Age - lifecycle stages
        'type': 'intervals',
        'intervals': [
            [0, 3],      # New customers (0-3 months)
            [3, 12],     # Growing customers (3-12 months)
            [12, 36],    # Mature customers (1-3 years)
            [36, 120]    # Veteran customers (3+ years)
        ]
    },
    5: {  # Support Tickets - service interaction levels
        'type': 'intervals',
        'intervals': [
            [0, 1],      # No/minimal support
            [1, 3],      # Low support
            [3, 6],      # Medium support  
            [6, 20]      # High support needs
        ]
    }
}

# Apply flexible binning
X = df.values
flexible_binner = ManualFlexibleBinning(binning_configs=segmentation_configs)
X_segmented = flexible_binner.fit_transform(X)

# Add segmented features to DataFrame
feature_names = list(df.columns)
for i, feature_name in enumerate(feature_names):
    df[f'{feature_name}_segment'] = X_segmented[:, i]

# Create RFM score (combine Recency, Frequency, Monetary)
# Higher scores are better, but recency is inverted (recent = high score)
df['recency_score'] = 4 - df['recency_days_segment']  # Invert recency
df['frequency_score'] = df['frequency_segment'] + 1
df['monetary_score'] = df['monetary_value_segment'] + 1

df['rfm_score'] = df['recency_score'] + df['frequency_score'] + df['monetary_score']

# Define customer segments based on RFM score
def categorize_customer(row):
    rfm = row['rfm_score'] 
    recency = row['recency_score']
    frequency = row['frequency_score']
    monetary = row['monetary_score']
    
    if rfm >= 10 and recency >= 3:
        return 'Champions'  # High value, recent, frequent
    elif rfm >= 8 and recency >= 2:
        return 'Loyal Customers'  # Good value, somewhat recent
    elif rfm >= 7 and frequency >= 3:
        return 'Potential Loyalists'  # Good frequency, may increase spending
    elif recency >= 3 and monetary >= 3:
        return 'New Customers'  # Recent but low frequency
    elif frequency >= 3 and monetary >= 2:
        return 'At Risk'  # Good history but not recent
    elif recency <= 1:
        return 'Cannot Lose Them'  # High value but not recent - critical
    elif monetary <= 1:
        return 'Price Sensitive'  # Low spending
    else:
        return 'Others'

df['customer_segment'] = df.apply(categorize_customer, axis=1)

# Analyze customer segments
print("\\nCustomer Segmentation Analysis:")
segment_analysis = df.groupby('customer_segment').agg({
    'recency_days': ['count', 'mean'],
    'frequency': 'mean',
    'monetary_value': 'mean',
    'avg_order_value': 'mean',
    'customer_age_months': 'mean',
    'support_tickets': 'mean'
}).round(2)

print(segment_analysis)

# Calculate segment values
segment_summary = df.groupby('customer_segment').agg({
    'monetary_value': ['count', 'sum', 'mean'],
    'frequency': 'sum',
    'recency_days': 'mean'
}).round(2)

segment_summary.columns = ['Customer_Count', 'Total_Revenue', 'Avg_Revenue', 'Total_Orders', 'Avg_Recency']
segment_summary['Revenue_Percentage'] = (segment_summary['Total_Revenue'] / segment_summary['Total_Revenue'].sum() * 100).round(1)
segment_summary['Customer_Percentage'] = (segment_summary['Customer_Count'] / segment_summary['Customer_Count'].sum() * 100).round(1)

print("\\nSegment Business Impact:")
print(segment_summary.sort_values('Total_Revenue', ascending=False))

# Visualization
plt.figure(figsize=(16, 12))

# Segment distribution
plt.subplot(3, 3, 1)
segment_counts = df['customer_segment'].value_counts()
plt.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Customer Segment Distribution')

# RFM score distribution
plt.subplot(3, 3, 2)
plt.hist(df['rfm_score'], bins=20, alpha=0.7, color='lightblue', edgecolor='black')
plt.xlabel('RFM Score')
plt.ylabel('Customer Count')
plt.title('RFM Score Distribution')

# Revenue by segment
plt.subplot(3, 3, 3)
segment_revenue = df.groupby('customer_segment')['monetary_value'].sum().sort_values(ascending=False)
bars = plt.bar(range(len(segment_revenue)), segment_revenue.values, alpha=0.7, color='lightgreen')
plt.xlabel('Customer Segment')
plt.ylabel('Total Revenue')
plt.title('Revenue by Customer Segment')
plt.xticks(range(len(segment_revenue)), segment_revenue.index, rotation=45, ha='right')

# Feature distributions by segment for top 3 segments
top_segments = segment_summary.head(3).index

for i, feature in enumerate(['recency_days', 'frequency', 'monetary_value']):
    plt.subplot(3, 3, 4 + i)
    
    for segment in top_segments:
        segment_data = df[df['customer_segment'] == segment][feature]
        plt.hist(segment_data, bins=20, alpha=0.6, label=segment)
    
    plt.xlabel(feature.replace('_', ' ').title())
    plt.ylabel('Frequency')
    plt.title(f'{feature.replace("_", " ").title()} by Top Segments')
    plt.legend()

# Correlation matrix of segment features
plt.subplot(3, 3, 7)
segment_features = ['recency_score', 'frequency_score', 'monetary_score', 'rfm_score']
correlation_matrix = df[segment_features].corr()

im = plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar(im)
plt.xticks(range(len(segment_features)), [f.replace('_score', '') for f in segment_features])
plt.yticks(range(len(segment_features)), [f.replace('_score', '') for f in segment_features])
plt.title('RFM Correlation Matrix')

# Add correlation values
for i in range(len(segment_features)):
    for j in range(len(segment_features)):
        plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                ha='center', va='center', color='black')

plt.tight_layout()
plt.show()

# Marketing recommendations
print("\\nMarketing Recommendations by Segment:")
recommendations = {
    'Champions': 'Reward them, ask for reviews, upsell premium products',
    'Loyal Customers': 'Recommend related products, loyalty programs',
    'Potential Loyalists': 'Membership programs, cross-sell opportunities',
    'New Customers': 'Onboarding campaigns, education about products',
    'At Risk': 'Win-back campaigns, special offers, surveys',
    'Cannot Lose Them': 'Urgent retention campaigns, personal outreach',
    'Price Sensitive': 'Discount campaigns, value products',
    'Others': 'General marketing, re-engagement campaigns'
}

for segment, recommendation in recommendations.items():
    if segment in df['customer_segment'].values:
        count = (df['customer_segment'] == segment).sum()
        revenue_pct = segment_summary.loc[segment, 'Revenue_Percentage']
        print(f"\\n{segment} ({count} customers, {revenue_pct}% of revenue):")
        print(f"  Strategy: {recommendation}")
```

### Healthcare Patient Risk Assessment

```python
import numpy as np
import pandas as pd
from binning import ManualFlexibleBinning

# Simulate healthcare patient dataset
np.random.seed(42)
n_patients = 3000

# Patient characteristics with medical significance
patient_data = {
    'age': np.random.normal(55, 20, n_patients),
    'bmi': np.random.normal(27, 6, n_patients),
    'systolic_bp': np.random.normal(130, 25, n_patients),
    'cholesterol': np.random.normal(200, 50, n_patients),
    'blood_glucose': np.random.normal(100, 30, n_patients),
    'hospital_visits_year': np.random.poisson(3, n_patients)
}

# Clip values to realistic ranges
patient_data['age'] = np.clip(patient_data['age'], 18, 100)
patient_data['bmi'] = np.clip(patient_data['bmi'], 15, 50)
patient_data['systolic_bp'] = np.clip(patient_data['systolic_bp'], 80, 220)
patient_data['cholesterol'] = np.clip(patient_data['cholesterol'], 100, 400)
patient_data['blood_glucose'] = np.clip(patient_data['blood_glucose'], 60, 300)

df_patients = pd.DataFrame(patient_data)

# Define medically-relevant binning for each biomarker
medical_configs = {
    0: {  # Age - life stage risk categories
        'type': 'intervals',
        'intervals': [
            [18, 40],    # Young adults - low baseline risk
            [40, 55],    # Middle age - increasing risk
            [55, 70],    # Older adults - elevated risk
            [70, 100]    # Elderly - high risk
        ]
    },
    1: {  # BMI - WHO classification
        'type': 'intervals',
        'intervals': [
            [15, 18.5],  # Underweight
            [18.5, 25],  # Normal weight
            [25, 30],    # Overweight
            [30, 50]     # Obese
        ]
    },
    2: {  # Systolic BP - AHA guidelines
        'type': 'intervals',
        'intervals': [
            [80, 120],   # Normal
            [120, 130],  # Elevated
            [130, 140],  # Stage 1 Hypertension
            [140, 220]   # Stage 2 Hypertension
        ]
    },
    3: {  # Cholesterol - ATP III guidelines
        'type': 'intervals',
        'intervals': [
            [100, 200],  # Desirable
            [200, 240],  # Borderline high
            [240, 300],  # High
            [300, 400]   # Very high
        ]
    },
    4: {  # Blood Glucose - ADA guidelines
        'type': 'intervals',
        'intervals': [
            [60, 100],   # Normal
            [100, 126],  # Prediabetes
            [126, 200],  # Diabetes
            [200, 300]   # Severe diabetes
        ]
    },
    5: {  # Hospital Visits - healthcare utilization
        'type': 'intervals',
        'intervals': [
            [0, 1],      # Minimal use
            [1, 3],      # Low use
            [3, 6],      # Moderate use
            [6, 20]      # High use
        ]
    }
}

# Apply medical binning
X_medical = df_patients.values
medical_binner = ManualFlexibleBinning(binning_configs=medical_configs)
X_risk_categories = medical_binner.fit_transform(X_medical)

# Add risk categories to DataFrame
risk_features = ['age_risk', 'bmi_risk', 'bp_risk', 'chol_risk', 'glucose_risk', 'utilization_risk']
for i, feature in enumerate(risk_features):
    df_patients[feature] = X_risk_categories[:, i]

# Define risk category labels
risk_labels = {
    'age_risk': ['Young Adult', 'Middle Age', 'Older Adult', 'Elderly'],
    'bmi_risk': ['Underweight', 'Normal', 'Overweight', 'Obese'],
    'bp_risk': ['Normal BP', 'Elevated', 'Stage 1 HTN', 'Stage 2 HTN'],
    'chol_risk': ['Desirable', 'Borderline', 'High', 'Very High'],
    'glucose_risk': ['Normal', 'Prediabetes', 'Diabetes', 'Severe'],
    'utilization_risk': ['Minimal', 'Low', 'Moderate', 'High']
}

# Calculate composite risk score
# Weight factors based on medical significance
risk_weights = {
    'age_risk': 1.0,
    'bmi_risk': 1.2,
    'bp_risk': 1.5,
    'chol_risk': 1.3,
    'glucose_risk': 1.8,
    'utilization_risk': 0.8
}

df_patients['composite_risk'] = sum(
    df_patients[feature] * weight 
    for feature, weight in risk_weights.items()
)

# Categorize overall risk
def categorize_overall_risk(risk_score):
    if risk_score <= 3:
        return 'Low Risk'
    elif risk_score <= 6:
        return 'Moderate Risk'
    elif risk_score <= 9:
        return 'High Risk'
    else:
        return 'Very High Risk'

df_patients['overall_risk_category'] = df_patients['composite_risk'].apply(categorize_overall_risk)

# Analysis
print("Healthcare Risk Assessment Analysis:")
print(f"Total patients assessed: {len(df_patients)}")

print("\\nRisk Factor Distributions:")
for feature, labels in risk_labels.items():
    print(f"\\n{feature.replace('_', ' ').title()}:")
    counts = df_patients[feature].value_counts().sort_index()
    for risk_level, label in enumerate(labels):
        if risk_level in counts.index:
            count = counts[risk_level]
            pct = count / len(df_patients) * 100
            print(f"  {label}: {count} patients ({pct:.1f}%)")

print("\\nOverall Risk Assessment:")
overall_risk_summary = df_patients['overall_risk_category'].value_counts()
for category in ['Low Risk', 'Moderate Risk', 'High Risk', 'Very High Risk']:
    if category in overall_risk_summary.index:
        count = overall_risk_summary[category]
        pct = count / len(df_patients) * 100
        print(f"{category}: {count} patients ({pct:.1f}%)")

# High-risk patient analysis
high_risk_patients = df_patients[df_patients['overall_risk_category'].isin(['High Risk', 'Very High Risk'])]
print(f"\\nHigh-risk patients requiring intervention: {len(high_risk_patients)} ({len(high_risk_patients)/len(df_patients)*100:.1f}%)")

# Identify most common risk factor combinations
print("\\nMost Common Risk Factor Combinations in High-Risk Patients:")
risk_combinations = high_risk_patients[list(risk_labels.keys())].value_counts().head(10)
for i, (combo, count) in enumerate(risk_combinations.items()):
    print(f"\\n{i+1}. Count: {count}")
    for j, (feature, value) in enumerate(zip(risk_labels.keys(), combo)):
        risk_label = risk_labels[feature][value]
        print(f"   {feature.replace('_', ' ').title()}: {risk_label}")

# Clinical recommendations by risk category
print("\\nClinical Recommendations by Risk Category:")
recommendations = {
    'Low Risk': 'Annual checkups, lifestyle maintenance',
    'Moderate Risk': 'Semi-annual monitoring, lifestyle interventions',
    'High Risk': 'Quarterly monitoring, medication review, specialist referral',
    'Very High Risk': 'Monthly monitoring, intensive management, multidisciplinary care'
}

for category, recommendation in recommendations.items():
    if category in df_patients['overall_risk_category'].values:
        count = (df_patients['overall_risk_category'] == category).sum()
        print(f"\\n{category} ({count} patients):")
        print(f"  Clinical Action: {recommendation}")
```

### Financial Portfolio Analysis

```python
import numpy as np
import pandas as pd
from binning import ManualFlexibleBinning

# Simulate investment portfolio data
np.random.seed(42)
n_investments = 2500

# Investment characteristics
portfolio_data = {
    'expected_return': np.random.normal(8, 4, n_investments),  # Annual return %
    'volatility': np.random.gamma(2, 3, n_investments),        # Risk measure
    'market_cap': np.random.lognormal(16, 2, n_investments),   # Company size
    'pe_ratio': np.random.lognormal(3, 0.8, n_investments),   # Valuation metric
    'dividend_yield': np.random.beta(1, 4, n_investments) * 8, # Dividend %
    'liquidity_score': np.random.uniform(1, 10, n_investments) # Ease of trading
}

# Clip to realistic ranges
portfolio_data['expected_return'] = np.clip(portfolio_data['expected_return'], -10, 25)
portfolio_data['volatility'] = np.clip(portfolio_data['volatility'], 1, 40)
portfolio_data['pe_ratio'] = np.clip(portfolio_data['pe_ratio'], 5, 100)

df_portfolio = pd.DataFrame(portfolio_data)

# Define investment-specific binning strategies
investment_configs = {
    0: {  # Expected Return - performance tiers
        'type': 'intervals',
        'intervals': [
            [-10, 0],    # Negative returns
            [0, 5],      # Conservative returns
            [5, 12],     # Moderate returns
            [12, 25]     # Aggressive returns
        ]
    },
    1: {  # Volatility - risk categories
        'type': 'intervals',
        'intervals': [
            [1, 8],      # Low risk
            [8, 15],     # Medium risk
            [15, 25],    # High risk
            [25, 40]     # Very high risk
        ]
    },
    2: {  # Market Cap - size categories
        'type': 'intervals',
        'intervals': [
            [0, 2e9],        # Small cap
            [2e9, 10e9],     # Mid cap
            [10e9, 200e9],   # Large cap
            [200e9, 1e15]    # Mega cap
        ]
    },
    3: {  # P/E Ratio - valuation categories
        'type': 'intervals',
        'intervals': [
            [5, 15],     # Value stocks
            [15, 25],    # Fair value
            [25, 40],    # Growth premium
            [40, 100]    # High growth/speculative
        ]
    },
    4: {  # Dividend Yield - income categories
        'type': 'intervals',
        'intervals': [
            [0, 1],      # No/low dividend
            [1, 3],      # Moderate dividend
            [3, 5],      # High dividend
            [5, 8]       # Very high dividend
        ]
    },
    5: {  # Liquidity Score - trading ease
        'type': 'intervals',
        'intervals': [
            [1, 3],      # Illiquid
            [3, 5],      # Moderate liquidity
            [5, 7],      # Good liquidity
            [7, 10]      # Highly liquid
        ]
    }
}

# Apply investment binning
X_portfolio = df_portfolio.values
portfolio_binner = ManualFlexibleBinning(binning_configs=investment_configs)
X_categorized = portfolio_binner.fit_transform(X_portfolio)

# Add categories to DataFrame
category_features = ['return_category', 'risk_category', 'size_category', 
                    'valuation_category', 'income_category', 'liquidity_category']

for i, feature in enumerate(category_features):
    df_portfolio[feature] = X_categorized[:, i]

# Define category labels
category_labels = {
    'return_category': ['Negative', 'Conservative', 'Moderate', 'Aggressive'],
    'risk_category': ['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk'],
    'size_category': ['Small Cap', 'Mid Cap', 'Large Cap', 'Mega Cap'],
    'valuation_category': ['Value', 'Fair Value', 'Growth Premium', 'Speculative'],
    'income_category': ['No Dividend', 'Moderate', 'High Income', 'Very High Income'],
    'liquidity_category': ['Illiquid', 'Moderate', 'Good', 'Highly Liquid']
}

# Create investment style classification
def classify_investment_style(row):
    return_cat = row['return_category']
    risk_cat = row['risk_category']
    size_cat = row['size_category']
    val_cat = row['valuation_category']
    
    # Conservative: Low risk, moderate returns
    if risk_cat <= 1 and return_cat <= 2:
        return 'Conservative'
    
    # Aggressive Growth: High risk, high expected returns
    elif risk_cat >= 2 and return_cat >= 2:
        return 'Aggressive Growth'
    
    # Value: Low P/E, moderate risk/return
    elif val_cat == 0 and risk_cat <= 2:
        return 'Value'
    
    # Large Cap Stable: Large companies, moderate risk
    elif size_cat >= 2 and risk_cat <= 1:
        return 'Large Cap Stable'
    
    # Small Cap Growth: Small companies, higher risk/return
    elif size_cat == 0 and return_cat >= 1:
        return 'Small Cap Growth'
    
    # Income: High dividend yield
    elif row['income_category'] >= 2:
        return 'Income'
    
    else:
        return 'Balanced'

df_portfolio['investment_style'] = df_portfolio.apply(classify_investment_style, axis=1)

# Portfolio analysis
print("Investment Portfolio Analysis:")
print(f"Total investments analyzed: {len(df_portfolio)}")

print("\\nInvestment Style Distribution:")
style_distribution = df_portfolio['investment_style'].value_counts()
for style, count in style_distribution.items():
    pct = count / len(df_portfolio) * 100
    avg_return = df_portfolio[df_portfolio['investment_style'] == style]['expected_return'].mean()
    avg_risk = df_portfolio[df_portfolio['investment_style'] == style]['volatility'].mean()
    print(f"{style}: {count} investments ({pct:.1f}%) - Avg Return: {avg_return:.1f}%, Avg Risk: {avg_risk:.1f}%")

# Risk-return analysis by style
print("\\nRisk-Return Profile by Investment Style:")
risk_return_analysis = df_portfolio.groupby('investment_style').agg({
    'expected_return': ['mean', 'std'],
    'volatility': 'mean',
    'market_cap': 'median',
    'pe_ratio': 'mean',
    'dividend_yield': 'mean'
}).round(2)

print(risk_return_analysis)

# Portfolio recommendations
print("\\nPortfolio Allocation Recommendations:")
total_investments = len(df_portfolio)

allocation_recommendations = {
    'Conservative': {'target_pct': 30, 'rationale': 'Stable foundation, capital preservation'},
    'Large Cap Stable': {'target_pct': 25, 'rationale': 'Core holdings, steady growth'},
    'Value': {'target_pct': 15, 'rationale': 'Undervalued opportunities'},
    'Balanced': {'target_pct': 15, 'rationale': 'Diversification, moderate risk'},
    'Income': {'target_pct': 10, 'rationale': 'Regular income, dividend growth'},
    'Small Cap Growth': {'target_pct': 3, 'rationale': 'High growth potential'},
    'Aggressive Growth': {'target_pct': 2, 'rationale': 'Speculative opportunities'}
}

print("\\nSuggested Allocation vs Current:")
for style, allocation in allocation_recommendations.items():
    if style in style_distribution.index:
        current_count = style_distribution[style]
        current_pct = current_count / total_investments * 100
        target_pct = allocation['target_pct']
        difference = target_pct - current_pct
        
        print(f"\\n{style}:")
        print(f"  Current: {current_pct:.1f}% ({current_count} investments)")
        print(f"  Target:  {target_pct}%")
        print(f"  Adjustment: {difference:+.1f}%")
        print(f"  Rationale: {allocation['rationale']}")

# Risk metrics
df_portfolio['risk_adjusted_return'] = df_portfolio['expected_return'] / df_portfolio['volatility']

print("\\nRisk-Adjusted Performance by Style:")
risk_adj_performance = df_portfolio.groupby('investment_style')['risk_adjusted_return'].mean().sort_values(ascending=False)
for style, ratio in risk_adj_performance.items():
    print(f"{style}: {ratio:.3f} (return per unit of risk)")
```

## Advanced Usage

### Combining Multiple Binning Strategies

```python
import numpy as np
from binning import ManualFlexibleBinning

# Demonstrate combining different binning approaches within one dataset
np.random.seed(42)
n_samples = 1500

# Create features that benefit from different binning strategies
features_data = {
    'percentile_feature': np.random.normal(100, 15, n_samples),      # Best with percentiles
    'domain_feature': np.random.uniform(0, 100, n_samples),         # Domain-specific intervals
    'skewed_feature': np.random.exponential(2, n_samples),          # Needs log-scale intervals
    'categorical_feature': np.random.poisson(5, n_samples),         # Count-based intervals
    'bimodal_feature': np.concatenate([                             # Custom intervals for modes
        np.random.normal(30, 5, n_samples//2),
        np.random.normal(70, 5, n_samples//2)
    ])
}

X_mixed = np.column_stack(list(features_data.values()))

# Define mixed binning strategies
mixed_configs = {
    0: {  # Percentile-based binning
        'type': 'percentiles',
        'percentiles': [0, 25, 50, 75, 100]
    },
    1: {  # Domain-specific intervals (e.g., test scores)
        'type': 'intervals',
        'intervals': [
            [0, 60],     # Fail
            [60, 70],    # Pass
            [70, 85],    # Good
            [85, 100]    # Excellent
        ]
    },
    2: {  # Log-scale intervals for skewed data
        'type': 'intervals',
        'intervals': [
            [0, 1],      # Very low
            [1, 3],      # Low
            [3, 6],      # Medium
            [6, 20]      # High
        ]
    },
    3: {  # Count-based intervals
        'type': 'intervals',
        'intervals': [
            [0, 2],      # Rare
            [2, 4],      # Low
            [4, 7],      # Moderate
            [7, 15]      # High
        ]
    },
    4: {  # Custom intervals for bimodal distribution
        'type': 'intervals',
        'intervals': [
            [0, 37],     # First mode region
            [37, 50],    # Transition
            [50, 63],    # Between modes
            [63, 100]    # Second mode region
        ]
    }
}

# Apply mixed binning
mixed_binner = ManualFlexibleBinning(binning_configs=mixed_configs)
X_mixed_binned = mixed_binner.fit_transform(X_mixed)

# Analyze the effectiveness of each strategy
feature_names = list(features_data.keys()) 
strategies = ['Percentiles', 'Domain-specific', 'Log-scale', 'Count-based', 'Bimodal-aware']

print("Mixed Binning Strategy Analysis:")
print("=" * 50)

for i, (feature_name, strategy) in enumerate(zip(feature_names, strategies)):
    print(f"\\n{feature_name.upper()} ({strategy}):")
    
    # Show bin distributions
    unique_bins = np.unique(X_mixed_binned[:, i])
    for bin_id in unique_bins:
        mask = X_mixed_binned[:, i] == bin_id
        bin_values = X_mixed[mask, i]
        count = len(bin_values)
        mean_val = bin_values.mean()
        range_val = f"[{bin_values.min():.2f}, {bin_values.max():.2f}]"
        
        print(f"  Bin {bin_id}: {count} samples, range {range_val}, mean {mean_val:.2f}")
    
    # Calculate balance metric
    bin_counts = [np.sum(X_mixed_binned[:, i] == bin_id) for bin_id in unique_bins]
    balance = np.std(bin_counts) / np.mean(bin_counts)
    print(f"  Balance metric: {balance:.3f} (lower = more balanced)")

# Visualize the mixed strategies
plt.figure(figsize=(20, 12))

for i, (feature_name, strategy) in enumerate(zip(feature_names, strategies)):
    # Original distribution
    plt.subplot(3, 5, i + 1)
    plt.hist(X_mixed[:, i], bins=30, alpha=0.7, color='lightblue', edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f'{feature_name}\\n({strategy})')
    
    # Binned distribution
    plt.subplot(3, 5, i + 6)
    bin_counts = [np.sum(X_mixed_binned[:, i] == j) for j in range(4)]
    plt.bar(range(4), bin_counts, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.xlabel('Bin')
    plt.ylabel('Count')
    plt.title(f'Bin Distribution')
    
    # Box plot by bins
    plt.subplot(3, 5, i + 11)
    bin_data = [X_mixed[X_mixed_binned[:, i] == j, i] for j in range(4)]
    plt.boxplot(bin_data, labels=[f'Bin {j}' for j in range(4)])
    plt.xlabel('Bin')
    plt.ylabel('Value')
    plt.title('Values by Bin')

plt.tight_layout()
plt.show()
```

### Dynamic Configuration Updates

```python
import numpy as np
from binning import ManualFlexibleBinning

# Demonstrate updating binning configurations based on data analysis
np.random.seed(42)

# Initial dataset
X_initial = np.,
random.normal(50, 20, (500, 3))

# Initial binning configuration
initial_config = {
    0: {'type': 'intervals', 'intervals': [[0, 25], [25, 50], [50, 75], [75, 100]]},
    1: {'type': 'intervals', 'intervals': [[0, 25], [25, 50], [50, 75], [75, 100]]},
    2: {'type': 'intervals', 'intervals': [[0, 25], [25, 50], [50, 75], [75, 100]]}
}

print("Dynamic Configuration Example:")
print("=" * 40)

# Apply initial binning
initial_binner = ManualFlexibleBinning(binning_configs=initial_config)
X_initial_binned = initial_binner.fit_transform(X_initial)

# Analyze initial results
print("Initial Binning Analysis:")
for i in range(3):
    bin_counts = [np.sum(X_initial_binned[:, i] == j) for j in range(4)]
    print(f"Feature {i} bin counts: {bin_counts}")
    
    # Identify issues
    min_count = min(bin_counts)
    max_count = max(bin_counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    if imbalance_ratio > 3:
        print(f"  ⚠️  High imbalance detected (ratio: {imbalance_ratio:.1f})")
    if min_count < 50:  # Less than 10% of data
        print(f"  ⚠️  Sparse bin detected (min count: {min_count})")

# New data arrives with different distribution
X_new = np.random.exponential(2, (300, 3)) * 30  # Highly skewed data
X_combined = np.vstack([X_initial, X_new])

print(f"\\nNew data characteristics:")
print(f"Combined dataset size: {len(X_combined)}")
print(f"New data distribution: Exponential (skewed)")

# Analyze combined data to update configuration
updated_config = {}

for i in range(3):
    feature_data = X_combined[:, i]
    
    # Calculate percentiles for adaptive intervals
    percentiles = np.percentile(feature_data, [0, 30, 60, 85, 100])
    
    # Create updated intervals based on data distribution
    updated_intervals = [
        [percentiles[0], percentiles[1]],
        [percentiles[1], percentiles[2]], 
        [percentiles[2], percentiles[3]],
        [percentiles[3], percentiles[4]]
    ]
    
    updated_config[i] = {
        'type': 'intervals',
        'intervals': updated_intervals
    }
    
    print(f"\\nFeature {i} updated intervals:")
    for j, interval in enumerate(updated_intervals):
        print(f"  Bin {j}: [{interval[0]:.1f}, {interval[1]:.1f}]")

# Apply updated configuration
updated_binner = ManualFlexibleBinning(binning_configs=updated_config)
X_updated_binned = updated_binner.fit_transform(X_combined)

# Compare results
print("\\nConfiguration Update Results:")
for i in range(3):
    initial_counts = [np.sum(X_initial_binned[:, i] == j) for j in range(4)]
    updated_counts = [np.sum(X_updated_binned[:, i] == j) for j in range(4)]
    
    initial_balance = np.std(initial_counts) / np.mean(initial_counts)
    updated_balance = np.std(updated_counts) / np.mean(updated_counts)
    
    print(f"\\nFeature {i}:")
    print(f"  Initial balance: {initial_balance:.3f}")
    print(f"  Updated balance: {updated_balance:.3f}")
    print(f"  Improvement: {((initial_balance - updated_balance) / initial_balance * 100):+.1f}%")
```

## Best Practices and Tips

### When to Use Manual Flexible Binning

```python
print("Guidelines for using Manual Flexible Binning:")
print("\\n✅ EXCELLENT for:")
print("  - Multi-modal datasets with mixed feature types")
print("  - Domain expertise available for different features")
print("  - Features requiring different binning philosophies")
print("  - Complex business rules with feature-specific requirements")
print("  - Interpretability crucial for each feature independently")
print("\\n⚠️  CONSIDER CAREFULLY for:")
print("  - Simple datasets with homogeneous features")
print("  - Automated/production systems requiring consistency")
print("  - Limited domain knowledge about feature relationships")
print("\\n❌ AVOID for:")
print("  - High-dimensional datasets (configuration complexity)")
print("  - Frequent retraining scenarios")
print("  - When feature relationships are more important than individual treatment")

# Best practices demonstration
def create_flexible_binning_best_practices():
    """
    Demonstrate best practices for ManualFlexibleBinning
    """
    
    # Practice 1: Document configuration rationale
    documented_config = {
        0: {
            'type': 'intervals',
            'intervals': [[18, 30], [30, 45], [45, 65], [65, 80]],
            'rationale': 'Life stage intervals for age-related analysis',
            'domain_expert': 'Demographics team',
            'last_updated': '2024-01-15'
        },
        1: {
            'type': 'percentiles', 
            'percentiles': [0, 25, 50, 75, 100],
            'rationale': 'Income requires equal-sized bins for fair comparison',
            'domain_expert': 'Economics team',
            'last_updated': '2024-01-15'
        }
    }
    
    # Practice 2: Validation function
    def validate_flexible_config(config, data):
        """Validate flexible binning configuration"""
        issues = []
        
        for feature_idx, feature_config in config.items():
            if feature_idx >= data.shape[1]:
                issues.append(f"Feature {feature_idx} not in data (only {data.shape[1]} features)")
                continue
                
            feature_data = data[:, feature_idx]
            
            if feature_config['type'] == 'intervals':
                intervals = feature_config['intervals']
                
                # Check coverage
                data_min, data_max = feature_data.min(), feature_data.max()
                config_min = min(interval[0] for interval in intervals)
                config_max = max(interval[1] for interval in intervals)
                
                if config_min > data_min or config_max < data_max:
                    issues.append(f"Feature {feature_idx}: Intervals don't cover data range")
                
                # Check for gaps
                sorted_intervals = sorted(intervals, key=lambda x: x[0])
                for i in range(len(sorted_intervals) - 1):
                    if sorted_intervals[i][1] != sorted_intervals[i+1][0]:
                        issues.append(f"Feature {feature_idx}: Gap in intervals")
        
        return issues
    
    # Practice 3: Performance monitoring
    def monitor_binning_performance(binner, data, config):
        """Monitor the performance of flexible binning"""
        results = {}
        binned_data = binner.transform(data)
        
        for feature_idx in config.keys():
            if feature_idx < data.shape[1]:
                bin_counts = [np.sum(binned_data[:, feature_idx] == i) 
                             for i in range(len(config[feature_idx]['intervals']))]
                
                # Balance metric
                balance = np.std(bin_counts) / np.mean(bin_counts)
                
                # Effective bins (non-empty)
                effective_bins = sum(1 for count in bin_counts if count > 0)
                
                results[feature_idx] = {
                    'balance': balance,
                    'effective_bins': effective_bins,
                    'bin_counts': bin_counts,
                    'sparsest_bin': min(bin_counts),
                    'fullest_bin': max(bin_counts)
                }
        
        return results
    
    return documented_config, validate_flexible_config, monitor_binning_performance

# Demonstrate best practices
config, validator, monitor = create_flexible_binning_best_practices()

print("\\nBest Practices Implementation:")
print("\\n1. Configuration Documentation:")
for feature_idx, feature_config in config.items():
    print(f"\\nFeature {feature_idx}:")
    print(f"  Type: {feature_config['type']}")
    print(f"  Rationale: {feature_config['rationale']}")
    print(f"  Expert: {feature_config['domain_expert']}")
    print(f"  Updated: {feature_config['last_updated']}")

print("\\n2. Validation Process:")
print("  ✓ Check data coverage")
print("  ✓ Verify interval continuity")
print("  ✓ Validate configuration format")
print("  ✓ Monitor bin distributions")

print("\\n3. Monitoring Metrics:")
print("  • Balance ratio (std/mean of bin counts)")
print("  • Effective bins (non-empty bins)")
print("  • Sparsest bin warning threshold")
print("  • Configuration drift detection")
```

This comprehensive example documentation for Manual Flexible Binning covers:

1. **Basic Usage**: Multi-feature binning with different strategies per feature
2. **Real-world Applications**: E-commerce segmentation, healthcare risk assessment, financial portfolio analysis
3. **Advanced Techniques**: Mixed strategies, dynamic configuration updates
4. **Best Practices**: Documentation, validation, monitoring, and when to use flexible binning

Each example demonstrates how manual flexible binning provides maximum control and customization for complex multi-feature datasets where different features require different binning approaches.
