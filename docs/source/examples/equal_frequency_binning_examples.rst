# Equal Frequency Binning Examples

This page demonstrates the use of `EqualFrequencyBinning` for creating bins with approximately equal sample counts.

## Basic Usage

### Understanding Equal Frequency vs Equal Width

```python
import numpy as np
import matplotlib.pyplot as plt
from binning import EqualWidthBinning, EqualFrequencyBinning

# Create skewed data to highlight differences
np.random.seed(42)
skewed_data = np.random.exponential(2, 1000).reshape(-1, 1)

# Apply both binning methods
equal_width = EqualWidthBinning(n_bins=5)
equal_freq = EqualFrequencyBinning(n_bins=5)

width_binned = equal_width.fit_transform(skewed_data)
freq_binned = equal_freq.fit_transform(skewed_data)

print("Equal Width Bin Edges:", equal_width.bin_edges_[0])
print("Equal Frequency Quantiles:", equal_freq.quantiles_[0])

# Compare distributions
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(skewed_data, bins=30, alpha=0.7, edgecolor='black')
plt.title('Original Skewed Data')
plt.xlabel('Value')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
unique, counts = np.unique(width_binned, return_counts=True)
plt.bar(unique, counts, alpha=0.7, edgecolor='black')
plt.title('Equal Width Binning\\n(Uneven frequencies)')
plt.xlabel('Bin')
plt.ylabel('Count')

plt.subplot(1, 3, 3)
unique, counts = np.unique(freq_binned, return_counts=True)
plt.bar(unique, counts, alpha=0.7, edgecolor='black')
plt.title('Equal Frequency Binning\\n(Even frequencies)')
plt.xlabel('Bin')
plt.ylabel('Count')

plt.tight_layout()
plt.show()
```

### Multi-feature Equal Frequency Binning

```python
import numpy as np
from binning import EqualFrequencyBinning

# Create multi-dimensional data with different distributions
np.random.seed(42)
feature1 = np.random.exponential(1, 800)      # Exponential
feature2 = np.random.pareto(1, 800)           # Pareto (heavy-tailed)
feature3 = np.random.lognormal(1, 0.5, 800)  # Log-normal

data = np.column_stack([feature1, feature2, feature3])

# Apply equal frequency binning
binner = EqualFrequencyBinning(n_bins=4)
binned_data = binner.fit_transform(data)

print("Original data shape:", data.shape)
print("Binned data shape:", binned_data.shape)

# Show quantiles for each feature
for i in range(3):
    print(f"\\nFeature {i+1} quantiles:")
    print(binner.quantiles_[i])
    
    # Verify equal frequencies
    unique, counts = np.unique(binned_data[:, i], return_counts=True)
    print(f"Bin counts: {counts}")
    print(f"Standard deviation of counts: {np.std(counts):.2f}")
```

## Real-world Applications

### Customer Lifetime Value Segmentation

```python
import numpy as np
import pandas as pd
from binning import EqualFrequencyBinning
import matplotlib.pyplot as plt

# Simulate customer lifetime value data (typically skewed)
np.random.seed(42)
n_customers = 2000

# Create realistic CLV distribution (log-normal with some adjustment)
base_clv = np.random.lognormal(mean=6, sigma=1, size=n_customers)
# Add some high-value customers
high_value = np.random.lognormal(mean=9, sigma=0.5, size=200)
clv_data = np.concatenate([base_clv, high_value]).reshape(-1, 1)

# Apply equal frequency binning to create customer tiers
clv_binner = EqualFrequencyBinning(n_bins=5)
customer_tiers = clv_binner.fit_transform(clv_data)

# Create DataFrame for analysis
df = pd.DataFrame({
    'customer_id': range(len(clv_data)),
    'clv': clv_data.flatten(),
    'tier': customer_tiers.flatten()
})

# Define tier labels
tier_labels = {0: 'Bronze', 1: 'Silver', 2: 'Gold', 3: 'Platinum', 4: 'Diamond'}
df['tier_label'] = df['tier'].map(tier_labels)

# Analysis
print("Customer Tier Distribution (Equal Frequency):")
tier_counts = df['tier_label'].value_counts()
print(tier_counts)

print("\\nCLV Statistics by Tier:")
tier_stats = df.groupby('tier_label')['clv'].agg(['count', 'mean', 'min', 'max'])
print(tier_stats)

# Visualization
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(clv_data, bins=50, alpha=0.7, edgecolor='black')
plt.title('Customer Lifetime Value Distribution')
plt.xlabel('CLV ($)')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
tier_counts.plot(kind='bar')
plt.title('Customer Tier Distribution')
plt.xlabel('Tier')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)

plt.subplot(1, 3, 3)
df.boxplot(column='clv', by='tier_label', ax=plt.gca())
plt.title('CLV Distribution by Tier')
plt.xlabel('Tier')
plt.ylabel('CLV ($)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
```

### Risk Score Binning for Credit Assessment

```python
import numpy as np
import pandas as pd
from binning import EqualFrequencyBinning
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Simulate credit risk dataset
np.random.seed(42)
n_samples = 5000

# Create features that influence credit risk
income = np.random.lognormal(10, 0.6, n_samples)
debt_to_income = np.random.beta(2, 5, n_samples)  # Skewed towards lower values
credit_history = np.random.exponential(5, n_samples)  # Years of credit history
utilization = np.random.beta(1.5, 3, n_samples)  # Credit utilization ratio

# Create target variable (default risk)
risk_score = (
    -0.3 * np.log(income/50000) +  # Higher income = lower risk
    2.0 * debt_to_income +         # Higher debt ratio = higher risk
    -0.1 * credit_history +        # Longer history = lower risk
    1.5 * utilization +            # Higher utilization = higher risk
    np.random.normal(0, 0.5, n_samples)  # Random noise
)

# Convert to binary default indicator
default_prob = 1 / (1 + np.exp(-risk_score))  # Sigmoid transformation
defaults = (default_prob > 0.5).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'income': income,
    'debt_to_income': debt_to_income,
    'credit_history': credit_history,
    'utilization': utilization,
    'risk_score': risk_score,
    'default': defaults
})

print("Dataset Overview:")
print(df.describe())
print(f"\\nDefault rate: {defaults.mean():.2%}")

# Apply equal frequency binning to create risk categories
features_to_bin = ['income', 'debt_to_income', 'credit_history', 'utilization']
X_original = df[features_to_bin].values

# Use equal frequency binning for balanced risk categories
risk_binner = EqualFrequencyBinning(n_bins=5)
X_binned = risk_binner.fit_transform(X_original)

# Create risk categories
df_risk = df.copy()
for i, feature in enumerate(features_to_bin):
    df_risk[f'{feature}_risk_cat'] = X_binned[:, i]

# Analyze risk by categories
print("\\nDefault Rate by Risk Categories:")
for feature in features_to_bin:
    risk_cat_col = f'{feature}_risk_cat'
    risk_analysis = df_risk.groupby(risk_cat_col)['default'].agg(['count', 'mean'])
    print(f"\\n{feature}:")
    print(risk_analysis)

# Model comparison
X_train, X_test, y_train, y_test = train_test_split(
    X_original, defaults, test_size=0.2, random_state=42
)

X_train_binned, X_test_binned, _, _ = train_test_split(
    X_binned, defaults, test_size=0.2, random_state=42
)

# Train models
gb_original = GradientBoostingClassifier(random_state=42)
gb_original.fit(X_train, y_train)

gb_binned = GradientBoostingClassifier(random_state=42)
gb_binned.fit(X_train_binned, y_train)

# Evaluate
y_pred_orig = gb_original.predict(X_test)
y_pred_binned = gb_binned.predict(X_test_binned)

y_prob_orig = gb_original.predict_proba(X_test)[:, 1]
y_prob_binned = gb_binned.predict_proba(X_test_binned)[:, 1]

print("\\nModel Performance Comparison:")
print("\\nOriginal Features:")
print(f"AUC: {roc_auc_score(y_test, y_prob_orig):.3f}")
print(classification_report(y_test, y_pred_orig))

print("\\nBinned Features (Equal Frequency):")
print(f"AUC: {roc_auc_score(y_test, y_prob_binned):.3f}")
print(classification_report(y_test, y_pred_binned))
```

### Market Research: Survey Response Analysis

```python
import numpy as np
import pandas as pd
from binning import EqualFrequencyBinning
import matplotlib.pyplot as plt

# Simulate survey response data (satisfaction scores)
np.random.seed(42)
n_responses = 1500

# Create different response patterns for different products
product_a_scores = np.random.beta(8, 2, 500) * 10  # Generally high satisfaction
product_b_scores = np.random.beta(3, 3, 500) * 10  # Mixed satisfaction
product_c_scores = np.random.beta(2, 5, 500) * 10  # Generally low satisfaction

all_scores = np.concatenate([product_a_scores, product_b_scores, product_c_scores])
products = ['Product A'] * 500 + ['Product B'] * 500 + ['Product C'] * 500

df = pd.DataFrame({
    'product': products,
    'satisfaction_score': all_scores
})

print("Survey Data Overview:")
print(df.groupby('product')['satisfaction_score'].describe())

# Apply equal frequency binning to create satisfaction levels
score_data = df['satisfaction_score'].values.reshape(-1, 1)
satisfaction_binner = EqualFrequencyBinning(n_bins=5)
satisfaction_levels = satisfaction_binner.fit_transform(score_data)

df['satisfaction_level'] = satisfaction_levels.flatten()

# Define level labels
level_labels = {
    0: 'Very Dissatisfied',
    1: 'Dissatisfied', 
    2: 'Neutral',
    3: 'Satisfied',
    4: 'Very Satisfied'
}
df['satisfaction_label'] = df['satisfaction_level'].map(level_labels)

print("\\nSatisfaction Level Distribution (Equal Frequency):")
print(df['satisfaction_label'].value_counts())

print("\\nScore Ranges by Satisfaction Level:")
level_ranges = df.groupby('satisfaction_label')['satisfaction_score'].agg(['min', 'max', 'mean'])
print(level_ranges)

# Cross-tabulation analysis
print("\\nSatisfaction by Product:")
cross_tab = pd.crosstab(df['product'], df['satisfaction_label'], normalize='index') * 100
print(cross_tab.round(1))

# Visualization
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
df['satisfaction_score'].hist(bins=30, alpha=0.7, edgecolor='black')
plt.title('Original Satisfaction Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')

plt.subplot(2, 2, 2)
df['satisfaction_label'].value_counts().plot(kind='bar')
plt.title('Equal Frequency Satisfaction Levels')
plt.xlabel('Satisfaction Level')
plt.ylabel('Count')
plt.xticks(rotation=45)

plt.subplot(2, 2, 3)
cross_tab.plot(kind='bar', stacked=True)
plt.title('Satisfaction Distribution by Product')
plt.xlabel('Product')
plt.ylabel('Percentage')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.subplot(2, 2, 4)
df.boxplot(column='satisfaction_score', by='product')
plt.title('Score Distribution by Product')
plt.xlabel('Product')
plt.ylabel('Satisfaction Score')

plt.tight_layout()
plt.show()
```

## Advanced Usage

### Handling Missing Values

```python
import numpy as np
import pandas as pd
from binning import EqualFrequencyBinning

# Create data with missing values
np.random.seed(42)
n_samples = 1000

# Generate base data
complete_data = np.random.exponential(2, n_samples)

# Introduce missing values (MCAR - Missing Completely At Random)
missing_indices = np.random.choice(n_samples, size=100, replace=False)
data_with_missing = complete_data.copy()
data_with_missing[missing_indices] = np.nan

print(f"Missing values: {np.sum(np.isnan(data_with_missing))}/{n_samples}")

# Strategy 1: Remove missing values before binning
data_clean = data_with_missing[~np.isnan(data_with_missing)].reshape(-1, 1)
binner_clean = EqualFrequencyBinning(n_bins=5)
binned_clean = binner_clean.fit_transform(data_clean)

print("\\nStrategy 1 - Remove missing values:")
print(f"Data shape after removal: {data_clean.shape}")
print("Quantiles:", binner_clean.quantiles_[0])

# Strategy 2: Impute missing values before binning
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
data_imputed = imputer.fit_transform(data_with_missing.reshape(-1, 1))

binner_imputed = EqualFrequencyBinning(n_bins=5)
binned_imputed = binner_imputed.fit_transform(data_imputed)

print("\\nStrategy 2 - Impute missing values:")
print(f"Imputed value (median): {imputer.statistics_[0]:.3f}")
print("Quantiles:", binner_imputed.quantiles_[0])

# Compare strategies
print("\\nComparison of strategies:")
print("Clean data quantiles:", binner_clean.quantiles_[0])
print("Imputed data quantiles:", binner_imputed.quantiles_[0])
```

### Custom Percentile Binning

```python
import numpy as np
from binning import EqualFrequencyBinning

# Create heavily skewed data
np.random.seed(42)
skewed_data = np.random.pareto(1, 10000).reshape(-1, 1)

# Standard equal frequency (quintiles)
standard_binner = EqualFrequencyBinning(n_bins=5)
standard_binned = standard_binner.fit_transform(skewed_data)

print("Standard Equal Frequency Binning (Quintiles):")
print("Quantiles:", standard_binner.quantiles_[0])

# Different bin counts for different analysis needs
bin_counts = [3, 4, 10]
for n_bins in bin_counts:
    binner = EqualFrequencyBinning(n_bins=n_bins)
    binned = binner.fit_transform(skewed_data)
    
    print(f"\\n{n_bins}-bin Equal Frequency:")
    print("Quantiles:", binner.quantiles_[0])
    
    # Check actual frequencies
    unique, counts = np.unique(binned, return_counts=True)
    print("Actual counts:", counts)
    print("Std dev of counts:", np.std(counts))
```

## Integration with Machine Learning Pipelines

### Scikit-learn Pipeline Integration

```python
import numpy as np
from binning import EqualFrequencyBinning
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification

# Create synthetic dataset
X, y = make_classification(
    n_samples=1000, 
    n_features=10, 
    n_informative=5,
    n_redundant=2,
    random_state=42
)

# Create pipeline with equal frequency binning
pipeline = Pipeline([
    ('binning', EqualFrequencyBinning(n_bins=5)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Cross-validation
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')

print("Pipeline Performance with Equal Frequency Binning:")
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Compare with original features
rf_original = RandomForestClassifier(random_state=42)
cv_scores_original = cross_val_score(rf_original, X, y, cv=5, scoring='accuracy')

print("\\nComparison with Original Features:")
print(f"Original features CV score: {cv_scores_original.mean():.3f} (+/- {cv_scores_original.std() * 2:.3f})")
print(f"Binned features CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
```

## Performance and Memory Considerations

### Large Dataset Processing

```python
import numpy as np
import time
from binning import EqualFrequencyBinning

def benchmark_equal_frequency_binning():
    """Benchmark equal frequency binning with different dataset sizes."""
    
    sizes = [1000, 10000, 100000, 1000000]
    results = []
    
    for size in sizes:
        # Create test data
        np.random.seed(42)
        data = np.random.exponential(2, size).reshape(-1, 1)
        
        # Time the binning operation
        start_time = time.time()
        binner = EqualFrequencyBinning(n_bins=10)
        binned_data = binner.fit_transform(data)
        end_time = time.time()
        
        duration = end_time - start_time
        memory_mb = data.nbytes / (1024**2)
        
        results.append({
            'size': size,
            'time': duration,
            'memory_mb': memory_mb
        })
        
        print(f"Size: {size:,} | Time: {duration:.4f}s | Memory: {memory_mb:.2f}MB")
    
    return results

print("Equal Frequency Binning Performance Benchmark:")
benchmark_results = benchmark_equal_frequency_binning()
```

## Best Practices and Tips

### When to Use Equal Frequency Binning

```python
import numpy as np
import matplotlib.pyplot as plt
from binning import EqualWidthBinning, EqualFrequencyBinning

# Demonstrate scenarios where equal frequency is preferred

scenarios = {
    'Normal Distribution': np.random.normal(0, 1, 1000),
    'Exponential Distribution': np.random.exponential(1, 1000),
    'Power Law Distribution': np.random.pareto(1, 1000),
    'Uniform Distribution': np.random.uniform(0, 10, 1000)
}

fig, axes = plt.subplots(4, 3, figsize=(18, 16))

for i, (name, data) in enumerate(scenarios.items()):
    data = data.reshape(-1, 1)
    
    # Original distribution
    axes[i, 0].hist(data, bins=30, alpha=0.7, edgecolor='black')
    axes[i, 0].set_title(f'{name}\\nOriginal Data')
    
    # Equal width binning
    ew_binner = EqualWidthBinning(n_bins=5)
    ew_binned = ew_binner.fit_transform(data)
    unique, counts = np.unique(ew_binned, return_counts=True)
    axes[i, 1].bar(unique, counts, alpha=0.7)
    axes[i, 1].set_title(f'Equal Width\\nStd: {np.std(counts):.1f}')
    
    # Equal frequency binning
    ef_binner = EqualFrequencyBinning(n_bins=5)
    ef_binned = ef_binner.fit_transform(data)
    unique, counts = np.unique(ef_binned, return_counts=True)
    axes[i, 2].bar(unique, counts, alpha=0.7)
    axes[i, 2].set_title(f'Equal Frequency\\nStd: {np.std(counts):.1f}')

plt.tight_layout()
plt.show()

print("Guidelines for using Equal Frequency Binning:")
print("✅ GOOD for:")
print("  - Skewed distributions (exponential, power-law)")
print("  - When you need balanced sample sizes in each bin")
print("  - Ranking and percentile-based analysis")
print("  - Reducing the impact of outliers")
print("\\n❌ AVOID when:")
print("  - You need interpretable bin boundaries")
print("  - Domain knowledge suggests specific cut-points")
print("  - Working with approximately normal distributions")
print("  - Bin boundaries have business meaning")
```

This comprehensive example documentation for Equal Frequency Binning covers:

1. **Basic Usage**: Comparison with equal width, multi-feature examples
2. **Real-world Applications**: Customer segmentation, risk assessment, survey analysis
3. **Advanced Techniques**: Missing value handling, custom percentiles
4. **ML Integration**: Pipeline usage, performance comparison
5. **Performance**: Benchmarking with large datasets
6. **Best Practices**: When to choose equal frequency over other methods

Each example includes practical scenarios where equal frequency binning provides advantages over other approaches.
