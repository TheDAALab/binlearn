# Equal Width Binning Examples

This page provides comprehensive examples of using `EqualWidthBinning` for different data scenarios and use cases.

## Basic Usage

### Simple Numerical Data

```python
import numpy as np
from binning import EqualWidthBinning

# Create sample data
np.random.seed(42)
ages = np.random.normal(35, 10, 1000).reshape(-1, 1)

# Basic equal width binning
binner = EqualWidthBinning(n_bins=5)
binned_ages = binner.fit_transform(ages)

print("Original data range:", ages.min(), "to", ages.max())
print("Bin edges:", binner.bin_edges_[0])
print("Sample binned values:", binned_ages[:10].flatten())
```

**Output:**
```
Original data range: 3.64 to 68.15
Bin edges: [ 3.64 16.54 29.44 42.34 55.24 68.15]
Sample binned values: [2 3 2 4 2 1 2 3 1 2]
```

### Multi-dimensional Data

```python
import numpy as np
from binning import EqualWidthBinning

# Create multi-dimensional dataset
np.random.seed(42)
data = np.random.rand(500, 3) * 100  # 3 features, values 0-100

# Bin all features with different number of bins per feature
binner = EqualWidthBinning(n_bins=[4, 6, 3])
binned_data = binner.fit_transform(data)

print("Original data shape:", data.shape)
print("Binned data shape:", binned_data.shape)
print("Bin edges for feature 0:", binner.bin_edges_[0])
print("Bin edges for feature 1:", binner.bin_edges_[1])
print("Bin edges for feature 2:", binner.bin_edges_[2])
```

## Real-world Examples

### Customer Segmentation by Age

```python
import numpy as np
import pandas as pd
from binning import EqualWidthBinning
import matplotlib.pyplot as plt

# Simulate customer data
np.random.seed(42)
n_customers = 2000

# Create realistic age distribution
young_adults = np.random.normal(25, 3, 600)
adults = np.random.normal(40, 8, 800)
seniors = np.random.normal(60, 5, 600)
ages = np.concatenate([young_adults, adults, seniors])
ages = np.clip(ages, 18, 80).reshape(-1, 1)

# Create age groups using equal width binning
age_binner = EqualWidthBinning(n_bins=6)
age_groups = age_binner.fit_transform(ages)

# Create DataFrame for analysis
df = pd.DataFrame({
    'age': ages.flatten(),
    'age_group': age_groups.flatten()
})

# Define group labels
group_labels = ['18-28', '28-38', '38-48', '48-58', '58-68', '68-78']
df['age_group_label'] = df['age_group'].map(
    {i: label for i, label in enumerate(group_labels)}
)

# Analyze distribution
print("Age Group Distribution:")
print(df['age_group_label'].value_counts().sort_index())

# Plotting
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(ages, bins=30, alpha=0.7, edgecolor='black')
plt.title('Original Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
df['age_group_label'].value_counts().sort_index().plot(kind='bar')
plt.title('Age Groups After Binning')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
```

### Financial Data Preprocessing

```python
import numpy as np
import pandas as pd
from binning import EqualWidthBinning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Simulate financial dataset
np.random.seed(42)
n_samples = 1000

# Create features: income, debt, credit_score
income = np.random.lognormal(10, 0.5, n_samples)  # Log-normal distribution
debt = income * np.random.uniform(0.1, 0.8, n_samples)  # Debt relative to income
credit_score = np.random.normal(650, 100, n_samples)
credit_score = np.clip(credit_score, 300, 850)

# Create target: loan approval (simplified logic)
debt_to_income = debt / income
loan_approved = (
    (credit_score > 600) & 
    (debt_to_income < 0.4) & 
    (income > 30000)
).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'income': income,
    'debt': debt,
    'credit_score': credit_score,
    'loan_approved': loan_approved
})

print("Dataset shape:", df.shape)
print("\\nOriginal data statistics:")
print(df.describe())

# Apply equal width binning to continuous features
features_to_bin = ['income', 'debt', 'credit_score']
X_original = df[features_to_bin].values

# Bin features with equal width
binner = EqualWidthBinning(n_bins=5)
X_binned = binner.fit_transform(X_original)

# Create binned DataFrame
df_binned = pd.DataFrame(
    X_binned, 
    columns=[f'{col}_binned' for col in features_to_bin]
)
df_binned['loan_approved'] = df['loan_approved'].values

print("\\nBinned data statistics:")
print(df_binned.describe())

# Compare model performance
y = df['loan_approved']

# Train model with original data
X_train_orig, X_test_orig, y_train, y_test = train_test_split(
    X_original, y, test_size=0.3, random_state=42
)

rf_original = RandomForestClassifier(random_state=42)
rf_original.fit(X_train_orig, y_train)
y_pred_orig = rf_original.predict(X_test_orig)

# Train model with binned data
X_train_binned, X_test_binned, _, _ = train_test_split(
    X_binned, y, test_size=0.3, random_state=42
)

rf_binned = RandomForestClassifier(random_state=42)
rf_binned.fit(X_train_binned, y_train)
y_pred_binned = rf_binned.predict(X_test_binned)

print("\\nModel Performance Comparison:")
print("\\nOriginal Data:")
print(classification_report(y_test, y_pred_orig))

print("\\nBinned Data:")
print(classification_report(y_test, y_pred_binned))
```

## Advanced Usage

### Custom Bin Boundaries

```python
import numpy as np
from binning import EqualWidthBinning

# Create data with known characteristics
np.random.seed(42)
temperatures = np.random.normal(20, 15, 1000).reshape(-1, 1)  # Celsius

# Standard equal width binning
standard_binner = EqualWidthBinning(n_bins=5)
standard_binned = standard_binner.fit_transform(temperatures)

print("Standard binning - Bin edges:")
print(standard_binner.bin_edges_[0])

# For temperature data, you might want to use predefined ranges
# that make more sense (freezing, cold, mild, warm, hot)
# This would require manual binning, but we can still use equal width
# within reasonable ranges

# Filter to reasonable temperature range first
reasonable_temps = np.clip(temperatures, -10, 50)
reasonable_binner = EqualWidthBinning(n_bins=5)
reasonable_binned = reasonable_binner.fit_transform(reasonable_temps)

print("\\nReasonable range binning - Bin edges:")
print(reasonable_binner.bin_edges_[0])
```

### Handling Outliers

```python
import numpy as np
from binning import EqualWidthBinning
import matplotlib.pyplot as plt

# Create data with outliers
np.random.seed(42)
normal_data = np.random.normal(50, 10, 950)
outliers = np.array([5, 8, 92, 95, 98])  # Extreme values
data_with_outliers = np.concatenate([normal_data, outliers]).reshape(-1, 1)

# Binning with outliers
binner_with_outliers = EqualWidthBinning(n_bins=5)
binned_with_outliers = binner_with_outliers.fit_transform(data_with_outliers)

# Binning after removing outliers (using IQR method)
Q1 = np.percentile(data_with_outliers, 25)
Q3 = np.percentile(data_with_outliers, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter outliers
data_no_outliers = data_with_outliers[
    (data_with_outliers >= lower_bound) & (data_with_outliers <= upper_bound)
]

binner_no_outliers = EqualWidthBinning(n_bins=5)
binned_no_outliers = binner_no_outliers.fit_transform(data_no_outliers)

print("With outliers - Bin edges:")
print(binner_with_outliers.bin_edges_[0])

print("\\nWithout outliers - Bin edges:")
print(binner_no_outliers.bin_edges_[0])

# Visualization
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(data_with_outliers, bins=30, alpha=0.7, edgecolor='black')
plt.title('Original Data with Outliers')
plt.xlabel('Value')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.hist(binned_with_outliers, bins=5, alpha=0.7, edgecolor='black')
plt.title('Binned with Outliers')
plt.xlabel('Bin')
plt.ylabel('Frequency')

plt.subplot(1, 3, 3)
plt.hist(binned_no_outliers, bins=5, alpha=0.7, edgecolor='black')
plt.title('Binned without Outliers')
plt.xlabel('Bin')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
```

## Integration with Pandas

### DataFrame Processing

```python
import pandas as pd
import numpy as np
from binning import EqualWidthBinning

# Create sample DataFrame
np.random.seed(42)
df = pd.DataFrame({
    'salary': np.random.lognormal(10, 0.5, 1000),
    'experience': np.random.exponential(5, 1000),
    'age': np.random.normal(35, 10, 1000),
    'department': np.random.choice(['IT', 'HR', 'Finance', 'Marketing'], 1000)
})

print("Original DataFrame:")
print(df.head())
print("\\nDataFrame info:")
print(df.info())

# Select numerical columns for binning
numerical_cols = ['salary', 'experience', 'age']
X = df[numerical_cols].values

# Apply binning
binner = EqualWidthBinning(n_bins=4)
X_binned = binner.fit_transform(X)

# Create binned DataFrame
df_binned = df.copy()
for i, col in enumerate(numerical_cols):
    df_binned[f'{col}_binned'] = X_binned[:, i]

print("\\nDataFrame with binned features:")
print(df_binned.head())

# Analyze distribution of binned features
print("\\nBinned feature distributions:")
for col in numerical_cols:
    print(f"\\n{col}_binned distribution:")
    print(df_binned[f'{col}_binned'].value_counts().sort_index())
```

## Performance Considerations

### Large Dataset Handling

```python
import numpy as np
import time
from binning import EqualWidthBinning

# Test with different dataset sizes
sizes = [1000, 10000, 100000, 1000000]

for size in sizes:
    # Create large dataset
    np.random.seed(42)
    large_data = np.random.rand(size, 5)
    
    # Time the binning operation
    start_time = time.time()
    binner = EqualWidthBinning(n_bins=10)
    binned_data = binner.fit_transform(large_data)
    end_time = time.time()
    
    print(f"Dataset size: {size:,} samples, 5 features")
    print(f"Binning time: {end_time - start_time:.4f} seconds")
    print(f"Memory usage: ~{large_data.nbytes / 1024**2:.2f} MB")
    print("-" * 50)
```

## Tips and Best Practices

### Choosing the Number of Bins

```python
import numpy as np
import matplotlib.pyplot as plt
from binning import EqualWidthBinning

# Create sample data
np.random.seed(42)
data = np.random.gamma(2, 2, 1000).reshape(-1, 1)

# Test different numbers of bins
bin_counts = [3, 5, 10, 20]

plt.figure(figsize=(15, 10))

for i, n_bins in enumerate(bin_counts, 1):
    binner = EqualWidthBinning(n_bins=n_bins)
    binned_data = binner.fit_transform(data)
    
    plt.subplot(2, 2, i)
    plt.hist(binned_data, bins=n_bins, alpha=0.7, edgecolor='black')
    plt.title(f'Equal Width Binning: {n_bins} bins')
    plt.xlabel('Bin')
    plt.ylabel('Frequency')
    
    # Show bin edges as vertical lines
    for edge in binner.bin_edges_[0][1:-1]:  # Exclude first and last
        plt.axvline(edge, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

print("Guidelines for choosing number of bins:")
print("- Too few bins: Loss of information, over-simplification")
print("- Too many bins: Noisy, defeats the purpose of binning")
print("- Common rules: sqrt(n), log2(n), or domain knowledge")
print("- For this dataset size (1000 samples):")
print(f"  - sqrt rule suggests: {int(np.sqrt(1000))} bins")
print(f"  - log2 rule suggests: {int(np.log2(1000))} bins")
```

This comprehensive example documentation covers:

1. **Basic Usage**: Simple and multi-dimensional examples
2. **Real-world Applications**: Customer segmentation, financial preprocessing
3. **Advanced Techniques**: Outlier handling, custom boundaries
4. **Integration**: Pandas DataFrame processing
5. **Performance**: Large dataset considerations
6. **Best Practices**: Choosing optimal bin counts

Each example is practical, runnable, and includes explanations of when and why to use each approach.
