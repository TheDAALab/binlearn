# Supervised Binning Examples

This page demonstrates the use of `SupervisedBinning` for creating bins that are optimized for predictive modeling by considering the target variable.

## Basic Usage

### Understanding Supervised Binning

```python
import numpy as np
import matplotlib.pyplot as plt
from binlearn.methods import SupervisedBinning, EqualWidthBinning, EqualFrequencyBinning

# Create data with clear relationship to target
np.random.seed(42)
n_samples = 2000

# Feature with non-linear relationship to target
X = np.random.uniform(0, 10, n_samples).reshape(-1, 1)

# Create target with step-wise relationship
y = np.zeros(n_samples)
y[X.flatten() < 2] = 0.1  # Low risk zone
y[(X.flatten() >= 2) & (X.flatten() < 4)] = 0.3  # Medium risk zone
y[(X.flatten() >= 4) & (X.flatten() < 7)] = 0.7  # High risk zone  
y[X.flatten() >= 7] = 0.9  # Very high risk zone

# Add noise and convert to binary classification
y += np.random.normal(0, 0.1, n_samples)
y = (y > 0.5).astype(int)

print(f"Feature range: {X.min():.2f} to {X.max():.2f}")
print(f"Target distribution: {np.bincount(y)}")

# Compare different binning methods
methods = {
    'Equal Width': EqualWidthBinning(n_bins=4),
    'Equal Frequency': EqualFrequencyBinning(n_bins=4),
    'Supervised': SupervisedBinning(n_bins=4, task_type='classification')
}

plt.figure(figsize=(15, 10))

# Original data
plt.subplot(2, 3, 1)
colors = ['blue', 'red']
for label in [0, 1]:
    mask = y == label
    plt.scatter(X[mask], np.random.normal(label, 0.1, mask.sum()), 
               c=colors[label], alpha=0.6, label=f'Target {label}')
plt.title('Original Data Distribution')
plt.xlabel('Feature Value')
plt.ylabel('Target + Noise')
plt.legend()

# Compare binning methods
for i, (name, method) in enumerate(methods.items(), 2):
    if name == 'Supervised':
        # Supervised binning uses guidance_data parameter
        X_binned = method.fit_transform(X, guidance_data=y)
    else:
        X_binned = method.fit_transform(X)
    
    plt.subplot(2, 3, i)
    
    # Calculate target rate per bin
    bin_target_rates = []
    bin_labels = []
    for bin_id in np.unique(X_binned):
        mask = X_binned.flatten() == bin_id
        target_rate = y[mask].mean()
        bin_size = mask.sum()
        bin_target_rates.append(target_rate)
        bin_labels.append(f'Bin {bin_id}\n(n={bin_size})')
    
    bars = plt.bar(range(len(bin_target_rates)), bin_target_rates, 
                   alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Bin')
    plt.ylabel('Target Rate')
    plt.title(f'{name} Binning')
    plt.xticks(range(len(bin_labels)), bin_labels)
    
    # Add target rate labels on bars
    for bar, rate in zip(bars, bin_target_rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rate:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Print bin boundaries for comparison
for name, method in methods.items():
    edges = method.bin_edges_[0]
    print(f"\n{name} bin edges: {edges}")
```

### Classification Task Example

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from binlearn.methods import SupervisedBinning

# Create classification dataset
X, y = make_classification(
    n_samples=1000, 
    n_features=5, 
    n_informative=3,
    n_redundant=1,
    n_clusters_per_class=1,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create supervised binner with custom tree parameters
sup_binner = SupervisedBinning(
    n_bins=4,
    task_type='classification',
    tree_params={
        'max_depth': 3,
        'min_samples_split': 50,
        'min_samples_leaf': 20,
        'random_state': 42
    }
)

# Fit and transform
X_train_binned = sup_binner.fit_transform(X_train, guidance_data=y_train)
X_test_binned = sup_binner.transform(X_test)

# Compare performance
classifier = RandomForestClassifier(random_state=42, n_estimators=100)

# Original data performance
classifier.fit(X_train, y_train)
y_pred_orig = classifier.predict(X_test)
accuracy_orig = accuracy_score(y_test, y_pred_orig)

# Binned data performance  
classifier.fit(X_train_binned, y_train)
y_pred_binned = classifier.predict(X_test_binned)
accuracy_binned = accuracy_score(y_test, y_pred_binned)

print(f"Original data accuracy: {accuracy_orig:.3f}")
print(f"Supervised binned accuracy: {accuracy_binned:.3f}")
print(f"Accuracy difference: {accuracy_binned - accuracy_orig:.3f}")

# Show bin information
print("\nBin edges per feature:")
for i in range(X.shape[1]):
    edges = sup_binner.bin_edges_[i]
    print(f"Feature {i}: {len(edges)-1} bins, edges={edges}")
```

### Regression Task Example

```python
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Create regression dataset
X, y = make_regression(
    n_samples=1000,
    n_features=4,
    noise=0.1,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create supervised binner for regression
reg_binner = SupervisedBinning(
    n_bins=5,
    task_type='regression',
    tree_params={
        'max_depth': 4,
        'min_samples_split': 30,
        'min_samples_leaf': 15
    }
)

# Bin the data
X_train_binned = reg_binner.fit_transform(X_train, guidance_data=y_train)
X_test_binned = reg_binner.transform(X_test)

# Compare regression performance
regressor = RandomForestRegressor(random_state=42, n_estimators=100)

# Original data performance
regressor.fit(X_train, y_train)
y_pred_orig = regressor.predict(X_test)
mse_orig = mean_squared_error(y_test, y_pred_orig)

# Binned data performance
regressor.fit(X_train_binned, y_train) 
y_pred_binned = regressor.predict(X_test_binned)
mse_binned = mean_squared_error(y_test, y_pred_binned)

print(f"Original data MSE: {mse_orig:.3f}")
print(f"Supervised binned MSE: {mse_binned:.3f}")
print(f"MSE improvement: {mse_orig - mse_binned:.3f}")
```

### DataFrame Support

```python
import pandas as pd
from binlearn.methods import SupervisedBinning

# Create DataFrame with mixed data types
df = pd.DataFrame({
    'age': np.random.normal(35, 10, 500),
    'income': np.random.lognormal(10, 1, 500),
    'score': np.random.uniform(0, 100, 500),
    'category': np.random.choice(['A', 'B', 'C'], 500)
})

# Create binary target based on complex rules
target = (
    (df['age'] > 40) & (df['income'] > 30000) & (df['score'] > 70)
).astype(int)

# Apply supervised binning to numeric columns only
numeric_cols = ['age', 'income', 'score']
sup_binner = SupervisedBinning(
    n_bins=3,
    task_type='classification',
    columns=numeric_cols,
    preserve_dataframe=True
)

df_binned = sup_binner.fit_transform(df, guidance_data=target)

print("Original DataFrame:")
print(df[numeric_cols].head())
print("\nBinned DataFrame:")
print(df_binned[numeric_cols].head())
print("\nBin edges:")
for col in numeric_cols:
    print(f"{col}: {sup_binner.bin_edges_[col]}")
```

### Advanced: Custom Tree Parameters

```python
from binlearn.methods import SupervisedBinning
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Load wine dataset
wine = load_wine()
X, y = wine.data, wine.target

# Define different tree parameter configurations
param_configs = [
    {
        'name': 'Conservative',
        'params': {
            'max_depth': 2,
            'min_samples_split': 100,
            'min_samples_leaf': 50
        }
    },
    {
        'name': 'Moderate', 
        'params': {
            'max_depth': 3,
            'min_samples_split': 50,
            'min_samples_leaf': 20
        }
    },
    {
        'name': 'Aggressive',
        'params': {
            'max_depth': 5,
            'min_samples_split': 20,
            'min_samples_leaf': 10
        }
    }
]

classifier = RandomForestClassifier(random_state=42, n_estimators=50)

print("Cross-validation results:")
print(f"Original data: {cross_val_score(classifier, X, y, cv=5).mean():.3f}")

for config in param_configs:
    binner = SupervisedBinning(
        n_bins=4,
        task_type='classification',
        tree_params=config['params']
    )
    
    X_binned = binner.fit_transform(X, guidance_data=y)
    score = cross_val_score(classifier, X_binned, y, cv=5).mean()
    
    print(f"{config['name']} binning: {score:.3f}")
    
    # Show average bins created per feature
    avg_bins = np.mean([len(edges)-1 for edges in binner.bin_edges_.values()])
    print(f"  Average bins per feature: {avg_bins:.1f}")
```

## Best Practices

### 1. Choose Appropriate Task Type

```python
# For classification problems
clf_binner = SupervisedBinning(task_type='classification')

# For regression problems  
reg_binner = SupervisedBinning(task_type='regression')
```

### 2. Regularize with Tree Parameters

```python
# Prevent overfitting with conservative parameters
conservative_binner = SupervisedBinning(
    n_bins=5,
    tree_params={
        'max_depth': 3,           # Limit tree depth
        'min_samples_split': 50,  # Require more samples to split
        'min_samples_leaf': 20    # Require more samples in leaves
    }
)
```

### 3. Validate Binning Quality

```python
def evaluate_binning_quality(binner, X, y):
    """Evaluate the quality of supervised binning."""
    X_binned = binner.fit_transform(X, guidance_data=y)
    
    # Check target separation per feature
    for i in range(X.shape[1]):
        bin_values = X_binned[:, i]
        print(f"Feature {i}:")
        
        for bin_id in np.unique(bin_values):
            mask = bin_values == bin_id
            target_mean = y[mask].mean()
            bin_size = mask.sum()
            print(f"  Bin {bin_id}: target_mean={target_mean:.3f}, size={bin_size}")
        print()

# Example usage
binner = SupervisedBinning(n_bins=3, task_type='classification')
evaluate_binning_quality(binner, X, y)
```

This comprehensive guide shows how to effectively use supervised binning for both classification and regression tasks, with practical examples and best practices.
