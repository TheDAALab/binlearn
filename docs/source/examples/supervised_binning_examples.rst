# Supervised Binning Examples

This page demonstrates the use of `SupervisedBinning` for creating bins that are optimized for predictive modeling by considering the target variable.

## Basic Usage

### Understanding Supervised Binning

```python
import numpy as np
import matplotlib.pyplot as plt
from binning import SupervisedBinning, EqualWidthBinning, EqualFrequencyBinning

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

# Add noise
y += np.random.normal(0, 0.1, n_samples)
y = (y > 0.5).astype(int)  # Convert to binary

print(f"Feature range: {X.min():.2f} to {X.max():.2f}")
print(f"Target distribution: {np.bincount(y)}")

# Compare different binning methods
methods = {
    'Equal Width': EqualWidthBinning(n_bins=4),
    'Equal Frequency': EqualFrequencyBinning(n_bins=4),
    'Supervised': SupervisedBinning(n_bins=4)
}

plt.figure(figsize=(15, 10))

# Original data
plt.subplot(2, 3, 1)
colors = ['blue', 'red']
for label in [0, 1]:
    mask = y == label
    plt.scatter(X[mask], np.random.normal(label, 0.1, mask.sum()), 
               c=colors[label], alpha=0.6, label=f'Target {label}')
plt.xlabel('Feature Value')
plt.ylabel('Target (with jitter)')
plt.title('Original Data')
plt.legend()

# Compare binning methods
for i, (name, binner) in enumerate(methods.items(), 2):
    if name == 'Supervised':
        X_binned = binner.fit_transform(X, y)
    else:
        X_binned = binner.fit_transform(X)
    
    plt.subplot(2, 3, i)
    
    # Calculate target rate per bin
    bin_target_rates = []
    bin_labels = []
    for bin_id in np.unique(X_binned):
        mask = X_binned.flatten() == bin_id
        target_rate = y[mask].mean()
        bin_size = mask.sum()
        bin_target_rates.append(target_rate)
        bin_labels.append(f'Bin {bin_id}\\n(n={bin_size})')
    
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

# Show bin boundaries
print("\\nBin Boundaries Comparison:")
for name, binner in methods.items():
    if hasattr(binner, 'bin_edges_'):
        print(f"{name}: {binner.bin_edges_[0]}")
    elif hasattr(binner, 'quantiles_'):
        print(f"{name} (quantiles): {binner.quantiles_[0]}")
```

### Multi-feature Supervised Binning

```python
import numpy as np
import pandas as pd
from binning import SupervisedBinning
from sklearn.datasets import make_classification

# Create synthetic classification dataset
X, y = make_classification(
    n_samples=3000,
    n_features=4,
    n_informative=3,
    n_redundant=1,
    n_clusters_per_class=2,
    random_state=42
)

print("Dataset shape:", X.shape)
print("Target distribution:", np.bincount(y))

# Apply supervised binning
supervised_binner = SupervisedBinning(n_bins=5)
X_supervised = supervised_binner.fit_transform(X, y)

print("\\nSupervised binned shape:", X_supervised.shape)

# Compare with unsupervised binning
from binning import EqualFrequencyBinning
unsupervised_binner = EqualFrequencyBinning(n_bins=5)
X_unsupervised = unsupervised_binner.fit_transform(X)

# Evaluate information gain for each feature
from sklearn.metrics import mutual_info_score

print("\\nMutual Information with Target:")
print("Feature\\tOriginal\\tUnsupervised\\tSupervised")
print("-" * 50)

for i in range(X.shape[1]):
    mi_original = mutual_info_score(y, np.digitize(X[:, i], bins=5))
    mi_unsupervised = mutual_info_score(y, X_unsupervised[:, i])
    mi_supervised = mutual_info_score(y, X_supervised[:, i])
    
    print(f"Feature {i}\\t{mi_original:.3f}\\t\\t{mi_unsupervised:.3f}\\t\\t{mi_supervised:.3f}")
```

## Real-world Applications

### Credit Risk Assessment

```python
import numpy as np
import pandas as pd
from binning import SupervisedBinning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

# Simulate credit risk dataset
np.random.seed(42)
n_applicants = 10000

# Applicant features
age = np.random.normal(40, 15, n_applicants)
age = np.clip(age, 18, 80)

income = np.random.lognormal(10.5, 0.7, n_applicants)
debt_to_income = np.random.beta(2, 6, n_applicants)  # Skewed towards lower values
credit_history_years = np.random.exponential(8, n_applicants)
existing_credit_accounts = np.random.poisson(3, n_applicants)

# Create default probability based on risk factors
risk_score = (
    -0.02 * (age - 25) +  # Younger = higher risk
    -0.00002 * income +   # Lower income = higher risk
    3.0 * debt_to_income +  # Higher debt ratio = higher risk
    -0.05 * credit_history_years +  # Shorter history = higher risk
    0.1 * existing_credit_accounts +  # More accounts = slight increase in risk
    np.random.normal(0, 0.5, n_applicants)  # Random component
)

# Convert to binary default (1) / no default (0)
default_prob = 1 / (1 + np.exp(-risk_score))
defaults = (default_prob > 0.15).astype(int)  # ~15% default rate

# Create DataFrame
df = pd.DataFrame({
    'age': age,
    'income': income,
    'debt_to_income': debt_to_income,
    'credit_history_years': credit_history_years,
    'existing_accounts': existing_credit_accounts,
    'default': defaults
})

print("Credit Risk Dataset Overview:")
print(df.describe())
print(f"\\nDefault rate: {defaults.mean():.2%}")

# Prepare features
features = ['age', 'income', 'debt_to_income', 'credit_history_years', 'existing_accounts']
X = df[features].values
y = df['default'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Original continuous features
lr_continuous = LogisticRegression(random_state=42)
lr_continuous.fit(X_train, y_train)
y_prob_continuous = lr_continuous.predict_proba(X_test)[:, 1]

# Model 2: Supervised binning
supervised_binner = SupervisedBinning(n_bins=5)
X_train_supervised = supervised_binner.fit_transform(X_train, y_train)
X_test_supervised = supervised_binner.transform(X_test)

lr_supervised = LogisticRegression(random_state=42)
lr_supervised.fit(X_train_supervised, y_train)
y_prob_supervised = lr_supervised.predict_proba(X_test_supervised)[:, 1]

# Model 3: Equal frequency binning (for comparison)
from binning import EqualFrequencyBinning
ef_binner = EqualFrequencyBinning(n_bins=5)
X_train_ef = ef_binner.fit_transform(X_train)
X_test_ef = ef_binner.transform(X_test)

lr_ef = LogisticRegression(random_state=42)
lr_ef.fit(X_train_ef, y_train)
y_prob_ef = lr_ef.predict_proba(X_test_ef)[:, 1]

# Compare performance
print("\\nModel Performance Comparison:")
print(f"Continuous Features AUC: {roc_auc_score(y_test, y_prob_continuous):.4f}")
print(f"Equal Frequency Binning AUC: {roc_auc_score(y_test, y_prob_ef):.4f}")
print(f"Supervised Binning AUC: {roc_auc_score(y_test, y_prob_supervised):.4f}")

# Analyze risk scores by bins
print("\\nRisk Analysis by Features (Supervised Binning):")
df_analysis = pd.DataFrame(X_train_supervised, columns=[f'feature_{i}' for i in range(X_train_supervised.shape[1])])
df_analysis['default'] = y_train

for i, feature_name in enumerate(features):
    print(f"\\n{feature_name}:")
    risk_by_bin = df_analysis.groupby(f'feature_{i}')['default'].agg(['count', 'mean'])
    print(risk_by_bin)
    
    # Show original value ranges for each bin
    print("Bin ranges:")
    for bin_id in range(5):
        mask = X_train_supervised[:, i] == bin_id
        if mask.any():
            feature_values = X_train[:, i][mask]
            print(f"  Bin {bin_id}: {feature_values.min():.2f} to {feature_values.max():.2f}")
```

### Medical Diagnosis: Biomarker Analysis

```python
import numpy as np
import pandas as pd
from binning import SupervisedBinning
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Simulate medical biomarker dataset
np.random.seed(42)
n_patients = 5000

# Biomarker levels
# Healthy patients
healthy_count = 3500
biomarker_a_healthy = np.random.lognormal(3, 0.5, healthy_count)
biomarker_b_healthy = np.random.normal(50, 15, healthy_count)
biomarker_c_healthy = np.random.gamma(2, 10, healthy_count)

# Disease patients  
disease_count = 1500
biomarker_a_disease = np.random.lognormal(4.5, 0.8, disease_count)  # Higher levels
biomarker_b_disease = np.random.normal(80, 20, disease_count)  # Higher levels
biomarker_c_disease = np.random.gamma(5, 15, disease_count)  # Higher levels

# Combine data
biomarker_a = np.concatenate([biomarker_a_healthy, biomarker_a_disease])
biomarker_b = np.concatenate([biomarker_b_healthy, biomarker_b_disease])
biomarker_c = np.concatenate([biomarker_c_healthy, biomarker_c_disease])

disease_labels = np.concatenate([
    np.zeros(healthy_count),  # Healthy = 0
    np.ones(disease_count)    # Disease = 1
])

# Create DataFrame
df = pd.DataFrame({
    'biomarker_a': biomarker_a,
    'biomarker_b': biomarker_b,
    'biomarker_c': biomarker_c,
    'disease': disease_labels
})

print("Medical Dataset Overview:")
print(df.describe())
print(f"\\nDisease prevalence: {disease_labels.mean():.2%}")

# Apply supervised binning for diagnostic cut-points
X = df[['biomarker_a', 'biomarker_b', 'biomarker_c']].values
y = df['disease'].values

supervised_binner = SupervisedBinning(n_bins=4)  # Create diagnostic ranges
X_supervised = supervised_binner.fit_transform(X, y)

# Analyze diagnostic performance of each bin
biomarker_names = ['biomarker_a', 'biomarker_b', 'biomarker_c']

plt.figure(figsize=(15, 12))

for i, biomarker_name in enumerate(biomarker_names):
    # Original distribution
    plt.subplot(3, 3, i*3 + 1)
    plt.hist(X[y==0, i], bins=50, alpha=0.7, label='Healthy', color='blue')
    plt.hist(X[y==1, i], bins=50, alpha=0.7, label='Disease', color='red')
    plt.xlabel(biomarker_name)
    plt.ylabel('Frequency')
    plt.title(f'{biomarker_name} Distribution')
    plt.legend()
    
    # Supervised binning results
    plt.subplot(3, 3, i*3 + 2)
    bin_disease_rates = []
    bin_labels = []
    bin_ranges = []
    
    for bin_id in range(4):
        mask = X_supervised[:, i] == bin_id
        if mask.any():
            disease_rate = y[mask].mean()
            count = mask.sum()
            bin_disease_rates.append(disease_rate)
            bin_labels.append(f'Bin {bin_id}')
            
            # Get value range for this bin
            values_in_bin = X[mask, i]
            bin_ranges.append(f'{values_in_bin.min():.1f}-{values_in_bin.max():.1f}')
    
    bars = plt.bar(range(len(bin_disease_rates)), bin_disease_rates, 
                   alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('Diagnostic Range')
    plt.ylabel('Disease Rate')
    plt.title(f'{biomarker_name} Diagnostic Ranges')
    plt.xticks(range(len(bin_labels)), bin_labels)
    
    # Add rate labels
    for bar, rate in zip(bars, bin_disease_rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rate:.2f}', ha='center', va='bottom')
    
    # ROC-like analysis
    plt.subplot(3, 3, i*3 + 3)
    sensitivities = []
    specificities = []
    
    # Calculate sensitivity and specificity for each cut-point
    sorted_values = np.sort(X[:, i])
    thresholds = np.percentile(sorted_values, [25, 50, 75, 90])
    
    for threshold in thresholds:
        predictions = (X[:, i] > threshold).astype(int)
        
        # Calculate confusion matrix components
        tp = np.sum((predictions == 1) & (y == 1))
        tn = np.sum((predictions == 0) & (y == 0))
        fp = np.sum((predictions == 1) & (y == 0))
        fn = np.sum((predictions == 0) & (y == 1))
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        sensitivities.append(sensitivity)
        specificities.append(specificity)
    
    plt.scatter(1 - np.array(specificities), sensitivities, s=100, alpha=0.7)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title(f'{biomarker_name} ROC Points')

plt.tight_layout()
plt.show()

# Print diagnostic cut-points
print("\\nDiagnostic Cut-points (Supervised Binning):")
for i, biomarker_name in enumerate(biomarker_names):
    print(f"\\n{biomarker_name}:")
    for bin_id in range(4):
        mask = X_supervised[:, i] == bin_id
        if mask.any():
            values_in_bin = X[mask, i]
            disease_rate = y[mask].mean()
            print(f"  Range {values_in_bin.min():.1f}-{values_in_bin.max():.1f}: "
                  f"{disease_rate:.1%} disease rate ({mask.sum()} patients)")
```

### Marketing: Customer Response Prediction

```python
import numpy as np
import pandas as pd
from binning import SupervisedBinning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Simulate marketing campaign dataset
np.random.seed(42)
n_customers = 8000

# Customer characteristics
age = np.random.normal(45, 18, n_customers)
age = np.clip(age, 18, 80)

income = np.random.lognormal(10.8, 0.6, n_customers)
previous_purchases = np.random.poisson(12, n_customers)
days_since_last_purchase = np.random.exponential(60, n_customers)
email_engagement_score = np.random.beta(2, 5, n_customers) * 100  # 0-100 scale
website_visits_month = np.random.poisson(8, n_customers)

# Create campaign response based on customer profile
response_prob = (
    0.01 * (age - 20) / 60 +  # Middle-aged customers more responsive
    0.0001 * income / 100000 +  # Higher income slightly more responsive
    0.02 * np.minimum(previous_purchases / 20, 1) +  # Loyal customers more responsive
    -0.005 * np.minimum(days_since_last_purchase / 100, 1) +  # Recent customers more responsive
    0.008 * email_engagement_score / 100 +  # Engaged customers more responsive
    0.015 * np.minimum(website_visits_month / 15, 1) +  # Active users more responsive
    np.random.normal(0, 0.1, n_customers)  # Random component
)

# Convert to binary response
campaign_response = (response_prob > 0.25).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'age': age,
    'income': income,
    'previous_purchases': previous_purchases,
    'days_since_last': days_since_last_purchase,
    'email_engagement': email_engagement_score,
    'website_visits': website_visits_month,
    'response': campaign_response
})

print("Marketing Dataset Overview:")
print(df.describe())
print(f"\\nCampaign response rate: {campaign_response.mean():.2%}")

# Prepare features
features = ['age', 'income', 'previous_purchases', 'days_since_last', 
           'email_engagement', 'website_visits']
X = df[features].values
y = df['response'].values

# Apply supervised binning for customer segmentation
supervised_binner = SupervisedBinning(n_bins=4)
X_supervised = supervised_binner.fit_transform(X, y)

# Create customer segments based on response likelihood
segment_scores = X_supervised.mean(axis=1)  # Average bin across features
df['segment_score'] = segment_scores
df['segment'] = pd.qcut(segment_scores, q=4, labels=['Low', 'Medium', 'High', 'Premium'])

print("\\nCustomer Segments by Response Rate:")
segment_analysis = df.groupby('segment').agg({
    'response': ['count', 'mean'],
    'income': 'mean',
    'previous_purchases': 'mean',
    'email_engagement': 'mean'
}).round(3)

print(segment_analysis)

# Compare model performance
rf_original = RandomForestClassifier(random_state=42)
rf_supervised = RandomForestClassifier(random_state=42)

cv_original = cross_val_score(rf_original, X, y, cv=5, scoring='roc_auc')
cv_supervised = cross_val_score(rf_supervised, X_supervised, y, cv=5, scoring='roc_auc')

print(f"\\nModel Performance (5-fold CV AUC):")
print(f"Original features: {cv_original.mean():.4f} (+/- {cv_original.std() * 2:.4f})")
print(f"Supervised binning: {cv_supervised.mean():.4f} (+/- {cv_supervised.std() * 2:.4f})")

# Feature importance analysis
rf_supervised.fit(X_supervised, y)
feature_importance = pd.DataFrame({
    'feature': [f'{features[i//4]}_bin_{i%4}' for i in range(len(features)*4)],
    'importance': rf_supervised.feature_importances_
}).sort_values('importance', ascending=False)

print("\\nTop 10 Features (Supervised Binning):")
print(feature_importance.head(10))

# Campaign targeting recommendations
print("\\nCampaign Targeting Recommendations:")
for segment in ['Low', 'Medium', 'High', 'Premium']:
    segment_data = df[df['segment'] == segment]
    response_rate = segment_data['response'].mean()
    size = len(segment_data)
    expected_responses = size * response_rate
    
    print(f"\\n{segment} Segment:")
    print(f"  Size: {size:,} customers ({size/len(df):.1%})")
    print(f"  Response rate: {response_rate:.1%}")
    print(f"  Expected responses: {expected_responses:.0f}")
    
    if response_rate > 0.4:
        print(f"  Recommendation: High priority for campaign targeting")
    elif response_rate > 0.25:
        print(f"  Recommendation: Good target for campaign")
    elif response_rate > 0.15:
        print(f"  Recommendation: Consider with discount incentive")
    else:
        print(f"  Recommendation: Low priority, focus on engagement first")
```

## Advanced Usage

### Optimizing Number of Bins

```python
import numpy as np
from binning import SupervisedBinning
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Create dataset with optimal binning structure
np.random.seed(42)
n_samples = 2000

# Create feature with 3 natural risk levels
X = np.random.uniform(0, 10, n_samples).reshape(-1, 1)
y = np.zeros(n_samples)

# Define true risk levels
y[X.flatten() < 3] = np.random.binomial(1, 0.2, (X.flatten() < 3).sum())  # Low risk
y[(X.flatten() >= 3) & (X.flatten() < 7)] = np.random.binomial(1, 0.5, ((X.flatten() >= 3) & (X.flatten() < 7)).sum())  # Medium risk
y[X.flatten() >= 7] = np.random.binomial(1, 0.8, (X.flatten() >= 7).sum())  # High risk

# Test different numbers of bins
bin_counts = range(2, 11)
cv_scores = []

for n_bins in bin_counts:
    binner = SupervisedBinning(n_bins=n_bins)
    X_binned = binner.fit_transform(X, y)
    
    # Use logistic regression to evaluate binning quality
    lr = LogisticRegression()
    scores = cross_val_score(lr, X_binned, y, cv=5, scoring='roc_auc')
    cv_scores.append(scores.mean())

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(bin_counts, cv_scores, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Bins')
plt.ylabel('Cross-validation AUC')
plt.title('Optimal Number of Bins for Supervised Binning')
plt.grid(True, alpha=0.3)

# Mark the optimal number of bins
optimal_bins = bin_counts[np.argmax(cv_scores)]
plt.axvline(optimal_bins, color='red', linestyle='--', alpha=0.7)
plt.text(optimal_bins + 0.1, max(cv_scores) - 0.01, 
         f'Optimal: {optimal_bins} bins', fontsize=12)

plt.tight_layout()
plt.show()

print(f"Optimal number of bins: {optimal_bins}")
print(f"Best CV AUC: {max(cv_scores):.4f}")

# Show the optimal binning result
optimal_binner = SupervisedBinning(n_bins=optimal_bins)
X_optimal = optimal_binner.fit_transform(X, y)

print("\\nOptimal Binning Analysis:")
for bin_id in range(optimal_bins):
    mask = X_optimal.flatten() == bin_id
    if mask.any():
        bin_values = X[mask].flatten()
        risk_rate = y[mask].mean()
        print(f"Bin {bin_id}: Range {bin_values.min():.2f}-{bin_values.max():.2f}, "
              f"Risk Rate: {risk_rate:.2%}, Count: {mask.sum()}")
```

### Handling Imbalanced Datasets

```python
import numpy as np
from binning import SupervisedBinning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Create imbalanced dataset
np.random.seed(42)
n_samples = 5000

# Majority class (95%)
majority_size = int(0.95 * n_samples)
X_majority = np.random.normal(0, 1, (majority_size, 3))
y_majority = np.zeros(majority_size)

# Minority class (5%) - with different distribution
minority_size = n_samples - majority_size
X_minority = np.random.normal(2, 1.5, (minority_size, 3))  # Different mean and std
y_minority = np.ones(minority_size)

# Combine data
X = np.vstack([X_majority, X_minority])
y = np.concatenate([y_majority, y_minority])

print(f"Dataset size: {len(X)}")
print(f"Class distribution: {np.bincount(y)}")
print(f"Imbalance ratio: {np.bincount(y)[0] / np.bincount(y)[1]:.1f}:1")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Method 1: Standard supervised binning
standard_binner = SupervisedBinning(n_bins=5)
X_train_standard = standard_binner.fit_transform(X_train, y_train)
X_test_standard = standard_binner.transform(X_test)

lr_standard = LogisticRegression(random_state=42)
lr_standard.fit(X_train_standard, y_train)
y_pred_standard = lr_standard.predict(X_test_standard)

# Method 2: SMOTE then supervised binning
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

smote_binner = SupervisedBinning(n_bins=5)
X_train_smote_binned = smote_binner.fit_transform(X_train_smote, y_train_smote)
X_test_smote_binned = smote_binner.transform(X_test)

lr_smote = LogisticRegression(random_state=42)
lr_smote.fit(X_train_smote_binned, y_train_smote)
y_pred_smote = lr_smote.predict(X_test_smote_binned)

# Method 3: Class-weighted supervised binning
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
sample_weights = np.array([class_weights[int(label)] for label in y_train])

weighted_binner = SupervisedBinning(n_bins=5)
X_train_weighted = weighted_binner.fit_transform(X_train, y_train)
X_test_weighted = weighted_binner.transform(X_test)

lr_weighted = LogisticRegression(class_weight='balanced', random_state=42)
lr_weighted.fit(X_train_weighted, y_train)
y_pred_weighted = lr_weighted.predict(X_test_weighted)

# Compare results
methods = {
    'Standard': y_pred_standard,
    'SMOTE + Binning': y_pred_smote,
    'Class Weighted': y_pred_weighted
}

print("\\nPerformance Comparison on Imbalanced Dataset:")
for method_name, y_pred in methods.items():
    print(f"\\n{method_name}:")
    print(classification_report(y_test, y_pred, target_names=['Majority', 'Minority']))
```

## Best Practices and Tips

### When to Use Supervised Binning

```python
print("Guidelines for using Supervised Binning:")
print("\\n✅ EXCELLENT for:")
print("  - Predictive modeling with clear target variable")
print("  - Risk scoring and credit assessment")
print("  - Medical diagnosis and biomarker analysis")
print("  - Marketing response prediction")
print("  - Any scenario where bins should reflect target relationship")
print("\\n⚠️  CONSIDER CAREFULLY for:")
print("  - Exploratory data analysis without clear target")
print("  - Unsupervised learning tasks")
print("  - When interpretability of bin boundaries is critical")
print("\\n❌ AVOID for:")
print("  - Datasets with no clear target variable")
print("  - When you need equal-sized bins")
print("  - Time series data where temporal order matters")
print("  - Text or categorical data")

# Demonstrate when supervised binning provides the most benefit
import numpy as np
from binning import SupervisedBinning, EqualWidthBinning
from sklearn.metrics import mutual_info_score

scenarios = {
    'Strong Relationship': {
        'X': np.random.uniform(0, 10, 1000),
        'y_func': lambda x: (x > 5).astype(int)  # Clear threshold
    },
    'Weak Relationship': {
        'X': np.random.uniform(0, 10, 1000),
        'y_func': lambda x: np.random.binomial(1, 0.5 + 0.05 * x / 10, len(x))  # Slight trend
    },
    'No Relationship': {
        'X': np.random.uniform(0, 10, 1000),
        'y_func': lambda x: np.random.binomial(1, 0.5, len(x))  # Random
    }
}

print("\\nMutual Information Comparison:")
print("Scenario\\t\\tEqual Width\\tSupervised\\tImprovement")
print("-" * 60)

for scenario_name, scenario in scenarios.items():
    X = scenario['X'].reshape(-1, 1)
    y = scenario['y_func'](scenario['X'])
    
    # Equal width binning
    ew_binner = EqualWidthBinning(n_bins=5)
    X_ew = ew_binner.fit_transform(X)
    mi_ew = mutual_info_score(y, X_ew.flatten())
    
    # Supervised binning
    sup_binner = SupervisedBinning(n_bins=5)
    X_sup = sup_binner.fit_transform(X, y)
    mi_sup = mutual_info_score(y, X_sup.flatten())
    
    improvement = (mi_sup - mi_ew) / mi_ew * 100 if mi_ew > 0 else 0
    
    print(f"{scenario_name:<15}\\t{mi_ew:.3f}\\t\\t{mi_sup:.3f}\\t\\t{improvement:+.1f}%")
```

This comprehensive example documentation for Supervised Binning covers:

1. **Basic Usage**: Comparison with other methods, multi-feature examples
2. **Real-world Applications**: Credit risk, medical diagnosis, marketing prediction
3. **Advanced Techniques**: Optimal bin selection, imbalanced data handling
4. **Best Practices**: When supervised binning provides maximum benefit

Each example demonstrates how supervised binning creates target-aware discretization that improves predictive modeling performance.
