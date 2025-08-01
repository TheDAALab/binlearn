# One-Hot Binning Examples

This page demonstrates the use of `OneHotBinning` for creating binary indicator features from continuous variables.

## Basic Usage

### Understanding One-Hot Binning

```python
import numpy as np
import pandas as pd
from binning import OneHotBinning, EqualWidthBinning
import matplotlib.pyplot as plt

# Create sample data
np.random.seed(42)
ages = np.random.normal(35, 12, 1000).reshape(-1, 1)

# Regular binning vs One-hot binning
regular_binner = EqualWidthBinning(n_bins=5)
onehot_binner = OneHotBinning(n_bins=5)

regular_binned = regular_binner.fit_transform(ages)
onehot_binned = onehot_binner.fit_transform(ages)

print("Original data shape:", ages.shape)
print("Regular binning shape:", regular_binned.shape)
print("One-hot binning shape:", onehot_binned.shape)

print("\\nRegular binning sample (first 10 rows):")
print(regular_binned[:10].flatten())

print("\\nOne-hot binning sample (first 10 rows):")
print(onehot_binned[:10])

# Show the transformation
df = pd.DataFrame({
    'age': ages.flatten()[:10],
    'regular_bin': regular_binned.flatten()[:10]
})

# Add one-hot columns
for i in range(5):
    df[f'bin_{i}'] = onehot_binned[:10, i]

print("\\nTransformation comparison:")
print(df)
```

### Multi-feature One-Hot Binning

```python
import numpy as np
import pandas as pd
from binning import OneHotBinning

# Create multi-dimensional data
np.random.seed(42)
income = np.random.lognormal(10, 0.5, 800)
experience = np.random.exponential(5, 800)
age = np.random.normal(35, 10, 800)

data = np.column_stack([income, experience, age])
feature_names = ['income', 'experience', 'age']

print("Original data shape:", data.shape)

# Apply one-hot binning with different bins per feature
onehot_binner = OneHotBinning(n_bins=[4, 3, 5])  # Different bins for each feature
onehot_features = onehot_binner.fit_transform(data)

print("One-hot encoded shape:", onehot_features.shape)
print("Total features created:", onehot_features.shape[1])

# Show bin edges for each feature
for i, feature in enumerate(feature_names):
    print(f"\\n{feature} bin edges:")
    print(onehot_binner.bin_edges_[i])

# Create feature names for the one-hot encoded columns
feature_names_onehot = []
start_idx = 0
for i, feature in enumerate(feature_names):
    n_bins = onehot_binner.n_bins[i] if isinstance(onehot_binner.n_bins, list) else onehot_binner.n_bins
    for j in range(n_bins):
        feature_names_onehot.append(f'{feature}_bin_{j}')

print("\\nOne-hot feature names:")
print(feature_names_onehot)

# Show sample of transformed data
df_onehot = pd.DataFrame(onehot_features[:10], columns=feature_names_onehot)
print("\\nSample one-hot encoded data:")
print(df_onehot)
```

## Real-world Applications

### E-commerce Customer Analysis

```python
import numpy as np
import pandas as pd
from binning import OneHotBinning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Simulate e-commerce customer data
np.random.seed(42)
n_customers = 5000

# Customer features
age = np.random.normal(40, 15, n_customers)
age = np.clip(age, 18, 80)

income = np.random.lognormal(10.5, 0.6, n_customers)
income = np.clip(income, 20000, 200000)

sessions_per_month = np.random.exponential(8, n_customers)
avg_session_duration = np.random.gamma(2, 5, n_customers)  # minutes
total_spent = np.random.lognormal(6, 1.2, n_customers)

# Create target: premium membership (based on customer value)
premium_score = (
    0.3 * (age - 18) / 62 +  # Older customers more likely
    0.4 * np.log(income) / np.log(200000) +  # Higher income more likely
    0.2 * np.minimum(sessions_per_month / 20, 1) +  # Active users more likely
    0.1 * np.minimum(avg_session_duration / 30, 1) +  # Engaged users more likely
    np.random.normal(0, 0.2, n_customers)  # Random component
)

premium_member = (premium_score > 0.6).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'age': age,
    'income': income,
    'sessions_per_month': sessions_per_month,
    'avg_session_duration': avg_session_duration,
    'total_spent': total_spent,
    'premium_member': premium_member
})

print("Customer Data Overview:")
print(df.describe())
print(f"\\nPremium membership rate: {premium_member.mean():.2%}")

# Prepare features for modeling
features = ['age', 'income', 'sessions_per_month', 'avg_session_duration', 'total_spent']
X = df[features].values
y = df['premium_member'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Original continuous features
lr_continuous = LogisticRegression(random_state=42)
lr_continuous.fit(X_train, y_train)
y_pred_continuous = lr_continuous.predict(X_test)
y_prob_continuous = lr_continuous.predict_proba(X_test)[:, 1]

# Model 2: One-hot binned features
onehot_binner = OneHotBinning(n_bins=4)  # 4 bins per feature
X_train_onehot = onehot_binner.fit_transform(X_train)
X_test_onehot = onehot_binner.transform(X_test)

lr_onehot = LogisticRegression(random_state=42)
lr_onehot.fit(X_train_onehot, y_train)
y_pred_onehot = lr_onehot.predict(X_test_onehot)
y_prob_onehot = lr_onehot.predict_proba(X_test_onehot)[:, 1]

# Model 3: Combined features
X_train_combined = np.concatenate([X_train, X_train_onehot], axis=1)
X_test_combined = np.concatenate([X_test, X_test_onehot], axis=1)

lr_combined = LogisticRegression(random_state=42)
lr_combined.fit(X_train_combined, y_train)
y_pred_combined = lr_combined.predict(X_test_combined)
y_prob_combined = lr_combined.predict_proba(X_test_combined)[:, 1]

# Compare performance
print("\\nModel Performance Comparison:")
print("\\nContinuous Features:")
print(f"AUC: {roc_auc_score(y_test, y_prob_continuous):.3f}")
print(classification_report(y_test, y_pred_continuous))

print("\\nOne-Hot Binned Features:")
print(f"AUC: {roc_auc_score(y_test, y_prob_onehot):.3f}")
print(classification_report(y_test, y_pred_onehot))

print("\\nCombined Features:")
print(f"AUC: {roc_auc_score(y_test, y_prob_combined):.3f}")
print(classification_report(y_test, y_pred_combined))

# Analyze feature importance for one-hot model
feature_names_onehot = []
for i, feature in enumerate(features):
    for j in range(4):  # 4 bins per feature
        feature_names_onehot.append(f'{feature}_bin_{j}')

importance_df = pd.DataFrame({
    'feature': feature_names_onehot,
    'coefficient': lr_onehot.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

print("\\nTop 10 One-Hot Features by Importance:")
print(importance_df.head(10))
```

### Medical Diagnosis: Risk Factor Analysis

```python
import numpy as np
import pandas as pd
from binning import OneHotBinning
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Simulate medical dataset
np.random.seed(42)
n_patients = 3000

# Patient characteristics
age = np.random.normal(55, 20, n_patients)
age = np.clip(age, 18, 90)

bmi = np.random.normal(26, 5, n_patients)
bmi = np.clip(bmi, 15, 50)

systolic_bp = np.random.normal(130, 20, n_patients)
systolic_bp = np.clip(systolic_bp, 90, 200)

cholesterol = np.random.normal(200, 40, n_patients)
cholesterol = np.clip(cholesterol, 120, 350)

glucose = np.random.normal(100, 30, n_patients)
glucose = np.clip(glucose, 70, 300)

# Create disease risk (simplified medical model)
risk_score = (
    0.02 * (age - 18) +  # Age factor
    0.05 * np.maximum(bmi - 25, 0) +  # BMI above normal
    0.01 * np.maximum(systolic_bp - 120, 0) +  # High blood pressure
    0.005 * np.maximum(cholesterol - 200, 0) +  # High cholesterol
    0.01 * np.maximum(glucose - 100, 0) +  # High glucose
    np.random.normal(0, 0.5, n_patients)  # Random component
)

# Convert to binary disease outcome
disease = (risk_score > 2.0).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'age': age,
    'bmi': bmi,
    'systolic_bp': systolic_bp,
    'cholesterol': cholesterol,
    'glucose': glucose,
    'disease': disease
})

print("Medical Dataset Overview:")
print(df.describe())
print(f"\\nDisease prevalence: {disease.mean():.2%}")

# Apply one-hot binning to create risk categories
features = ['age', 'bmi', 'systolic_bp', 'cholesterol', 'glucose']
X = df[features].values
y = df['disease'].values

# Use clinically meaningful number of bins
# Age: 4 groups (young, middle-aged, senior, elderly)
# BMI: 4 groups (underweight, normal, overweight, obese)
# BP: 3 groups (normal, elevated, high)
# Cholesterol: 3 groups (normal, borderline, high)
# Glucose: 3 groups (normal, prediabetic, diabetic)
bins_per_feature = [4, 4, 3, 3, 3]

onehot_binner = OneHotBinning(n_bins=bins_per_feature)
X_onehot = onehot_binner.fit_transform(X)

print(f"\\nOriginal features: {X.shape[1]}")
print(f"One-hot encoded features: {X_onehot.shape[1]}")

# Create meaningful feature names
feature_bins = {
    'age': ['18-35', '35-50', '50-65', '65+'],
    'bmi': ['Underweight', 'Normal', 'Overweight', 'Obese'],
    'systolic_bp': ['Normal', 'Elevated', 'High'],
    'cholesterol': ['Normal', 'Borderline', 'High'],
    'glucose': ['Normal', 'Prediabetic', 'Diabetic']
}

feature_names_onehot = []
for i, feature in enumerate(features):
    for j, category in enumerate(feature_bins[feature]):
        feature_names_onehot.append(f'{feature}_{category}')

# Train model with one-hot features
X_train, X_test, y_train, y_test = train_test_split(X_onehot, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print("\\nModel Performance:")
print(classification_report(y_test, y_pred))

# Feature importance analysis
importance_df = pd.DataFrame({
    'feature': feature_names_onehot,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\\nTop 10 Risk Factors:")
print(importance_df.head(10))

# Analyze risk by categories
risk_analysis = pd.DataFrame(X_onehot, columns=feature_names_onehot)
risk_analysis['disease'] = y

print("\\nRisk Analysis by Categories:")
for feature in feature_names_onehot[:10]:  # Show top 10
    category_risk = risk_analysis.groupby(feature)['disease'].agg(['count', 'mean'])
    if category_risk.loc[1, 'count'] > 50:  # Only show if sufficient samples
        risk_rate = category_risk.loc[1, 'mean']
        sample_size = category_risk.loc[1, 'count']
        print(f"{feature}: {risk_rate:.1%} risk ({sample_size} patients)")
```

### Text Classification with Numerical Features

```python
import numpy as np
import pandas as pd
from binning import OneHotBinning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Simulate text classification dataset with numerical features
np.random.seed(42)
n_documents = 2000

# Simulate document characteristics
doc_length = np.random.lognormal(6, 0.8, n_documents)  # Number of words
readability_score = np.random.normal(50, 15, n_documents)  # Flesch reading ease
sentiment_score = np.random.normal(0, 1, n_documents)  # Sentiment (-3 to +3)
entity_count = np.random.poisson(5, n_documents)  # Named entities

# Create document categories based on characteristics
category_score = (
    0.001 * doc_length +  # Longer documents -> category 1
    0.02 * readability_score +  # Higher readability -> category 1
    0.3 * sentiment_score +  # Positive sentiment -> category 1
    0.1 * entity_count +  # More entities -> category 1
    np.random.normal(0, 2, n_documents)
)

categories = (category_score > np.median(category_score)).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'doc_id': range(n_documents),
    'doc_length': doc_length,
    'readability_score': readability_score,
    'sentiment_score': sentiment_score,
    'entity_count': entity_count,
    'category': categories
})

print("Text Dataset Overview:")
print(df.describe())
print(f"\\nCategory distribution: {np.bincount(categories)}")

# Prepare numerical features
numerical_features = ['doc_length', 'readability_score', 'sentiment_score', 'entity_count']
X_numerical = df[numerical_features].values
y = df['category'].values

# Apply one-hot binning to numerical features
onehot_binner = OneHotBinning(n_bins=4)
X_numerical_onehot = onehot_binner.fit_transform(X_numerical)

print(f"\\nNumerical features: {X_numerical.shape[1]}")
print(f"One-hot encoded numerical features: {X_numerical_onehot.shape[1]}")

# Split data
X_train_num, X_test_num, y_train, y_test = train_test_split(
    X_numerical_onehot, y, test_size=0.2, random_state=42
)

# Train classifier with one-hot numerical features
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_num, y_train)
y_pred_num = lr_model.predict(X_test_num)

print("\\nClassification with One-Hot Numerical Features:")
print(classification_report(y_test, y_pred_num))

# Create feature names
feature_names_onehot = []
for feature in numerical_features:
    for i in range(4):
        feature_names_onehot.append(f'{feature}_bin_{i}')

# Feature importance
importance_df = pd.DataFrame({
    'feature': feature_names_onehot,
    'coefficient': lr_model.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

print("\\nFeature Importance (One-Hot Numerical):")
print(importance_df.head(10))

# Show bin ranges for interpretation
print("\\nBin Ranges for Interpretation:")
for i, feature in enumerate(numerical_features):
    bin_edges = onehot_binner.bin_edges_[i]
    print(f"\\n{feature}:")
    for j in range(len(bin_edges) - 1):
        print(f"  Bin {j}: {bin_edges[j]:.2f} to {bin_edges[j+1]:.2f}")
```

## Advanced Usage

### Handling Sparse Data and Memory Optimization

```python
import numpy as np
from binning import OneHotBinning
from scipy.sparse import csr_matrix
import pandas as pd

# Create sparse-like data (many zeros)
np.random.seed(42)
n_samples = 10000
n_features = 5

# Create data where most values fall into specific bins
data = []
for i in range(n_features):
    # Create bimodal distribution - most values near 0 or 10
    feature_data = np.concatenate([
        np.random.normal(0, 0.5, n_samples // 2),
        np.random.normal(10, 0.5, n_samples // 2)
    ])
    np.random.shuffle(feature_data)
    data.append(feature_data)

X = np.column_stack(data)

print("Original data shape:", X.shape)
print("Memory usage (MB):", X.nbytes / 1024**2)

# Apply one-hot binning
onehot_binner = OneHotBinning(n_bins=5)
X_onehot = onehot_binner.fit_transform(X)

print("One-hot data shape:", X_onehot.shape)
print("Memory usage (MB):", X_onehot.nbytes / 1024**2)

# Check sparsity
sparsity = np.mean(X_onehot == 0)
print(f"Sparsity: {sparsity:.2%}")

# Convert to sparse matrix for memory efficiency
X_sparse = csr_matrix(X_onehot)
print(f"Sparse matrix memory (MB): {X_sparse.data.nbytes / 1024**2:.2f}")
print(f"Memory reduction: {X_onehot.nbytes / X_sparse.data.nbytes:.1f}x")

# Demonstrate working with sparse matrices
from sklearn.linear_model import LogisticRegression

# Create dummy target
y_dummy = np.random.randint(0, 2, n_samples)

# Train model with sparse matrix
lr_sparse = LogisticRegression(random_state=42)
lr_sparse.fit(X_sparse, y_dummy)

print("\\nSuccessfully trained model with sparse one-hot features!")
```

### Integration with Categorical Features

```python
import numpy as np
import pandas as pd
from binning import OneHotBinning
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Create mixed dataset with numerical and categorical features
np.random.seed(42)
n_samples = 2000

# Numerical features
age = np.random.normal(35, 12, n_samples)
income = np.random.lognormal(10, 0.6, n_samples)
credit_score = np.random.normal(650, 100, n_samples)

# Categorical features
cities = np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n_samples)
education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples)
employment = np.random.choice(['Full-time', 'Part-time', 'Self-employed', 'Unemployed'], n_samples)

# Target variable
target_score = (
    0.01 * age +
    0.00001 * income +
    0.001 * credit_score +
    np.random.normal(0, 2, n_samples)
)
target = (target_score > np.median(target_score)).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'age': age,
    'income': income,
    'credit_score': credit_score,
    'city': cities,
    'education': education,
    'employment': employment,
    'target': target
})

print("Mixed Dataset Overview:")
print(df.head())
print("\\nData types:")
print(df.dtypes)

# Separate numerical and categorical features
numerical_features = ['age', 'income', 'credit_score']
categorical_features = ['city', 'education', 'employment']

# Create preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num_onehot', OneHotBinning(n_bins=4), numerical_features),
    ('cat_onehot', OneHotEncoder(drop='first'), categorical_features)
])

# Create full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])

# Prepare data
X = df[numerical_features + categorical_features]
y = df['target']

# Train pipeline
pipeline.fit(X, y)

# Get feature names after preprocessing
num_feature_names = []
for feature in numerical_features:
    for i in range(4):
        num_feature_names.append(f'{feature}_bin_{i}')

cat_feature_names = pipeline.named_steps['preprocessor'].named_transformers_['cat_onehot'].get_feature_names_out(categorical_features)

all_feature_names = num_feature_names + list(cat_feature_names)

print(f"\\nTotal features after preprocessing: {len(all_feature_names)}")
print("Feature names:", all_feature_names[:10], "...")

# Evaluate pipeline
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(pipeline, X, y, cv=5)
print(f"\\nCross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
```

## Visualization and Interpretation

### Visualizing One-Hot Encoded Features

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from binning import OneHotBinning

# Create dataset for visualization
np.random.seed(42)
n_samples = 1000

# Create data with clear patterns
feature1 = np.concatenate([
    np.random.normal(2, 0.5, 300),  # Group 1
    np.random.normal(5, 0.7, 400),  # Group 2
    np.random.normal(8, 0.6, 300)   # Group 3
])

feature2 = np.concatenate([
    np.random.exponential(1, 400),   # Skewed distribution
    np.random.exponential(3, 600)    # Different scale
])

# Create target based on features
target = (
    (feature1 > 6) | (feature2 > 4)
).astype(int)

df = pd.DataFrame({
    'feature1': feature1,
    'feature2': feature2,
    'target': target
})

# Apply one-hot binning
X = df[['feature1', 'feature2']].values
onehot_binner = OneHotBinning(n_bins=4)
X_onehot = onehot_binner.fit_transform(X)

# Create one-hot DataFrame
onehot_columns = []
for i, feature in enumerate(['feature1', 'feature2']):
    for j in range(4):
        onehot_columns.append(f'{feature}_bin_{j}')

df_onehot = pd.DataFrame(X_onehot, columns=onehot_columns)
df_onehot['target'] = target

# Visualization
plt.figure(figsize=(20, 15))

# Original features
plt.subplot(3, 4, 1)
plt.scatter(feature1, feature2, c=target, alpha=0.6, cmap='RdYlBu')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Original Features')
plt.colorbar()

# Feature distributions
plt.subplot(3, 4, 2)
plt.hist(feature1, bins=30, alpha=0.7, edgecolor='black')
plt.title('Feature 1 Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Add bin boundaries
for edge in onehot_binner.bin_edges_[0][1:-1]:
    plt.axvline(edge, color='red', linestyle='--', alpha=0.7)

plt.subplot(3, 4, 3)
plt.hist(feature2, bins=30, alpha=0.7, edgecolor='black')
plt.title('Feature 2 Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Add bin boundaries
for edge in onehot_binner.bin_edges_[1][1:-1]:
    plt.axvline(edge, color='red', linestyle='--', alpha=0.7)

# One-hot feature distributions
for i, col in enumerate(onehot_columns):
    plt.subplot(3, 4, i + 5)
    target_0_count = df_onehot[df_onehot['target'] == 0][col].sum()
    target_1_count = df_onehot[df_onehot['target'] == 1][col].sum()
    
    plt.bar(['Target 0', 'Target 1'], [target_0_count, target_1_count], 
            alpha=0.7, color=['blue', 'red'])
    plt.title(f'{col}')
    plt.ylabel('Count')

plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = df_onehot.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f')
plt.title('One-Hot Features Correlation Matrix')
plt.tight_layout()
plt.show()

# Feature importance for each bin
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_onehot, target)

importance_df = pd.DataFrame({
    'feature': onehot_columns,
    'coefficient': lr.coef_[0],
    'abs_coefficient': np.abs(lr.coef_[0])
}).sort_values('abs_coefficient', ascending=False)

print("Feature Importance (One-Hot Bins):")
print(importance_df)

# Plot feature importance
plt.figure(figsize=(12, 6))
plt.bar(range(len(importance_df)), importance_df['coefficient'], 
        color=['red' if x < 0 else 'blue' for x in importance_df['coefficient']])
plt.xlabel('Feature Index')
plt.ylabel('Coefficient')
plt.title('Logistic Regression Coefficients for One-Hot Bins')
plt.xticks(range(len(importance_df)), importance_df['feature'], rotation=45)
plt.tight_layout()
plt.show()
```

## Best Practices and Tips

### When to Use One-Hot Binning

```python
print("Guidelines for using One-Hot Binning:")
print("\\n✅ EXCELLENT for:")
print("  - Linear models (Logistic Regression, Linear SVM)")
print("  - Neural networks with categorical-like inputs")
print("  - When you need to capture non-linear relationships in linear models")
print("  - Sparse datasets where memory efficiency matters")
print("  - Feature selection scenarios")
print("\\n⚠️  CONSIDER CAREFULLY for:")
print("  - Tree-based models (they handle continuous features well)")
print("  - High-dimensional data (curse of dimensionality)")
print("  - When interpretability of individual bins is not important")
print("\\n❌ AVOID for:")
print("  - Very large datasets with memory constraints")
print("  - When the number of bins creates too many features")
print("  - Distance-based algorithms without proper scaling")

# Demonstrate computational complexity
import time

sizes = [1000, 5000, 10000, 50000]
features = [2, 5, 10, 20]

print("\\nComputational Complexity Analysis:")
print("Size\\tFeatures\\tTime(s)\\tMemory(MB)")
print("-" * 40)

for n_samples in [1000, 10000]:
    for n_features in [2, 10]:
        np.random.seed(42)
        X = np.random.rand(n_samples, n_features)
        
        start_time = time.time()
        binner = OneHotBinning(n_bins=5)
        X_onehot = binner.fit_transform(X)
        end_time = time.time()
        
        memory_mb = X_onehot.nbytes / (1024**2)
        
        print(f"{n_samples}\\t{n_features}\\t\\t{end_time-start_time:.3f}\\t{memory_mb:.1f}")
```

This comprehensive example documentation for One-Hot Binning covers:

1. **Basic Usage**: Understanding the transformation, multi-feature examples
2. **Real-world Applications**: E-commerce analysis, medical diagnosis, text classification
3. **Advanced Techniques**: Sparse data handling, mixed data types
4. **Visualization**: Feature interpretation, correlation analysis
5. **Best Practices**: When to use one-hot binning, computational considerations

Each example shows how one-hot binning creates interpretable binary features that work well with linear models and provides feature selection capabilities.
