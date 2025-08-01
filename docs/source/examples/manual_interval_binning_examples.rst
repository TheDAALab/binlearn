# Manual Interval Binning Examples

This page demonstrates the use of `ManualIntervalBinning` for creating bins with precisely defined boundaries, ideal for domain-specific requirements and standardized classifications.

## Basic Usage

### Understanding Manual Interval Binning

```python
import numpy as np
import matplotlib.pyplot as plt
from binning import ManualIntervalBinning, EqualWidthBinning

# Create sample data
np.random.seed(42)
data = np.random.normal(100, 25, 2000)  # Mean=100, Std=25
data = np.clip(data, 0, 200)  # Clip to reasonable range

# Define custom intervals
custom_intervals = [
    [0, 50],      # Very Low
    [50, 75],     # Low  
    [75, 100],    # Below Average
    [100, 125],   # Above Average
    [125, 150],   # High
    [150, 200]    # Very High
]

# Create manual interval binner
manual_binner = ManualIntervalBinning(intervals=custom_intervals)
data_manual = manual_binner.fit_transform(data.reshape(-1, 1))

# Compare with equal width binning
equal_width_binner = EqualWidthBinning(n_bins=6)
data_equal_width = equal_width_binner.fit_transform(data.reshape(-1, 1))

# Visualize the comparison
plt.figure(figsize=(15, 10))

# Original data distribution
plt.subplot(2, 3, 1)
plt.hist(data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Original Data Distribution')

# Manual interval binning
plt.subplot(2, 3, 2)
bin_counts_manual = [np.sum(data_manual.flatten() == i) for i in range(6)]
labels_manual = ['Very Low\\n(0-50)', 'Low\\n(50-75)', 'Below Avg\\n(75-100)', 
                'Above Avg\\n(100-125)', 'High\\n(125-150)', 'Very High\\n(150-200)']

bars_manual = plt.bar(range(6), bin_counts_manual, alpha=0.7, color='lightgreen', edgecolor='black')
plt.xlabel('Bin')
plt.ylabel('Count')
plt.title('Manual Interval Binning')
plt.xticks(range(6), labels_manual, rotation=45, ha='right')

# Add count labels
for bar, count in zip(bars_manual, bin_counts_manual):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
             f'{count}', ha='center', va='bottom')

# Equal width binning
plt.subplot(2, 3, 3)
bin_counts_ew = [np.sum(data_equal_width.flatten() == i) for i in range(6)]
bars_ew = plt.bar(range(6), bin_counts_ew, alpha=0.7, color='lightcoral', edgecolor='black')
plt.xlabel('Bin')
plt.ylabel('Count')
plt.title('Equal Width Binning')

# Add count labels
for bar, count in zip(bars_ew, bin_counts_ew):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
             f'{count}', ha='center', va='bottom')

# Show bin boundaries
plt.subplot(2, 3, 4)
manual_boundaries = [interval[0] for interval in custom_intervals] + [custom_intervals[-1][1]]
equal_width_boundaries = equal_width_binner.bin_edges_[0]

plt.plot(manual_boundaries[:-1], [1]*len(manual_boundaries[:-1]), 'go-', 
         label='Manual Intervals', markersize=8, linewidth=2)
plt.plot(equal_width_boundaries[:-1], [0.5]*len(equal_width_boundaries[:-1]), 'ro-', 
         label='Equal Width', markersize=8, linewidth=2)

plt.xlabel('Value')
plt.ylabel('Method')
plt.title('Bin Boundaries Comparison')
plt.legend()
plt.ylim(0, 1.5)

# Data distribution within bins
plt.subplot(2, 3, 5)
for i, (start, end) in enumerate(custom_intervals):
    mask = (data >= start) & (data < end)
    if i == len(custom_intervals) - 1:  # Last bin includes upper boundary
        mask = (data >= start) & (data <= end)
    
    if mask.any():
        plt.scatter(data[mask], np.full(mask.sum(), i), alpha=0.6, s=20)

plt.xlabel('Value')
plt.ylabel('Bin')
plt.title('Data Points by Manual Bins')
plt.yticks(range(6), [f'Bin {i}' for i in range(6)])

plt.tight_layout()
plt.show()

print("Manual Interval Binning Summary:")
print(f"Total data points: {len(data)}")
print("\\nBin distributions:")
for i, (start, end) in enumerate(custom_intervals):
    count = bin_counts_manual[i]
    percentage = count / len(data) * 100
    print(f"Bin {i} ({start}-{end}): {count} points ({percentage:.1f}%)")
```

### Multi-feature Manual Binning

```python
import numpy as np
import pandas as pd
from binning import ManualIntervalBinning

# Create multi-feature dataset
np.random.seed(42)
n_samples = 1000

# Feature 1: Age (different intervals)
age = np.random.normal(40, 15, n_samples)
age = np.clip(age, 18, 80)

# Feature 2: Income (different intervals)  
income = np.random.lognormal(10.5, 0.6, n_samples)

# Feature 3: Score (different intervals)
score = np.random.beta(3, 2, n_samples) * 100

# Define different intervals for each feature
age_intervals = [
    [18, 25],   # Young adults
    [25, 35],   # Early career
    [35, 50],   # Mid career
    [50, 65],   # Late career
    [65, 80]    # Retirement age
]

income_intervals = [
    [0, 30000],      # Low income
    [30000, 50000],  # Lower middle
    [50000, 75000],  # Middle income
    [75000, 100000], # Upper middle
    [100000, 200000] # High income
]

score_intervals = [
    [0, 20],     # Poor
    [20, 40],    # Fair
    [40, 60],    # Good
    [60, 80],    # Very Good
    [80, 100]    # Excellent
]

# Create separate binners for each feature
age_binner = ManualIntervalBinning(intervals=age_intervals)
income_binner = ManualIntervalBinning(intervals=income_intervals)
score_binner = ManualIntervalBinning(intervals=score_intervals)

# Apply binning
age_binned = age_binner.fit_transform(age.reshape(-1, 1))
income_binned = income_binner.fit_transform(income.reshape(-1, 1))
score_binned = score_binner.fit_transform(score.reshape(-1, 1))

# Create DataFrame for analysis
df = pd.DataFrame({
    'age': age,
    'income': income,
    'score': score,
    'age_bin': age_binned.flatten(),
    'income_bin': income_binned.flatten(),
    'score_bin': score_binned.flatten()
})

# Define labels for better interpretation
age_labels = ['Young (18-25)', 'Early Career (25-35)', 'Mid Career (35-50)', 
              'Late Career (50-65)', 'Retirement (65-80)']
income_labels = ['Low (<30k)', 'Lower Mid (30-50k)', 'Middle (50-75k)', 
                'Upper Mid (75-100k)', 'High (100k+)']
score_labels = ['Poor (0-20)', 'Fair (20-40)', 'Good (40-60)', 
               'Very Good (60-80)', 'Excellent (80-100)']

# Analysis by bins
print("Multi-feature Manual Binning Analysis:")
print("\\nAge Distribution:")
for i, label in enumerate(age_labels):
    count = (df['age_bin'] == i).sum()
    print(f"{label}: {count} ({count/len(df)*100:.1f}%)")

print("\\nIncome Distribution:")
for i, label in enumerate(income_labels):
    count = (df['income_bin'] == i).sum()
    print(f"{label}: {count} ({count/len(df)*100:.1f}%)")

print("\\nScore Distribution:")
for i, label in enumerate(score_labels):
    count = (df['score_bin'] == i).sum()
    print(f"{label}: {count} ({count/len(df)*100:.1f}%)")

# Cross-tabulation analysis
print("\\nCross-tabulation: Age vs Income Bins")
crosstab = pd.crosstab(df['age_bin'], df['income_bin'], margins=True)
crosstab.index = age_labels + ['Total']
crosstab.columns = income_labels + ['Total']
print(crosstab)
```

## Real-world Applications

### Academic Grading System

```python
import numpy as np
import pandas as pd
from binning import ManualIntervalBinning
import matplotlib.pyplot as plt

# Simulate student scores
np.random.seed(42)
n_students = 2500

# Generate scores with realistic distribution
# Most students score in middle range, fewer at extremes
scores = np.concatenate([
    np.random.normal(85, 8, int(0.4 * n_students)),   # High performers
    np.random.normal(75, 10, int(0.35 * n_students)), # Average performers
    np.random.normal(65, 12, int(0.2 * n_students)),  # Below average
    np.random.normal(45, 15, int(0.05 * n_students))  # Struggling students
])

scores = np.clip(scores, 0, 100)  # Ensure scores are 0-100

# Define standard grading intervals
grading_intervals = [
    [0, 60],    # F (Fail)
    [60, 70],   # D (Poor)
    [70, 80],   # C (Satisfactory) 
    [80, 90],   # B (Good)
    [90, 100]   # A (Excellent)
]

# Apply manual interval binning for grades
grade_binner = ManualIntervalBinning(intervals=grading_intervals)
grade_bins = grade_binner.fit_transform(scores.reshape(-1, 1))

# Create DataFrame for analysis
df_grades = pd.DataFrame({
    'student_id': range(1, len(scores) + 1),
    'score': scores,
    'grade_bin': grade_bins.flatten()
})

# Define grade labels
grade_labels = ['F (0-59)', 'D (60-69)', 'C (70-79)', 'B (80-89)', 'A (90-100)']
grade_letters = ['F', 'D', 'C', 'B', 'A']

# Add grade letters to DataFrame
df_grades['grade'] = df_grades['grade_bin'].map({i: grade_letters[i] for i in range(5)})

# Analysis
print("Academic Grading Analysis:")
print(f"Total students: {len(scores)}")
print("\\nGrade Distribution:")

grade_stats = df_grades.groupby('grade_bin').agg({
    'score': ['count', 'mean', 'std', 'min', 'max']
}).round(2)

for i, label in enumerate(grade_labels):
    count = (df_grades['grade_bin'] == i).sum()
    if count > 0:
        avg_score = df_grades[df_grades['grade_bin'] == i]['score'].mean()
        print(f"{label}: {count} students ({count/len(df_grades)*100:.1f}%), avg score: {avg_score:.1f}")
    else:
        print(f"{label}: 0 students (0.0%)")

# Visualization
plt.figure(figsize=(15, 10))

# Score distribution with grade boundaries
plt.subplot(2, 2, 1)
plt.hist(scores, bins=50, alpha=0.7, color='lightblue', edgecolor='black')

# Add grade boundary lines
boundaries = [60, 70, 80, 90]
colors = ['red', 'orange', 'yellow', 'lightgreen']
for boundary, color in zip(boundaries, colors):
    plt.axvline(boundary, color=color, linestyle='--', linewidth=2, alpha=0.8)

plt.xlabel('Score')
plt.ylabel('Number of Students')
plt.title('Score Distribution with Grade Boundaries')

# Grade distribution
plt.subplot(2, 2, 2)
grade_counts = [np.sum(grade_bins.flatten() == i) for i in range(5)]
bars = plt.bar(range(5), grade_counts, alpha=0.7, 
               color=['red', 'orange', 'yellow', 'lightgreen', 'darkgreen'], 
               edgecolor='black')

plt.xlabel('Grade')
plt.ylabel('Number of Students')
plt.title('Grade Distribution')
plt.xticks(range(5), grade_letters)

# Add count labels
for bar, count in zip(bars, grade_counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f'{count}', ha='center', va='bottom')

# Box plot by grade
plt.subplot(2, 2, 3)
scores_by_grade = [scores[grade_bins.flatten() == i] for i in range(5) if np.sum(grade_bins.flatten() == i) > 0]
grade_labels_existing = [grade_letters[i] for i in range(5) if np.sum(grade_bins.flatten() == i) > 0]

plt.boxplot(scores_by_grade, labels=grade_labels_existing)
plt.xlabel('Grade')
plt.ylabel('Score')
plt.title('Score Distribution by Grade')

# Cumulative distribution
plt.subplot(2, 2, 4)
cumulative_counts = np.cumsum(grade_counts)
cumulative_percentages = cumulative_counts / len(scores) * 100

plt.plot(range(5), cumulative_percentages, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Grade Level')
plt.ylabel('Cumulative Percentage')
plt.title('Cumulative Grade Distribution')
plt.xticks(range(5), grade_letters)
plt.grid(True, alpha=0.3)

# Add percentage labels
for i, pct in enumerate(cumulative_percentages):
    plt.text(i, pct + 2, f'{pct:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Statistical analysis
print("\\nDetailed Grade Statistics:")
print(grade_stats)

# Performance indicators
pass_rate = (df_grades['grade_bin'] >= 2).mean() * 100  # C or better
honor_rate = (df_grades['grade_bin'] >= 3).mean() * 100  # B or better
distinction_rate = (df_grades['grade_bin'] == 4).mean() * 100  # A grade

print(f"\\nPerformance Indicators:")
print(f"Pass rate (C or better): {pass_rate:.1f}%")
print(f"Honor roll (B or better): {honor_rate:.1f}%")
print(f"Distinction (A grade): {distinction_rate:.1f}%")
```

### Medical Risk Stratification

```python
import numpy as np
import pandas as pd
from binning import ManualIntervalBinning
import matplotlib.pyplot as plt

# Simulate patient risk assessment data
np.random.seed(42)
n_patients = 3000

# Blood pressure readings (systolic)
blood_pressure = np.random.normal(130, 20, n_patients)
blood_pressure = np.clip(blood_pressure, 80, 200)

# Cholesterol levels (mg/dL)
cholesterol = np.random.normal(200, 40, n_patients)
cholesterol = np.clip(cholesterol, 120, 350)

# Blood glucose (fasting, mg/dL)
glucose = np.random.normal(95, 25, n_patients)
glucose = np.clip(glucose, 60, 200)

# Define medical standard intervals
bp_intervals = [
    [80, 120],   # Normal
    [120, 130],  # Elevated
    [130, 140],  # Stage 1 Hypertension
    [140, 180],  # Stage 2 Hypertension
    [180, 200]   # Hypertensive Crisis
]

cholesterol_intervals = [
    [120, 200],  # Desirable
    [200, 240],  # Borderline High
    [240, 350]   # High
]

glucose_intervals = [
    [60, 100],   # Normal
    [100, 126],  # Prediabetes
    [126, 200]   # Diabetes
]

# Apply manual interval binning
bp_binner = ManualIntervalBinning(intervals=bp_intervals)
chol_binner = ManualIntervalBinning(intervals=cholesterol_intervals)
glucose_binner = ManualIntervalBinning(intervals=glucose_intervals)

bp_bins = bp_binner.fit_transform(blood_pressure.reshape(-1, 1))
chol_bins = chol_binner.fit_transform(cholesterol.reshape(-1, 1))
glucose_bins = glucose_binner.fit_transform(glucose.reshape(-1, 1))

# Create patient DataFrame
df_patients = pd.DataFrame({
    'patient_id': range(1, n_patients + 1),
    'blood_pressure': blood_pressure,
    'cholesterol': cholesterol,
    'glucose': glucose,
    'bp_category': bp_bins.flatten(),
    'chol_category': chol_bins.flatten(),
    'glucose_category': glucose_bins.flatten()
})

# Define category labels
bp_labels = ['Normal (<120)', 'Elevated (120-129)', 'Stage 1 (130-139)', 
            'Stage 2 (140-179)', 'Crisis (≥180)']
chol_labels = ['Desirable (<200)', 'Borderline (200-239)', 'High (≥240)']
glucose_labels = ['Normal (<100)', 'Prediabetes (100-125)', 'Diabetes (≥126)']

# Risk stratification analysis
print("Medical Risk Stratification Analysis:")
print(f"Total patients assessed: {n_patients}")

print("\\nBlood Pressure Categories:")
for i, label in enumerate(bp_labels):
    count = (df_patients['bp_category'] == i).sum()
    avg_bp = df_patients[df_patients['bp_category'] == i]['blood_pressure'].mean()
    print(f"{label}: {count} patients ({count/n_patients*100:.1f}%), avg BP: {avg_bp:.1f}")

print("\\nCholesterol Categories:")
for i, label in enumerate(chol_labels):
    count = (df_patients['chol_category'] == i).sum()
    avg_chol = df_patients[df_patients['chol_category'] == i]['cholesterol'].mean()
    print(f"{label}: {count} patients ({count/n_patients*100:.1f}%), avg: {avg_chol:.1f} mg/dL")

print("\\nGlucose Categories:")
for i, label in enumerate(glucose_labels):
    count = (df_patients['glucose_category'] == i).sum()
    avg_glucose = df_patients[df_patients['glucose_category'] == i]['glucose'].mean()
    print(f"{label}: {count} patients ({count/n_patients*100:.1f}%), avg: {avg_glucose:.1f} mg/dL")

# Combined risk assessment
df_patients['total_risk_score'] = (
    df_patients['bp_category'] * 2 +      # BP has higher weight
    df_patients['chol_category'] * 1.5 +  # Cholesterol moderate weight
    df_patients['glucose_category'] * 1.5 # Glucose moderate weight
)

# Define risk levels based on combined score
risk_intervals = [
    [0, 2],    # Low risk
    [2, 4],    # Moderate risk
    [4, 6],    # High risk
    [6, 12]    # Very high risk
]

risk_binner = ManualIntervalBinning(intervals=risk_intervals)
risk_bins = risk_binner.fit_transform(df_patients['total_risk_score'].values.reshape(-1, 1))
df_patients['risk_level'] = risk_bins.flatten()

risk_labels = ['Low Risk', 'Moderate Risk', 'High Risk', 'Very High Risk']

print("\\nOverall Risk Stratification:")
for i, label in enumerate(risk_labels):
    count = (df_patients['risk_level'] == i).sum()
    print(f"{label}: {count} patients ({count/n_patients*100:.1f}%)")

# Visualization
plt.figure(figsize=(16, 12))

# Blood pressure distribution
plt.subplot(3, 3, 1)
bp_counts = [np.sum(bp_bins.flatten() == i) for i in range(5)]
plt.bar(range(5), bp_counts, alpha=0.7, color='lightcoral', edgecolor='black')
plt.xlabel('BP Category')
plt.ylabel('Patient Count')
plt.title('Blood Pressure Distribution')
plt.xticks(range(5), ['Normal', 'Elevated', 'Stage 1', 'Stage 2', 'Crisis'], rotation=45)

# Cholesterol distribution
plt.subplot(3, 3, 2)
chol_counts = [np.sum(chol_bins.flatten() == i) for i in range(3)]
plt.bar(range(3), chol_counts, alpha=0.7, color='lightblue', edgecolor='black')
plt.xlabel('Cholesterol Category')
plt.ylabel('Patient Count')
plt.title('Cholesterol Distribution')
plt.xticks(range(3), ['Desirable', 'Borderline', 'High'])

# Glucose distribution
plt.subplot(3, 3, 3)
glucose_counts = [np.sum(glucose_bins.flatten() == i) for i in range(3)]
plt.bar(range(3), glucose_counts, alpha=0.7, color='lightgreen', edgecolor='black')
plt.xlabel('Glucose Category')
plt.ylabel('Patient Count')
plt.title('Glucose Distribution')
plt.xticks(range(3), ['Normal', 'Prediabetes', 'Diabetes'])

# Overall risk distribution
plt.subplot(3, 3, 4)
risk_counts = [np.sum(risk_bins.flatten() == i) for i in range(4)]
colors = ['green', 'yellow', 'orange', 'red']
plt.bar(range(4), risk_counts, alpha=0.7, color=colors, edgecolor='black')
plt.xlabel('Risk Level')
plt.ylabel('Patient Count')
plt.title('Overall Risk Distribution')
plt.xticks(range(4), ['Low', 'Moderate', 'High', 'Very High'])

# Risk score distribution
plt.subplot(3, 3, 5)
plt.hist(df_patients['total_risk_score'], bins=30, alpha=0.7, color='purple', edgecolor='black')
plt.xlabel('Total Risk Score')
plt.ylabel('Patient Count')
plt.title('Risk Score Distribution')

# Correlation matrix
plt.subplot(3, 3, 6)
correlation_data = df_patients[['bp_category', 'chol_category', 'glucose_category', 'risk_level']]
correlation_matrix = correlation_data.corr()

im = plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar(im)
plt.xticks(range(4), ['BP', 'Cholesterol', 'Glucose', 'Risk'])
plt.yticks(range(4), ['BP', 'Cholesterol', 'Glucose', 'Risk'])
plt.title('Risk Factor Correlations')

# Add correlation values
for i in range(4):
    for j in range(4):
        plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                ha='center', va='center', color='black')

plt.tight_layout()
plt.show()

# High-risk patient analysis
high_risk_patients = df_patients[df_patients['risk_level'] >= 2]  # High or Very High risk
print(f"\\nHigh-risk patients requiring intervention: {len(high_risk_patients)} ({len(high_risk_patients)/n_patients*100:.1f}%)")

print("\\nHigh-risk patient characteristics:")
print(high_risk_patients[['blood_pressure', 'cholesterol', 'glucose', 'total_risk_score']].describe())
```

### Financial Credit Scoring

```python
import numpy as np
import pandas as pd
from binning import ManualIntervalBinning

# Simulate credit scoring dataset
np.random.seed(42)
n_applicants = 5000

# Generate financial indicators
credit_score = np.random.normal(650, 100, n_applicants)
credit_score = np.clip(credit_score, 300, 850)

debt_to_income = np.random.beta(2, 8, n_applicants)  # Most people have low DTI
payment_history = np.random.normal(85, 15, n_applicants)  # Percentage of on-time payments
payment_history = np.clip(payment_history, 0, 100)

# Define industry-standard credit intervals
credit_score_intervals = [
    [300, 580],  # Poor
    [580, 670],  # Fair
    [670, 740],  # Good
    [740, 800],  # Very Good
    [800, 850]   # Exceptional
]

dti_intervals = [
    [0, 0.20],    # Excellent DTI (0-20%)
    [0.20, 0.36], # Good DTI (20-36%)
    [0.36, 0.50], # Fair DTI (36-50%)
    [0.50, 1.0]   # Poor DTI (>50%)
]

payment_intervals = [
    [0, 60],      # Poor payment history
    [60, 80],     # Fair payment history
    [80, 95],     # Good payment history
    [95, 100]     # Excellent payment history
]

# Apply manual interval binning
credit_binner = ManualIntervalBinning(intervals=credit_score_intervals)
dti_binner = ManualIntervalBinning(intervals=dti_intervals)
payment_binner = ManualIntervalBinning(intervals=payment_intervals)

credit_bins = credit_binner.fit_transform(credit_score.reshape(-1, 1))
dti_bins = dti_binner.fit_transform(debt_to_income.reshape(-1, 1))
payment_bins = payment_binner.fit_transform(payment_history.reshape(-1, 1))

# Create credit assessment DataFrame
df_credit = pd.DataFrame({
    'applicant_id': range(1, n_applicants + 1),
    'credit_score': credit_score,
    'debt_to_income': debt_to_income,
    'payment_history': payment_history,
    'credit_tier': credit_bins.flatten(),
    'dti_tier': dti_bins.flatten(),
    'payment_tier': payment_bins.flatten()
})

# Define tier labels
credit_labels = ['Poor (300-579)', 'Fair (580-669)', 'Good (670-739)', 
                'Very Good (740-799)', 'Exceptional (800-850)']
dti_labels = ['Excellent (0-20%)', 'Good (20-36%)', 'Fair (36-50%)', 'Poor (>50%)']
payment_labels = ['Poor (0-59%)', 'Fair (60-79%)', 'Good (80-94%)', 'Excellent (95-100%)']

# Calculate loan approval recommendations
def calculate_loan_decision(row):
    # Weighted scoring system
    credit_weight = 0.5
    dti_weight = 0.3
    payment_weight = 0.2
    
    # Higher tier numbers = better scores (reverse for DTI where lower is better)
    credit_score_points = row['credit_tier'] * credit_weight
    dti_score_points = (3 - row['dti_tier']) * dti_weight  # Reverse DTI scoring
    payment_score_points = row['payment_tier'] * payment_weight
    
    total_score = credit_score_points + dti_score_points + payment_score_points
    
    if total_score >= 2.5:
        return 'Approved'
    elif total_score >= 1.5:
        return 'Conditional'
    else:
        return 'Declined'

df_credit['loan_decision'] = df_credit.apply(calculate_loan_decision, axis=1)

print("Credit Scoring Analysis:")
print(f"Total loan applications: {n_applicants}")

print("\\nCredit Score Distribution:")
for i, label in enumerate(credit_labels):
    count = (df_credit['credit_tier'] == i).sum()
    avg_score = df_credit[df_credit['credit_tier'] == i]['credit_score'].mean()
    print(f"{label}: {count} applicants ({count/n_applicants*100:.1f}%), avg: {avg_score:.0f}")

print("\\nDebt-to-Income Distribution:")
for i, label in enumerate(dti_labels):
    count = (df_credit['dti_tier'] == i).sum()
    avg_dti = df_credit[df_credit['dti_tier'] == i]['debt_to_income'].mean()
    print(f"{label}: {count} applicants ({count/n_applicants*100:.1f}%), avg: {avg_dti:.1%}")

print("\\nPayment History Distribution:")
for i, label in enumerate(payment_labels):
    count = (df_credit['payment_tier'] == i).sum()
    avg_payment = df_credit[df_credit['payment_tier'] == i]['payment_history'].mean()
    print(f"{label}: {count} applicants ({count/n_applicants*100:.1f}%), avg: {avg_payment:.1f}%")

print("\\nLoan Decision Summary:")
decision_summary = df_credit['loan_decision'].value_counts()
for decision, count in decision_summary.items():
    print(f"{decision}: {count} applicants ({count/n_applicants*100:.1f}%)")

# Risk analysis by credit tiers
print("\\nRisk Analysis by Credit Tier:")
risk_analysis = df_credit.groupby('credit_tier')['loan_decision'].value_counts(normalize=True).unstack(fill_value=0)
risk_analysis.index = [f"Tier {i}" for i in range(len(credit_labels))]
print(risk_analysis.round(3))
```

## Advanced Usage

### Dynamic Interval Adjustment

```python
import numpy as np
from binning import ManualIntervalBinning
import matplotlib.pyplot as plt

# Create data that might require interval adjustment
np.random.seed(42)
data = np.concatenate([
    np.random.normal(20, 5, 500),   # Low cluster
    np.random.normal(60, 8, 1200),  # Main cluster
    np.random.normal(90, 3, 300)    # High cluster
])

# Initial intervals (equal width)
initial_intervals = [
    [0, 25],
    [25, 50],
    [50, 75],
    [75, 100]
]

# Apply initial binning
initial_binner = ManualIntervalBinning(intervals=initial_intervals)
initial_bins = initial_binner.fit_transform(data.reshape(-1, 1))

# Analyze bin occupancy
bin_counts = [np.sum(initial_bins.flatten() == i) for i in range(4)]
print("Initial bin occupancy:", bin_counts)
print("Bin occupancy percentages:", [f"{count/len(data)*100:.1f}%" for count in bin_counts])

# Identify bins with very low or very high occupancy
min_occupancy = len(data) * 0.05  # 5% minimum
max_occupancy = len(data) * 0.70  # 70% maximum

print(f"\\nBins requiring adjustment (< {min_occupancy:.0f} or > {max_occupancy:.0f} samples):")
for i, count in enumerate(bin_counts):
    if count < min_occupancy:
        print(f"Bin {i}: Too few samples ({count})")
    elif count > max_occupancy:
        print(f"Bin {i}: Too many samples ({count})")

# Adjust intervals based on data distribution
data_percentiles = np.percentile(data, [20, 45, 75, 100])
adjusted_intervals = [
    [0, data_percentiles[0]],
    [data_percentiles[0], data_percentiles[1]],
    [data_percentiles[1], data_percentiles[2]],
    [data_percentiles[2], data_percentiles[3]]
]

print(f"\\nAdjusted intervals based on data distribution:")
for i, (start, end) in enumerate(adjusted_intervals):
    print(f"Bin {i}: [{start:.1f}, {end:.1f}]")

# Apply adjusted binning
adjusted_binner = ManualIntervalBinning(intervals=adjusted_intervals)
adjusted_bins = adjusted_binner.fit_transform(data.reshape(-1, 1))

adjusted_counts = [np.sum(adjusted_bins.flatten() == i) for i in range(4)]
print("\\nAdjusted bin occupancy:", adjusted_counts)
print("Adjusted percentages:", [f"{count/len(data)*100:.1f}%" for count in adjusted_counts])

# Visualize the comparison
plt.figure(figsize=(15, 10))

# Original data
plt.subplot(2, 3, 1)
plt.hist(data, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Original Data Distribution')

# Initial binning
plt.subplot(2, 3, 2)
plt.bar(range(4), bin_counts, alpha=0.7, color='lightcoral', edgecolor='black')
plt.xlabel('Bin')
plt.ylabel('Count')
plt.title('Initial Equal-Width Binning')
for i, count in enumerate(bin_counts):
    plt.text(i, count + 10, str(count), ha='center')

# Adjusted binning
plt.subplot(2, 3, 3)
plt.bar(range(4), adjusted_counts, alpha=0.7, color='lightgreen', edgecolor='black')
plt.xlabel('Bin')
plt.ylabel('Count')
plt.title('Adjusted Data-Driven Binning')
for i, count in enumerate(adjusted_counts):
    plt.text(i, count + 10, str(count), ha='center')

# Bin boundaries comparison
plt.subplot(2, 3, 4)
initial_boundaries = [interval[0] for interval in initial_intervals] + [initial_intervals[-1][1]]
adjusted_boundaries = [interval[0] for interval in adjusted_intervals] + [adjusted_intervals[-1][1]]

plt.step(initial_boundaries[:-1], [1]*len(initial_boundaries[:-1]), 'ro-', 
         label='Initial Boundaries', where='post', linewidth=2)
plt.step(adjusted_boundaries[:-1], [0.5]*len(adjusted_boundaries[:-1]), 'go-', 
         label='Adjusted Boundaries', where='post', linewidth=2)

plt.xlabel('Value')
plt.ylabel('Method')
plt.title('Boundary Comparison')
plt.legend()
plt.ylim(0, 1.5)

# Data distribution within adjusted bins
plt.subplot(2, 3, 5)
colors = ['red', 'blue', 'green', 'orange']
for i in range(4):
    mask = adjusted_bins.flatten() == i
    if mask.any():
        plt.scatter(data[mask], np.full(mask.sum(), i), 
                   alpha=0.6, s=20, color=colors[i], label=f'Bin {i}')

plt.xlabel('Value')
plt.ylabel('Bin')
plt.title('Data Points by Adjusted Bins')
plt.legend()

plt.tight_layout()
plt.show()
```

### Handling Edge Cases and Outliers

```python
import numpy as np
from binning import ManualIntervalBinning
import matplotlib.pyplot as plt

# Create data with outliers and edge cases
np.random.seed(42)
normal_data = np.random.normal(50, 10, 1800)
outliers = np.array([5, 8, 95, 98, 102, 105])  # Extreme values
data_with_outliers = np.concatenate([normal_data, outliers])

print(f"Data range: {data_with_outliers.min():.1f} to {data_with_outliers.max():.1f}")
print(f"Data with outliers: {len(data_with_outliers)} points")
print(f"Outliers: {len(outliers)} points")

# Strategy 1: Include outliers in extreme bins
strategy1_intervals = [
    [0, 30],      # Includes low outliers
    [30, 45],     # Low normal
    [45, 55],     # Mid normal
    [55, 70],     # High normal
    [70, 110]     # Includes high outliers
]

# Strategy 2: Create separate outlier bins
strategy2_intervals = [
    [0, 20],      # Low outliers bin
    [20, 40],     # Low normal
    [40, 50],     # Below average
    [50, 60],     # Above average
    [60, 80],     # High normal
    [80, 110]     # High outliers bin
]

# Strategy 3: Clip outliers to boundary values
data_clipped = np.clip(data_with_outliers, 20, 80)
strategy3_intervals = [
    [20, 35],
    [35, 50],
    [50, 65],
    [65, 80]
]

# Apply different strategies
binner1 = ManualIntervalBinning(intervals=strategy1_intervals)
binner2 = ManualIntervalBinning(intervals=strategy2_intervals)
binner3 = ManualIntervalBinning(intervals=strategy3_intervals)

bins1 = binner1.fit_transform(data_with_outliers.reshape(-1, 1))
bins2 = binner2.fit_transform(data_with_outliers.reshape(-1, 1))
bins3 = binner3.fit_transform(data_clipped.reshape(-1, 1))

# Analyze each strategy
strategies = {
    'Strategy 1 (Outliers in Extreme Bins)': (bins1, 5, strategy1_intervals),
    'Strategy 2 (Separate Outlier Bins)': (bins2, 6, strategy2_intervals),
    'Strategy 3 (Clip Outliers)': (bins3, 4, strategy3_intervals)
}

plt.figure(figsize=(15, 12))

# Original data with outliers highlighted
plt.subplot(3, 3, 1)
plt.hist(normal_data, bins=30, alpha=0.7, color='lightblue', label='Normal data')
plt.scatter(outliers, np.zeros_like(outliers), color='red', s=100, 
           label='Outliers', marker='^', zorder=5)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Original Data with Outliers')
plt.legend()

plot_idx = 2
for strategy_name, (bins, n_bins, intervals) in strategies.items():
    # Bin distribution
    plt.subplot(3, 3, plot_idx)
    
    if strategy_name.startswith('Strategy 3'):
        # For clipped data, show distribution of clipped values
        bin_counts = [np.sum(bins.flatten() == i) for i in range(n_bins)]
        data_for_analysis = data_clipped
    else:
        bin_counts = [np.sum(bins.flatten() == i) for i in range(n_bins)]
        data_for_analysis = data_with_outliers
    
    bars = plt.bar(range(n_bins), bin_counts, alpha=0.7, edgecolor='black')
    plt.xlabel('Bin')
    plt.ylabel('Count')
    plt.title(f'{strategy_name}\\nBin Distribution')
    
    # Add count labels
    for bar, count in zip(bars, bin_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha='center', va='bottom', fontsize=8)
    
    # Statistics per bin
    plt.subplot(3, 3, plot_idx + 3)
    bin_stats = []
    bin_labels = []
    
    for i in range(n_bins):
        mask = bins.flatten() == i
        if mask.any():
            bin_data = data_for_analysis[mask]
            mean_val = bin_data.mean()
            std_val = bin_data.std()
            bin_stats.append([mean_val, std_val])
            bin_labels.append(f'Bin {i}\\n[{intervals[i][0]:.0f}-{intervals[i][1]:.0f}]')
    
    bin_stats = np.array(bin_stats)
    
    # Plot means with error bars (std)
    plt.errorbar(range(len(bin_stats)), bin_stats[:, 0], yerr=bin_stats[:, 1],
                fmt='o-', capsize=5, capthick=2, linewidth=2)
    plt.xlabel('Bin')
    plt.ylabel('Value (Mean ± Std)')
    plt.title(f'{strategy_name}\\nBin Statistics')
    plt.xticks(range(len(bin_labels)), [label.split('\\n')[0] for label in bin_labels])
    
    plot_idx += 1

# Summary comparison
plt.subplot(3, 3, 7)
strategy_names = list(strategies.keys())
outlier_handling = []

for strategy_name, (bins, n_bins, intervals) in strategies.items():
    if strategy_name.startswith('Strategy 3'):
        # Count how many original outliers were clipped
        clipped_count = len(outliers)
        outlier_handling.append(clipped_count)
    else:
        # Count outliers in extreme bins
        extreme_bins = [0, n_bins-1]  # First and last bins
        outlier_count = 0
        for bin_idx in extreme_bins:
            mask = bins.flatten() == bin_idx
            bin_data = data_with_outliers[mask]
            # Count values that are actual outliers (< 20 or > 80)
            outlier_count += np.sum((bin_data < 20) | (bin_data > 80))
        outlier_handling.append(outlier_count)

bars = plt.bar(range(len(strategy_names)), outlier_handling, alpha=0.7, 
               color=['lightcoral', 'lightblue', 'lightgreen'])
plt.xlabel('Strategy')
plt.ylabel('Outliers Handled')
plt.title('Outlier Handling Comparison')
plt.xticks(range(len(strategy_names)), ['Strategy 1', 'Strategy 2', 'Strategy 3'], 
          rotation=45, ha='right')

# Add count labels
for bar, count in zip(bars, outlier_handling):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            str(count), ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Detailed analysis
print("\\nDetailed Strategy Comparison:")
for strategy_name, (bins, n_bins, intervals) in strategies.items():
    print(f"\\n{strategy_name}:")
    print(f"  Number of bins: {n_bins}")
    print(f"  Bin intervals: {intervals}")
    
    bin_counts = [np.sum(bins.flatten() == i) for i in range(n_bins)]
    print(f"  Bin occupancy: {bin_counts}")
    
    # Calculate balance metric (coefficient of variation of bin sizes)
    balance_metric = np.std(bin_counts) / np.mean(bin_counts)
    print(f"  Balance metric (lower is better): {balance_metric:.3f}")
    
    # Identify which bins contain outliers
    if strategy_name.startswith('Strategy 3'):
        print(f"  Note: Original outliers were clipped to range [20, 80]")
    else:
        outlier_bins = []
        for i in range(n_bins):
            mask = bins.flatten() == i
            bin_data = data_with_outliers[mask]
            if np.any((bin_data < 20) | (bin_data > 80)):
                outlier_count = np.sum((bin_data < 20) | (bin_data > 80))
                outlier_bins.append(f"Bin {i} ({outlier_count} outliers)")
        
        if outlier_bins:
            print(f"  Bins containing outliers: {', '.join(outlier_bins)}")
        else:
            print(f"  No bins contain outliers")
```

## Best Practices and Tips

### When to Use Manual Interval Binning

```python
print("Guidelines for using Manual Interval Binning:")
print("\\n✅ EXCELLENT for:")
print("  - Domain-specific standards (medical, financial, academic)")
print("  - Regulatory compliance requirements")
print("  - Established industry benchmarks")
print("  - Risk stratification with predefined levels")
print("  - Consistent binning across different datasets")
print("  - Interpretable business rules")
print("\\n⚠️  CONSIDER CAREFULLY for:")
print("  - Exploratory data analysis")
print("  - Data with unknown distribution")
print("  - Rapidly changing domain standards")
print("\\n❌ AVOID for:")
print("  - Data-driven optimization tasks")
print("  - Machine learning feature engineering (unless domain-specific)")
print("  - Datasets with extreme outliers requiring data-driven boundaries")
print("  - Highly skewed distributions without domain knowledge")

# Demonstrate best practices
import numpy as np
from binning import ManualIntervalBinning

# Best Practice 1: Document your interval choices
def create_documented_binner(intervals, descriptions):
    """
    Create a manual interval binner with documented rationale.
    
    Parameters:
    intervals: list of [start, end] pairs
    descriptions: list of descriptions for each interval
    """
    binner = ManualIntervalBinning(intervals=intervals)
    binner.interval_descriptions = descriptions
    return binner

# Example: Credit scoring with documented intervals
credit_intervals = [
    [300, 580],  # Poor
    [580, 670],  # Fair  
    [670, 740],  # Good
    [740, 800],  # Very Good
    [800, 850]   # Exceptional
]

credit_descriptions = [
    "Poor: High risk, likely to default, requires collateral",
    "Fair: Elevated risk, higher interest rates, may require co-signer", 
    "Good: Average risk, standard terms and rates",
    "Very Good: Low risk, favorable terms and rates",
    "Exceptional: Minimal risk, best available terms"
]

documented_binner = create_documented_binner(credit_intervals, credit_descriptions)

print("\\nExample: Documented Credit Score Intervals")
for i, (interval, description) in enumerate(zip(credit_intervals, credit_descriptions)):
    print(f"Interval {i}: {interval} - {description}")

# Best Practice 2: Validate interval coverage
def validate_intervals(intervals, data_range):
    """
    Validate that intervals cover the expected data range.
    """
    min_boundary = min(interval[0] for interval in intervals)
    max_boundary = max(interval[1] for interval in intervals)
    
    issues = []
    
    if min_boundary > data_range[0]:
        issues.append(f"Gap below minimum: data starts at {data_range[0]}, intervals start at {min_boundary}")
    
    if max_boundary < data_range[1]:
        issues.append(f"Gap above maximum: data ends at {data_range[1]}, intervals end at {max_boundary}")
    
    # Check for gaps between intervals
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    for i in range(len(sorted_intervals) - 1):
        current_end = sorted_intervals[i][1]
        next_start = sorted_intervals[i+1][0]
        if current_end != next_start:
            issues.append(f"Gap between intervals: {current_end} to {next_start}")
    
    return issues

# Test validation
test_data_range = (250, 900)  # Credit scores might go beyond standard range
validation_issues = validate_intervals(credit_intervals, test_data_range)

print("\\nInterval Validation Results:")
if not validation_issues:
    print("✅ No issues found")
else:
    for issue in validation_issues:
        print(f"⚠️  {issue}")

# Best Practice 3: Handle boundary conditions explicitly
def apply_manual_binning_with_boundary_handling(data, intervals, boundary_strategy='inclusive_upper'):
    """
    Apply manual binning with explicit boundary handling.
    
    Parameters:
    boundary_strategy: 'inclusive_upper' (default), 'inclusive_lower', 'midpoint'
    """
    binner = ManualIntervalBinning(intervals=intervals)
    
    # Check for boundary cases
    boundary_cases = []
    for i, (start, end) in enumerate(intervals):
        if boundary_strategy == 'inclusive_upper':
            # Values exactly on boundary go to upper bin (except last bin)
            if i < len(intervals) - 1:
                boundary_cases.extend(data[data == end])
            else:
                boundary_cases.extend(data[data == end])
    
    if boundary_cases:
        print(f"Found {len(boundary_cases)} values on bin boundaries")
        print(f"Boundary strategy: {boundary_strategy}")
    
    return binner.fit_transform(data.reshape(-1, 1))

# Test boundary handling
test_data = np.array([579.9, 580.0, 580.1, 669.9, 670.0, 670.1])
print(f"\\nBoundary Handling Test:")
print(f"Test data: {test_data}")

result = apply_manual_binning_with_boundary_handling(test_data, credit_intervals)
print(f"Bin assignments: {result.flatten()}")

for i, (value, bin_id) in enumerate(zip(test_data, result.flatten())):
    interval = credit_intervals[bin_id]
    print(f"  {value} → Bin {bin_id} {interval}")
```

This comprehensive example documentation for Manual Interval Binning covers:

1. **Basic Usage**: Single and multi-feature manual binning with domain-specific intervals
2. **Real-world Applications**: Academic grading, medical risk stratification, financial credit scoring
3. **Advanced Techniques**: Dynamic interval adjustment, outlier handling strategies
4. **Best Practices**: Documentation, validation, boundary handling

Each example demonstrates how manual interval binning provides precise control over bin boundaries for domain-specific requirements and standardized classifications.
