# K-Means Binning Examples

This page demonstrates the use of `KMeansBinning` for creating bins based on data clustering using the K-means algorithm.

## Basic Usage

### Understanding K-Means Binning

```python
import numpy as np
import matplotlib.pyplot as plt
from binning import KMeansBinning, EqualWidthBinning, EqualFrequencyBinning

# Create data with natural clusters
np.random.seed(42)
cluster1 = np.random.normal(2, 0.5, 300)
cluster2 = np.random.normal(6, 0.8, 400)
cluster3 = np.random.normal(10, 0.6, 300)
clustered_data = np.concatenate([cluster1, cluster2, cluster3]).reshape(-1, 1)

# Apply different binning methods
kmeans_binner = KMeansBinning(n_bins=3)
equal_width_binner = EqualWidthBinning(n_bins=3)
equal_freq_binner = EqualFrequencyBinning(n_bins=3)

kmeans_binned = kmeans_binner.fit_transform(clustered_data)
width_binned = equal_width_binner.fit_transform(clustered_data)
freq_binned = equal_freq_binner.fit_transform(clustered_data)

print("K-Means Cluster Centers:", kmeans_binner.cluster_centers_[0])
print("Equal Width Bin Edges:", equal_width_binner.bin_edges_[0])
print("Equal Frequency Quantiles:", equal_freq_binner.quantiles_[0])

# Visualization
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.hist(clustered_data, bins=30, alpha=0.7, edgecolor='black')
plt.title('Original Data with Natural Clusters')
plt.xlabel('Value')
plt.ylabel('Frequency')

for method, binned_data, title in [
    ('K-Means', kmeans_binned, 'K-Means Binning\\n(Cluster-based)'),
    ('Equal Width', width_binned, 'Equal Width Binning\\n(Fixed intervals)'),
    ('Equal Frequency', freq_binned, 'Equal Frequency Binning\\n(Equal counts)')
]:
    plt.subplot(2, 2, [2, 3, 4][['K-Means', 'Equal Width', 'Equal Frequency'].index(method)])
    unique, counts = np.unique(binned_data, return_counts=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique)))
    plt.bar(unique, counts, alpha=0.7, color=colors, edgecolor='black')
    plt.title(title)
    plt.xlabel('Bin')
    plt.ylabel('Count')

plt.tight_layout()
plt.show()
```

### Multi-dimensional K-Means Binning

```python
import numpy as np
from binning import KMeansBinning
import matplotlib.pyplot as plt

# Create 2D data with natural clusters
np.random.seed(42)
n_samples = 800

# Create three distinct clusters in 2D space
cluster1 = np.random.multivariate_normal([2, 2], [[1, 0.3], [0.3, 1]], 250)
cluster2 = np.random.multivariate_normal([8, 3], [[1.5, -0.5], [-0.5, 1.5]], 300)
cluster3 = np.random.multivariate_normal([5, 8], [[0.8, 0.2], [0.2, 0.8]], 250)

data_2d = np.vstack([cluster1, cluster2, cluster3])

# Apply K-means binning
kmeans_binner = KMeansBinning(n_bins=3)
binned_2d = kmeans_binner.fit_transform(data_2d)

print("2D Data shape:", data_2d.shape)
print("Binned data shape:", binned_2d.shape)
print("\\nCluster centers:")
for i, centers in enumerate(kmeans_binner.cluster_centers_):
    print(f"Feature {i}: {centers}")

# Visualize 2D clustering
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.6, s=30)
plt.title('Original 2D Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(1, 2, 2)
colors = ['red', 'blue', 'green']
for bin_id in np.unique(binned_2d[:, 0]):  # Using first feature's bins for coloring
    mask = binned_2d[:, 0] == bin_id
    plt.scatter(data_2d[mask, 0], data_2d[mask, 1], 
               c=colors[bin_id], alpha=0.6, s=30, label=f'Bin {bin_id}')

# Plot cluster centers
centers_x = kmeans_binner.cluster_centers_[0]
centers_y = kmeans_binner.cluster_centers_[1]
plt.scatter(centers_x, centers_y, c='black', marker='x', s=200, linewidths=3)

plt.title('K-Means Binning Result')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.tight_layout()
plt.show()
```

## Real-world Applications

### Customer Segmentation Based on Behavior

```python
import numpy as np
import pandas as pd
from binning import KMeansBinning
import matplotlib.pyplot as plt

# Simulate customer behavior data
np.random.seed(42)
n_customers = 2000

# Create realistic customer segments
# Segment 1: High-value, frequent customers
high_value = {
    'purchase_frequency': np.random.normal(15, 3, 400),  # purchases per month
    'avg_order_value': np.random.normal(150, 30, 400),   # dollars
    'days_since_last': np.random.exponential(5, 400)     # days
}

# Segment 2: Medium-value, occasional customers  
medium_value = {
    'purchase_frequency': np.random.normal(6, 2, 800),
    'avg_order_value': np.random.normal(80, 20, 800),
    'days_since_last': np.random.exponential(15, 800)
}

# Segment 3: Low-value, rare customers
low_value = {
    'purchase_frequency': np.random.normal(2, 1, 600),
    'avg_order_value': np.random.normal(35, 10, 600),
    'days_since_last': np.random.exponential(45, 600)
}

# Segment 4: Dormant customers
dormant = {
    'purchase_frequency': np.random.normal(0.5, 0.2, 200),
    'avg_order_value': np.random.normal(45, 15, 200),
    'days_since_last': np.random.exponential(120, 200)
}

# Combine all segments
purchase_frequency = np.concatenate([
    high_value['purchase_frequency'], medium_value['purchase_frequency'],
    low_value['purchase_frequency'], dormant['purchase_frequency']
])

avg_order_value = np.concatenate([
    high_value['avg_order_value'], medium_value['avg_order_value'],
    low_value['avg_order_value'], dormant['avg_order_value']
])

days_since_last = np.concatenate([
    high_value['days_since_last'], medium_value['days_since_last'],
    low_value['days_since_last'], dormant['days_since_last']
])

# Create DataFrame
df = pd.DataFrame({
    'customer_id': range(n_customers),
    'purchase_frequency': np.clip(purchase_frequency, 0, None),
    'avg_order_value': np.clip(avg_order_value, 10, None),
    'days_since_last': np.clip(days_since_last, 0, None)
})

print("Customer Data Overview:")
print(df.describe())

# Apply K-means binning for segmentation
features = ['purchase_frequency', 'avg_order_value', 'days_since_last']
X = df[features].values

# Use K-means binning to identify natural customer segments
kmeans_binner = KMeansBinning(n_bins=4)  # Expecting 4 natural segments
customer_segments = kmeans_binner.fit_transform(X)

# Add segment labels to DataFrame
df['segment'] = customer_segments[:, 0]  # Use first feature's clustering

# Define segment names based on characteristics
segment_stats = df.groupby('segment')[features].mean()
print("\\nSegment Characteristics:")
print(segment_stats)

# Assign meaningful names to segments
segment_names = {
    0: 'VIP Customers',
    1: 'Regular Customers', 
    2: 'Occasional Customers',
    3: 'At-Risk Customers'
}

df['segment_name'] = df['segment'].map(segment_names)

print("\\nSegment Distribution:")
print(df['segment_name'].value_counts())

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 3D scatter plot representation using 2D projections
feature_pairs = [
    ('purchase_frequency', 'avg_order_value'),
    ('purchase_frequency', 'days_since_last'),
    ('avg_order_value', 'days_since_last')
]

colors = ['red', 'blue', 'green', 'orange']

for i, (x_col, y_col) in enumerate(feature_pairs[:3]):
    row, col = i // 2, i % 2
    ax = axes[row, col] if i < 2 else axes[1, 1]
    
    for segment in df['segment'].unique():
        mask = df['segment'] == segment
        ax.scatter(df[mask][x_col], df[mask][y_col], 
                  c=colors[segment], alpha=0.6, s=30, 
                  label=segment_names[segment])
    
    ax.set_xlabel(x_col.replace('_', ' ').title())
    ax.set_ylabel(y_col.replace('_', ' ').title())
    ax.set_title(f'{x_col.replace("_", " ").title()} vs {y_col.replace("_", " ").title()}')
    ax.legend()

# Segment summary
axes[1, 0].axis('off')
summary_text = "Segment Summary:\\n\\n"
for segment in df['segment'].unique():
    name = segment_names[segment]
    count = (df['segment'] == segment).sum()
    summary_text += f"{name}: {count} customers\\n"

axes[1, 0].text(0.1, 0.5, summary_text, fontsize=12, 
               verticalalignment='center', transform=axes[1, 0].transAxes)

plt.tight_layout()
plt.show()
```

### Image Processing: Color Quantization

```python
import numpy as np
import matplotlib.pyplot as plt
from binning import KMeansBinning
from sklearn.datasets import load_sample_image

# Load a sample image
china = load_sample_image("china.jpg")
print(f"Original image shape: {china.shape}")

# Reshape image to 2D array of pixels
original_shape = china.shape
pixel_data = china.reshape(-1, 3)  # Each row is an RGB pixel
print(f"Pixel data shape: {pixel_data.shape}")

# Apply K-means binning for color quantization
color_levels = [4, 8, 16, 32]

plt.figure(figsize=(20, 15))

# Original image
plt.subplot(3, 3, 1)
plt.imshow(china)
plt.title('Original Image')
plt.axis('off')

for i, n_colors in enumerate(color_levels):
    # Apply K-means binning to RGB values
    kmeans_binner = KMeansBinning(n_bins=n_colors)
    quantized_pixels = kmeans_binner.fit_transform(pixel_data.astype(float))
    
    # Get the color palette (cluster centers)
    color_palette = []
    for channel in range(3):  # RGB channels
        color_palette.append(kmeans_binner.cluster_centers_[channel])
    
    # Reconstruct image using cluster centers
    reconstructed_pixels = np.zeros_like(pixel_data, dtype=float)
    for channel in range(3):
        for bin_id in range(n_colors):
            mask = quantized_pixels[:, channel] == bin_id
            reconstructed_pixels[mask, channel] = color_palette[channel][bin_id]
    
    # Reshape back to original image shape
    reconstructed_image = reconstructed_pixels.reshape(original_shape)
    reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)
    
    # Display quantized image
    plt.subplot(3, 3, i + 2)
    plt.imshow(reconstructed_image)
    plt.title(f'{n_colors} Colors (K-means)')
    plt.axis('off')
    
    # Display color palette
    plt.subplot(3, 3, i + 6)
    palette_display = np.array(color_palette).T.reshape(1, n_colors, 3)
    palette_display = np.clip(palette_display, 0, 255).astype(np.uint8)
    plt.imshow(palette_display)
    plt.title(f'Color Palette ({n_colors} colors)')
    plt.axis('off')

plt.tight_layout()
plt.show()

# Show file size reduction potential
original_bits = china.size * 8  # 8 bits per color channel
for n_colors in color_levels:
    bits_per_pixel = np.ceil(np.log2(n_colors))  # Bits needed to represent n_colors
    compressed_bits = china.shape[0] * china.shape[1] * bits_per_pixel + n_colors * 3 * 8
    compression_ratio = original_bits / compressed_bits
    print(f"{n_colors:2d} colors: {compression_ratio:.1f}x compression ratio")
```

### Financial Data: Price Movement Analysis

```python
import numpy as np
import pandas as pd
from binning import KMeansBinning
import matplotlib.pyplot as plt

# Simulate stock price movements
np.random.seed(42)
n_days = 1000

# Generate price movements with different volatility regimes
regime1 = np.random.normal(0.001, 0.02, 300)  # Low volatility bull market
regime2 = np.random.normal(-0.002, 0.04, 200)  # High volatility bear market  
regime3 = np.random.normal(0.0005, 0.015, 300)  # Stable market
regime4 = np.random.normal(0.003, 0.06, 200)   # Very volatile bull market

returns = np.concatenate([regime1, regime2, regime3, regime4])

# Calculate additional features
rolling_volatility = pd.Series(returns).rolling(20).std().fillna(0).values
price_momentum = pd.Series(returns).rolling(10).mean().fillna(0).values

# Create feature matrix
feature_data = np.column_stack([returns, rolling_volatility, price_momentum])

print("Financial data shape:", feature_data.shape)
print("\\nFeature statistics:")
feature_names = ['Daily Returns', 'Rolling Volatility', 'Price Momentum']
for i, name in enumerate(feature_names):
    print(f"{name}: mean={feature_data[:, i].mean():.4f}, std={feature_data[:, i].std():.4f}")

# Apply K-means binning to identify market regimes
regime_binner = KMeansBinning(n_bins=4)
market_regimes = regime_binner.fit_transform(feature_data)

# Create DataFrame for analysis
df = pd.DataFrame({
    'day': range(n_days),
    'returns': returns,
    'volatility': rolling_volatility,
    'momentum': price_momentum,
    'regime': market_regimes[:, 0]  # Use returns-based clustering
})

# Define regime characteristics
regime_stats = df.groupby('regime')[['returns', 'volatility', 'momentum']].agg(['mean', 'std'])
print("\\nMarket Regime Characteristics:")
print(regime_stats)

# Assign regime names based on characteristics
regime_names = {}
for regime in df['regime'].unique():
    avg_return = regime_stats.loc[regime, ('returns', 'mean')]
    avg_vol = regime_stats.loc[regime, ('volatility', 'mean')]
    
    if avg_return > 0.001 and avg_vol < 0.03:
        regime_names[regime] = 'Bull Market (Low Vol)'
    elif avg_return > 0.001 and avg_vol >= 0.03:
        regime_names[regime] = 'Bull Market (High Vol)'
    elif avg_return < -0.001:
        regime_names[regime] = 'Bear Market'
    else:
        regime_names[regime] = 'Sideways Market'

df['regime_name'] = df['regime'].map(regime_names)

print("\\nRegime Distribution:")
print(df['regime_name'].value_counts())

# Visualization
plt.figure(figsize=(15, 12))

# Time series of returns with regime coloring
plt.subplot(3, 1, 1)
colors = ['red', 'blue', 'green', 'orange']
for regime in df['regime'].unique():
    mask = df['regime'] == regime
    plt.scatter(df[mask]['day'], df[mask]['returns'], 
               c=colors[regime], alpha=0.6, s=10, label=regime_names[regime])
plt.title('Daily Returns by Market Regime')
plt.xlabel('Day')
plt.ylabel('Daily Return')
plt.legend()

# Cumulative returns by regime
plt.subplot(3, 1, 2)
cumulative_returns = (1 + df['returns']).cumprod()
plt.plot(df['day'], cumulative_returns, 'black', alpha=0.7, linewidth=2)
plt.title('Cumulative Returns')
plt.xlabel('Day')
plt.ylabel('Cumulative Return')

# Regime distribution over time
plt.subplot(3, 1, 3)
for regime in df['regime'].unique():
    mask = df['regime'] == regime
    plt.scatter(df[mask]['day'], [regime] * mask.sum(), 
               c=colors[regime], alpha=0.6, s=20, label=regime_names[regime])
plt.title('Market Regime Evolution')
plt.xlabel('Day')
plt.ylabel('Regime')
plt.yticks(df['regime'].unique(), [regime_names[r] for r in df['regime'].unique()])
plt.legend()

plt.tight_layout()
plt.show()
```

## Advanced Usage

### Optimizing Number of Clusters

```python
import numpy as np
from binning import KMeansBinning
import matplotlib.pyplot as plt

# Create data with unknown number of natural clusters
np.random.seed(42)
cluster_centers = [1, 4, 7, 11, 15]  # 5 natural clusters
data_points = []

for center in cluster_centers:
    cluster_data = np.random.normal(center, 0.8, 200)
    data_points.extend(cluster_data)

data = np.array(data_points).reshape(-1, 1)

# Test different numbers of clusters
k_values = range(2, 11)
inertias = []
silhouette_scores = []

from sklearn.metrics import silhouette_score

for k in k_values:
    binner = KMeansBinning(n_bins=k)
    binned_data = binner.fit_transform(data)
    
    # Calculate inertia (within-cluster sum of squares)
    inertia = 0
    for i in range(k):
        cluster_center = binner.cluster_centers_[0][i]
        cluster_points = data[binned_data[:, 0] == i]
        if len(cluster_points) > 0:
            inertia += np.sum((cluster_points - cluster_center) ** 2)
    
    inertias.append(inertia)
    
    # Calculate silhouette score
    if k > 1:
        sil_score = silhouette_score(data, binned_data[:, 0])
        silhouette_scores.append(sil_score)

# Plot elbow curve and silhouette scores
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(k_values, inertias, 'bo-')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(k_values[1:], silhouette_scores, 'ro-')
plt.title('Silhouette Score vs k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)

# Show the optimal clustering
optimal_k = k_values[np.argmax(silhouette_scores) + 1]
optimal_binner = KMeansBinning(n_bins=optimal_k)
optimal_binned = optimal_binner.fit_transform(data)

plt.subplot(1, 3, 3)
colors = plt.cm.Set3(np.linspace(0, 1, optimal_k))
for i in range(optimal_k):
    cluster_data = data[optimal_binned[:, 0] == i]
    plt.hist(cluster_data, alpha=0.7, color=colors[i], 
             label=f'Cluster {i}', bins=20)
    plt.axvline(optimal_binner.cluster_centers_[0][i], 
               color=colors[i], linestyle='--', linewidth=2)

plt.title(f'Optimal Clustering (k={optimal_k})')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()

print(f"Optimal number of clusters: {optimal_k}")
print(f"Best silhouette score: {max(silhouette_scores):.3f}")
```

### Handling Different Data Scales

```python
import numpy as np
from binning import KMeansBinning
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Create data with different scales
np.random.seed(42)
n_samples = 800

# Feature 1: Small scale (0-10)
feature1 = np.random.exponential(2, n_samples)

# Feature 2: Large scale (1000-10000)  
feature2 = np.random.normal(5000, 1500, n_samples)

# Feature 3: Medium scale (0-100)
feature3 = np.random.uniform(0, 100, n_samples)

data_unscaled = np.column_stack([feature1, feature2, feature3])

print("Data statistics before scaling:")
print(f"Feature 1: mean={feature1.mean():.2f}, std={feature1.std():.2f}")
print(f"Feature 2: mean={feature2.mean():.2f}, std={feature2.std():.2f}")
print(f"Feature 3: mean={feature3.mean():.2f}, std={feature3.std():.2f}")

# Apply K-means binning without scaling
binner_unscaled = KMeansBinning(n_bins=4)
binned_unscaled = binner_unscaled.fit_transform(data_unscaled)

# Apply scaling then K-means binning
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_unscaled)

binner_scaled = KMeansBinning(n_bins=4)
binned_scaled = binner_scaled.fit_transform(data_scaled)

print("\\nCluster centers (unscaled data):")
for i, centers in enumerate(binner_unscaled.cluster_centers_):
    print(f"Feature {i+1}: {centers}")

print("\\nCluster centers (scaled data):")
for i, centers in enumerate(binner_scaled.cluster_centers_):
    print(f"Feature {i+1}: {centers}")

# Compare clustering results
from sklearn.metrics import adjusted_rand_score

ari_score = adjusted_rand_score(binned_unscaled[:, 0], binned_scaled[:, 0])
print(f"\\nAdjusted Rand Index between scaled/unscaled: {ari_score:.3f}")

# Visualization
plt.figure(figsize=(15, 10))

# Unscaled data clustering
plt.subplot(2, 3, 1)
colors = ['red', 'blue', 'green', 'orange']
for bin_id in range(4):
    mask = binned_unscaled[:, 0] == bin_id
    plt.scatter(data_unscaled[mask, 0], data_unscaled[mask, 1], 
               c=colors[bin_id], alpha=0.6, s=30, label=f'Cluster {bin_id}')
plt.xlabel('Feature 1 (Small Scale)')
plt.ylabel('Feature 2 (Large Scale)')
plt.title('Unscaled Data Clustering')
plt.legend()

plt.subplot(2, 3, 2)
for bin_id in range(4):
    mask = binned_unscaled[:, 0] == bin_id
    plt.scatter(data_unscaled[mask, 0], data_unscaled[mask, 2], 
               c=colors[bin_id], alpha=0.6, s=30, label=f'Cluster {bin_id}')
plt.xlabel('Feature 1 (Small Scale)')
plt.ylabel('Feature 3 (Medium Scale)')
plt.title('Unscaled Data Clustering')
plt.legend()

plt.subplot(2, 3, 3)
for bin_id in range(4):
    mask = binned_unscaled[:, 0] == bin_id
    plt.scatter(data_unscaled[mask, 1], data_unscaled[mask, 2], 
               c=colors[bin_id], alpha=0.6, s=30, label=f'Cluster {bin_id}')
plt.xlabel('Feature 2 (Large Scale)')
plt.ylabel('Feature 3 (Medium Scale)')
plt.title('Unscaled Data Clustering')
plt.legend()

# Scaled data clustering
plt.subplot(2, 3, 4)
for bin_id in range(4):
    mask = binned_scaled[:, 0] == bin_id
    plt.scatter(data_scaled[mask, 0], data_scaled[mask, 1], 
               c=colors[bin_id], alpha=0.6, s=30, label=f'Cluster {bin_id}')
plt.xlabel('Feature 1 (Standardized)')
plt.ylabel('Feature 2 (Standardized)')
plt.title('Scaled Data Clustering')
plt.legend()

plt.subplot(2, 3, 5)
for bin_id in range(4):
    mask = binned_scaled[:, 0] == bin_id
    plt.scatter(data_scaled[mask, 0], data_scaled[mask, 2], 
               c=colors[bin_id], alpha=0.6, s=30, label=f'Cluster {bin_id}')
plt.xlabel('Feature 1 (Standardized)')
plt.ylabel('Feature 3 (Standardized)')
plt.title('Scaled Data Clustering')
plt.legend()

plt.subplot(2, 3, 6)
for bin_id in range(4):
    mask = binned_scaled[:, 0] == bin_id
    plt.scatter(data_scaled[mask, 1], data_scaled[mask, 2], 
               c=colors[bin_id], alpha=0.6, s=30, label=f'Cluster {bin_id}')
plt.xlabel('Feature 2 (Standardized)')
plt.ylabel('Feature 3 (Standardized)')
plt.title('Scaled Data Clustering')
plt.legend()

plt.tight_layout()
plt.show()
```

## Integration with Machine Learning

### Feature Engineering with K-Means Binning

```python
import numpy as np
from binning import KMeansBinning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import make_classification

# Create synthetic dataset
X, y = make_classification(
    n_samples=2000,
    n_features=8,
    n_informative=6,
    n_redundant=2,
    n_clusters_per_class=2,
    random_state=42
)

print("Original dataset shape:", X.shape)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Baseline model with original features
rf_baseline = RandomForestClassifier(random_state=42)
rf_baseline.fit(X_train, y_train)
y_pred_baseline = rf_baseline.predict(X_test)
baseline_accuracy = accuracy_score(y_test, y_pred_baseline)

print(f"Baseline accuracy: {baseline_accuracy:.3f}")

# Model with K-means binned features
kmeans_binner = KMeansBinning(n_bins=5)
X_train_binned = kmeans_binner.fit_transform(X_train)
X_test_binned = kmeans_binner.transform(X_test)

rf_binned = RandomForestClassifier(random_state=42)
rf_binned.fit(X_train_binned, y_train)
y_pred_binned = rf_binned.predict(X_test_binned)
binned_accuracy = accuracy_score(y_test, y_pred_binned)

print(f"K-means binned accuracy: {binned_accuracy:.3f}")

# Combined model (original + binned features)
X_train_combined = np.concatenate([X_train, X_train_binned], axis=1)
X_test_combined = np.concatenate([X_test, X_test_binned], axis=1)

rf_combined = RandomForestClassifier(random_state=42)
rf_combined.fit(X_train_combined, y_train)
y_pred_combined = rf_combined.predict(X_test_combined)
combined_accuracy = accuracy_score(y_test, y_pred_combined)

print(f"Combined features accuracy: {combined_accuracy:.3f}")

# Feature importance analysis
feature_names = [f'Original_{i}' for i in range(X.shape[1])] + \\
                [f'KMeans_{i}' for i in range(X_train_binned.shape[1])]

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_combined.feature_importances_
}).sort_values('importance', ascending=False)

print("\\nTop 10 most important features:")
print(importance_df.head(10))
```

## Performance and Scalability

### Large Dataset Processing

```python
import numpy as np
import time
from binning import KMeansBinning

def benchmark_kmeans_binning():
    """Benchmark K-means binning with different dataset sizes and dimensions."""
    
    print("K-Means Binning Performance Benchmark:")
    print("=" * 50)
    
    # Test different dataset sizes
    sizes = [1000, 5000, 10000, 50000]
    dimensions = [1, 3, 5, 10]
    
    for n_features in dimensions:
        print(f"\\nFeatures: {n_features}")
        print("-" * 30)
        
        for size in sizes:
            # Create test data
            np.random.seed(42)
            data = np.random.rand(size, n_features)
            
            # Time the binning operation
            start_time = time.time()
            binner = KMeansBinning(n_bins=5)
            binned_data = binner.fit_transform(data)
            end_time = time.time()
            
            duration = end_time - start_time
            memory_mb = data.nbytes / (1024**2)
            
            print(f"Size: {size:5d} | Time: {duration:.3f}s | Memory: {memory_mb:.1f}MB")

benchmark_kmeans_binning()
```

## Best Practices and Tips

### When to Use K-Means Binning

```python
import numpy as np
import matplotlib.pyplot as plt
from binning import KMeansBinning, EqualWidthBinning, EqualFrequencyBinning

# Create different data distributions to demonstrate when K-means is best

distributions = {
    'Normal Distribution': np.random.normal(0, 1, 1000),
    'Bimodal Distribution': np.concatenate([
        np.random.normal(-2, 0.5, 500),
        np.random.normal(2, 0.5, 500)
    ]),
    'Multimodal Distribution': np.concatenate([
        np.random.normal(-3, 0.3, 250),
        np.random.normal(-1, 0.3, 250),
        np.random.normal(1, 0.3, 250),
        np.random.normal(3, 0.3, 250)
    ]),
    'Uniform Distribution': np.random.uniform(-3, 3, 1000)
}

fig, axes = plt.subplots(4, 4, figsize=(20, 16))

for i, (name, data) in enumerate(distributions.items()):
    data = data.reshape(-1, 1)
    
    # Original distribution
    axes[i, 0].hist(data, bins=30, alpha=0.7, edgecolor='black')
    axes[i, 0].set_title(f'{name}\\nOriginal Data')
    
    # K-means binning
    kmeans_binner = KMeansBinning(n_bins=4)
    kmeans_binned = kmeans_binner.fit_transform(data)
    unique, counts = np.unique(kmeans_binned, return_counts=True)
    axes[i, 1].bar(unique, counts, alpha=0.7)
    axes[i, 1].set_title('K-Means Binning')
    
    # Equal width binning
    ew_binner = EqualWidthBinning(n_bins=4)
    ew_binned = ew_binner.fit_transform(data)
    unique, counts = np.unique(ew_binned, return_counts=True)
    axes[i, 2].bar(unique, counts, alpha=0.7)
    axes[i, 2].set_title('Equal Width Binning')
    
    # Equal frequency binning
    ef_binner = EqualFrequencyBinning(n_bins=4)
    ef_binned = ef_binner.fit_transform(data)
    unique, counts = np.unique(ef_binned, return_counts=True)
    axes[i, 3].bar(unique, counts, alpha=0.7)
    axes[i, 3].set_title('Equal Frequency Binning')

plt.tight_layout()
plt.show()

print("Guidelines for using K-Means Binning:")
print("\\n✅ BEST for:")
print("  - Data with natural clusters or modes")
print("  - Multimodal distributions")
print("  - When cluster-based grouping makes domain sense")
print("  - Image processing and color quantization")
print("  - Customer segmentation based on behavior")
print("\\n⚠️  CONSIDER CAREFULLY for:")
print("  - Uniform distributions (no natural clusters)")
print("  - Single-modal normal distributions")
print("  - When interpretable bin boundaries are needed")
print("\\n❌ AVOID for:")
print("  - Very high-dimensional sparse data")
print("  - Data where distance-based clustering doesn't make sense")
print("  - When you need exactly equal-sized bins")
```

This comprehensive example documentation for K-Means Binning covers:

1. **Basic Usage**: Comparison with other methods, multi-dimensional clustering
2. **Real-world Applications**: Customer segmentation, image processing, financial analysis
3. **Advanced Techniques**: Cluster optimization, handling different scales
4. **ML Integration**: Feature engineering, combined with original features
5. **Performance**: Benchmarking with different sizes and dimensions
6. **Best Practices**: When to choose K-means over other binning methods

Each example demonstrates the unique advantages of K-means binning for data with natural cluster structures.
