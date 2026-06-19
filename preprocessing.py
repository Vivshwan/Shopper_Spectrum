# ============================================================================
# COMPLETE CODE - FORCE 4 CLUSTERS WITH MEMORY OPTIMIZATION
# ============================================================================

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from mpl_toolkits.mplot3d import Axes3D
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Create models directory if it doesn't exist
models_dir = 'models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"✅ Created directory: {models_dir}")

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.width', None)

# Set style for better visuals
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("ONLINE RETAIL ANALYSIS - FORCE 4 CLUSTERS (MEMORY OPTIMIZED)")
print("="*80)

# ============================================================================
# STEP 1: DATASET COLLECTION AND UNDERSTANDING
# ============================================================================

print("\n" + "="*80)
print("STEP 1: DATASET COLLECTION AND UNDERSTANDING")
print("="*80)

# Load the dataset
file_path = 'online_retail.csv'
df = pd.read_csv(file_path)

print(f"\n📊 Dataset Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")

# ============================================================================
# STEP 2: DATA PREPROCESSING
# ============================================================================

print("\n" + "="*80)
print("STEP 2: DATA PREPROCESSING")
print("="*80)

# Make a copy for preprocessing
df_clean = df.copy()

# Convert InvoiceDate to datetime
df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'], errors='coerce')

# 1. Remove rows with missing CustomerID
df_clean = df_clean.dropna(subset=['CustomerID'])

# 2. Exclude cancelled invoices (InvoiceNo starting with 'C')
df_clean = df_clean[~df_clean['InvoiceNo'].astype(str).str.startswith('C')]

# 3. Remove negative or zero quantities
df_clean = df_clean[df_clean['Quantity'] > 0]

# 4. Remove negative or zero prices
df_clean = df_clean[df_clean['UnitPrice'] > 0]

# Calculate TotalAmount for each transaction
df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['UnitPrice']

print(f"\n📊 Cleaned Dataset:")
print(f"   Total transactions: {len(df_clean):,}")
print(f"   Unique customers: {df_clean['CustomerID'].nunique():,}")
print(f"   Unique products: {df_clean['StockCode'].nunique():,}")
print(f"   Total revenue: ${df_clean['TotalAmount'].sum():,.2f}")

# ============================================================================
# STEP 3: RFM ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("STEP 3: RFM ANALYSIS")
print("="*80)

# Calculate RFM
snapshot_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df_clean.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalAmount': 'sum'
}).rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalAmount': 'Monetary'
})

print(f"\n📊 RFM Statistics:")
print(rfm.describe())

# ============================================================================
# STEP 4: CLUSTERING METHODOLOGY - FORCE 4 CLUSTERS
# ============================================================================

print("\n" + "="*80)
print("STEP 4: CLUSTERING METHODOLOGY - FORCE 4 CLUSTERS")
print("="*80)

# Scale the RFM values
scaler_standard = StandardScaler()
rfm_scaled = scaler_standard.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

print("\n✅ Applied StandardScaler for clustering")

# Save scaler
with open(f'{models_dir}/rfm_scaler.pkl', 'wb') as file:
    pickle.dump(scaler_standard, file)
print(f"✅ Saved scaler to: {models_dir}/rfm_scaler.pkl")

# ============================================================================
# 4.4: ELBOW METHOD WITH MEMORY OPTIMIZATION
# ============================================================================

print("\n" + "="*50)
print("4.4: OPTIMAL CLUSTER DETERMINATION (MEMORY OPTIMIZED)")
print("="*50)

# Calculate metrics for different k values (using sampling for silhouette)
k_range = range(2, 11)
wcss = []
silhouette_scores = []
calinski_scores = []
davies_scores = []

# Sample data for silhouette score calculation (to avoid memory issues)
sample_size = min(2000, len(rfm_scaled))  # Use max 2000 samples
np.random.seed(42)
sample_indices = np.random.choice(len(rfm_scaled), sample_size, replace=False)
rfm_scaled_sample = rfm_scaled[sample_indices]

print(f"📊 Using {sample_size} samples for silhouette score calculation")

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(rfm_scaled)
    wcss.append(kmeans.inertia_)
    
    # Calculate silhouette on sampled data
    sampled_labels = labels[sample_indices]
    sil_score = silhouette_score(rfm_scaled_sample, sampled_labels)
    silhouette_scores.append(sil_score)
    
    # Calinski-Harabasz and Davies-Bouldin on full data (these are memory efficient)
    calinski_scores.append(calinski_harabasz_score(rfm_scaled, labels))
    davies_scores.append(davies_bouldin_score(rfm_scaled, labels))

# Find best k
best_k = k_range[np.argmax(silhouette_scores)]

# ============================================================================
# FORCE 4 CLUSTERS - OVERRIDE THE OPTIMAL K
# ============================================================================

print(f"\n📊 Elbow Method suggests k = {best_k}")
print(f"📊 Silhouette Score: Best k = {best_k} (Score: {max(silhouette_scores):.4f})")

# FORCE 4 CLUSTERS - We want exactly 4 clusters for our 4 segment labels
FORCED_K = 4
print(f"\n✅ FORCING 4 CLUSTERS for exact segment labeling:")
print(f"   • High-Value")
print(f"   • Regular")
print(f"   • Occasional")
print(f"   • At-Risk")

optimal_k = FORCED_K

# Create elbow and silhouette plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Elbow Curve
axes[0, 0].plot(k_range, wcss, 'bo-', linewidth=2, markersize=8)
axes[0, 0].set_title('Elbow Method - WCSS', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Number of Clusters (k)')
axes[0, 0].set_ylabel('WCSS')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axvline(x=optimal_k, color='red', linestyle='--', label=f'Forced k={optimal_k}')
axes[0, 0].axvline(x=best_k, color='green', linestyle=':', label=f'Optimal k={best_k}')
axes[0, 0].legend()

# 2. Silhouette Scores
axes[0, 1].plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
axes[0, 1].set_title('Silhouette Score Analysis (Sampled)', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Number of Clusters (k)')
axes[0, 1].set_ylabel('Silhouette Score')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axvline(x=optimal_k, color='red', linestyle='--', label=f'Forced k={optimal_k}')
axes[0, 1].axvline(x=best_k, color='green', linestyle=':', label=f'Optimal k={best_k}')
axes[0, 1].legend()
axes[0, 1].axhline(y=0.25, color='green', linestyle=':', alpha=0.5, label='Good threshold')

# 3. Calinski-Harabasz Score
axes[1, 0].plot(k_range, calinski_scores, 'go-', linewidth=2, markersize=8)
axes[1, 0].set_title('Calinski-Harabasz Score', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Number of Clusters (k)')
axes[1, 0].set_ylabel('Score')
axes[1, 0].grid(True, alpha=0.3)

# 4. Davies-Bouldin Score
axes[1, 1].plot(k_range, davies_scores, 'mo-', linewidth=2, markersize=8)
axes[1, 1].set_title('Davies-Bouldin Score', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Number of Clusters (k)')
axes[1, 1].set_ylabel('Score (lower is better)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{models_dir}/cluster_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: cluster_metrics_comparison.png")

# ============================================================================
# 4.5: RUN CLUSTERING WITH FORCED 4 CLUSTERS
# ============================================================================

print("\n" + "="*50)
print("4.5: RUN CLUSTERING WITH 4 CLUSTERS")
print("="*50)

# Apply K-Means with 4 clusters
kmeans_final = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm['Cluster'] = kmeans_final.fit_predict(rfm_scaled)

# Calculate final metrics (using sampled data for silhouette)
final_silhouette = silhouette_score(rfm_scaled_sample, rfm['Cluster'].iloc[sample_indices])
final_calinski = calinski_harabasz_score(rfm_scaled, rfm['Cluster'])
final_davies = davies_bouldin_score(rfm_scaled, rfm['Cluster'])

print(f"\n📊 Final Clustering Metrics (4 Clusters):")
print(f"  • Silhouette Score: {final_silhouette:.4f} (based on {sample_size} samples)")
print(f"  • Calinski-Harabasz: {final_calinski:.4f}")
print(f"  • Davies-Bouldin: {final_davies:.4f}")

# Save the K-Means model
with open(f'{models_dir}/kmeans_final_model.pkl', 'wb') as file:
    pickle.dump(kmeans_final, file)
print(f"✅ Saved K-Means model to: {models_dir}/kmeans_final_model.pkl")

# ============================================================================
# 4.6: LABEL CLUSTERS BASED ON RFM CHARACTERISTICS
# ============================================================================

print("\n" + "="*50)
print("4.6: LABELING CLUSTERS BASED ON RFM CHARACTERISTICS")
print("="*50)

# Calculate cluster profiles
cluster_profile = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
cluster_count = rfm.groupby('Cluster').size()

print("\n📊 Raw Cluster Profiles:")
print(cluster_profile)

# Normalize for comparison (0-1 scale)
cluster_profile_norm = (cluster_profile - cluster_profile.min()) / (cluster_profile.max() - cluster_profile.min())

print("\n📊 Normalized Cluster Profiles:")
print(cluster_profile_norm)

# ============================================================================
# FUNCTION TO LABEL CLUSTERS EXACTLY AS PER THE TABLE
# ============================================================================

def label_cluster_based_on_rfm(row):
    """
    Label clusters based on RFM characteristics exactly as specified:
    
    | Characteristics                    | Segment Label |
    |-----------------------------------|---------------|
    | High R, High F, High M            | High-Value    |
    | Medium F, Medium M                | Regular       |
    | Low F, Low M, older R             | Occasional    |
    | High R, Low F, Low M              | At-Risk       |
    """
    r_score = row['Recency']   # Lower is better (0-1 scale, 0=recent, 1=old)
    f_score = row['Frequency'] # Higher is better (0-1 scale, 0=low, 1=high)
    m_score = row['Monetary']  # Higher is better (0-1 scale, 0=low, 1=high)
    
    # Define thresholds for each characteristic
    r_high = 0.6   # High Recency (old customers)
    r_low = 0.3    # Low Recency (recent customers)
    f_high = 0.6   # High Frequency
    f_low = 0.3    # Low Frequency
    f_medium = 0.4 # Medium Frequency (between 0.4 and 0.6)
    m_high = 0.6   # High Monetary
    m_low = 0.3    # Low Monetary
    m_medium = 0.4 # Medium Monetary (between 0.4 and 0.6)
    
    # 1. High R, High F, High M → High-Value
    if r_score <= r_low and f_score >= f_high and m_score >= m_high:
        return 'High-Value'
    
    # 2. Medium F, Medium M → Regular
    elif f_score >= f_medium and f_score < f_high and m_score >= m_medium and m_score < m_high:
        return 'Regular'
    
    # 3. Low F, Low M, older R → Occasional
    elif f_score <= f_low and m_score <= m_low and r_score >= r_high:
        return 'Occasional'
    
    # 4. High R, Low F, Low M → At-Risk
    elif r_score >= r_high and f_score <= f_low and m_score <= m_low:
        return 'At-Risk'
    
    # Fallback: Additional categories for remaining clusters
    else:
        # If none of the above match, classify based on closest pattern
        if f_score >= f_high and m_score >= m_high:
            return 'High-Value'
        elif f_score >= f_medium and m_score >= m_medium:
            return 'Regular'
        elif r_score >= r_high:
            return 'At-Risk'
        else:
            return 'Occasional'

# Apply labeling to each cluster using normalized values
cluster_labels = {}
segment_descriptions = {}
descriptions = {
    'High-Value': 'Regular, frequent, recent, and big spenders',
    'Regular': 'Steady purchasers but not premium',
    'Occasional': 'Rare, occasional purchases',
    'At-Risk': 'Haven\'t purchased in a long time'
}

# First pass: label each cluster
for cluster_id in range(4):
    row = cluster_profile_norm.loc[cluster_id]
    label = label_cluster_based_on_rfm(row)
    cluster_labels[cluster_id] = label

# Check if we got all 4 unique labels
unique_labels = set(cluster_labels.values())
print(f"\n📊 Unique labels found: {unique_labels}")

# If we don't have all 4 labels, manually assign based on characteristics
if len(unique_labels) < 4:
    print("\n⚠️ Not all 4 labels were assigned. Manually assigning based on RFM characteristics...")
    
    # Get cluster characteristics
    cluster_r = cluster_profile['Recency']
    cluster_f = cluster_profile['Frequency']
    cluster_m = cluster_profile['Monetary']
    
    # Sort clusters by characteristics
    # Highest Recency (oldest) + Lowest Frequency + Lowest Monetary → At-Risk
    at_risk_cluster = cluster_r.idxmax()
    
    # Lowest Recency (recent) + Highest Frequency + Highest Monetary → High-Value
    high_value_cluster = (cluster_f + cluster_m).idxmax()
    
    # Remaining clusters: one with medium F/M → Regular, the other → Occasional
    remaining = [i for i in range(4) if i not in [at_risk_cluster, high_value_cluster]]
    
    if len(remaining) >= 2:
        # Sort remaining by frequency+monetary (higher is more regular)
        remaining_scores = {i: cluster_f[i] + cluster_m[i] for i in remaining}
        sorted_remaining = sorted(remaining_scores.items(), key=lambda x: x[1], reverse=True)
        regular_cluster = sorted_remaining[0][0]
        occasional_cluster = sorted_remaining[1][0]
    else:
        regular_cluster = remaining[0] if len(remaining) > 0 else 2
        occasional_cluster = 3
    
    # Assign labels
    cluster_labels = {
        high_value_cluster: 'High-Value',
        regular_cluster: 'Regular',
        occasional_cluster: 'Occasional',
        at_risk_cluster: 'At-Risk'
    }
    print(f"   Manually assigned: {cluster_labels}")

# Store descriptions for each segment
for label in cluster_labels.values():
    segment_descriptions[label] = descriptions.get(label, 'Mixed behavior')

# Map labels back to the rfm dataframe
rfm['Segment_Label'] = rfm['Cluster'].map(cluster_labels)

# ============================================================================
# CREATE CLUSTER SUMMARY WITH EXACT LABELS
# ============================================================================

cluster_summary = pd.DataFrame({
    'Cluster': range(4),
    'Size': cluster_count.values,
    'Percentage': (cluster_count.values / len(rfm) * 100).round(1),
    'Avg_Recency': cluster_profile['Recency'].round(1),
    'Avg_Frequency': cluster_profile['Frequency'].round(1),
    'Avg_Monetary': cluster_profile['Monetary'].round(2),
    'Segment_Label': [cluster_labels.get(i, 'Unknown') for i in range(4)],
    'Characteristics': [segment_descriptions.get(cluster_labels.get(i, 'Unknown'), 'Mixed') for i in range(4)]
})

# Sort by Segment Label for better readability
cluster_summary = cluster_summary.sort_values('Segment_Label').reset_index(drop=True)

print("\n" + "="*50)
print("📊 FINAL CLUSTER SUMMARY WITH LABELS")
print("="*50)
print(cluster_summary.to_string(index=False))

# ============================================================================
# DISPLAY DETAILED CLUSTER INFORMATION
# ============================================================================

print("\n" + "="*50)
print("📊 DETAILED CLUSTER INTERPRETATION")
print("="*50)

for idx, row in cluster_summary.iterrows():
    print(f"\n🔹 {row['Segment_Label']} (Cluster {row['Cluster']})")
    print(f"   • Size: {row['Size']:,} customers ({row['Percentage']:.1f}%)")
    print(f"   • Average Recency: {row['Avg_Recency']:.1f} days since last purchase")
    print(f"   • Average Frequency: {row['Avg_Frequency']:.1f} purchases")
    print(f"   • Average Monetary: ${row['Avg_Monetary']:.2f}")
    print(f"   • Characteristics: {row['Characteristics']}")

# ============================================================================
# SAVE CLUSTER RESULTS (ONLY PKL FILES)
# ============================================================================

print("\n" + "="*50)
print("4.7: SAVING CLUSTER RESULTS")
print("="*50)

# Save cluster labels
with open(f'{models_dir}/cluster_labels.pkl', 'wb') as file:
    pickle.dump(cluster_labels, file)
print(f"✅ Saved cluster labels to: {models_dir}/cluster_labels.pkl")

# Save cluster centroids
with open(f'{models_dir}/cluster_centroids.pkl', 'wb') as file:
    pickle.dump(kmeans_final.cluster_centers_, file)
print(f"✅ Saved cluster centroids to: {models_dir}/cluster_centroids.pkl")

# ============================================================================
# 4.8: VISUALIZE CLUSTERS
# ============================================================================

print("\n" + "="*50)
print("4.8: CLUSTER VISUALIZATIONS")
print("="*50)

# PCA for 2D visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(rfm_scaled)
rfm['PCA1'] = pca_result[:, 0]
rfm['PCA2'] = pca_result[:, 1]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Define segment colors
segment_colors = {
    'High-Value': '#2ecc71',   # Green
    'Regular': '#3498db',      # Blue
    'Occasional': '#f39c12',   # Orange
    'At-Risk': '#e74c3c'       # Red
}

# 1. PCA Scatter - Clusters
scatter = axes[0, 0].scatter(rfm['PCA1'], rfm['PCA2'], c=rfm['Cluster'], 
                            cmap='viridis', alpha=0.6, s=50)
axes[0, 0].set_title('Customer Clusters (PCA)', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.colorbar(scatter, ax=axes[0, 0])

# 2. PCA Scatter - Segment Labels
for segment in rfm['Segment_Label'].unique():
    mask = rfm['Segment_Label'] == segment
    color = segment_colors.get(segment, 'gray')
    axes[0, 1].scatter(rfm.loc[mask, 'PCA1'], rfm.loc[mask, 'PCA2'], 
                      label=segment, c=color, alpha=0.6, s=30)
axes[0, 1].set_title('Customer Segments (PCA)', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Principal Component 1')
axes[0, 1].set_ylabel('Principal Component 2')
axes[0, 1].legend(loc='best')

# 3. 3D Scatter Plot
ax3d = fig.add_subplot(2, 3, 3, projection='3d')
scatter3d = ax3d.scatter(rfm['Recency'], rfm['Frequency'], rfm['Monetary'],
                        c=rfm['Cluster'], cmap='viridis', alpha=0.6)
ax3d.set_title('3D RFM Visualization', fontsize=14, fontweight='bold')
ax3d.set_xlabel('Recency')
ax3d.set_ylabel('Frequency')
ax3d.set_zlabel('Monetary')
ax3d.view_init(elev=20, azim=45)

# 4. Cluster Profiles (Bar Chart)
cluster_profile_norm.plot(kind='bar', ax=axes[1, 0], width=0.8)
axes[1, 0].set_title('Cluster Profiles (Normalized)', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Cluster')
axes[1, 0].set_ylabel('Normalized Value')
axes[1, 0].legend(['Recency', 'Frequency', 'Monetary'])
axes[1, 0].grid(True, alpha=0.3)

# 5. Cluster Size Distribution with Segment Labels
colors = [segment_colors.get(cluster_labels.get(i, 'Unknown'), 'gray') for i in range(4)]
bars = axes[1, 1].bar(range(4), cluster_summary['Size'], color=colors, alpha=0.7)
axes[1, 1].set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Cluster')
axes[1, 1].set_ylabel('Number of Customers')
axes[1, 1].grid(True, alpha=0.3)

# Add labels on bars
for bar, row in zip(bars, cluster_summary.iterrows()):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                    f"{row[1]['Segment_Label']}\n{int(bar.get_height())}", 
                    ha='center', fontsize=9)

# 6. Segment Distribution Pie Chart
segment_counts = rfm['Segment_Label'].value_counts()
colors_pie = [segment_colors.get(seg, 'gray') for seg in segment_counts.index]
axes[1, 2].pie(segment_counts.values, labels=segment_counts.index, 
               autopct='%1.1f%%', colors=colors_pie, startangle=90)
axes[1, 2].set_title('Segment Distribution', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{models_dir}/cluster_visualizations_complete.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: cluster_visualizations_complete.png")

# ============================================================================
# 4.9: SAVE STREAMLIT MODEL PACKAGE
# ============================================================================

print("\n" + "="*50)
print("4.9: SAVING STREAMLIT MODEL PACKAGE")
print("="*50)

# Create complete model package
model_package = {
    'model': kmeans_final,
    'scaler': scaler_standard,
    'cluster_summary': cluster_summary,
    'cluster_profiles': cluster_profile,
    'rfm_data': rfm,
    'feature_names': ['Recency', 'Frequency', 'Monetary'],
    'n_clusters': 4,
    'model_metrics': {
        'silhouette_score': final_silhouette,
        'calinski_harabasz': final_calinski,
        'davies_bouldin': final_davies
    },
    'segment_labels': cluster_labels,
    'segment_descriptions': segment_descriptions,
    'segment_colors': segment_colors
}

with open(f'{models_dir}/streamlit_model_package.pkl', 'wb') as file:
    pickle.dump(model_package, file)
print(f"✅ Saved Streamlit model package to: {models_dir}/streamlit_model_package.pkl")

# ============================================================================
# STEP 5: EDA VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("STEP 5: EDA VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(20, 20))

# 1. Transaction volume by country
ax1 = plt.subplot(3, 3, 1)
country_volume = df_clean.groupby('Country').size().sort_values(ascending=True).tail(10)
country_volume.plot(kind='barh', ax=ax1, color='steelblue')
ax1.set_title('Top 10 Countries by Volume', fontsize=12, fontweight='bold')
ax1.set_xlabel('Number of Transactions')

# 2. Revenue by country
ax2 = plt.subplot(3, 3, 2)
country_revenue = df_clean.groupby('Country')['TotalAmount'].sum().sort_values(ascending=True).tail(10)
country_revenue.plot(kind='barh', ax=ax2, color='coral')
ax2.set_title('Top 10 Countries by Revenue', fontsize=12, fontweight='bold')
ax2.set_xlabel('Total Revenue ($)')

# 3. Top selling products
ax3 = plt.subplot(3, 3, 3)
top_products = df_clean.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
top_products.plot(kind='barh', ax=ax3, color='forestgreen')
ax3.set_title('Top 10 Products by Quantity', fontsize=12, fontweight='bold')
ax3.set_xlabel('Total Quantity Sold')

# 4. Daily sales trend
ax4 = plt.subplot(3, 3, 4)
daily_sales = df_clean.groupby(df_clean['InvoiceDate'].dt.date)['TotalAmount'].sum()
daily_sales.plot(ax=ax4, color='darkblue', linewidth=1)
ax4.set_title('Daily Sales Trend', fontsize=12, fontweight='bold')
ax4.set_xlabel('Date')
ax4.set_ylabel('Daily Revenue ($)')
ax4.tick_params(axis='x', rotation=45)

# 5. Monthly sales trend
ax5 = plt.subplot(3, 3, 5)
monthly_sales = df_clean.groupby(df_clean['InvoiceDate'].dt.to_period('M'))['TotalAmount'].sum()
monthly_sales.plot(kind='bar', ax=ax5, color='purple')
ax5.set_title('Monthly Sales Trend', fontsize=12, fontweight='bold')
ax5.set_xlabel('Month')
ax5.set_ylabel('Monthly Revenue ($)')
ax5.tick_params(axis='x', rotation=45)

# 6. Hourly transactions
ax6 = plt.subplot(3, 3, 6)
hourly_transactions = df_clean.groupby(df_clean['InvoiceDate'].dt.hour).size()
hourly_transactions.plot(kind='bar', ax=ax6, color='orange')
ax6.set_title('Transactions by Hour', fontsize=12, fontweight='bold')
ax6.set_xlabel('Hour')
ax6.set_ylabel('Number of Transactions')

# 7. Transaction value distribution
ax7 = plt.subplot(3, 3, 7)
df_clean['TotalAmount'].hist(bins=50, ax=ax7, color='skyblue', edgecolor='black', alpha=0.7)
ax7.set_title('Transaction Value Distribution', fontsize=12, fontweight='bold')
ax7.set_xlabel('Transaction Amount ($)')
ax7.set_ylabel('Frequency')
ax7.set_xlim(0, df_clean['TotalAmount'].quantile(0.95))

# 8. Customer lifetime value
ax8 = plt.subplot(3, 3, 8)
rfm['Monetary'].hist(bins=50, ax=ax8, color='lightgreen', edgecolor='black', alpha=0.7)
ax8.set_title('Customer Lifetime Value', fontsize=12, fontweight='bold')
ax8.set_xlabel('Total Spend ($)')
ax8.set_ylabel('Number of Customers')
ax8.set_xlim(0, rfm['Monetary'].quantile(0.95))

# 9. RFM Distributions
ax9 = plt.subplot(3, 3, 9)
rfm[['Recency', 'Frequency', 'Monetary']].hist(bins=30, ax=ax9, alpha=0.5)
ax9.set_title('RFM Distributions', fontsize=12, fontweight='bold')
ax9.set_xlabel('Value')

plt.tight_layout()
plt.savefig(f'{models_dir}/eda_visualizations.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: eda_visualizations.png")

# ============================================================================
# STEP 6: PRODUCT RECOMMENDATION SYSTEM (FULL DATASET)
# ============================================================================

print("\n" + "="*80)
print("STEP 6: PRODUCT RECOMMENDATION SYSTEM")
print("="*80)

try:
    # Create user-item matrix (full dataset)
    print("Creating user-item matrix...")
    user_item_matrix = df_clean.pivot_table(
        index='CustomerID', 
        columns='StockCode', 
        values='Quantity',
        fill_value=0,
        aggfunc='sum'
    )

    print(f"✅ User-Item Matrix: {user_item_matrix.shape[0]:,} customers x {user_item_matrix.shape[1]:,} products")
    print(f"   Matrix density: {user_item_matrix.astype(bool).sum().sum() / (user_item_matrix.shape[0] * user_item_matrix.shape[1]) * 100:.4f}%")

    # Calculate item similarity using cosine similarity (full matrix)
    print("\n📊 Calculating cosine similarity for all products...")
    item_similarity = cosine_similarity(user_item_matrix.T)
    item_similarity_df = pd.DataFrame(
        item_similarity,
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns
    )

    print(f"✅ Similarity matrix shape: {item_similarity_df.shape[0]:,} x {item_similarity_df.shape[1]:,}")

    # Save the recommendation model
    recommendation_data = {
        'item_similarity_matrix': item_similarity_df,
        'product_descriptions': df_clean[['StockCode', 'Description']].drop_duplicates(subset=['StockCode']),
        'user_item_matrix': user_item_matrix
    }

    with open(f'{models_dir}/recommendation_model.pkl', 'wb') as file:
        pickle.dump(recommendation_data, file)
    print(f"✅ Saved recommendation model to: {models_dir}/recommendation_model.pkl")

    # ============================================================================
    # RECOMMENDATION VISUALIZATION
    # ============================================================================

    print("\n📊 Generating recommendation visualizations...")

    # Get top products for heatmap
    top_products_vis = df_clean.groupby('StockCode')['Quantity'].sum().sort_values(ascending=False).head(15).index

    # Create product names
    product_names = {}
    for code in top_products_vis:
        desc = df_clean[df_clean['StockCode'] == code]['Description'].iloc[0] if len(df_clean[df_clean['StockCode'] == code]) > 0 else f'Product_{code}'
        if len(desc) > 25:
            desc = desc[:22] + '...'
        product_names[code] = f"{desc} ({code})"

    # Get similarity matrix for top products
    top_similarity = item_similarity_df.loc[top_products_vis, top_products_vis]
    
    # Rename index and columns for better readability
    renamed_index = [product_names[code] for code in top_similarity.index]
    top_similarity.index = renamed_index
    top_similarity.columns = renamed_index

    # Plot heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Heatmap of product similarities
    sns.heatmap(top_similarity, cmap='coolwarm', center=0, ax=ax1, square=True, 
                cbar_kws={'label': 'Similarity Score'})
    ax1.set_title('Product Similarity Heatmap (Top 15 Products)', fontsize=14, fontweight='bold')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)

    # Get recommendations for sample customer
    sample_customer = user_item_matrix.index[0]
    print(f"\n📊 Generating recommendations for sample customer: {sample_customer}")
    
    purchased = user_item_matrix.loc[sample_customer]
    purchased_products = purchased[purchased > 0].index.tolist()
    print(f"   Customer purchased {len(purchased_products)} products")

    if purchased_products:
        # Filter to products that exist in similarity matrix
        valid_products = [p for p in purchased_products if p in item_similarity_df.index]
        print(f"   Valid products for recommendation: {len(valid_products)}")
        
        if valid_products:
            # Calculate average similarity scores
            sim_scores = item_similarity_df.loc[valid_products].mean(axis=0)
            
            # Remove already purchased products
            recommendations = sim_scores.drop(valid_products, errors='ignore').sort_values(ascending=False).head(10)
            
            if not recommendations.empty:
                rec_names = []
                rec_values = []
                for code in recommendations.index:
                    desc = df_clean[df_clean['StockCode'] == code]['Description'].iloc[0] if len(df_clean[df_clean['StockCode'] == code]) > 0 else f'Product_{code}'
                    if len(desc) > 35:
                        desc = desc[:32] + '...'
                    rec_names.append(desc)
                    rec_values.append(recommendations[code])
                
                # Plot recommendations
                bars = ax2.barh(rec_names, rec_values, color='teal', alpha=0.7)
                ax2.set_title(f'Top 10 Recommendations for Customer {sample_customer}', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Similarity Score')
                ax2.grid(True, alpha=0.3)
                
                for bar, val in zip(bars, rec_values):
                    ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center')
                
                print(f"   ✅ Generated {len(recommendations)} recommendations")
            else:
                print("   ⚠️ No recommendations available for this customer")
                ax2.text(0.5, 0.5, 'No recommendations available', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        else:
            print("   ⚠️ No valid products found for similarity calculation")
            ax2.text(0.5, 0.5, 'No products found for similarity calculation', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=14)
    else:
        print("   ⚠️ Customer has no purchases")
        ax2.text(0.5, 0.5, 'Customer has no purchases', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=14)

    plt.tight_layout()
    plt.savefig(f'{models_dir}/recommendations_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: recommendations_visualization.png")

    # ============================================================================
    # RECOMMENDATION STATISTICS
    # ============================================================================

    print("\n📊 Recommendation System Statistics:")
    print(f"   • Total Products: {len(item_similarity_df):,}")
    print(f"   • Total Customers: {len(user_item_matrix):,}")
    print(f"   • Similarity Matrix Size: {item_similarity_df.shape[0]:,} x {item_similarity_df.shape[1]:,}")
    print(f"   • Memory Usage: {item_similarity_df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

except Exception as e:
    print(f"❌ Recommendation system error: {e}")
    import traceback
    traceback.print_exc()
    print("   Continuing without recommendation model...")

# ============================================================================
# FINAL OUTPUT
# ============================================================================

print("\n" + "="*80)
print("✅ PROJECT COMPLETED SUCCESSFULLY!")
print("="*80)

print(f"\n📁 Files saved in '{models_dir}' folder (PKL & PNG only):")
print("-"*50)

# List all files in models directory
files = os.listdir(models_dir)
pkl_files = [f for f in files if f.endswith('.pkl')]
png_files = [f for f in files if f.endswith('.png')]

print("\n🤖 MODEL FILES (.pkl):")
for file in sorted(pkl_files):
    file_path = os.path.join(models_dir, file)
    size = os.path.getsize(file_path)
    size_str = f"{size/1024:.1f} KB" if size < 1024*1024 else f"{size/(1024*1024):.1f} MB"
    print(f"  📄 {file} ({size_str})")

print("\n📊 VISUALIZATION FILES (.png):")
for file in sorted(png_files):
    file_path = os.path.join(models_dir, file)
    size = os.path.getsize(file_path)
    size_str = f"{size/1024:.1f} KB" if size < 1024*1024 else f"{size/(1024*1024):.1f} MB"
    print(f"  🖼️ {file} ({size_str})")

print("\n" + "="*80)
print("📊 SEGMENT DISTRIBUTION (4 Clusters):")
print("="*80)
segment_order = ['High-Value', 'Regular', 'Occasional', 'At-Risk']
for segment in segment_order:
    count = len(rfm[rfm['Segment_Label'] == segment])
    pct = count / len(rfm) * 100 if len(rfm) > 0 else 0
    print(f"  • {segment}: {count:,} customers ({pct:.1f}%)")

print("\n" + "="*80)
print("📊 FILES SAVED IN 'models' FOLDER:")
print("="*80)
print("\n🤖 Models (.pkl):")
print("  1. rfm_scaler.pkl - Standard scaler for new data")
print("  2. kmeans_final_model.pkl - Trained K-Means model (4 clusters)")
print("  3. cluster_labels.pkl - Cluster to segment mapping")
print("  4. cluster_centroids.pkl - Cluster centroids")
print("  5. streamlit_model_package.pkl - Complete model package for Streamlit")
if 'recommendation_model.pkl' in pkl_files:
    print("  6. recommendation_model.pkl - Product recommendation system")

print("\n📊 Visualizations (.png):")
print("  7. cluster_metrics_comparison.png - Elbow, Silhouette, etc.")
print("  8. cluster_visualizations_complete.png - PCA, 3D, Profiles")
print("  9. eda_visualizations.png - EDA plots")
print("  10. recommendations_visualization.png - Product recommendations")

print("\n" + "="*80)
print("✅ COMPLETE! 4 CLUSTERS FOR EXACT LABELING")
print("="*80)
