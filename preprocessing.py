# ============================================================================
# COMPLETE CODE WITH SAVING FUNCTIONALITY
# ============================================================================

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
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
print("ONLINE RETAIL ANALYSIS - COMPLETE PIPELINE")
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
print(f"\n📋 Columns: {df.columns.tolist()}")

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
before_count = len(df_clean)
df_clean = df_clean.dropna(subset=['CustomerID'])
print(f"\n✅ Removed {before_count - len(df_clean):,} rows with missing CustomerID")

# 2. Exclude cancelled invoices (InvoiceNo starting with 'C')
before_count = len(df_clean)
df_clean = df_clean[~df_clean['InvoiceNo'].astype(str).str.startswith('C')]
print(f"✅ Removed {before_count - len(df_clean):,} cancelled invoices")

# 3. Remove negative or zero quantities
before_count = len(df_clean)
df_clean = df_clean[df_clean['Quantity'] > 0]
print(f"✅ Removed {before_count - len(df_clean):,} rows with negative/zero quantity")

# 4. Remove negative or zero prices
before_count = len(df_clean)
df_clean = df_clean[df_clean['UnitPrice'] > 0]
print(f"✅ Removed {before_count - len(df_clean):,} rows with negative/zero price")

# Calculate TotalAmount for each transaction
df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['UnitPrice']

print(f"\n📊 Cleaned Dataset:")
print(f"   Total transactions: {len(df_clean):,}")
print(f"   Unique customers: {df_clean['CustomerID'].nunique():,}")
print(f"   Unique products: {df_clean['StockCode'].nunique():,}")
print(f"   Total revenue: ${df_clean['TotalAmount'].sum():,.2f}")

# Save cleaned data
df_clean.to_csv(f'{models_dir}/cleaned_retail_data.csv', index=False)
print(f"\n✅ Saved cleaned data to: {models_dir}/cleaned_retail_data.csv")

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
    'InvoiceNo': 'count',
    'TotalAmount': 'sum'
}).rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalAmount': 'Monetary'
})

print(f"\n📊 RFM Statistics:")
print(rfm.describe())

# Save RFM data
rfm.to_csv(f'{models_dir}/rfm_data.csv')
print(f"\n✅ Saved RFM data to: {models_dir}/rfm_data.csv")

# ============================================================================
# STEP 4: CUSTOMER SEGMENTATION WITH K-MEANS
# ============================================================================

print("\n" + "="*80)
print("STEP 4: CUSTOMER SEGMENTATION")
print("="*80)

# Prepare data for clustering
rfm_scaled = StandardScaler().fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# Find optimal number of clusters (Elbow method)
wcss = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(rfm_scaled)
    wcss.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, 'bo-', linewidth=2, markersize=8)
plt.title('Elbow Method for Optimal Clusters', fontsize=14, fontweight='bold')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.grid(True, alpha=0.3)
plt.savefig(f'{models_dir}/elbow_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved elbow curve to: {models_dir}/elbow_curve.png")

# Apply KMeans with optimal k=4
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Save the model
with open(f'{models_dir}/kmeans_model.pkl', 'wb') as file:
    pickle.dump(kmeans, file)
print(f"✅ Saved K-Means model to: {models_dir}/kmeans_model.pkl")

# Save scaler
with open(f'{models_dir}/scaler.pkl', 'wb') as file:
    pickle.dump(StandardScaler().fit(rfm[['Recency', 'Frequency', 'Monetary']]), file)
print(f"✅ Saved scaler to: {models_dir}/scaler.pkl")

# Cluster profiles
cluster_profile = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
cluster_profile.to_csv(f'{models_dir}/cluster_profiles.csv')
print(f"✅ Saved cluster profiles to: {models_dir}/cluster_profiles.csv")

# Create RFM scores
rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=['4', '3', '2', '1']).astype(int)
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=['1', '2', '3', '4']).astype(int)
rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 4, labels=['1', '2', '3', '4']).astype(int)
rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

# Segment customers
def segment_customers(row):
    if row['R_Score'] >= 4 and row['F_Score'] >= 4 and row['M_Score'] >= 4:
        return 'Champions'
    elif row['R_Score'] >= 3 and row['F_Score'] >= 3:
        return 'Loyal Customers'
    elif row['R_Score'] >= 4 and row['F_Score'] <= 2:
        return 'New Customers'
    elif row['R_Score'] <= 2 and row['F_Score'] >= 3:
        return 'At-Risk Customers'
    elif row['R_Score'] <= 1 and row['F_Score'] >= 2:
        return 'Lost Customers'
    else:
        return 'Others'

rfm['Segment'] = rfm.apply(segment_customers, axis=1)

# Save RFM with segments
rfm.to_csv(f'{models_dir}/rfm_with_segments.csv')
print(f"✅ Saved RFM with segments to: {models_dir}/rfm_with_segments.csv")

# ============================================================================
# STEP 5: VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("STEP 5: GENERATING VISUALIZATIONS")
print("="*80)

# Create comprehensive visualizations
fig = plt.figure(figsize=(20, 25))

# 1. Transaction volume by country
ax1 = plt.subplot(4, 3, 1)
country_volume = df_clean.groupby('Country').size().sort_values(ascending=True).tail(10)
country_volume.plot(kind='barh', ax=ax1, color='steelblue')
ax1.set_title('Top 10 Countries by Transaction Volume', fontsize=14, fontweight='bold')
ax1.set_xlabel('Number of Transactions')

# 2. Revenue by country
ax2 = plt.subplot(4, 3, 2)
country_revenue = df_clean.groupby('Country')['TotalAmount'].sum().sort_values(ascending=True).tail(10)
country_revenue.plot(kind='barh', ax=ax2, color='coral')
ax2.set_title('Top 10 Countries by Revenue', fontsize=14, fontweight='bold')
ax2.set_xlabel('Total Revenue ($)')

# 3. Top selling products
ax3 = plt.subplot(4, 3, 3)
top_products = df_clean.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
top_products.plot(kind='barh', ax=ax3, color='forestgreen')
ax3.set_title('Top 10 Products by Quantity Sold', fontsize=14, fontweight='bold')
ax3.set_xlabel('Total Quantity Sold')

# 4. Daily sales trend
ax4 = plt.subplot(4, 3, 4)
daily_sales = df_clean.groupby(df_clean['InvoiceDate'].dt.date)['TotalAmount'].sum()
daily_sales.plot(ax=ax4, color='darkblue', linewidth=1)
ax4.set_title('Daily Sales Trend', fontsize=14, fontweight='bold')
ax4.set_xlabel('Date')
ax4.set_ylabel('Daily Revenue ($)')
ax4.tick_params(axis='x', rotation=45)

# 5. Monthly sales trend
ax5 = plt.subplot(4, 3, 5)
monthly_sales = df_clean.groupby(df_clean['InvoiceDate'].dt.to_period('M'))['TotalAmount'].sum()
monthly_sales.plot(kind='bar', ax=ax5, color='purple')
ax5.set_title('Monthly Sales Trend', fontsize=14, fontweight='bold')
ax5.set_xlabel('Month')
ax5.set_ylabel('Monthly Revenue ($)')
ax5.tick_params(axis='x', rotation=45)

# 6. Hourly transactions
ax6 = plt.subplot(4, 3, 6)
hourly_transactions = df_clean.groupby(df_clean['InvoiceDate'].dt.hour).size()
hourly_transactions.plot(kind='bar', ax=ax6, color='orange')
ax6.set_title('Transactions by Hour of Day', fontsize=14, fontweight='bold')
ax6.set_xlabel('Hour')
ax6.set_ylabel('Number of Transactions')

# 7. Transaction value distribution
ax7 = plt.subplot(4, 3, 7)
df_clean['TotalAmount'].hist(bins=50, ax=ax7, color='skyblue', edgecolor='black', alpha=0.7)
ax7.set_title('Transaction Value Distribution', fontsize=14, fontweight='bold')
ax7.set_xlabel('Transaction Amount ($)')
ax7.set_ylabel('Frequency')
ax7.set_xlim(0, df_clean['TotalAmount'].quantile(0.95))

# 8. Customer lifetime value
ax8 = plt.subplot(4, 3, 8)
rfm['Monetary'].hist(bins=50, ax=ax8, color='lightgreen', edgecolor='black', alpha=0.7)
ax8.set_title('Customer Lifetime Value Distribution', fontsize=14, fontweight='bold')
ax8.set_xlabel('Total Spend per Customer ($)')
ax8.set_ylabel('Number of Customers')
ax8.set_xlim(0, rfm['Monetary'].quantile(0.95))

# 9. Recency distribution
ax9 = plt.subplot(4, 3, 9)
rfm['Recency'].hist(bins=30, ax=ax9, color='salmon', edgecolor='black', alpha=0.7)
ax9.set_title('Recency Distribution (Days Since Last Purchase)', fontsize=14, fontweight='bold')
ax9.set_xlabel('Days')
ax9.set_ylabel('Number of Customers')

# 10. Frequency distribution
ax10 = plt.subplot(4, 3, 10)
rfm['Frequency'].hist(bins=30, ax=ax10, color='gold', edgecolor='black', alpha=0.7)
ax10.set_title('Frequency Distribution (Number of Purchases)', fontsize=14, fontweight='bold')
ax10.set_xlabel('Number of Transactions')
ax10.set_ylabel('Number of Customers')
ax10.set_xlim(0, rfm['Frequency'].quantile(0.95))

# 11. Monetary distribution
ax11 = plt.subplot(4, 3, 11)
rfm['Monetary'].hist(bins=30, ax=ax11, color='lightcoral', edgecolor='black', alpha=0.7)
ax11.set_title('Monetary Distribution (Total Spend)', fontsize=14, fontweight='bold')
ax11.set_xlabel('Total Spend ($)')
ax11.set_ylabel('Number of Customers')
ax11.set_xlim(0, rfm['Monetary'].quantile(0.95))

# 12. Elbow curve (already plotted, adding reference)
ax12 = plt.subplot(4, 3, 12)
ax12.plot(k_range, wcss, 'bo-', linewidth=2, markersize=8)
ax12.set_title('Elbow Method for Optimal Clusters', fontsize=14, fontweight='bold')
ax12.set_xlabel('Number of Clusters (k)')
ax12.set_ylabel('WCSS')
ax12.grid(True, alpha=0.3)
ax12.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal k={optimal_k}')
ax12.legend()

plt.tight_layout()
plt.savefig(f'{models_dir}/eda_visualizations.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved EDA visualizations to: {models_dir}/eda_visualizations.png")

# ============================================================================
# STEP 6: CLUSTER VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("STEP 6: CLUSTER VISUALIZATIONS")
print("="*80)

fig2, axes = plt.subplots(2, 2, figsize=(16, 12))

# RFM Score distribution
rfm['RFM_Score'].value_counts().head(15).sort_index().plot(kind='bar', ax=axes[0, 0], color='teal')
axes[0, 0].set_title('Top 15 RFM Score Combinations', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('RFM Score (R-F-M)')
axes[0, 0].set_ylabel('Number of Customers')

# Segment distribution
segment_counts = rfm['Segment'].value_counts()
colors = ['#2ecc71', '#3498db', '#f1c40f', '#e74c3c', '#95a5a6', '#34495e']
segment_counts.plot(kind='pie', ax=axes[0, 1], autopct='%1.1f%%', colors=colors, startangle=90)
axes[0, 1].set_title('Customer Segment Distribution', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('')

# PCA visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(rfm_scaled)
rfm['PCA1'] = pca_result[:, 0]
rfm['PCA2'] = pca_result[:, 1]

scatter = axes[1, 0].scatter(rfm['PCA1'], rfm['PCA2'], c=rfm['Cluster'], cmap='viridis', alpha=0.6)
axes[1, 0].set_title('Customer Clusters (PCA Visualization)', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Principal Component 1')
axes[1, 0].set_ylabel('Principal Component 2')
plt.colorbar(scatter, ax=axes[1, 0])

# Cluster profiles
cluster_profile_norm = (cluster_profile - cluster_profile.min()) / (cluster_profile.max() - cluster_profile.min())
cluster_profile_norm.plot(kind='bar', ax=axes[1, 1], width=0.8)
axes[1, 1].set_title('Customer Cluster Profiles', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Cluster')
axes[1, 1].set_ylabel('Normalized Value')
axes[1, 1].legend(['Recency (lower is better)', 'Frequency', 'Monetary'])
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{models_dir}/cluster_visualizations.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved cluster visualizations to: {models_dir}/cluster_visualizations.png")

# ============================================================================
# STEP 7: PRODUCT RECOMMENDATION SYSTEM
# ============================================================================

print("\n" + "="*80)
print("STEP 7: PRODUCT RECOMMENDATION SYSTEM")
print("="*80)

# Create user-item matrix
user_item_matrix = df_clean.pivot_table(
    index='CustomerID', 
    columns='StockCode', 
    values='Quantity',
    fill_value=0,
    aggfunc='sum'
)

print(f"User-Item Matrix: {user_item_matrix.shape[0]:,} customers x {user_item_matrix.shape[1]:,} products")

# Save user-item matrix
user_item_matrix.to_csv(f'{models_dir}/user_item_matrix.csv')
print(f"✅ Saved user-item matrix to: {models_dir}/user_item_matrix.csv")

# Calculate item similarity
item_similarity = cosine_similarity(user_item_matrix.T)
item_similarity_df = pd.DataFrame(
    item_similarity,
    index=user_item_matrix.columns,
    columns=user_item_matrix.columns
)

# Save similarity matrix
item_similarity_df.to_csv(f'{models_dir}/item_similarity_matrix.csv')
print(f"✅ Saved item similarity matrix to: {models_dir}/item_similarity_matrix.csv")

# Get top products for heatmap
top_products = df_clean.groupby('StockCode')['Quantity'].sum().sort_values(ascending=False).head(15).index

# Create product names
product_names = {}
for code in top_products:
    desc = df_clean[df_clean['StockCode'] == code]['Description'].iloc[0] if len(df_clean[df_clean['StockCode'] == code]) > 0 else f'Product_{code}'
    if len(desc) > 25:
        desc = desc[:22] + '...'
    product_names[code] = f"{desc} ({code})"

# Get similarity matrix for top products
top_similarity = item_similarity_df.loc[top_products, top_products]
renamed_index = [product_names[code] for code in top_similarity.index]
top_similarity.index = renamed_index
top_similarity.columns = renamed_index

# Plot heatmap
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

sns.heatmap(top_similarity, cmap='coolwarm', center=0, ax=ax1, square=True, 
            cbar_kws={'label': 'Similarity Score'})
ax1.set_title('Product Similarity Heatmap (Top 15 Products)', fontsize=14, fontweight='bold')
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)

# Get recommendations for sample customer
sample_customer = user_item_matrix.index[0]
purchased = user_item_matrix.loc[sample_customer]
purchased_products = purchased[purchased > 0].index.tolist()

if purchased_products:
    sim_scores = item_similarity_df.loc[purchased_products].mean(axis=0)
    recommendations = sim_scores.drop(purchased_products, errors='ignore').sort_values(ascending=False).head(10)
    
    # Plot recommendations
    rec_names = []
    rec_values = []
    for code in recommendations.index:
        desc = df_clean[df_clean['StockCode'] == code]['Description'].iloc[0] if len(df_clean[df_clean['StockCode'] == code]) > 0 else f'Product_{code}'
        if len(desc) > 35:
            desc = desc[:32] + '...'
        rec_names.append(desc)
        rec_values.append(recommendations[code])
    
    bars = ax2.barh(rec_names, rec_values, color='teal', alpha=0.7)
    ax2.set_title(f'Top Recommendations for Customer {sample_customer}', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Similarity Score')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, rec_values):
        ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center')

plt.tight_layout()
plt.savefig(f'{models_dir}/recommendations_visualization.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved recommendations visualization to: {models_dir}/recommendations_visualization.png")

# Save recommendations for top customers
print("\n📊 Generating recommendations for top customers...")
top_customers = rfm.nlargest(10, 'Monetary').index.tolist()
recommendations_dict = {}

for customer in top_customers:
    purchased = user_item_matrix.loc[customer]
    purchased_products = purchased[purchased > 0].index.tolist()
    
    if purchased_products:
        sim_scores = item_similarity_df.loc[purchased_products].mean(axis=0)
        recs = sim_scores.drop(purchased_products, errors='ignore').sort_values(ascending=False).head(5)
        recommendations_dict[customer] = recs.index.tolist()

# Save recommendations
rec_df = pd.DataFrame.from_dict(recommendations_dict, orient='index')
rec_df.columns = [f'Recommendation_{i+1}' for i in range(rec_df.shape[1])]
rec_df.to_csv(f'{models_dir}/top_customer_recommendations.csv')
print(f"✅ Saved top customer recommendations to: {models_dir}/top_customer_recommendations.csv")

# ============================================================================
# STEP 8: SAVE SUMMARY REPORT
# ============================================================================

print("\n" + "="*80)
print("STEP 8: GENERATING SUMMARY REPORT")
print("="*80)

# Create summary report
summary = {
    'Dataset Information': {
        'Total Transactions': len(df_clean),
        'Unique Customers': df_clean['CustomerID'].nunique(),
        'Unique Products': df_clean['StockCode'].nunique(),
        'Total Revenue': f"${df_clean['TotalAmount'].sum():,.2f}",
        'Date Range': f"{df_clean['InvoiceDate'].min()} to {df_clean['InvoiceDate'].max()}"
    },
    'RFM Statistics': {
        'Avg Recency': f"{rfm['Recency'].mean():.1f} days",
        'Avg Frequency': f"{rfm['Frequency'].mean():.1f} purchases",
        'Avg Monetary': f"${rfm['Monetary'].mean():.2f}"
    },
    'Customer Segments': rfm['Segment'].value_counts().to_dict(),
    'Cluster Information': {
        f'Cluster {i}': {
            'Size': len(rfm[rfm['Cluster'] == i]),
            'Recency': f"{rfm[rfm['Cluster'] == i]['Recency'].mean():.1f} days",
            'Frequency': f"{rfm[rfm['Cluster'] == i]['Frequency'].mean():.1f}",
            'Monetary': f"${rfm[rfm['Cluster'] == i]['Monetary'].mean():.2f}"
        } for i in range(optimal_k)
    }
}

# Save summary as CSV
summary_df = pd.DataFrame({
    'Metric': [],
    'Value': []
})

for category, values in summary.items():
    if isinstance(values, dict):
        for key, value in values.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    summary_df = pd.concat([summary_df, pd.DataFrame({
                        'Metric': [f"{category} - {key} - {sub_key}"],
                        'Value': [sub_value]
                    })], ignore_index=True)
            else:
                summary_df = pd.concat([summary_df, pd.DataFrame({
                    'Metric': [f"{category} - {key}"],
                    'Value': [value]
                })], ignore_index=True)

summary_df.to_csv(f'{models_dir}/summary_report.csv', index=False)
print(f"✅ Saved summary report to: {models_dir}/summary_report.csv")

# Save summary as text file
with open(f'{models_dir}/summary_report.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("ONLINE RETAIL ANALYSIS - SUMMARY REPORT\n")
    f.write("="*80 + "\n\n")
    
    for category, values in summary.items():
        f.write(f"\n{category}:\n")
        f.write("-"*40 + "\n")
        if isinstance(values, dict):
            for key, value in values.items():
                if isinstance(value, dict):
                    f.write(f"  {key}:\n")
                    for sub_key, sub_value in value.items():
                        f.write(f"    {sub_key}: {sub_value}\n")
                else:
                    f.write(f"  {key}: {value}\n")
        else:
            f.write(f"  {values}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("END OF REPORT\n")
    f.write("="*80 + "\n")

print(f"✅ Saved summary report to: {models_dir}/summary_report.txt")

# ============================================================================
# FINAL OUTPUT
# ============================================================================

print("\n" + "="*80)
print("✅ PROJECT COMPLETED SUCCESSFULLY!")
print("="*80)

print(f"\n📁 All outputs saved in '{models_dir}' folder:")
print("-"*40)

# List all files in models directory
files = os.listdir(models_dir)
for file in sorted(files):
    file_path = os.path.join(models_dir, file)
    size = os.path.getsize(file_path)
    if size < 1024:
        size_str = f"{size} bytes"
    elif size < 1024 * 1024:
        size_str = f"{size/1024:.1f} KB"
    else:
        size_str = f"{size/(1024*1024):.1f} MB"
    print(f"  📄 {file} ({size_str})")

print("\n" + "="*80)
print("📊 FILES SAVED IN 'models' FOLDER:")
print("="*80)
print("1. cleaned_retail_data.csv - Cleaned transaction data")
print("2. rfm_data.csv - RFM metrics for each customer")
print("3. rfm_with_segments.csv - RFM with customer segments")
print("4. kmeans_model.pkl - Trained K-Means model")
print("5. scaler.pkl - Standard scaler for new data")
print("6. cluster_profiles.csv - Cluster characteristics")
print("7. user_item_matrix.csv - Customer-product interaction matrix")
print("8. item_similarity_matrix.csv - Product similarity matrix")
print("9. top_customer_recommendations.csv - Recommendations for top customers")
print("10. summary_report.csv - Project summary in CSV format")
print("11. summary_report.txt - Project summary in text format")
print("12. eda_visualizations.png - EDA plots")
print("13. cluster_visualizations.png - Cluster analysis plots")
print("14. recommendations_visualization.png - Recommendation visualizations")
print("15. elbow_curve.png - Elbow method for optimal clusters")
print("="*80)


# ============================================================================
# STEP 4: CLUSTERING METHODOLOGY
# ============================================================================

print("\n" + "="*80)
print("STEP 4: CLUSTERING METHODOLOGY")
print("="*80)

# Import additional libraries for clustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 4.1: FEATURE ENGINEERING - RFM CALCULATION
# ============================================================================

print("\n" + "="*50)
print("4.1: FEATURE ENGINEERING - RFM CALCULATION")
print("="*50)

# Use the cleaned data from previous steps
# Calculate RFM metrics

# 1️⃣ Recency: Days since last purchase
snapshot_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df_clean.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',  # Unique invoice count
    'TotalAmount': 'sum'
}).rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalAmount': 'Monetary'
})

print(f"\n📊 RFM Data Shape: {rfm.shape}")
print(f"📊 RFM Statistics:")
print(rfm.describe())

# Save RFM data
rfm.to_csv(f'{models_dir}/rfm_calculation.csv')
print(f"\n✅ Saved RFM calculation to: {models_dir}/rfm_calculation.csv")

# ============================================================================
# 4.2: STANDARDIZE/NORMALIZE RFM VALUES
# ============================================================================

print("\n" + "="*50)
print("4.2: STANDARDIZE/NORMALIZE RFM VALUES")
print("="*50)

# Multiple scaling techniques for comparison
# StandardScaler (Z-score normalization)
scaler_standard = StandardScaler()
rfm_scaled_standard = scaler_standard.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# MinMaxScaler (0-1 normalization)
scaler_minmax = MinMaxScaler()
rfm_scaled_minmax = scaler_minmax.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# RobustScaler (for outliers)
from sklearn.preprocessing import RobustScaler
scaler_robust = RobustScaler()
rfm_scaled_robust = scaler_robust.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

print("\n📊 Scaling Methods Applied:")
print(f"  • StandardScaler: Mean ≈ 0, Std ≈ 1")
print(f"  • MinMaxScaler: Range [0, 1]")
print(f"  • RobustScaler: Median-based (robust to outliers)")

# Choose StandardScaler for K-Means (most commonly used)
rfm_scaled = rfm_scaled_standard
print(f"\n✅ Using StandardScaler for clustering")

# Save scaler for future use
with open(f'{models_dir}/rfm_scaler.pkl', 'wb') as file:
    pickle.dump(scaler_standard, file)
print(f"✅ Saved scaler to: {models_dir}/rfm_scaler.pkl")

# ============================================================================
# 4.3: CHOOSE CLUSTERING ALGORITHM
# ============================================================================

print("\n" + "="*50)
print("4.3: CLUSTERING ALGORITHMS COMPARISON")
print("="*50)

# Test different algorithms
algorithms = {
    'K-Means': KMeans(n_clusters=4, random_state=42, n_init=10),
    'Agglomerative': AgglomerativeClustering(n_clusters=4),
    'DBSCAN': DBSCAN(eps=0.5, min_samples=5)
}

results = {}
for name, algo in algorithms.items():
    try:
        if name == 'DBSCAN':
            labels = algo.fit_predict(rfm_scaled)
            # Check if DBSCAN found clusters
            if len(set(labels)) > 1 and -1 not in labels:
                score = silhouette_score(rfm_scaled, labels)
                results[name] = {'labels': labels, 'silhouette': score}
            else:
                # If DBSCAN fails, try with different parameters
                algo_dbscan = DBSCAN(eps=0.8, min_samples=3)
                labels = algo_dbscan.fit_predict(rfm_scaled)
                if len(set(labels)) > 1:
                    score = silhouette_score(rfm_scaled, labels)
                    results[name] = {'labels': labels, 'silhouette': score}
                else:
                    print(f"⚠️ DBSCAN: Only found {len(set(labels))} clusters")
        else:
            labels = algo.fit_predict(rfm_scaled)
            score = silhouette_score(rfm_scaled, labels)
            results[name] = {'labels': labels, 'silhouette': score}
    except Exception as e:
        print(f"⚠️ {name} failed: {str(e)}")

# Display results
print("\n📊 Algorithm Comparison Results:")
for name, result in results.items():
    print(f"  • {name}: Silhouette Score = {result['silhouette']:.4f}")

# Choose best algorithm
best_algo = max(results, key=lambda x: results[x]['silhouette'])
print(f"\n✅ Best Algorithm: {best_algo} (Silhouette: {results[best_algo]['silhouette']:.4f})")

# Use K-Means as default (most interpretable)
chosen_algo = 'K-Means'
print(f"\n✅ Using {chosen_algo} for final clustering (most interpretable for business)")

# ============================================================================
# 4.4: ELBOW METHOD & SILHOUETTE SCORE FOR OPTIMAL CLUSTERS
# ============================================================================

print("\n" + "="*50)
print("4.4: OPTIMAL CLUSTER DETERMINATION")
print("="*50)

# Calculate metrics for different k values
k_range = range(2, 11)
wcss = []
silhouette_scores = []
calinski_scores = []
davies_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(rfm_scaled)
    
    # WCSS (Within-Cluster Sum of Squares)
    wcss.append(kmeans.inertia_)
    
    # Silhouette Score
    sil_score = silhouette_score(rfm_scaled, labels)
    silhouette_scores.append(sil_score)
    
    # Calinski-Harabasz Score
    cal_score = calinski_harabasz_score(rfm_scaled, labels)
    calinski_scores.append(cal_score)
    
    # Davies-Bouldin Score
    dav_score = davies_bouldin_score(rfm_scaled, labels)
    davies_scores.append(dav_score)

# Create elbow and silhouette plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Elbow Curve (WCSS)
axes[0, 0].plot(k_range, wcss, 'bo-', linewidth=2, markersize=8)
axes[0, 0].set_title('Elbow Method - WCSS', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Number of Clusters (k)')
axes[0, 0].set_ylabel('WCSS (Within-Cluster Sum of Squares)')
axes[0, 0].grid(True, alpha=0.3)
# Mark the elbow point (k=4)
axes[0, 0].axvline(x=4, color='red', linestyle='--', label='Optimal k=4')
axes[0, 0].legend()

# 2. Silhouette Scores
axes[0, 1].plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
axes[0, 1].set_title('Silhouette Score Analysis', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Number of Clusters (k)')
axes[0, 1].set_ylabel('Silhouette Score')
axes[0, 1].grid(True, alpha=0.3)
# Mark the best silhouette score
best_k = k_range[np.argmax(silhouette_scores)]
axes[0, 1].axvline(x=best_k, color='red', linestyle='--', label=f'Best k={best_k}')
axes[0, 1].legend()
axes[0, 1].axhline(y=0.25, color='green', linestyle=':', alpha=0.5, label='Good threshold')

# 3. Calinski-Harabasz Score
axes[1, 0].plot(k_range, calinski_scores, 'go-', linewidth=2, markersize=8)
axes[1, 0].set_title('Calinski-Harabasz Score', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Number of Clusters (k)')
axes[1, 0].set_ylabel('Score')
axes[1, 0].grid(True, alpha=0.3)
# Mark the best score
best_cal_k = k_range[np.argmax(calinski_scores)]
axes[1, 0].axvline(x=best_cal_k, color='red', linestyle='--', label=f'Best k={best_cal_k}')
axes[1, 0].legend()

# 4. Davies-Bouldin Score (lower is better)
axes[1, 1].plot(k_range, davies_scores, 'mo-', linewidth=2, markersize=8)
axes[1, 1].set_title('Davies-Bouldin Score', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Number of Clusters (k)')
axes[1, 1].set_ylabel('Score (lower is better)')
axes[1, 1].grid(True, alpha=0.3)
# Mark the best score
best_db_k = k_range[np.argmin(davies_scores)]
axes[1, 1].axvline(x=best_db_k, color='red', linestyle='--', label=f'Best k={best_db_k}')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(f'{models_dir}/cluster_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved cluster metrics comparison to: {models_dir}/cluster_metrics_comparison.png")

# Print optimal cluster recommendations
print("\n📊 Optimal Cluster Recommendations:")
print(f"  • Elbow Method: k ≈ 4-5")
print(f"  • Silhouette Score: Best k = {best_k} (Score: {max(silhouette_scores):.4f})")
print(f"  • Calinski-Harabasz: Best k = {best_cal_k}")
print(f"  • Davies-Bouldin: Best k = {best_db_k} (Score: {min(davies_scores):.4f})")

# Choose optimal k (using silhouette score as primary)
optimal_k = best_k
print(f"\n✅ Selected Optimal K = {optimal_k} (based on Silhouette Score)")

# ============================================================================
# 4.5: RUN CLUSTERING
# ============================================================================

print("\n" + "="*50)
print("4.5: RUN CLUSTERING WITH OPTIMAL K")
print("="*50)

# Apply K-Means with optimal k
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
rfm['Cluster'] = kmeans_final.fit_predict(rfm_scaled)

# Calculate final metrics
final_silhouette = silhouette_score(rfm_scaled, rfm['Cluster'])
final_calinski = calinski_harabasz_score(rfm_scaled, rfm['Cluster'])
final_davies = davies_bouldin_score(rfm_scaled, rfm['Cluster'])

print(f"\n📊 Final Clustering Metrics:")
print(f"  • Silhouette Score: {final_silhouette:.4f}")
print(f"  • Calinski-Harabasz: {final_calinski:.4f}")
print(f"  • Davies-Bouldin: {final_davies:.4f}")

# Save the K-Means model
with open(f'{models_dir}/kmeans_final_model.pkl', 'wb') as file:
    pickle.dump(kmeans_final, file)
print(f"\n✅ Saved final K-Means model to: {models_dir}/kmeans_final_model.pkl")

# ============================================================================
# 4.6: LABEL CLUSTERS BY INTERPRETING RFM AVERAGES
# ============================================================================

print("\n" + "="*50)
print("4.6: CLUSTER LABELING AND INTERPRETATION")
print("="*50)

# Calculate cluster profiles
cluster_profile = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
cluster_count = rfm.groupby('Cluster').size()

# Normalize for comparison
cluster_profile_norm = (cluster_profile - cluster_profile.min()) / (cluster_profile.max() - cluster_profile.min())

# Function to label clusters
def label_cluster(row):
    r_score = row['Recency']  # Lower is better
    f_score = row['Frequency']  # Higher is better
    m_score = row['Monetary']  # Higher is better
    
    # High-Value: High Frequency, High Monetary, Low Recency
    if f_score > 0.7 and m_score > 0.7 and r_score < 0.3:
        return 'High-Value'
    # Regular: Medium Frequency, Medium Monetary
    elif f_score > 0.4 and f_score <= 0.7 and m_score > 0.4 and m_score <= 0.7:
        return 'Regular'
    # Occasional: Low Frequency, Low Monetary, High Recency (older)
    elif f_score <= 0.4 and m_score <= 0.4 and r_score > 0.5:
        return 'Occasional'
    # At-Risk: High Recency, Low Frequency, Low Monetary
    elif r_score > 0.7 and f_score <= 0.3 and m_score <= 0.3:
        return 'At-Risk'
    else:
        # Additional categories for remaining clusters
        if r_score < 0.3 and f_score > 0.5:
            return 'Loyal'
        elif r_score > 0.5:
            return 'Dormant'
        else:
            return 'Active'

# Apply labeling
rfm['Segment_Label'] = rfm.apply(lambda row: label_cluster(row), axis=1)

# Create cluster summary
cluster_summary = pd.DataFrame({
    'Cluster': range(optimal_k),
    'Size': cluster_count.values,
    'Percentage': (cluster_count.values / len(rfm) * 100).round(1),
    'Avg_Recency': cluster_profile['Recency'].round(1),
    'Avg_Frequency': cluster_profile['Frequency'].round(1),
    'Avg_Monetary': cluster_profile['Monetary'].round(2),
    'Segment_Label': rfm.groupby('Cluster')['Segment_Label'].first().values
})

print("\n📊 Cluster Summary:")
print(cluster_summary.to_string(index=False))

# Save cluster summary
cluster_summary.to_csv(f'{models_dir}/cluster_summary_with_labels.csv', index=False)
print(f"\n✅ Saved cluster summary to: {models_dir}/cluster_summary_with_labels.csv")

# Detailed cluster interpretation
print("\n" + "="*50)
print("📊 CLUSTER INTERPRETATION DETAILS")
print("="*50)

for idx, row in cluster_summary.iterrows():
    print(f"\n🔹 Cluster {row['Cluster']}: {row['Segment_Label']}")
    print(f"   • Size: {row['Size']:,} customers ({row['Percentage']:.1f}%)")
    print(f"   • Average Recency: {row['Avg_Recency']:.1f} days since last purchase")
    print(f"   • Average Frequency: {row['Avg_Frequency']:.1f} purchases")
    print(f"   • Average Monetary: ${row['Avg_Monetary']:.2f}")
    print(f"   • Characteristics: ", end="")
    if row['Segment_Label'] == 'High-Value':
        print("High-value customers who purchase frequently and recently")
    elif row['Segment_Label'] == 'Regular':
        print("Steady purchasers with moderate spending")
    elif row['Segment_Label'] == 'Occasional':
        print("Occasional buyers with low engagement")
    elif row['Segment_Label'] == 'At-Risk':
        print("At-risk customers who haven't purchased recently")
    elif row['Segment_Label'] == 'Loyal':
        print("Loyal customers with good engagement")
    elif row['Segment_Label'] == 'Dormant':
        print("Dormant customers needing re-engagement")
    else:
        print("Active customers with mixed behavior")

# ============================================================================
# 4.7: VISUALIZE CLUSTERS
# ============================================================================

print("\n" + "="*50)
print("4.7: CLUSTER VISUALIZATIONS")
print("="*50)

# 2D Scatter Plot (PCA)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# PCA for 2D visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(rfm_scaled)
rfm['PCA1'] = pca_result[:, 0]
rfm['PCA2'] = pca_result[:, 1]

# 1. PCA Scatter Plot with Cluster Labels
scatter = axes[0, 0].scatter(rfm['PCA1'], rfm['PCA2'], 
                            c=rfm['Cluster'], cmap='viridis', 
                            alpha=0.6, s=50)
axes[0, 0].set_title('Customer Clusters (PCA Visualization)', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[0, 0].set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.colorbar(scatter, ax=axes[0, 0])

# 2. PCA Scatter Plot with Segment Labels
segment_colors = {'High-Value': 'green', 'Regular': 'blue', 'Occasional': 'orange', 
                  'At-Risk': 'red', 'Loyal': 'purple', 'Dormant': 'brown', 'Active': 'gray'}
for segment in rfm['Segment_Label'].unique():
    mask = rfm['Segment_Label'] == segment
    color = segment_colors.get(segment, 'gray')
    axes[0, 1].scatter(rfm.loc[mask, 'PCA1'], rfm.loc[mask, 'PCA2'], 
                      label=segment, c=color, alpha=0.6, s=30)
axes[0, 1].set_title('Customer Segments (PCA Visualization)', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Principal Component 1')
axes[0, 1].set_ylabel('Principal Component 2')
axes[0, 1].legend(loc='best')

# 3. 3D Scatter Plot (projected to 2D for visualization)
from mpl_toolkits.mplot3d import Axes3D
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
axes[1, 0].legend(['Recency (lower is better)', 'Frequency', 'Monetary'])
axes[1, 0].grid(True, alpha=0.3)

# 5. Cluster Size Distribution
cluster_summary.plot(kind='bar', x='Cluster', y='Size', ax=axes[1, 1], color='teal', legend=False)
axes[1, 1].set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Cluster')
axes[1, 1].set_ylabel('Number of Customers')
axes[1, 1].grid(True, alpha=0.3)
# Add labels on bars
for i, v in enumerate(cluster_summary['Size']):
    axes[1, 1].text(i, v + 10, str(v), ha='center', fontsize=9)

# 6. Radar Chart for Cluster Profiles (simplified using bar chart)
metrics = ['Recency', 'Frequency', 'Monetary']
x = np.arange(len(metrics))
width = 0.2

for i in range(optimal_k):
    values = cluster_profile.iloc[i].values
    normalized_values = (values - values.min()) / (values.max() - values.min())
    axes[1, 2].bar(x + i*width, normalized_values, width, label=f'Cluster {i}')

axes[1, 2].set_title('Cluster Profiles Comparison', fontsize=14, fontweight='bold')
axes[1, 2].set_xlabel('RFM Metrics')
axes[1, 2].set_ylabel('Normalized Value')
axes[1, 2].set_xticks(x + width * (optimal_k-1) / 2)
axes[1, 2].set_xticklabels(metrics)
axes[1, 2].legend(loc='best')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{models_dir}/cluster_visualizations_complete.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved cluster visualizations to: {models_dir}/cluster_visualizations_complete.png")

# ============================================================================
# 4.8: SAVE BEST PERFORMING MODEL FOR STREAMLIT USAGE
# ============================================================================

print("\n" + "="*50)
print("4.8: SAVING MODEL FOR STREAMLIT DEPLOYMENT")
print("="*50)

# Create a complete model package for Streamlit
model_package = {
    'model': kmeans_final,
    'scaler': scaler_standard,
    'cluster_summary': cluster_summary,
    'cluster_profiles': cluster_profile,
    'rfm_data': rfm,
    'feature_names': ['Recency', 'Frequency', 'Monetary'],
    'n_clusters': optimal_k,
    'model_metrics': {
        'silhouette_score': final_silhouette,
        'calinski_harabasz': final_calinski,
        'davies_bouldin': final_davies
    },
    'segment_labels': {
        row['Cluster']: row['Segment_Label'] 
        for idx, row in cluster_summary.iterrows()
    }
}

# Save the complete model package
with open(f'{models_dir}/streamlit_model_package.pkl', 'wb') as file:
    pickle.dump(model_package, file)
print(f"✅ Saved Streamlit model package to: {models_dir}/streamlit_model_package.pkl")

# Save individual components for easy loading
# 1. Cluster labels mapping
cluster_labels = {row['Cluster']: row['Segment_Label'] for idx, row in cluster_summary.iterrows()}
with open(f'{models_dir}/cluster_labels.pkl', 'wb') as file:
    pickle.dump(cluster_labels, file)

# 2. Cluster centroids
with open(f'{models_dir}/cluster_centroids.pkl', 'wb') as file:
    pickle.dump(kmeans_final.cluster_centers_, file)

print(f"✅ Saved cluster labels to: {models_dir}/cluster_labels.pkl")
print(f"✅ Saved cluster centroids to: {models_dir}/cluster_centroids.pkl")
