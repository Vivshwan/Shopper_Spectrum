# ============================================================================
# STREAMLIT APP - SHOPPER SPECTRUM
# Customer Segmentation & Product Recommendation System
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Shopper Spectrum - Customer Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    
    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E3A5F;
        padding: 0.5rem 0;
        border-bottom: 3px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 1rem;
        opacity: 0.9;
    }
    .metric-card h2 {
        margin: 0.5rem 0;
        font-size: 2rem;
    }
    
    .segment-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.2rem;
        margin: 0.2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .product-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        transition: transform 0.3s;
    }
    .product-card:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .product-card .rank {
        font-size: 1.5rem;
        font-weight: bold;
        color: #667eea;
    }
    .product-card .name {
        font-size: 1.1rem;
        font-weight: bold;
        color: #1E3A5F;
    }
    .product-card .score {
        font-size: 0.9rem;
        color: #666;
    }
    
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        font-size: 1.1rem;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        border: none;
        transition: all 0.3s;
    }
    .stButton button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .info-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        border-top: 1px solid #e0e0e0;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS
# ============================================================================

@st.cache_resource
def load_models():
    """Load all saved models and data"""
    try:
        # Load segmentation model package
        with open('models/streamlit_model_package.pkl', 'rb') as file:
            segmentation_model = pickle.load(file)
        
        # Load cluster summary
        cluster_summary = pd.read_csv('models/cluster_summary_with_labels.csv')
        
        # Load the original data for product recommendations
        # Try to load cleaned data if available
        try:
            df_clean = pd.read_csv('models/cleaned_retail_data.csv')
        except:
            # If not available, try to load from original
            df_clean = pd.read_csv('online_retail.csv')
            # Basic preprocessing
            df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'], errors='coerce')
            df_clean = df_clean.dropna(subset=['CustomerID'])
            df_clean = df_clean[~df_clean['InvoiceNo'].astype(str).str.startswith('C')]
            df_clean = df_clean[df_clean['Quantity'] > 0]
            df_clean = df_clean[df_clean['UnitPrice'] > 0]
            df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['UnitPrice']
        
        # Create product descriptions lookup
        product_descriptions = df_clean[['StockCode', 'Description']].drop_duplicates(subset=['StockCode'])
        product_descriptions = product_descriptions.dropna(subset=['Description'])
        
        # Create user-item matrix for recommendations
        user_item_matrix = df_clean.pivot_table(
            index='CustomerID',
            columns='StockCode',
            values='Quantity',
            fill_value=0,
            aggfunc='sum'
        )
        
        # Calculate item similarity
        from sklearn.metrics.pairwise import cosine_similarity
        item_similarity = cosine_similarity(user_item_matrix.T)
        item_similarity_df = pd.DataFrame(
            item_similarity,
            index=user_item_matrix.columns,
            columns=user_item_matrix.columns
        )
        
        return {
            'segmentation': segmentation_model,
            'cluster_summary': cluster_summary,
            'product_descriptions': product_descriptions,
            'item_similarity': item_similarity_df,
            'df_clean': df_clean
        }
        
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}")
        st.info("Please run the analysis script first to generate the models.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_segment_color(segment):
    """Return color for each segment"""
    colors = {
        'High-Value': '#2ecc71',
        'Regular': '#3498db',
        'Occasional': '#f39c12',
        'At-Risk': '#e74c3c',
        'Loyal': '#9b59b6',
        'Dormant': '#95a5a6',
        'Active': '#1abc9c'
    }
    return colors.get(segment, '#95a5a6')

def get_segment_emoji(segment):
    """Return emoji for each segment"""
    emojis = {
        'High-Value': '👑',
        'Regular': '⭐',
        'Occasional': '🔄',
        'At-Risk': '⚠️',
        'Loyal': '💎',
        'Dormant': '💤',
        'Active': '✅'
    }
    return emojis.get(segment, '📌')

def get_segment_description(segment):
    """Return description for each segment"""
    descriptions = {
        'High-Value': 'Premium customers who purchase frequently and spend the most',
        'Regular': 'Steady purchasers with consistent buying behavior',
        'Occasional': 'Infrequent buyers who purchase occasionally',
        'At-Risk': 'Customers who haven\'t purchased recently - need re-engagement',
        'Loyal': 'Loyal customers with strong engagement',
        'Dormant': 'Inactive customers needing re-activation',
        'Active': 'Active customers with mixed behavior'
    }
    return descriptions.get(segment, 'Active customer with mixed behavior')

def get_product_recommendations(product_input, n_recommendations=5):
    """Get product recommendations by description or stock code"""
    if not product_input or models is None:
        return None
    
    try:
        # Search by description
        matches = models['product_descriptions'][
            models['product_descriptions']['Description'].str.contains(
                product_input, case=False, na=False
            )
        ]
        
        if len(matches) == 0:
            # Try as stock code
            matches = models['product_descriptions'][
                models['product_descriptions']['StockCode'].str.contains(
                    product_input, case=False, na=False
                )
            ]
        
        if len(matches) == 0:
            return None
        
        # Get the most popular product
        product_code = matches['StockCode'].iloc[0]
        
        # Get similarity scores
        if product_code not in models['item_similarity'].index:
            return None
        
        similarities = models['item_similarity'][product_code].sort_values(ascending=False)
        similarities = similarities.drop(product_code, errors='ignore')
        top_similar = similarities.head(n_recommendations)
        
        # Get product details
        recommendations = []
        for code, score in top_similar.items():
            desc = models['product_descriptions'][
                models['product_descriptions']['StockCode'] == code
            ]['Description'].iloc[0] if len(models['product_descriptions'][
                models['product_descriptions']['StockCode'] == code
            ]) > 0 else f'Product_{code}'
            
            # Get additional stats
            df = models['df_clean']
            total_sold = df[df['StockCode'] == code]['Quantity'].sum()
            unique_buyers = df[df['StockCode'] == code]['CustomerID'].nunique()
            
            recommendations.append({
                'StockCode': code,
                'Description': desc,
                'Similarity_Score': round(score, 4),
                'Total_Quantity_Sold': int(total_sold),
                'Unique_Buyers': int(unique_buyers)
            })
        
        return pd.DataFrame(recommendations)
        
    except Exception as e:
        st.error(f"Error getting recommendations: {e}")
        return None

def predict_customer_segment(recency, frequency, monetary):
    """Predict customer segment based on RFM values"""
    if models is None:
        return None
    
    try:
        segmentation = models['segmentation']
        scaler = segmentation['scaler']
        model = segmentation['model']
        segment_labels = segmentation['segment_labels']
        
        # Scale and predict
        scaled_data = scaler.transform([[recency, frequency, monetary]])
        cluster = model.predict(scaled_data)[0]
        segment = segment_labels.get(cluster, 'Unknown')
        
        # Get cluster statistics
        cluster_stats = models['cluster_summary']
        stats = cluster_stats[cluster_stats['Cluster'] == cluster]
        
        return {
            'cluster': int(cluster),
            'segment': segment,
            'avg_recency': float(stats['Avg_Recency'].values[0]) if len(stats) > 0 else None,
            'avg_frequency': float(stats['Avg_Frequency'].values[0]) if len(stats) > 0 else None,
            'avg_monetary': float(stats['Avg_Monetary'].values[0]) if len(stats) > 0 else None,
            'cluster_size': int(stats['Size'].values[0]) if len(stats) > 0 else None,
            'cluster_percentage': float(stats['Percentage'].values[0]) if len(stats) > 0 else None
        }
    except Exception as e:
        st.error(f"Error predicting segment: {e}")
        return None

# ============================================================================
# LOAD MODELS
# ============================================================================

models = load_models()

if models is None:
    st.stop()

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("# 🛍️ Shopper Spectrum")
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "📌 Navigate",
        ["🎯 Customer Segmentation", "🔍 Product Recommendations", "📊 Dashboard"],
        index=0
    )
    
    st.markdown("---")
    
    # System info
    st.markdown("### 📊 System Info")
    st.markdown(f"**Total Customers:** {models['segmentation']['rfm_data'].shape[0]:,}")
    st.markdown(f"**Clusters:** {models['segmentation']['n_clusters']}")
    
    st.markdown("---")
    
    # Quick stats
    st.markdown("### 📈 Quick Stats")
    for _, row in models['cluster_summary'].iterrows():
        color = get_segment_color(row['Segment_Label'])
        st.markdown(
            f"<span style='color:{color};font-weight:bold'>●</span> "
            f"{row['Segment_Label']}: {row['Size']:,} ({row['Percentage']:.1f}%)",
            unsafe_allow_html=True
        )

# ============================================================================
# PAGE: CUSTOMER SEGMENTATION
# ============================================================================

if page == "🎯 Customer Segmentation":
    st.markdown('<div class="main-header">🎯 Customer Segmentation</div>', unsafe_allow_html=True)
    st.markdown("Identify customer segments based on Recency, Frequency, and Monetary (RFM) values")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown('<div class="sub-header">📝 Enter Customer Details</div>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown("### 📅 Recency")
            recency = st.number_input(
                "Days since last purchase",
                min_value=0,
                max_value=730,
                value=30,
                step=1,
                help="Number of days since the customer's last purchase"
            )
            
            st.markdown("### 🔄 Frequency")
            frequency = st.number_input(
                "Number of purchases",
                min_value=0,
                max_value=100,
                value=5,
                step=1,
                help="Total number of purchases made by the customer"
            )
            
            st.markdown("### 💰 Monetary")
            monetary = st.number_input(
                "Total amount spent ($)",
                min_value=0,
                max_value=100000,
                value=500,
                step=50,
                help="Total amount spent by the customer"
            )
        
        predict_clicked = st.button("🔮 Predict Customer Segment", use_container_width=True)
        
        with st.expander("📖 Understanding RFM"):
            st.markdown("""
            **Recency** ⏰
            - How recently did the customer purchase?
            - Lower is better (recent purchasers)
            
            **Frequency** 🔄
            - How often does the customer purchase?
            - Higher is better (frequent purchasers)
            
            **Monetary** 💰
            - How much does the customer spend?
            - Higher is better (big spenders)
            """)
    
    with col2:
        if predict_clicked:
            with st.spinner("Analyzing customer data..."):
                result = predict_customer_segment(recency, frequency, monetary)
                
                if result:
                    segment = result['segment']
                    color = get_segment_color(segment)
                    emoji = get_segment_emoji(segment)
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {color}22 0%, {color}11 100%); 
                                padding: 2rem; border-radius: 15px; border: 2px solid {color};
                                margin: 1rem 0;">
                        <div style="text-align: center;">
                            <div style="font-size: 4rem;">{emoji}</div>
                            <div style="font-size: 2.5rem; font-weight: bold; color: {color};">
                                {segment}
                            </div>
                            <div style="font-size: 1rem; color: #666; margin-top: 0.5rem;">
                                {get_segment_description(segment)}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric(
                            "👥 Cluster Size",
                            f"{result['cluster_size']:,}",
                            f"{result['cluster_percentage']:.1f}% of customers"
                        )
                    with col_b:
                        st.metric(
                            "📊 Cluster ID",
                            f"#{result['cluster']}"
                        )
                    with col_c:
                        avg_spend = result['avg_monetary']
                        st.metric(
                            "💰 Avg. Spend in Cluster",
                            f"${avg_spend:,.2f}" if avg_spend else "N/A"
                        )
                    
                    # Display cluster comparison
                    st.markdown("---")
                    st.markdown("### 📊 How This Customer Compares")
                    
                    # Create comparison chart
                    fig = go.Figure()
                    
                    customer_values = [recency, frequency, monetary]
                    cluster_avg = [
                        result['avg_recency'] if result['avg_recency'] else 0,
                        result['avg_frequency'] if result['avg_frequency'] else 0,
                        result['avg_monetary'] if result['avg_monetary'] else 0
                    ]
                    
                    # Normalize
                    max_values = [365, 100, 10000]
                    customer_norm = [v/max_values[i] for i, v in enumerate(customer_values)]
                    cluster_norm = [v/max_values[i] for i, v in enumerate(cluster_avg)]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=customer_norm,
                        theta=['Recency', 'Frequency', 'Monetary'],
                        fill='toself',
                        name='Customer',
                        line_color='#667eea',
                        fillcolor='rgba(102, 126, 234, 0.3)'
                    ))
                    
                    fig.add_trace(go.Scatterpolar(
                        r=cluster_norm,
                        theta=['Recency', 'Frequency', 'Monetary'],
                        fill='toself',
                        name=f'Cluster {segment} Avg',
                        line_color='#e74c3c',
                        fillcolor='rgba(231, 76, 60, 0.2)'
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )
                        ),
                        showlegend=True,
                        title="Customer vs Cluster Average",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("❌ Could not predict segment. Please check your input values.")

# ============================================================================
# PAGE: PRODUCT RECOMMENDATIONS
# ============================================================================

elif page == "🔍 Product Recommendations":
    st.markdown('<div class="main-header">🔍 Product Recommendations</div>', unsafe_allow_html=True)
    st.markdown("Find similar products based on collaborative filtering")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="sub-header">🔎 Search for a Product</div>', unsafe_allow_html=True)
        
        product_input = st.text_input(
            "Enter product name or code",
            placeholder="e.g., 'WHITE HANGING HEART T-LIGHT HOLDER' or '85123A'",
            help="Type a product name or StockCode to find similar products"
        )
        
        n_recs = st.slider(
            "Number of recommendations",
            min_value=1,
            max_value=10,
            value=5,
            step=1
        )
        
        search_clicked = st.button("🔍 Find Similar Products", use_container_width=True)
        
        if product_input and not search_clicked:
            matches = models['product_descriptions'][
                models['product_descriptions']['Description'].str.contains(product_input, case=False, na=False)
            ].head(10)
            
            if len(matches) > 0:
                with st.expander("💡 Did you mean:"):
                    for _, row in matches.iterrows():
                        st.markdown(f"- {row['Description']} (Code: {row['StockCode']})")
    
    with col2:
        if search_clicked and product_input:
            with st.spinner("Finding similar products..."):
                recommendations = get_product_recommendations(product_input, n_recs)
                
                if recommendations is not None and len(recommendations) > 0:
                    original = models['product_descriptions'][
                        models['product_descriptions']['Description'].str.contains(product_input, case=False, na=False)
                    ].iloc[0] if len(models['product_descriptions'][
                        models['product_descriptions']['Description'].str.contains(product_input, case=False, na=False)
                    ]) > 0 else None
                    
                    if original is not None:
                        st.markdown(f"""
                        <div style="background: #f0f4ff; padding: 1rem; border-radius: 10px; border: 2px solid #667eea; margin-bottom: 1rem;">
                            <div style="font-size: 0.9rem; color: #666;">🔍 You searched for:</div>
                            <div style="font-size: 1.2rem; font-weight: bold; color: #1E3A5F;">
                                {original['Description']}
                            </div>
                            <div style="font-size: 0.9rem; color: #666;">Stock Code: {original['StockCode']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("### 🎯 Recommended Products")
                    
                    for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
                        score = row['Similarity_Score']
                        similarity_pct = score * 100
                        
                        st.markdown(f"""
                        <div class="product-card">
                            <div style="display: flex; align-items: center; gap: 1rem;">
                                <div class="rank">#{idx}</div>
                                <div style="flex: 1;">
                                    <div class="name">{row['Description']}</div>
                                    <div style="display: flex; gap: 1rem; margin-top: 0.3rem;">
                                        <span style="color: #666; font-size: 0.85rem;">
                                            📦 {row['Total_Quantity_Sold']:,} sold
                                        </span>
                                        <span style="color: #666; font-size: 0.85rem;">
                                            👤 {row['Unique_Buyers']:,} buyers
                                        </span>
                                    </div>
                                </div>
                                <div style="text-align: right;">
                                    <div style="font-size: 1.2rem; font-weight: bold; color: #667eea;">
                                        {similarity_pct:.1f}%
                                    </div>
                                    <div style="font-size: 0.8rem; color: #666;">similar</div>
                                </div>
                            </div>
                            <div style="margin-top: 0.5rem; background: #e8edf5; border-radius: 5px; height: 6px;">
                                <div style="background: linear-gradient(90deg, #667eea, #764ba2); width: {similarity_pct}%; height: 100%; border-radius: 5px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with st.expander("📊 View All Recommendations as Table"):
                        st.dataframe(
                            recommendations,
                            use_container_width=True,
                            hide_index=True
                        )
                else:
                    st.warning("❌ No similar products found. Please try a different product name or code.")
        elif search_clicked and not product_input:
            st.warning("⚠️ Please enter a product name or code.")

# ============================================================================
# PAGE: DASHBOARD
# ============================================================================

else:
    st.markdown('<div class="main-header">📊 Dashboard</div>', unsafe_allow_html=True)
    st.markdown("Overview of customer segments and system statistics")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>👥 Total Customers</h3>
            <h2>{models['segmentation']['rfm_data'].shape[0]:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_revenue = models['segmentation']['rfm_data']['Monetary'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <h3>💰 Total Revenue</h3>
            <h2>${total_revenue:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_spend = models['segmentation']['rfm_data']['Monetary'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>💵 Avg. Customer Spend</h3>
            <h2>${avg_spend:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_products = len(models['product_descriptions'])
        st.markdown(f"""
        <div class="metric-card">
            <h3>📦 Total Products</h3>
            <h2>{total_products:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Segment Distribution")
        
        fig = px.pie(
            models['cluster_summary'],
            values='Size',
            names='Segment_Label',
            title='Customer Segment Distribution',
            color='Segment_Label',
            color_discrete_map={
                'High-Value': '#2ecc71',
                'Regular': '#3498db',
                'Occasional': '#f39c12',
                'At-Risk': '#e74c3c',
                'Loyal': '#9b59b6',
                'Dormant': '#95a5a6',
                'Active': '#1abc9c'
            }
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Segment Details")
        
        display_df = models['cluster_summary'].copy()
        display_df['Percentage'] = display_df['Percentage'].apply(lambda x: f"{x:.1f}%")
        display_df['Avg_Recency'] = display_df['Avg_Recency'].apply(lambda x: f"{x:.1f} days")
        display_df['Avg_Frequency'] = display_df['Avg_Frequency'].apply(lambda x: f"{x:.1f}")
        display_df['Avg_Monetary'] = display_df['Avg_Monetary'].apply(lambda x: f"${x:,.2f}")
        
        st.dataframe(
            display_df[['Segment_Label', 'Size', 'Percentage', 'Avg_Recency', 'Avg_Frequency', 'Avg_Monetary']],
            use_container_width=True,
            hide_index=True,
            column_config={
                'Segment_Label': 'Segment',
                'Size': 'Customers',
                'Percentage': '%',
                'Avg_Recency': 'Avg Recency',
                'Avg_Frequency': 'Avg Frequency',
                'Avg_Monetary': 'Avg Spend'
            }
        )
    
    st.markdown("---")
    
    st.markdown("### 📈 RFM Distributions")
    
    col1, col2, col3 = st.columns(3)
    
    rfm_data = models['segmentation']['rfm_data']
    
    with col1:
        fig = px.histogram(
            rfm_data,
            x='Recency',
            nbins=30,
            title='Recency Distribution',
            color_discrete_sequence=['#667eea']
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(
            rfm_data,
            x='Frequency',
            nbins=30,
            title='Frequency Distribution',
            color_discrete_sequence=['#2ecc71']
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = px.histogram(
            rfm_data,
            x='Monetary',
            nbins=30,
            title='Monetary Distribution',
            color_discrete_sequence=['#e74c3c']
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("""
<div class="footer">
    <p>🛍️ Shopper Spectrum - Customer Intelligence Platform</p>
    <p style="font-size: 0.8rem; color: #999;">
        Powered by Machine Learning | RFM Analysis | Collaborative Filtering
    </p>
</div>
""", unsafe_allow_html=True)