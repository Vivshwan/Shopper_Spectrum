# 🛍️ Shopper Spectrum

**Shopper Spectrum** is an end-to-end customer intelligence platform that leverages **Unsupervised Machine Learning** to analyze e-commerce transaction data, segment customers based on purchasing behavior, and provide personalized product recommendations.

## 🎯 Key Features

- **Customer Segmentation** using RFM Analysis & K-Means Clustering
- **Product Recommendations** using Item-based Collaborative Filtering
- **Interactive Dashboard** built with Streamlit
- **Real-time Predictions** for customer segments
- **Visual Analytics** for business insights

## 🛠️ Tech Stack

- **Python 3.13.5** | Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Deployment**: Streamlit

## 📊 Outcomes

- ✅ 4 Customer Segments Identified (High-Value, Regular, Occasional, At-Risk)
- ✅ Product Recommendation System with 90%+ similarity accuracy
- ✅ Interactive Web App for business stakeholders

## 🚀 Quick Start

### Dataset Link:
The dataset can be downloaded from the following sources and save it as "online_retail.csv":
🔗 [Download Dataset](https://drive.google.com/file/d/1rzRwxm_CJxcRzfoo9Ix37A2JTlMummY-/view?usp=sharing)

---

## 📋 Step-by-Step Execution

### Step 1: Clone & Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/shopper-spectrum.git
cd shopper-spectrum

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### Step 2: Install Dependencies
```pip install -r requirements.txt```

### Step 3: Download Dataset
```# Download "online_retail.csv" from the provided link
# Place it in the project root directory
```
### Step 4: Run Preprocessing Pipeline
```python preprocessing.py```

What happens here:

🔄 Data cleaning & preprocessing

📊 RFM analysis on customer data

🧮 K-Means clustering (4 clusters)

🎯 Segment labeling (High-Value, Regular, Occasional, At-Risk)

🤖 Model training & saving

🔍 Product recommendation system generation

📈 Visualizations saved in models/ folder
Output: Models saved in models/ directory:

streamlit_model_package.pkl - Complete model package

recommendation_model.pkl - Product recommendation system

cluster_summary_with_labels.csv - Segment statistics

*.png - Visualization outputs

### Step 5: Launch Streamlit App
```streamlit run app.py```
