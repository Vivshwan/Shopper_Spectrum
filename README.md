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
https://drive.google.com/file/d/1rzRwxm_CJxcRzfoo9Ix37A2JTlMummY-/view?usp=sharing

```bash
# Install dependencies
pip install -r requirements.txt

# Run analysis pipeline
python preprocessing.py

# Launch Streamlit app
streamlit run app.py
