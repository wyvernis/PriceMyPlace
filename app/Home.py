"""
PriceMyPlace Mumbai - Home Page
AI-Powered Property Price Intelligence Platform
"""

import sys
import os
import io

# Fix Windows encoding
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Page config
st.set_page_config(
    page_title="PriceMyPlace Mumbai",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }

    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 70%);
        animation: shimmer 8s ease-in-out infinite;
    }

    @keyframes shimmer {
        0%, 100% { transform: translateX(-30%) translateY(-30%); }
        50% { transform: translateX(10%) translateY(10%); }
    }

    .main-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        position: relative;
        z-index: 1;
        letter-spacing: -0.5px;
    }

    .main-header p {
        color: rgba(255, 255, 255, 0.75);
        font-size: 1.05rem;
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
        font-weight: 300;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9ff 100%);
        border: 1px solid rgba(99, 102, 241, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.15);
        border-color: rgba(99, 102, 241, 0.3);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }

    .metric-label {
        color: #64748b;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    .metric-sublabel {
        color: #94a3b8;
        font-size: 0.75rem;
        margin-top: 0.25rem;
    }

    /* Section cards */
    .section-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
    }

    /* Feature cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }

    .feature-card {
        background: linear-gradient(145deg, #f8fafc, #ffffff);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        transition: all 0.2s ease;
    }

    .feature-card:hover {
        border-color: #6366f1;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.1);
    }

    .feature-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }

    .feature-title {
        font-weight: 600;
        color: #1e293b;
        font-size: 0.95rem;
    }

    .feature-desc {
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 0.25rem;
        line-height: 1.4;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    }

    section[data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }

    /* Info/warning boxes */
    .disclaimer-box {
        background: linear-gradient(135deg, #fef3c7, #fde68a);
        border: 1px solid #f59e0b;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        font-size: 0.8rem;
        color: #92400e;
        line-height: 1.5;
    }

    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


from app.utils import (
    get_project_root,
    load_processed_data,
    load_model_artifacts,
    format_inr,
)

@st.cache_data
def load_data():
    """Load the processed dataset."""
    return load_processed_data()


@st.cache_data
def load_metadata():
    """Load model metadata."""
    _, _, _, meta = load_model_artifacts()
    return meta


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏠 PriceMyPlace Mumbai</h1>
        <p>AI-Powered Property Price Intelligence for Mumbai & Navi Mumbai</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    df = load_data()
    metadata = load_metadata()

    if df is None:
        st.error("⚠️ No data found. Please run the pipeline first: `python run_pipeline.py`")
        st.code("cd pricemyplace-mumbai\npython run_pipeline.py", language="bash")
        return

    # Key metrics row
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Listings</div>
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-sublabel">Properties analyzed</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        median_price = df["Price"].median()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Median Price</div>
            <div class="metric-value">{format_inr(median_price)}</div>
            <div class="metric-sublabel">Property value</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        if "Price_Per_Sqft" in df.columns:
            median_ppsf = df["Price_Per_Sqft"].median()
        elif "price_per_sqft" in df.columns:
            median_ppsf = df["price_per_sqft"].median()
        else:
            median_ppsf = (df["Price"] / df["Area"]).median()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Median Rs./sqft</div>
            <div class="metric-value">Rs.{median_ppsf:,.0f}</div>
            <div class="metric-sublabel">Price per sqft</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        median_area = df["Area"].median()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Median Area</div>
            <div class="metric-value">{median_area:,.0f}</div>
            <div class="metric-sublabel">Square feet</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        if "fair_price_score" in df.columns:
            median_fps = df["fair_price_score"].median()
        else:
            median_fps = 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Fair Price Score</div>
            <div class="metric-value">{median_fps:.0f}</div>
            <div class="metric-sublabel">Median score</div>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        if "affordability_index" in df.columns:
            median_aff = df["affordability_index"].median()
        else:
            median_aff = 50
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Affordability</div>
            <div class="metric-value">{median_aff:.0f}</div>
            <div class="metric-sublabel">Median index</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Two-column layout for charts
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📊 Price Distribution")
        import plotly.express as px

        fig = px.histogram(
            df, x="Price", nbins=50,
            color_discrete_sequence=["#6366f1"],
            labels={"Price": "Property Price (Rs.)"},
        )
        fig.update_layout(
            template="plotly_white",
            showlegend=False,
            margin=dict(l=20, r=20, t=30, b=20),
            font=dict(family="Inter"),
            xaxis_title="Price (Rs.)",
            yaxis_title="Count",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("🏘️ Top 15 Localities by Median Price")
        locality_stats = df.groupby("Locality").agg(
            median_price=("Price", "median"),
            count=("Price", "count"),
        ).sort_values("median_price", ascending=True).tail(15)

        fig = px.bar(
            locality_stats.reset_index(),
            x="median_price", y="Locality",
            orientation="h",
            color="median_price",
            color_continuous_scale="Viridis",
            labels={"median_price": "Median Price (Rs.)", "Locality": ""},
        )
        fig.update_layout(
            template="plotly_white",
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=30, b=20),
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Property Type & Furnishing Distribution
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("🏗️ Property Types")
        pt_counts = df["Property_Type"].value_counts()
        fig = px.pie(
            values=pt_counts.values, names=pt_counts.index,
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.4,
        )
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=20, r=20, t=30, b=20),
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("🛋️ Furnishing Status")
        furn_price = df.groupby("Furnishing")["Price"].median().sort_values()
        fig = px.bar(
            x=furn_price.index, y=furn_price.values,
            color=furn_price.index,
            color_discrete_sequence=["#6366f1", "#8b5cf6", "#a78bfa"],
            labels={"x": "Furnishing", "y": "Median Price (Rs.)"},
        )
        fig.update_layout(
            template="plotly_white",
            showlegend=False,
            margin=dict(l=20, r=20, t=30, b=20),
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # BHK vs Price
    st.subheader("🛏️ BHK vs Price")
    fig = px.box(
        df, x="Bedrooms", y="Price",
        color_discrete_sequence=["#6366f1"],
        labels={"Bedrooms": "BHK", "Price": "Price (Rs.)"},
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=20),
        font=dict(family="Inter"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Model performance (if trained)
    if metadata:
        st.subheader("🤖 Model Performance")
        test = metadata.get("test_metrics", {})
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Best Model", metadata.get("best_model", "N/A"))
        with col_b:
            st.metric("Test R²", f"{test.get('r2', 0):.4f}")
        with col_c:
            st.metric("Test MAE", format_inr(test.get("mae", 0)))
        with col_d:
            st.metric("Test MAPE", f"{test.get('mape', 0):.1f}%")

    # Features overview
    st.markdown("---")
    st.subheader("🔍 Platform Features")
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">💰</div>
            <div class="feature-title">Price Prediction</div>
            <div class="feature-desc">ML-powered price estimation with explainable AI insights</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">⚖️</div>
            <div class="feature-title">Fair Price Score</div>
            <div class="feature-desc">Compare listing prices against model-estimated fair values</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🏘️</div>
            <div class="feature-title">Locality Explorer</div>
            <div class="feature-desc">Compare Mumbai & Navi Mumbai localities side-by-side</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🗺️</div>
            <div class="feature-title">Interactive Map</div>
            <div class="feature-desc">Geospatial visualization with price heatmaps</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Affordability Index</div>
            <div class="feature-desc">Amenity access vs price analysis across neighborhoods</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🏗️</div>
            <div class="feature-title">Infrastructure Explorer</div>
            <div class="feature-desc">Infrastructure proximity & exploratory investment signals</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Disclaimer:</strong> All predictions are model estimates based on historical data.
        The Fair Price Score, Affordability Index, and Investment Opportunity Score are informational
        and educational outputs — not financial or investment advice. Users should verify current
        information independently and consult qualified professionals for actual purchasing decisions.
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 🏠 PriceMyPlace")
        st.markdown("**Mumbai & Navi Mumbai**")
        st.markdown("---")
        st.markdown("#### Navigation")
        st.markdown("""
        - 🏠 **Home** — Overview & Key Metrics
        - 💰 **Price Prediction** — Predict Property Prices
        - 🏘️ **Locality Explorer** — Compare Localities
        - 🗺️ **Map** — Interactive Property Map
        - 📊 **Affordability** — Amenity vs Price Analysis
        - 🏗️ **Investment Explorer** — Infrastructure Signals
        - 🤖 **Model Insights** — ML Performance & SHAP
        - 📖 **Documentation** — Methodology & Guide
        """)
        st.markdown("---")
        if df is not None:
            st.markdown(f"**Data:** {len(df):,} properties")
            st.markdown(f"**Localities:** {df['Locality'].nunique()}")
            if metadata:
                st.markdown(f"**Model:** {metadata.get('best_model', 'N/A')}")


if __name__ == "__main__":
    main()
