"""
Affordability Page — PriceMyPlace Mumbai
Amenity access vs price analysis.
"""

import sys, os, io
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except: pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Affordability | PriceMyPlace", page_icon="📊", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


from app.utils import (
    get_project_root,
    load_processed_data,
    format_inr,
)

@st.cache_data
def load_data():
    return load_processed_data()


def main():
    st.markdown("# 📊 Affordability Analysis")
    st.markdown("*How much neighborhood value does each locality offer relative to its price?*")
    st.markdown("---")

    df = load_data()
    if df is None:
        st.error("No data available.")
        return

    if "affordability_index" not in df.columns:
        st.warning("Affordability Index not computed. Run the pipeline first.")
        # Still show amenity analysis
        pass

    # Formula explanation
    with st.expander("📐 How is the Affordability Index calculated?"):
        st.markdown("""
        **Affordability Index = Amenity Access Composite / Price per sqft**

        The amenity access composite combines:
        - Proximity to hospitals, schools, metro stations
        - Proximity to bus stops, parks, malls
        - Inverse crime rate (lower = better)
        - Inverse AQI (lower = better)

        Normalized to a **0-100 scale** across the dataset.

        **A higher score** means more amenity access per rupee spent — not just the cheapest
        locality, but the one offering the best value.
        """)

    # Top localities by affordability
    if "affordability_index" in df.columns:
        st.subheader("🏆 Most Affordable Localities (Value for Money)")

        loc_afford = df.groupby("Locality").agg({
            "affordability_index": "mean",
            "Price": ["median", "count"],
        })
        loc_afford.columns = ["Avg Affordability", "Median Price", "Listings"]
        loc_afford = loc_afford[loc_afford["Listings"] >= 10]  # Min 10 listings
        loc_afford = loc_afford.sort_values("Avg Affordability", ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Top 15 — Best Value**")
            top = loc_afford.head(15).reset_index()
            fig = px.bar(
                top, x="Avg Affordability", y="Locality", orientation="h",
                color="Avg Affordability", color_continuous_scale="Greens",
                text="Avg Affordability",
            )
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig.update_layout(template="plotly_white", showlegend=False,
                              coloraxis_showscale=False, font=dict(family="Inter"),
                              margin=dict(l=20, r=60, t=30, b=20), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Bottom 15 — Least Affordable**")
            bottom = loc_afford.tail(15).reset_index()
            fig = px.bar(
                bottom.sort_values("Avg Affordability", ascending=True),
                x="Avg Affordability", y="Locality", orientation="h",
                color="Avg Affordability", color_continuous_scale="Reds_r",
                text="Avg Affordability",
            )
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig.update_layout(template="plotly_white", showlegend=False,
                              coloraxis_showscale=False, font=dict(family="Inter"),
                              margin=dict(l=20, r=60, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

        # Affordability vs Price scatter
        st.subheader("💰 Affordability vs Median Price")
        fig = px.scatter(
            loc_afford.reset_index(),
            x="Median Price", y="Avg Affordability",
            size="Listings", color="Avg Affordability",
            hover_name="Locality",
            color_continuous_scale="Viridis",
            size_max=30,
        )
        fig.update_layout(template="plotly_white", font=dict(family="Inter"),
                          margin=dict(l=20, r=20, t=30, b=20),
                          xaxis_title="Median Property Price (Rs.)",
                          yaxis_title="Affordability Index")
        st.plotly_chart(fig, use_container_width=True)

    # Amenity Distance Analysis
    st.subheader("🏥 Amenity Distance by Locality")
    amenity_cols = [c for c in df.columns if c.startswith("Nearest_") and c.endswith("_km")]

    if amenity_cols:
        selected_amenity = st.selectbox("Select Amenity", amenity_cols,
                                         format_func=lambda x: x.replace("Nearest_", "").replace("_km", ""))

        loc_amenity = df.groupby("Locality")[selected_amenity].median().sort_values()

        fig = px.bar(
            x=loc_amenity.values, y=loc_amenity.index,
            orientation="h",
            color=loc_amenity.values,
            color_continuous_scale="RdYlGn_r",
            labels={"x": f"Median Distance (km)", "y": "Locality"},
        )
        fig.update_layout(template="plotly_white", showlegend=False,
                          coloraxis_showscale=False, font=dict(family="Inter"),
                          margin=dict(l=20, r=20, t=30, b=20),
                          height=max(400, len(loc_amenity) * 20))
        st.plotly_chart(fig, use_container_width=True)

    # Price vs Amenity Access Scatter
    if "Amenity_Access_Score" in df.columns:
        st.subheader("🎯 Amenity Access Score vs Price")
        fig = px.scatter(
            df.sample(min(1000, len(df)), random_state=42),
            x="Amenity_Access_Score", y="Price",
            color="Locality", hover_data=["Bedrooms", "Area"],
            opacity=0.6,
        )
        fig.update_layout(template="plotly_white", font=dict(family="Inter"),
                          margin=dict(l=20, r=20, t=30, b=20),
                          xaxis_title="Amenity Access Score",
                          yaxis_title="Property Price (Rs.)")
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
