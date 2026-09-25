"""
Locality Explorer Page — PriceMyPlace Mumbai
Compare localities across Mumbai & Navi Mumbai.
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

st.set_page_config(page_title="Locality Explorer | PriceMyPlace", page_icon="🏘️", layout="wide")

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


def compute_locality_stats(df):
    """Compute comprehensive stats per locality."""
    agg_dict = {
        "Price": ["median", "mean", "count", "std"],
        "Area": ["median", "mean"],
        "Bedrooms": ["median"],
    }

    if "fair_price_score" in df.columns:
        agg_dict["fair_price_score"] = ["mean"]
    if "affordability_index" in df.columns:
        agg_dict["affordability_index"] = ["mean"]
    if "investment_score" in df.columns:
        agg_dict["investment_score"] = ["mean"]
    if "predicted_price" in df.columns:
        agg_dict["predicted_price"] = ["median"]

    stats = df.groupby("Locality").agg(agg_dict)
    stats.columns = ["_".join(col).strip("_") for col in stats.columns]
    stats = stats.rename(columns={
        "Price_median": "Median Price",
        "Price_mean": "Mean Price",
        "Price_count": "Listings",
        "Price_std": "Price Std",
        "Area_median": "Median Area",
        "Area_mean": "Mean Area",
        "Bedrooms_median": "Median BHK",
    })

    if "predicted_price_median" in stats.columns:
        stats = stats.rename(columns={"predicted_price_median": "Predicted Median"})
    if "fair_price_score_mean" in stats.columns:
        stats = stats.rename(columns={"fair_price_score_mean": "Avg Fair Score"})
    if "affordability_index_mean" in stats.columns:
        stats = stats.rename(columns={"affordability_index_mean": "Avg Affordability"})
    if "investment_score_mean" in stats.columns:
        stats = stats.rename(columns={"investment_score_mean": "Avg Investment"})

    # Price per sqft
    stats["Median Rs/sqft"] = stats["Median Price"] / stats["Median Area"]

    return stats.sort_values("Median Price", ascending=False)


def main():
    st.markdown("# 🏘️ Locality Explorer")
    st.markdown("*Compare Mumbai & Navi Mumbai localities — prices, affordability, and infrastructure*")
    st.markdown("---")

    df = load_data()
    if df is None:
        st.error("No data available. Run the pipeline first.")
        return

    locality_stats = compute_locality_stats(df)

    # Locality selector
    all_localities = sorted(df["Locality"].unique())
    selected = st.multiselect(
        "Select localities to compare",
        all_localities,
        default=all_localities[:5]
    )

    if not selected:
        st.info("Please select at least one locality to compare.")
        return

    filtered_stats = locality_stats.loc[locality_stats.index.isin(selected)]
    filtered_df = df[df["Locality"].isin(selected)]

    # Summary table
    st.subheader("📊 Locality Comparison Table")
    display_cols = [c for c in ["Listings", "Median Price", "Median Rs/sqft", "Median Area",
                                 "Median BHK", "Avg Fair Score", "Avg Affordability", "Avg Investment"]
                    if c in filtered_stats.columns]

    formatted_stats = filtered_stats[display_cols].copy()
    if "Median Price" in formatted_stats.columns:
        formatted_stats["Median Price"] = formatted_stats["Median Price"].apply(format_inr)
    if "Median Rs/sqft" in formatted_stats.columns:
        formatted_stats["Median Rs/sqft"] = formatted_stats["Median Rs/sqft"].apply(lambda x: f"Rs.{x:,.0f}" if pd.notna(x) else "N/A")

    st.dataframe(formatted_stats, use_container_width=True)

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💰 Median Price by Locality")
        fig = px.bar(
            filtered_stats.reset_index().sort_values("Median Price", ascending=True),
            x="Median Price", y="Locality", orientation="h",
            color="Median Price", color_continuous_scale="Viridis",
        )
        fig.update_layout(template="plotly_white", showlegend=False,
                          coloraxis_showscale=False, font=dict(family="Inter"),
                          margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📐 Price per Sqft by Locality")
        fig = px.bar(
            filtered_stats.reset_index().sort_values("Median Rs/sqft", ascending=True),
            x="Median Rs/sqft", y="Locality", orientation="h",
            color="Median Rs/sqft", color_continuous_scale="Plasma",
        )
        fig.update_layout(template="plotly_white", showlegend=False,
                          coloraxis_showscale=False, font=dict(family="Inter"),
                          margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Rent distribution per locality
    st.subheader("📈 Price Distribution by Locality")
    fig = px.box(
        filtered_df, x="Locality", y="Price",
        color="Locality",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(template="plotly_white", showlegend=False,
                      font=dict(family="Inter"), margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # BHK vs Price
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("🛏️ BHK vs Price")
        fig = px.scatter(
            filtered_df, x="Bedrooms", y="Price", color="Locality",
            size="Area", hover_data=["Furnishing"],
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig.update_layout(template="plotly_white", font=dict(family="Inter"),
                          margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("📐 Area vs Price")
        fig = px.scatter(
            filtered_df, x="Area", y="Price", color="Locality",
            trendline="ols",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig.update_layout(template="plotly_white", font=dict(family="Inter"),
                          margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Scores comparison
    score_cols = [c for c in ["fair_price_score", "affordability_index", "investment_score"]
                  if c in df.columns]
    if score_cols:
        st.subheader("🎯 Score Comparison")
        score_data = []
        for loc in selected:
            loc_df = df[df["Locality"] == loc]
            row = {"Locality": loc}
            for c in score_cols:
                row[c.replace("_", " ").title()] = loc_df[c].mean()
            score_data.append(row)

        score_df = pd.DataFrame(score_data)

        fig = go.Figure()
        for col in [c for c in score_df.columns if c != "Locality"]:
            fig.add_trace(go.Bar(
                name=col, x=score_df["Locality"], y=score_df[col],
            ))
        fig.update_layout(
            barmode="group", template="plotly_white",
            font=dict(family="Inter"), margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Amenity distances radar
    amenity_cols = [c for c in df.columns if c.startswith("Nearest_")]
    if amenity_cols and len(selected) <= 6:
        st.subheader("🏥 Amenity Access Comparison")
        fig = go.Figure()
        for loc in selected:
            loc_df = df[df["Locality"] == loc]
            values = [loc_df[c].median() for c in amenity_cols]
            labels = [c.replace("Nearest_", "").replace("_km", "") for c in amenity_cols]
            fig.add_trace(go.Scatterpolar(
                r=values, theta=labels, fill="toself", name=loc,
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            template="plotly_white", font=dict(family="Inter"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Lower values = closer to amenity = better access")


if __name__ == "__main__":
    main()
