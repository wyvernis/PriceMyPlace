"""
Investment Explorer Page — PriceMyPlace Mumbai
Infrastructure proximity & exploratory investment signals.
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

st.set_page_config(page_title="Investment Explorer | PriceMyPlace", page_icon="🏗️", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    .disclaimer-box {
        background: linear-gradient(135deg, #fef3c7, #fde68a);
        border: 1px solid #f59e0b;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        font-size: 0.85rem;
        color: #92400e;
        line-height: 1.5;
        margin-bottom: 1.5rem;
    }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


from app.utils import (
    get_project_root,
    load_processed_data,
    load_infrastructure_data,
    format_inr,
)

@st.cache_data
def load_data():
    return load_processed_data()

@st.cache_data
def load_infrastructure():
    return load_infrastructure_data()


def main():
    st.markdown("# 🏗️ Investment Explorer")
    st.markdown("*Infrastructure context & exploratory opportunity signals for Mumbai/Navi Mumbai*")

    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Important Disclaimer:</strong> The Investment Opportunity Score is an <strong>exploratory
        analytical indicator</strong> based on historical listing data and publicly available infrastructure
        information. It is <strong>NOT financial or investment advice</strong>. Infrastructure proximity does
        not guarantee price appreciation. Users should verify current information independently and consult
        qualified professionals before making any decisions.
    </div>
    """, unsafe_allow_html=True)

    df = load_data()
    infra = load_infrastructure()

    if df is None:
        st.error("No data available. Run the pipeline first.")
        return

    # Infrastructure Projects Table
    if infra is not None:
        st.subheader("🏗️ Major Infrastructure Projects")

        display_infra = infra[["project_name", "project_type", "status",
                                "expected_completion", "source"]].copy()
        display_infra.columns = ["Project", "Type", "Status", "Expected Completion", "Source"]

        # Color code status
        st.dataframe(display_infra, use_container_width=True, hide_index=True)

        # Infrastructure map
        st.subheader("📍 Infrastructure Locations")
        import folium
        from streamlit_folium import st_folium

        m = folium.Map(location=[19.076, 72.877], zoom_start=11, tiles="cartodbpositron")
        status_colors = {"Completed": "green", "Under Construction": "orange", "Planned": "blue"}

        for _, proj in infra.iterrows():
            folium.Marker(
                location=[proj["latitude"], proj["longitude"]],
                popup=f"<b>{proj['project_name']}</b><br>Status: {proj['status']}<br>Type: {proj['project_type']}",
                tooltip=proj["project_name"],
                icon=folium.Icon(color=status_colors.get(proj["status"], "gray"), icon="wrench", prefix="fa"),
            ).add_to(m)

        st_folium(m, width=None, height=450, use_container_width=True)

    # Investment Scores
    if "investment_score" in df.columns:
        st.markdown("---")
        st.subheader("🎯 Investment Opportunity Scores by Locality")

        with st.expander("📐 How is the Investment Score calculated?"):
            st.markdown("""
            **Investment Score = w1 × Valuation + w2 × Infra Proximity + w3 × Infra Horizon**

            - **Valuation** (40%): Based on Fair Price Score — underpriced properties score higher
            - **Infra Proximity** (35%): Distance to nearest major infrastructure project
            - **Infra Horizon** (25%): Under-construction projects score highest (price hasn't
              fully absorbed the effect), followed by planned, then completed

            Weights are configurable in `config/feature_mapping.yaml`.
            Initial weights are equal until calibrated against historical price trends.
            """)

        loc_invest = df.groupby("Locality").agg({
            "investment_score": "mean",
            "nearest_upcoming_infra_km": "mean",
            "Price": ["median", "count"],
        })
        loc_invest.columns = ["Avg Investment Score", "Avg Infra Distance (km)",
                               "Median Price", "Listings"]
        loc_invest = loc_invest[loc_invest["Listings"] >= 5]
        loc_invest = loc_invest.sort_values("Avg Investment Score", ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Top 15 — Highest Investment Signal**")
            top = loc_invest.head(15).reset_index()
            fig = px.bar(
                top, x="Avg Investment Score", y="Locality", orientation="h",
                color="Avg Investment Score", color_continuous_scale="Viridis",
                text="Avg Investment Score",
            )
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig.update_layout(template="plotly_white", showlegend=False,
                              coloraxis_showscale=False, font=dict(family="Inter"),
                              margin=dict(l=20, r=60, t=30, b=20), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Infrastructure Distance vs Investment Score**")
            fig = px.scatter(
                loc_invest.reset_index(),
                x="Avg Infra Distance (km)", y="Avg Investment Score",
                size="Listings", color="Avg Investment Score",
                hover_name="Locality", color_continuous_scale="Viridis",
                size_max=25,
            )
            fig.update_layout(template="plotly_white", font=dict(family="Inter"),
                              margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

        # Infrastructure proximity distribution
        if "nearest_upcoming_infra_km" in df.columns:
            st.subheader("📏 Distance to Nearest Infrastructure Project")
            fig = px.histogram(
                df, x="nearest_upcoming_infra_km", nbins=50,
                color_discrete_sequence=["#6366f1"],
                labels={"nearest_upcoming_infra_km": "Distance to Nearest Infrastructure (km)"},
            )
            fig.update_layout(template="plotly_white", showlegend=False,
                              font=dict(family="Inter"),
                              margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

        # Nearest infra project distribution
        if "infra_completion_status" in df.columns:
            st.subheader("📊 Infrastructure Status Distribution")
            status_counts = df["infra_completion_status"].value_counts()
            fig = px.pie(
                values=status_counts.values, names=status_counts.index,
                color_discrete_sequence=["#10b981", "#f59e0b", "#6366f1"],
                hole=0.4,
            )
            fig.update_layout(template="plotly_white", font=dict(family="Inter"),
                              margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
