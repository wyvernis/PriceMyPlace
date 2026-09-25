"""
Model Insights Page — PriceMyPlace Mumbai
ML performance metrics, feature importance, and SHAP explainability.
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
import json
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Model Insights | PriceMyPlace", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


from app.utils import (
    get_project_root,
    load_model_artifacts,
    load_shap_importance,
    format_inr,
)

@st.cache_data
def load_metadata():
    _, _, _, meta = load_model_artifacts()
    return meta


def main():
    st.markdown("# 🤖 Model Insights")
    st.markdown("*Model performance, feature importance, and SHAP explainability*")
    st.markdown("---")

    metadata = load_metadata()
    shap_importance = load_shap_importance()

    if metadata is None:
        st.error("No model metadata found. Run the pipeline first.")
        return

    # Model overview
    st.subheader("📊 Model Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Best Model", metadata.get("best_model", "N/A"))
    with col2:
        st.metric("Training Samples", f"{metadata.get('n_train', 0):,}")
    with col3:
        st.metric("Validation Samples", f"{metadata.get('n_val', 0):,}")
    with col4:
        st.metric("Test Samples", f"{metadata.get('n_test', 0):,}")

    # Test metrics
    st.subheader("📈 Test Set Performance")
    test_metrics = metadata.get("test_metrics", {})

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("R² Score", f"{test_metrics.get('r2', 0):.4f}")
    with m2:
        st.metric("MAE", format_inr(test_metrics.get("mae", 0)))
    with m3:
        st.metric("RMSE", format_inr(test_metrics.get("rmse", 0)))
    with m4:
        st.metric("MAPE", f"{test_metrics.get('mape', 0):.1f}%")
    with m5:
        st.metric("Median AE", format_inr(test_metrics.get("median_ae", 0)))

    # Model comparison
    st.subheader("🔬 Model Comparison")
    all_results = metadata.get("all_results", {})

    if all_results:
        comparison_data = []
        for model_name, result in all_results.items():
            if "validation" in result:
                val = result["validation"]
                row = {
                    "Model": model_name,
                    "Val R²": val.get("r2", 0),
                    "Val MAE": val.get("mae", 0),
                    "Val RMSE": val.get("rmse", 0),
                    "Val MAPE (%)": val.get("mape", 0),
                }
                if "test" in result:
                    row["Test R²"] = result["test"].get("r2", 0)
                    row["Test MAE"] = result["test"].get("mae", 0)
                comparison_data.append(row)

        comp_df = pd.DataFrame(comparison_data).sort_values("Val R²", ascending=False)

        # Highlight best model
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # Comparison chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Val R²", x=comp_df["Model"], y=comp_df["Val R²"],
            marker_color="#6366f1",
        ))
        if "Test R²" in comp_df.columns:
            fig.add_trace(go.Bar(
                name="Test R²", x=comp_df["Model"], y=comp_df["Test R²"],
                marker_color="#10b981",
            ))
        fig.update_layout(
            template="plotly_white", barmode="group",
            font=dict(family="Inter"),
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis_title="R² Score",
        )
        st.plotly_chart(fig, use_container_width=True)

    # SHAP Feature Importance
    if shap_importance is not None:
        st.subheader("🔍 SHAP Feature Importance")
        st.markdown("*Mean absolute SHAP value — how much each feature influences the prediction on average*")

        top_n = st.slider("Show top N features", 5, min(30, len(shap_importance)), 15)
        top_features = shap_importance.head(top_n)

        fig = px.bar(
            top_features.sort_values("importance", ascending=True),
            x="importance", y="feature", orientation="h",
            color="importance", color_continuous_scale="Viridis",
            labels={"importance": "Mean |SHAP Value|", "feature": "Feature"},
        )
        fig.update_layout(
            template="plotly_white", showlegend=False,
            coloraxis_showscale=False, font=dict(family="Inter"),
            margin=dict(l=20, r=20, t=30, b=20),
            height=max(400, top_n * 28),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Feature columns used
    st.subheader("📋 Features Used")
    feature_cols = metadata.get("feature_columns", [])
    numeric_cols = metadata.get("numeric_columns", [])
    categorical_cols = metadata.get("categorical_columns", [])

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Numeric Features ({len(numeric_cols)})**")
        for c in sorted(numeric_cols):
            st.markdown(f"- `{c}`")
    with col_b:
        st.markdown(f"**Categorical Features ({len(categorical_cols)})**")
        for c in sorted(categorical_cols):
            st.markdown(f"- `{c}`")


if __name__ == "__main__":
    main()
