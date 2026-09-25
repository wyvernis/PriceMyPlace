"""
Price Prediction Page — PriceMyPlace Mumbai
Interactive property price prediction with SHAP explanations.
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
import plotly.graph_objects as go

st.set_page_config(page_title="Price Prediction | PriceMyPlace", page_icon="💰", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    .prediction-result {
        background: linear-gradient(135deg, #0f0c29, #302b63);
        border-radius: 16px;
        padding: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    .prediction-result h2 { color: #a5b4fc; font-size: 1rem; text-transform: uppercase; letter-spacing: 2px; }
    .prediction-result .price { font-size: 3rem; font-weight: 800; color: #ffffff; margin: 0.5rem 0; }
    .score-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .score-green { background: #10b981; color: white; }
    .score-yellow { background: #f59e0b; color: white; }
    .score-red { background: #ef4444; color: white; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
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
    return load_processed_data()


def main():
    st.markdown("# 💰 Price Prediction")
    st.markdown("*Predict property prices using our ML model and understand the key drivers*")

    df = load_data()
    model, preprocessor, feature_info, metadata = load_model_artifacts()

    if model is None:
        st.error("⚠️ Model not trained yet. Please run: `python run_pipeline.py`")
        return

    st.markdown("---")

    # Input form
    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.subheader("🏠 Property Details")

        localities = sorted(df["Locality"].unique().tolist()) if df is not None else ["Andheri", "Bandra"]
        locality = st.selectbox("Locality", localities, index=0)

        col_a, col_b = st.columns(2)
        with col_a:
            bhk = st.number_input("BHK (Bedrooms)", min_value=1, max_value=10, value=2)
            area = st.number_input("Area (sq ft)", min_value=100, max_value=15000, value=900)
        with col_b:
            bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
            listed_price = st.number_input("Listed Price (Rs.)", min_value=0, value=10000000, step=500000)

        property_types = df["Property_Type"].unique().tolist() if df is not None else ["Apartment"]
        property_type = st.selectbox("Property Type", property_types)

        furnishing_options = df["Furnishing"].unique().tolist() if df is not None else ["Unfurnished"]
        furnishing = st.selectbox("Furnishing", furnishing_options)

        predict_btn = st.button("🔮 Predict Price", type="primary", use_container_width=True)

    with col_result:
        if predict_btn:
            # Get locality defaults from data
            loc_data = df[df["Locality"] == locality] if df is not None else None

            # Build input features
            input_data = {
                "Area": area,
                "Bedrooms": bhk,
                "Bathrooms": bathrooms,
                "Latitude": loc_data["Latitude"].median() if loc_data is not None and len(loc_data) > 0 else 19.076,
                "Longitude": loc_data["Longitude"].median() if loc_data is not None and len(loc_data) > 0 else 72.877,
                "Locality": locality,
                "Property_Type": property_type,
                "Furnishing": furnishing,
            }

            # Add distance features from locality medians
            distance_cols = [c for c in feature_info["feature_cols"] if c not in input_data]
            if loc_data is not None and len(loc_data) > 0:
                for col in distance_cols:
                    if col in loc_data.columns:
                        input_data[col] = float(loc_data[col].median())
                    else:
                        input_data[col] = 0
            else:
                for col in distance_cols:
                    input_data[col] = 0

            # Predict
            try:
                input_df = pd.DataFrame([input_data])
                for col in feature_info["feature_cols"]:
                    if col not in input_df.columns:
                        input_df[col] = 0
                input_df = input_df[feature_info["feature_cols"]]

                X_proc = preprocessor.transform(input_df)
                predicted_price = float(model.predict(X_proc)[0])

                # Fair price score
                if listed_price > 0:
                    fair_score = 100 * (predicted_price / listed_price)
                    difference = predicted_price - listed_price
                    diff_pct = ((predicted_price - listed_price) / listed_price) * 100
                else:
                    fair_score = 100
                    difference = 0
                    diff_pct = 0

                # Display results
                st.markdown(f"""
                <div class="prediction-result">
                    <h2>Predicted Price</h2>
                    <div class="price">{format_inr(predicted_price)}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Metrics row
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Listed Price", format_inr(listed_price))
                with m2:
                    st.metric("Difference", format_inr(difference),
                              delta=f"{diff_pct:+.1f}%")
                with m3:
                    if fair_score > 105:
                        badge_class = "score-green"
                        label = "Underpriced"
                    elif fair_score >= 95:
                        badge_class = "score-yellow"
                        label = "Fair"
                    else:
                        badge_class = "score-red"
                        label = "Overpriced"
                    st.metric("Fair Price Score", f"{fair_score:.1f}")
                    st.markdown(f'<span class="score-badge {badge_class}">{label}</span>',
                                unsafe_allow_html=True)

                # Model confidence
                if metadata:
                    test_mae = metadata.get("test_metrics", {}).get("mae", 0)
                    test_mape = metadata.get("test_metrics", {}).get("mape", 0)
                    st.info(f"**Model Confidence:** MAE = {format_inr(test_mae)} | "
                            f"MAPE = {test_mape:.1f}% | "
                            f"Small differences within this range may not be meaningful.")

                # SHAP explanation
                try:
                    shap_explainer_path = Path(PROJECT_ROOT) / "models" / "shap_explainer.pkl"
                    if shap_explainer_path.exists():
                        import shap
                        explainer = joblib.load(shap_explainer_path)
                        shap_vals = explainer.shap_values(X_proc)

                        if len(shap_vals.shape) == 1:
                            shap_vals = shap_vals.reshape(1, -1)

                        feature_names = feature_info["feature_cols"]
                        n_shap = shap_vals.shape[1]
                        if len(feature_names) > n_shap:
                            feature_names = feature_names[:n_shap]
                        elif len(feature_names) < n_shap:
                            feature_names = feature_names + [f"f_{i}" for i in range(len(feature_names), n_shap)]

                        # Get top contributions
                        contrib = pd.DataFrame({
                            "feature": feature_names,
                            "contribution": shap_vals[0],
                        }).sort_values("contribution", key=abs, ascending=False).head(10)

                        st.subheader("🔍 What Drives This Prediction")

                        base_value = float(explainer.expected_value) if hasattr(explainer, 'expected_value') else predicted_price - shap_vals[0].sum()

                        st.markdown(f"**Base price:** {format_inr(base_value)}")
                        for _, row in contrib.iterrows():
                            sign = "+" if row["contribution"] > 0 else ""
                            st.markdown(f"- **{sign}{format_inr(row['contribution'])}** — {row['feature']}")

                except Exception as e:
                    st.caption(f"SHAP explanation unavailable: {e}")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                import traceback
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
