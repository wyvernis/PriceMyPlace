"""
Documentation Page — PriceMyPlace Mumbai
Methodology, dataset, model, scores, assumptions, limitations, responsible use.
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
from pathlib import Path

st.set_page_config(page_title="Documentation | PriceMyPlace", page_icon="📖", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def main():
    st.markdown("# 📖 Documentation")
    st.markdown("*Comprehensive guide to PriceMyPlace Mumbai methodology, models, and responsible use*")
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Methodology", "📁 Dataset", "🤖 Model", "📐 Scores",
        "⚠️ Limitations", "🛡️ Responsible Use"
    ])

    with tab1:
        st.markdown("""
        ## Methodology

        PriceMyPlace Mumbai follows a structured data science pipeline:

        ### 1. Data Collection & Cleaning
        - Property listings from publicly available sources in Mumbai & Navi Mumbai
        - 5,994 property records with 28 features covering property details,
          location, amenity distances, demographics, and air quality
        - Cleaning includes: duplicate removal, coordinate validation, category standardization,
          outlier detection, and missing value imputation

        ### 2. Feature Engineering
        - **Property features**: Price per sqft, price per BHK, area per BHK, log transforms
        - **Geographic features**: Distance to city center (CST), nearest major transit hub,
          nearest railway station, nearest metro station
        - **Locality statistics**: Median/mean prices, listing counts
        - **Infrastructure features**: Distance to major infrastructure projects,
          completion status, months since milestone

        ### 3. Model Training
        - Train/validation/test split (65%/15%/20%)
        - Preprocessing: StandardScaler for numeric, OrdinalEncoder for categorical
        - Candidate models: Mean/Median baselines, Linear Regression, Random Forest,
          Gradient Boosting, XGBoost, LightGBM, CatBoost
        - Best model selected on validation RMSE
        - Final evaluation on held-out test set

        ### 4. Scoring
        - Fair Price Score, Affordability Index, and Investment Opportunity Score
          computed using trained model predictions
        - Scores are generated post-training and are not model inputs

        ### 5. Explainability
        - SHAP (SHapley Additive exPlanations) for feature importance and
          individual prediction explanations
        """)

    with tab2:
        st.markdown("""
        ## Dataset

        ### Overview
        | Attribute | Value |
        |-----------|-------|
        | Total Records | 5,994 |
        | Total Features | 28 |
        | Geographic Coverage | Mumbai, Navi Mumbai, and extended MMR |
        | Unique Localities | 113 |
        | Missing Values | 0 |
        | Duplicate Records | 0 |

        ### Feature Categories

        #### Property Features
        | Feature | Type | Description |
        |---------|------|-------------|
        | Price | int64 | Property price in INR (target variable) |
        | Area | int64 | Property area in square feet |
        | Bedrooms | int64 | Number of bedrooms (BHK) |
        | Bathrooms | int64 | Number of bathrooms |
        | Property_Type | string | Apartment, Villa, Independent House, Independent Floor |
        | Furnishing | string | Unfurnished, Semi-Furnished, Furnished |

        #### Location Features
        | Feature | Type | Description |
        |---------|------|-------------|
        | Latitude | float64 | Geographic latitude |
        | Longitude | float64 | Geographic longitude |
        | Locality | string | Neighborhood/locality name |

        #### Amenity Distances (km)
        | Feature | Description |
        |---------|-------------|
        | Nearest_Hospital_km | Distance to nearest hospital |
        | Nearest_School_km | Distance to nearest school |
        | Nearest_Metro_km | Distance to nearest metro station |
        | Nearest_Bus_Stop_km | Distance to nearest bus stop |
        | Nearest_Park_km | Distance to nearest park |
        | Nearest_Mall_km | Distance to nearest mall |

        #### Demographics & Safety
        | Feature | Description |
        |---------|-------------|
        | Crime_Rate | Locality-level crime rate |
        | Population_Density | Population density per sq km |
        | Literacy_Rate | Literacy rate percentage |
        | Household_Count | Number of households |

        #### Air Quality
        | Feature | Description |
        |---------|-------------|
        | AQI | Air Quality Index |
        | PM2.5 | Fine particulate matter |
        | PM10 | Coarse particulate matter |
        """)

    with tab3:
        st.markdown("""
        ## Model

        ### Training Pipeline
        1. **Data leakage prevention**: Target-derived columns (Price_Per_Sqft,
           Price_Tier, etc.) are excluded from features
        2. **Preprocessing**: StandardScaler for numeric, OrdinalEncoder for categorical
        3. **Split**: Train (65%) / Validation (15%) / Test (20%)
        4. **Preprocessor fitted only on training data**

        ### Candidate Models
        | Model | Description |
        |-------|-------------|
        | Mean Baseline | Predicts the mean of training prices |
        | Median Baseline | Predicts the median of training prices |
        | Linear Regression | Ordinary least squares |
        | Random Forest | 200 trees, max_depth=20 |
        | Gradient Boosting | 200 estimators, learning_rate=0.1 |
        | XGBoost | 300 estimators, learning_rate=0.05 |
        | LightGBM | 300 estimators, learning_rate=0.05 |
        | CatBoost | 300 iterations, depth=8 |

        ### Model Selection
        - Best model selected by **lowest validation RMSE**
        - Final performance reported on held-out test set

        ### Evaluation Metrics
        | Metric | Description |
        |--------|-------------|
        | MAE | Mean Absolute Error — average prediction error in Rs. |
        | RMSE | Root Mean Squared Error — penalizes larger errors |
        | R² | Coefficient of determination — proportion of variance explained |
        | MAPE | Mean Absolute Percentage Error |
        | Median AE | Median Absolute Error — robust to outliers |
        """)

    with tab4:
        st.markdown("""
        ## Score Formulas

        ### 1. Fair Price Score
        ```
        Fair_Price_Score = 100 × (Predicted_Price / Listed_Price)
        ```

        | Score Range | Label | Interpretation |
        |-------------|-------|----------------|
        | > 115 | Underpriced | Significantly below model estimate |
        | 105 - 115 | Below Market | Somewhat below model estimate |
        | 95 - 105 | Fairly Priced | Close to model estimate |
        | 85 - 95 | Above Market | Somewhat above model estimate |
        | < 85 | Overpriced | Significantly above model estimate |

        **Important**: Small differences within the model's error margin (MAE/MAPE)
        may not be meaningful. Always consider the model's confidence interval.

        ### 2. Affordability Index
        ```
        Amenity_Access_Composite = weighted combination of:
            1/Hospital_dist, 1/School_dist, 1/Metro_dist,
            1/Bus_dist, 1/Park_dist, 1/Mall_dist,
            1/Crime_Rate, 1/AQI

        Affordability_Index = Amenity_Access_Composite / Price_per_sqft
        ```
        Normalized to 0-100 scale. Higher = more value for money.

        **Default Weights:**
        | Component | Weight |
        |-----------|--------|
        | Hospital | 0.20 |
        | Metro | 0.20 |
        | School | 0.15 |
        | Park | 0.15 |
        | Bus Stop | 0.10 |
        | Mall | 0.10 |
        | Crime Rate | 0.05 |
        | AQI | 0.05 |

        Weights are configurable in `config/feature_mapping.yaml`.

        ### 3. Investment Opportunity Score
        ```
        Score = w1 × Valuation_Component
             + w2 × Infrastructure_Proximity
             + w3 × Infrastructure_Horizon
        ```

        | Component | Weight | Description |
        |-----------|--------|-------------|
        | Valuation | 0.40 | Based on Fair Price Score (underpriced = higher) |
        | Infra Proximity | 0.35 | Distance to nearest major infrastructure project |
        | Infra Horizon | 0.25 | Under-construction projects score highest |

        **This is exploratory and directional, not financial advice.**
        """)

    with tab5:
        st.markdown("""
        ## Assumptions & Limitations

        ### Assumptions
        - Property prices follow patterns related to the available features
        - Amenity distances from the dataset are reasonably accurate
        - Crime rates and demographic data are representative of locality conditions
        - Infrastructure project timelines are based on publicly available information

        ### Limitations
        1. **Static snapshot**: The model reflects listing prices at a point in time,
           not current market conditions
        2. **Geographic bias**: Some localities may have insufficient data for reliable predictions
        3. **Missing features**: Factors like floor number, age of building, view, society amenities,
           and upcoming supply are not captured
        4. **Demographic bias**: Crime and population density data may encode historical
           or systemic biases
        5. **Infrastructure uncertainty**: Project timelines can be delayed or modified;
           the Investment Score uses information that may become outdated
        6. **Model error**: All predictions have uncertainty — the MAE/MAPE represents
           the typical prediction error
        7. **Sample bias**: The dataset may not represent all property segments equally
        8. **No temporal modeling**: The model does not account for market trends over time
        """)

    with tab6:
        st.markdown("""
        ## Responsible Use

        ### Guidelines
        - **Predictions are estimates**, not guaranteed market values
        - **Crime and demographic variables** may encode historical or social bias.
          Treat them with appropriate care
        - **Infrastructure proximity does not guarantee future price appreciation.**
          Many factors affect property values
        - **The Investment Opportunity Score is exploratory and educational** —
          not financial or investment advice
        - **External infrastructure data can change.** Project timelines, scopes,
          and statuses are based on publicly available information at a point in time
        - **Users should independently verify** current rental/sale listings and
          infrastructure status before making any decisions
        - **Consult qualified professionals** for actual buying, selling, or
          investment decisions

        ### Data Ethics
        - Only publicly available data is used
        - No personal or sensitive information is retained
        - The platform does not make decisions for users — it provides information
          to support their own analysis

        ### Bias Monitoring
        - Model performance should be monitored across localities and price ranges
        - The Investment Score should be checked for systematic bias favoring
          already-affluent areas
        - Feature importance should be reviewed to ensure no single demographic
          variable dominates predictions in ways that could reinforce inequality
        """)


if __name__ == "__main__":
    main()
