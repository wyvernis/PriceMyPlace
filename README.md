# PriceMyPlace Mumbai

**AI-Powered Property Price Intelligence Platform for Mumbai & Navi Mumbai**

An interactive data science application that predicts property prices, evaluates fair pricing, analyzes locality affordability, and provides infrastructure-based contextual signals across the Mumbai Metropolitan Region.

---

## Features

| Feature | Description |
|---------|-------------|
| **Price Prediction** | ML-powered property price estimation with SHAP explanations |
| **Fair Price Score** | Compare listing prices against model-estimated fair values |
| **Locality Explorer** | Compare 113 Mumbai/Navi Mumbai localities side-by-side |
| **Interactive Map** | Geospatial property visualization with price heatmaps |
| **Affordability Index** | Amenity access vs price analysis across neighborhoods |
| **Investment Explorer** | Infrastructure proximity & exploratory opportunity signals |
| **Model Insights** | ML performance metrics, model comparison, SHAP feature importance |
| **Documentation** | Complete methodology, formulas, limitations, and responsible use |

## Quick Start

### 1. Install Dependencies

```bash
cd pricemyplace-mumbai
pip install -r requirements.txt
```

### 2. Run the ML Pipeline

```bash
python run_pipeline.py
```

This executes the complete pipeline:
- Data preprocessing & cleaning
- Feature engineering (geographic distances, property ratios)
- Infrastructure enrichment (12 major Mumbai projects)
- Model training & comparison (7 models including XGBoost, LightGBM, CatBoost)
- SHAP explainability
- Score computation (Fair Price, Affordability, Investment)

### 3. Launch the Dashboard

```bash
streamlit run app/Home.py
```

## Project Structure

```
pricemyplace-mumbai/
├── app/                          # Streamlit dashboard
│   ├── Home.py                   # Main overview page
│   └── pages/
│       ├── 1_Price_Prediction.py # Interactive price predictor
│       ├── 2_Locality_Explorer.py # Locality comparison
│       ├── 3_Map.py              # Interactive Folium map
│       ├── 4_Affordability.py    # Affordability analysis
│       ├── 5_Investment_Explorer.py # Infrastructure signals
│       ├── 6_Model_Insights.py   # Model performance & SHAP
│       └── 7_Documentation.py    # Full methodology docs
├── config/
│   └── feature_mapping.yaml      # Column mapping & scoring weights
├── data/
│   ├── raw/                      # Original dataset
│   ├── processed/                # Cleaned & enriched data
│   └── external/                 # Infrastructure projects
├── models/                       # Trained model artifacts
│   ├── best_model.pkl
│   ├── preprocessor.pkl
│   ├── feature_info.pkl
│   ├── model_metadata.json
│   ├── shap_explainer.pkl
│   └── shap_importance.csv
├── src/
│   ├── data/preprocessing.py     # Data loading & cleaning
│   ├── features/engineering.py   # Feature engineering
│   ├── models/trainer.py         # ML training pipeline
│   ├── scoring/scores.py         # Fair Price, Affordability, Investment
│   └── enrichment/infrastructure.py # Mumbai infrastructure data
├── tests/test_pipeline.py        # Comprehensive test suite
├── outputs/                      # Pipeline logs & reports
├── run_pipeline.py               # Master pipeline script
├── data_dictionary.md            # Dataset schema documentation
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Dataset

- **5,994 properties** across 113 localities in Mumbai Metropolitan Region
- **28 features** covering property details, location, amenities, demographics, air quality
- **No missing values or duplicates** in the cleaned dataset
- See [data_dictionary.md](data_dictionary.md) for complete schema

## Scoring System

### Fair Price Score
```
Fair_Price_Score = 100 × (Predicted_Price / Listed_Price)
```
- ≈100: Fairly priced | >100: Potentially underpriced | <100: Potentially overpriced

### Affordability Index
```
Affordability = Amenity_Access_Composite / Price_per_sqft (normalized 0-100)
```

### Investment Opportunity Score
```
Score = 0.4×Valuation + 0.35×Infrastructure_Proximity + 0.25×Infrastructure_Horizon
```
⚠️ **Exploratory indicator only — not financial advice**

## Technology Stack

- **Python 3.11+** | **Streamlit** | **Pandas** | **scikit-learn**
- **XGBoost** | **LightGBM** | **CatBoost** | **SHAP**
- **Plotly** | **Folium** | **NumPy**

## Responsible Use

- All predictions are model estimates, not guaranteed market values
- Crime/demographic variables may encode historical bias
- Infrastructure proximity does not guarantee price appreciation
- The Investment Score is exploratory — not financial advice
- Users should verify information independently

## License

This project is for educational and analytical purposes.
