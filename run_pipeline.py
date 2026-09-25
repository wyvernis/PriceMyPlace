"""
Master pipeline script for PriceMyPlace Mumbai.
Runs data preprocessing, feature engineering, model training,
scoring, and SHAP explainability in sequence.
"""

import sys
import os
import io
import logging
import json
import warnings

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(PROJECT_ROOT, "outputs", "pipeline.log"), encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import joblib
from pathlib import Path


def run_pipeline():
    """Execute the complete pipeline."""

    logger.info("=" * 70)
    logger.info("PRICEMYPLACE MUMBAI - COMPLETE PIPELINE")
    logger.info("=" * 70)

    # ============================================================
    # PHASE 1: Data Loading & Preprocessing
    # ============================================================
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 1: Data Loading & Preprocessing")
    logger.info("=" * 50)

    from src.data.preprocessing import load_raw_data, clean_data, load_config

    config = load_config()
    df = load_raw_data()
    df_clean, clean_report = clean_data(df)

    logger.info(f"Rows: {clean_report['rows_before']} -> {clean_report['rows_after']}")
    logger.info(f"Duplicates removed: {clean_report['duplicates_removed']}")

    # Save processed data
    processed_path = Path(PROJECT_ROOT) / "data" / "processed" / "housing_processed.csv"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(processed_path, index=False)

    # ============================================================
    # PHASE 2: Feature Engineering
    # ============================================================
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 2: Feature Engineering")
    logger.info("=" * 50)

    from src.features.engineering import engineer_features

    df_feat = engineer_features(df_clean)
    logger.info(f"Total columns after feature engineering: {len(df_feat.columns)}")

    # Save feature-engineered data
    feat_path = Path(PROJECT_ROOT) / "data" / "processed" / "housing_features.csv"
    df_feat.to_csv(feat_path, index=False)

    # ============================================================
    # PHASE 3: Infrastructure Enrichment
    # ============================================================
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 3: Infrastructure Enrichment")
    logger.info("=" * 50)

    from src.enrichment.infrastructure import compute_infrastructure_features, get_infrastructure_df

    # Save infrastructure dataset
    infra_df = get_infrastructure_df()
    infra_path = Path(PROJECT_ROOT) / "data" / "external" / "infrastructure_projects.csv"
    infra_df.to_csv(infra_path, index=False)
    logger.info(f"Saved {len(infra_df)} infrastructure projects")

    # Compute infrastructure features for each property
    infra_features = compute_infrastructure_features(
        df_feat["Latitude"], df_feat["Longitude"]
    )
    for col in infra_features.columns:
        df_feat[col] = infra_features[col].values

    logger.info(f"Added infrastructure features: {list(infra_features.columns)}")

    # ============================================================
    # PHASE 4: Model Training
    # ============================================================
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 4: Model Training & Selection")
    logger.info("=" * 50)

    from src.models.trainer import train_and_compare

    # Prepare modeling dataframe (drop leakage columns)
    leakage_cols = [
        "Price_Per_Sqft", "Price_Per_Sqft_Outlier", "Price_Tier",
        "Price_vs_Locality_Median_Pct", "Price_Outlier",
        "price_per_sqft", "price_per_bhk", "price_per_bathroom",
        "locality_median_price", "locality_mean_price", "locality_price_std",
        "log_price", "nearest_infra_name",
    ]
    drop_cols = [c for c in leakage_cols if c in df_feat.columns]
    df_model = df_feat.drop(columns=drop_cols)

    results = train_and_compare(df_model, target_col="Price")

    logger.info(f"\nBest model: {results['best_name']}")
    logger.info(f"Test MAE:  Rs.{results['test_metrics']['mae']:,.0f}")
    logger.info(f"Test RMSE: Rs.{results['test_metrics']['rmse']:,.0f}")
    logger.info(f"Test R2:   {results['test_metrics']['r2']:.4f}")
    logger.info(f"Test MAPE: {results['test_metrics']['mape']:.1f}%")

    # ============================================================
    # PHASE 5: Generate Predictions for All Data
    # ============================================================
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 5: Generate Predictions & Scores")
    logger.info("=" * 50)

    model = results["best_model"]
    preprocessor = results["preprocessor"]
    feature_info = results["feature_info"]

    # Predict on all data
    X_all = df_model[feature_info["feature_cols"]]
    X_all_proc = preprocessor.transform(X_all)
    df_feat["predicted_price"] = model.predict(X_all_proc)

    # ============================================================
    # PHASE 6: Scoring
    # ============================================================
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 6: Computing Scores")
    logger.info("=" * 50)

    from src.scoring.scores import (
        compute_fair_price_scores_batch,
        compute_affordability_index,
        compute_investment_score,
    )

    # Fair Price Score
    fps = compute_fair_price_scores_batch(
        df_feat["predicted_price"], df_feat["Price"]
    )
    for col in fps.columns:
        df_feat[col] = fps[col].values
    logger.info(f"Fair Price Score distribution:")
    logger.info(f"  Mean: {df_feat['fair_price_score'].mean():.1f}")
    logger.info(f"  Median: {df_feat['fair_price_score'].median():.1f}")
    logger.info(f"  Labels: {df_feat['fair_price_label'].value_counts().to_dict()}")

    # Affordability Index
    df_feat["affordability_index"] = compute_affordability_index(df_feat)
    logger.info(f"Affordability Index - Mean: {df_feat['affordability_index'].mean():.1f}")

    # Investment Opportunity Score
    df_feat["investment_score"] = compute_investment_score(
        df_feat["fair_price_score"],
        df_feat["nearest_upcoming_infra_km"],
        df_feat["infra_completion_status"],
        df_feat["months_since_milestone"],
    )
    logger.info(f"Investment Score - Mean: {df_feat['investment_score'].mean():.1f}")

    # ============================================================
    # PHASE 7: SHAP Explainability
    # ============================================================
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 7: SHAP Explainability")
    logger.info("=" * 50)

    try:
        import shap

        # Use a sample for SHAP (faster)
        sample_size = min(500, len(X_all_proc))
        np.random.seed(42)
        sample_idx = np.random.choice(len(X_all_proc), sample_size, replace=False)

        if hasattr(X_all_proc, 'toarray'):
            X_sample = X_all_proc[sample_idx]
        else:
            X_sample = X_all_proc[sample_idx]

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # Save SHAP values
        models_dir = Path(PROJECT_ROOT) / "models"
        joblib.dump(explainer, models_dir / "shap_explainer.pkl")
        joblib.dump(shap_values, models_dir / "shap_values.pkl")
        joblib.dump(sample_idx, models_dir / "shap_sample_idx.pkl")

        # Feature importance from SHAP
        feature_names = feature_info["feature_cols"]

        # Handle case where preprocessor renames features
        n_features = shap_values.shape[1] if len(shap_values.shape) > 1 else len(feature_names)
        if len(feature_names) != n_features:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_importance = pd.DataFrame({
            "feature": feature_names[:len(mean_abs_shap)],
            "importance": mean_abs_shap,
        }).sort_values("importance", ascending=False)

        shap_importance.to_csv(models_dir / "shap_importance.csv", index=False)
        logger.info(f"Top 10 SHAP features:")
        for _, row in shap_importance.head(10).iterrows():
            logger.info(f"  {row['feature']:30s} | {row['importance']:,.0f}")

        logger.info("SHAP explainability saved successfully")

    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        logger.warning("Continuing without SHAP explainability...")

    # ============================================================
    # PHASE 8: Save Final Dataset
    # ============================================================
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 8: Saving Final Dataset")
    logger.info("=" * 50)

    final_path = Path(PROJECT_ROOT) / "data" / "processed" / "housing_final.csv"
    df_feat.to_csv(final_path, index=False)
    logger.info(f"Saved final dataset: {final_path}")
    logger.info(f"Final dataset: {df_feat.shape[0]} rows, {df_feat.shape[1]} columns")

    # Save preprocessing report
    report = {
        "cleaning": clean_report,
        "n_features_engineered": len(df_feat.columns) - len(df.columns),
        "best_model": results["best_name"],
        "test_metrics": results["test_metrics"],
        "fair_price_distribution": df_feat["fair_price_label"].value_counts().to_dict(),
        "mean_affordability": float(df_feat["affordability_index"].mean()),
        "mean_investment_score": float(df_feat["investment_score"].mean()),
    }
    report_path = Path(PROJECT_ROOT) / "outputs" / "pipeline_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"  Final dataset: {final_path}")
    logger.info(f"  Model:         models/best_model.pkl")
    logger.info(f"  Report:        {report_path}")
    logger.info("  Run Streamlit:  streamlit run app/Home.py")

    return df_feat, results


if __name__ == "__main__":
    run_pipeline()
