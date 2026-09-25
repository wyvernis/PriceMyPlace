"""
ML training pipeline for PriceMyPlace Mumbai.
Trains, evaluates, and selects the best regression model for property price prediction.
"""

import logging
import json
import os
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Tuple, Optional, List

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    median_absolute_error, mean_absolute_percentage_error,
)
from sklearn.dummy import DummyRegressor

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def prepare_features_and_target(
    df: pd.DataFrame,
    target_col: str = "Price",
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Separate features and target, identify numeric vs categorical columns.
    """
    if feature_cols is None:
        # Auto-detect features, excluding target and leakage columns
        exclude = {
            target_col, "log_price",
            "Price_Per_Sqft", "Price_Per_Sqft_Outlier", "Price_Tier",
            "Price_vs_Locality_Median_Pct", "Price_Outlier",
            "price_per_sqft", "price_per_bhk", "price_per_bathroom",
            "locality_median_price", "locality_mean_price", "locality_price_std",
        }
        feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "string", "str", "category"]).columns.tolist()
    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()

    # Convert bools to int
    for c in bool_cols:
        X[c] = X[c].astype(int)
        if c not in numeric_cols:
            numeric_cols.append(c)

    logger.info(f"Features: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")
    return X, y, numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for preprocessing."""
    transformers = []

    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))

    if categorical_cols:
        transformers.append(("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


def get_candidate_models() -> Dict[str, object]:
    """Return dictionary of candidate regression models."""
    models = {
        "Mean Baseline": DummyRegressor(strategy="mean"),
        "Median Baseline": DummyRegressor(strategy="median"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=20, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            min_samples_leaf=10, random_state=42
        ),
    }

    # Try importing XGBoost
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbosity=0
        )
    except ImportError:
        logger.warning("XGBoost not available, skipping")

    # Try importing LightGBM
    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = LGBMRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1
        )
    except ImportError:
        logger.warning("LightGBM not available, skipping")

    # Try importing CatBoost
    try:
        from catboost import CatBoostRegressor
        models["CatBoost"] = CatBoostRegressor(
            iterations=300, depth=8, learning_rate=0.05,
            random_seed=42, verbose=0
        )
    except ImportError:
        logger.warning("CatBoost not available, skipping")

    return models


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute evaluation metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        "median_ae": float(median_absolute_error(y_true, y_pred)),
    }


def train_and_compare(
    df: pd.DataFrame,
    target_col: str = "Price",
    feature_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Dict:
    """
    Train all candidate models, evaluate on validation and test sets.
    Select the best model based on validation RMSE.
    """
    logger.info("=" * 60)
    logger.info("STARTING MODEL TRAINING PIPELINE")
    logger.info("=" * 60)

    X, y, numeric_cols, categorical_cols = prepare_features_and_target(
        df, target_col, feature_cols
    )

    # Train/val/test split (prevent data leakage by splitting first)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size / (1 - test_size),
        random_state=random_state
    )

    logger.info(f"Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # Build preprocessor (fit only on training data!)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    preprocessor.fit(X_train)

    X_train_proc = preprocessor.transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    # Train all models
    results = {}
    trained_models = {}
    candidate_models = get_candidate_models()

    for name, model in candidate_models.items():
        logger.info(f"\nTraining: {name}")
        try:
            model.fit(X_train_proc, y_train)
            trained_models[name] = model

            y_val_pred = model.predict(X_val_proc)
            y_train_pred = model.predict(X_train_proc)

            val_metrics = evaluate_model(y_val, y_val_pred)
            train_metrics = evaluate_model(y_train, y_train_pred)

            results[name] = {
                "train": train_metrics,
                "validation": val_metrics,
            }
            logger.info(f"  Val MAE: Rs.{val_metrics['mae']:,.0f} | "
                        f"Val RMSE: Rs.{val_metrics['rmse']:,.0f} | "
                        f"Val R2: {val_metrics['r2']:.4f}")
        except Exception as e:
            logger.error(f"  Failed: {e}")
            results[name] = {"error": str(e)}

    # Select best model (lowest validation RMSE)
    valid_results = {k: v for k, v in results.items() if "validation" in v}
    best_name = min(valid_results, key=lambda k: valid_results[k]["validation"]["rmse"])
    best_model = trained_models[best_name]

    logger.info(f"\n{'=' * 60}")
    logger.info(f"BEST MODEL: {best_name}")
    logger.info(f"{'=' * 60}")

    # Evaluate best model on test set
    y_test_pred = best_model.predict(X_test_proc)
    test_metrics = evaluate_model(y_test, y_test_pred)
    results[best_name]["test"] = test_metrics
    logger.info(f"  Test MAE:  Rs.{test_metrics['mae']:,.0f}")
    logger.info(f"  Test RMSE: Rs.{test_metrics['rmse']:,.0f}")
    logger.info(f"  Test R2:   {test_metrics['r2']:.4f}")
    logger.info(f"  Test MAPE: {test_metrics['mape']:.1f}%")

    # Save artifacts
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, models_dir / "best_model.pkl")
    joblib.dump(preprocessor, models_dir / "preprocessor.pkl")

    # Save feature names for prediction
    feature_info = {
        "feature_cols": list(X.columns),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "target_col": target_col,
    }
    joblib.dump(feature_info, models_dir / "feature_info.pkl")

    # Save metadata
    metadata = {
        "best_model": best_name,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "n_features": X_train_proc.shape[1],
        "feature_columns": list(X.columns),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "all_results": results,
        "test_metrics": test_metrics,
    }
    with open(models_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"\nSaved: best_model.pkl, preprocessor.pkl, model_metadata.json")

    return {
        "best_name": best_name,
        "best_model": best_model,
        "preprocessor": preprocessor,
        "feature_info": feature_info,
        "all_results": results,
        "test_metrics": test_metrics,
        "X_test": X_test,
        "y_test": y_test,
        "y_test_pred": y_test_pred,
    }


def load_trained_model() -> Tuple[object, object, dict, dict]:
    """Load the trained model, preprocessor, feature info, and metadata."""
    models_dir = PROJECT_ROOT / "models"
    model = joblib.load(models_dir / "best_model.pkl")
    preprocessor = joblib.load(models_dir / "preprocessor.pkl")
    feature_info = joblib.load(models_dir / "feature_info.pkl")
    with open(models_dir / "model_metadata.json", "r") as f:
        metadata = json.load(f)
    return model, preprocessor, feature_info, metadata


def predict_price(
    model, preprocessor, feature_info: dict,
    input_data: dict
) -> dict:
    """Make a single property price prediction."""
    # Create DataFrame from input
    df_input = pd.DataFrame([input_data])

    # Ensure all feature columns exist
    for col in feature_info["feature_cols"]:
        if col not in df_input.columns:
            df_input[col] = 0  # Default for missing features

    df_input = df_input[feature_info["feature_cols"]]

    # Transform
    X_proc = preprocessor.transform(df_input)
    predicted_price = float(model.predict(X_proc)[0])

    return {
        "predicted_price": predicted_price,
        "predicted_price_formatted": f"Rs.{predicted_price:,.0f}",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from src.data.preprocessing import load_raw_data, clean_data
    from src.features.engineering import engineer_features

    # Load and preprocess
    df = load_raw_data()
    df, _ = clean_data(df)

    # Engineer features (without target-leaking ones)
    df = engineer_features(df)

    # Drop leakage columns
    leakage = ["Price_Per_Sqft", "Price_Per_Sqft_Outlier", "Price_Tier",
               "Price_vs_Locality_Median_Pct", "Price_Outlier",
               "price_per_sqft", "price_per_bhk", "price_per_bathroom",
               "locality_median_price", "locality_mean_price", "locality_price_std",
               "log_price"]
    drop_cols = [c for c in leakage if c in df.columns]
    df_model = df.drop(columns=drop_cols)

    # Train
    results = train_and_compare(df_model)
    print(f"\nBest model: {results['best_name']}")
    print(f"Test metrics: {results['test_metrics']}")
