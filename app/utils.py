"""
Shared utilities and robust data/model loaders for PriceMyPlace Mumbai.
Guarantees paths resolve regardless of working directory or launch method.
"""

import os
import sys
import json
import logging
import joblib
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Any, Dict

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """
    Find project root dynamically by probing known marker files/directories.
    Works whether running from project root, app/, tests/, or parent directory.
    """
    candidates = [
        Path(__file__).resolve().parent.parent,                     # app/utils.py -> root
        Path(__file__).resolve().parent.parent.parent,              # defensive
        Path.cwd(),                                                 # current working dir
        Path(r"c:\Users\ASUS\Downloads\ads-project\PriceMyPlace"),  # explicit workspace
        Path(r"c:\Users\ASUS\Downloads\ads-project\pricemyplace-mumbai"),
    ]

    for c in candidates:
        if (c / "data" / "processed" / "housing_final.csv").exists() or \
           (c / "models" / "best_model.pkl").exists() or \
           (c / "data" / "raw" / "final cleaned dataset.csv").exists():
            return c

    # Fallback
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = get_project_root()


def format_inr(value: Optional[float]) -> str:
    """Format numeric rupee amount into standard Indian denomination (Crores/Lakhs)."""
    if value is None or pd.isna(value):
        return "N/A"
    abs_val = abs(value)
    if abs_val >= 1e7:
        return f"Rs. {value / 1e7:.2f} Cr"
    elif abs_val >= 1e5:
        return f"Rs. {value / 1e5:.2f} L"
    else:
        return f"Rs. {value:,.0f}"


def load_processed_data() -> Optional[pd.DataFrame]:
    """Load final processed dataset with scores and predictions."""
    root = get_project_root()
    final_path = root / "data" / "processed" / "housing_final.csv"
    if final_path.exists():
        return pd.read_csv(final_path)

    features_path = root / "data" / "processed" / "housing_features.csv"
    if features_path.exists():
        return pd.read_csv(features_path)

    raw_path = root / "data" / "raw" / "final cleaned dataset.csv"
    if raw_path.exists():
        return pd.read_csv(raw_path)

    logger.warning("No housing data file found in project data paths.")
    return None


def load_infrastructure_data() -> Optional[pd.DataFrame]:
    """Load infrastructure mega-projects dataset."""
    root = get_project_root()
    path = root / "data" / "external" / "infrastructure_projects.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def load_model_artifacts() -> Tuple[Any, Any, Any, Optional[Dict]]:
    """
    Load trained ML model, preprocessor, feature information, and metadata.
    Logs explicit errors if any step fails.
    """
    root = get_project_root()
    models_dir = root / "models"

    model = None
    preprocessor = None
    feature_info = None
    metadata = None

    model_path = models_dir / "best_model.pkl"
    prep_path = models_dir / "preprocessor.pkl"
    feat_path = models_dir / "feature_info.pkl"
    meta_path = models_dir / "model_metadata.json"

    if model_path.exists():
        try:
            model = joblib.load(model_path)
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")

    if prep_path.exists():
        try:
            preprocessor = joblib.load(prep_path)
        except Exception as e:
            logger.error(f"Error loading preprocessor from {prep_path}: {e}")

    if feat_path.exists():
        try:
            feature_info = joblib.load(feat_path)
        except Exception as e:
            logger.error(f"Error loading feature info from {feat_path}: {e}")

    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata from {meta_path}: {e}")

    return model, preprocessor, feature_info, metadata


def load_shap_importance() -> Optional[pd.DataFrame]:
    """Load precomputed SHAP feature importance table."""
    root = get_project_root()
    path = root / "models" / "shap_importance.csv"
    if path.exists():
        return pd.read_csv(path)
    return None
