"""
Data loading and preprocessing pipeline for PriceMyPlace Mumbai.
Handles CSV ingestion, validation, cleaning, and transformation.
"""

import logging
import pandas as pd
import numpy as np
import yaml
import os
from pathlib import Path
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "feature_mapping.yaml"


def load_config() -> dict:
    """Load feature mapping configuration."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load the raw dataset from CSV."""
    if filepath is None:
        filepath = PROJECT_ROOT / "data" / "raw" / "final cleaned dataset.csv"
    logger.info(f"Loading raw data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def validate_columns(df: pd.DataFrame, config: dict) -> Dict[str, list]:
    """Validate that expected columns exist in the dataframe."""
    report = {"found": [], "missing": [], "extra": []}
    expected = set()

    # Collect all expected column names from config
    for group_key in ["property_features", "location_features", "amenity_distances",
                       "demographics", "air_quality", "derived_features"]:
        if group_key in config:
            for _, col_name in config[group_key].items():
                expected.add(col_name)

    # Add target
    if "target" in config:
        expected.add(config["target"]["price"])

    actual = set(df.columns)
    report["found"] = sorted(expected & actual)
    report["missing"] = sorted(expected - actual)
    report["extra"] = sorted(actual - expected)

    for col in report["missing"]:
        logger.warning(f"Expected column '{col}' not found in dataset")
    logger.info(f"Column validation: {len(report['found'])} found, "
                f"{len(report['missing'])} missing, {len(report['extra'])} extra")
    return report


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Clean the dataset: handle duplicates, outliers, missing values,
    invalid coordinates, and inconsistent categories.
    Returns cleaned DataFrame and cleaning report.
    """
    report = {
        "rows_before": len(df),
        "duplicates_removed": 0,
        "outliers_flagged": 0,
        "invalid_coords_fixed": 0,
        "missing_filled": {},
        "transformations": [],
    }

    df = df.copy()

    # 1. Remove exact duplicates
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        report["duplicates_removed"] = n_dupes
        logger.info(f"Removed {n_dupes} duplicate rows")

    # 2. Handle missing values
    for col in df.columns:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            if df[col].dtype in [np.float64, np.int64, float, int]:
                fill_val = df[col].median()
                df[col] = df[col].fillna(fill_val)
                report["missing_filled"][col] = {"count": int(n_missing), "strategy": "median", "value": float(fill_val)}
            else:
                fill_val = df[col].mode()[0] if len(df[col].mode()) > 0 else "Unknown"
                df[col] = df[col].fillna(fill_val)
                report["missing_filled"][col] = {"count": int(n_missing), "strategy": "mode", "value": str(fill_val)}
            logger.info(f"Filled {n_missing} missing values in '{col}'")

    # 3. Validate coordinates (Mumbai region: lat 18.5-19.9, lon 72.5-73.6)
    if "Latitude" in df.columns and "Longitude" in df.columns:
        valid_lat = df["Latitude"].between(18.5, 19.9)
        valid_lon = df["Longitude"].between(72.5, 73.6)
        invalid_coords = ~(valid_lat & valid_lon)
        n_invalid = invalid_coords.sum()
        if n_invalid > 0:
            # Replace with locality median coordinates
            for locality in df.loc[invalid_coords, "Locality"].unique():
                mask = (df["Locality"] == locality) & valid_lat & valid_lon
                if mask.sum() > 0:
                    med_lat = df.loc[mask, "Latitude"].median()
                    med_lon = df.loc[mask, "Longitude"].median()
                    fix_mask = (df["Locality"] == locality) & invalid_coords
                    df.loc[fix_mask, "Latitude"] = med_lat
                    df.loc[fix_mask, "Longitude"] = med_lon
            report["invalid_coords_fixed"] = int(n_invalid)
            logger.info(f"Fixed {n_invalid} invalid coordinates")

    # 4. Standardize categorical values
    if "Property_Type" in df.columns:
        df["Property_Type"] = df["Property_Type"].str.strip().str.title()
        report["transformations"].append("Standardized Property_Type")

    if "Furnishing" in df.columns:
        df["Furnishing"] = df["Furnishing"].str.strip().str.title()
        furnishing_map = {
            "Semi Furnished": "Semi-Furnished",
            "Semi-Furnished": "Semi-Furnished",
            "Semifurnished": "Semi-Furnished",
            "Un Furnished": "Unfurnished",
            "Un-Furnished": "Unfurnished",
            "Full Furnished": "Furnished",
            "Fully Furnished": "Furnished",
        }
        df["Furnishing"] = df["Furnishing"].replace(furnishing_map)
        report["transformations"].append("Standardized Furnishing categories")

    if "Locality" in df.columns:
        df["Locality"] = df["Locality"].str.strip().str.title()
        report["transformations"].append("Standardized Locality names")

    # 5. Price outlier detection (IQR method)
    if "Price" in df.columns:
        Q1 = df["Price"].quantile(0.01)
        Q3 = df["Price"].quantile(0.99)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = (df["Price"] < lower) | (df["Price"] > upper)
        report["outliers_flagged"] = int(outliers.sum())
        # Don't remove - just flag
        df["Price_Outlier"] = outliers
        logger.info(f"Flagged {outliers.sum()} price outliers")
        report["transformations"].append("Flagged price outliers using IQR(1%-99%)")

    # 6. Area validation
    if "Area" in df.columns:
        # Flag impossibly small or large areas
        area_outliers = (df["Area"] < 50) | (df["Area"] > 20000)
        n_area_out = area_outliers.sum()
        if n_area_out > 0:
            logger.info(f"Found {n_area_out} area outliers (< 50 or > 20,000 sqft)")

    report["rows_after"] = len(df)
    report["columns_after"] = len(df.columns)

    return df, report


def preprocess_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data specifically for ML modeling.
    Drop target-derived columns to prevent data leakage.
    """
    df = df.copy()

    # Columns that are derived from the target and must NOT be used as features
    leakage_columns = [
        "Price_Per_Sqft",
        "Price_Per_Sqft_Outlier",
        "Price_Tier",
        "Price_vs_Locality_Median_Pct",
        "Price_Outlier",
    ]

    drop_cols = [c for c in leakage_columns if c in df.columns]
    if drop_cols:
        logger.info(f"Dropping leakage columns: {drop_cols}")
        df = df.drop(columns=drop_cols)

    return df


def run_preprocessing_pipeline(raw_path: Optional[str] = None) -> Tuple[pd.DataFrame, dict]:
    """Run the complete preprocessing pipeline."""
    config = load_config()
    df = load_raw_data(raw_path)

    # Validate columns
    col_report = validate_columns(df, config)

    # Clean data
    df_clean, clean_report = clean_data(df)

    # Save processed data
    processed_path = PROJECT_ROOT / "data" / "processed" / "housing_processed.csv"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(processed_path, index=False)
    logger.info(f"Saved processed data to {processed_path}")

    full_report = {
        "column_validation": col_report,
        "cleaning": clean_report,
        "output_path": str(processed_path),
    }

    return df_clean, full_report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    df, report = run_preprocessing_pipeline()
    print(f"\nPreprocessing complete: {report['cleaning']['rows_before']} -> {report['cleaning']['rows_after']} rows")
