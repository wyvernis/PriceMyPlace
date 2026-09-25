"""
Feature engineering module for PriceMyPlace Mumbai.
Creates derived features for the ML pipeline.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Optional

logger = logging.getLogger(__name__)

# Mumbai CST / Fort area (city center)
MUMBAI_CENTER_LAT = 18.9398
MUMBAI_CENTER_LON = 72.8355

# Major transit hubs
MAJOR_TRANSIT = {
    "CST": (18.9398, 72.8355),
    "Dadar": (19.0178, 72.8478),
    "Bandra": (19.0544, 72.8402),
    "Andheri": (19.1197, 72.8468),
    "Borivali": (19.2288, 72.8567),
    "Thane": (19.1860, 72.9757),
    "Panvel": (18.9894, 73.1175),
    "Vashi": (19.0771, 73.0003),
}

# Major railway stations
RAILWAY_STATIONS = {
    "Mumbai CST": (18.9398, 72.8355),
    "Churchgate": (18.9352, 72.8272),
    "Dadar": (19.0178, 72.8478),
    "Kurla LTT": (19.0726, 72.8794),
    "Bandra Terminus": (19.0544, 72.8402),
    "Thane": (19.1860, 72.9757),
    "Kalyan": (19.2437, 73.1355),
    "Panvel": (18.9894, 73.1175),
}

# Metro stations (operational as of 2024-2025)
METRO_STATIONS = {
    "Ghatkopar Metro": (19.0863, 72.9085),
    "Andheri Metro": (19.1197, 72.8468),
    "DN Nagar Metro": (19.1265, 72.8346),
    "Versova Metro": (19.1321, 72.8173),
    "WEH Metro": (19.1087, 72.8570),
    "Chakala Metro": (19.1135, 72.8580),
    "Airport Road Metro": (19.0998, 72.8655),
    "Marol Naka Metro": (19.1021, 72.8804),
    "Saki Naka Metro": (19.0904, 72.8883),
    "Asalpha Metro": (19.0880, 72.8982),
}


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points in km."""
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def compute_distance_to_nearest(lat_series: pd.Series, lon_series: pd.Series,
                                 locations: dict) -> pd.Series:
    """Compute distance to the nearest location from a dictionary of named locations."""
    min_distances = pd.Series(np.inf, index=lat_series.index)
    for name, (ref_lat, ref_lon) in locations.items():
        dist = haversine_distance(lat_series, lon_series, ref_lat, ref_lon)
        min_distances = np.minimum(min_distances, dist)
    return min_distances


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create all derived features for the ML pipeline."""
    df = df.copy()
    logger.info("Starting feature engineering...")

    # ---------- Property-based features ----------
    if "Price" in df.columns and "Area" in df.columns:
        df["price_per_sqft"] = df["Price"] / df["Area"].replace(0, np.nan)
        logger.info("Created: price_per_sqft")

    if "Price" in df.columns and "Bedrooms" in df.columns:
        df["price_per_bhk"] = df["Price"] / df["Bedrooms"].replace(0, np.nan)
        logger.info("Created: price_per_bhk")

    if "Price" in df.columns and "Bathrooms" in df.columns:
        df["price_per_bathroom"] = df["Price"] / df["Bathrooms"].replace(0, np.nan)
        logger.info("Created: price_per_bathroom")

    if "Area" in df.columns and "Bedrooms" in df.columns:
        df["area_per_bhk"] = df["Area"] / df["Bedrooms"].replace(0, np.nan)
        logger.info("Created: area_per_bhk")

    if "Bedrooms" in df.columns and "Bathrooms" in df.columns:
        df["bhk_bathroom_ratio"] = df["Bedrooms"] / df["Bathrooms"].replace(0, np.nan)
        logger.info("Created: bhk_bathroom_ratio")

    # Log transforms for skewed distributions
    if "Price" in df.columns:
        df["log_price"] = np.log1p(df["Price"])
        logger.info("Created: log_price")

    if "Area" in df.columns:
        df["log_area"] = np.log1p(df["Area"])
        logger.info("Created: log_area")

    # ---------- Geographic features ----------
    if "Latitude" in df.columns and "Longitude" in df.columns:
        # Distance to city center (CST)
        df["distance_to_city_center"] = haversine_distance(
            df["Latitude"], df["Longitude"],
            MUMBAI_CENTER_LAT, MUMBAI_CENTER_LON
        )
        logger.info("Created: distance_to_city_center")

        # Distance to nearest major transit hub
        df["distance_to_major_transit"] = compute_distance_to_nearest(
            df["Latitude"], df["Longitude"], MAJOR_TRANSIT
        )
        logger.info("Created: distance_to_major_transit")

        # Distance to nearest railway station
        df["distance_to_railway"] = compute_distance_to_nearest(
            df["Latitude"], df["Longitude"], RAILWAY_STATIONS
        )
        logger.info("Created: distance_to_railway")

        # Distance to nearest metro station
        df["distance_to_metro_station"] = compute_distance_to_nearest(
            df["Latitude"], df["Longitude"], METRO_STATIONS
        )
        logger.info("Created: distance_to_metro_station")

    # ---------- Locality-level features ----------
    if "Locality" in df.columns and "Price" in df.columns:
        locality_stats = df.groupby("Locality")["Price"].agg(
            locality_median_price="median",
            locality_mean_price="mean",
            locality_listing_count="count",
            locality_price_std="std",
        )
        df = df.merge(locality_stats, on="Locality", how="left")
        logger.info("Created: locality_median_price, locality_mean_price, locality_listing_count, locality_price_std")

    if "Locality" in df.columns and "Area" in df.columns:
        area_stats = df.groupby("Locality")["Area"].agg(
            locality_median_area="median",
        )
        df = df.merge(area_stats, on="Locality", how="left")
        logger.info("Created: locality_median_area")

    # ---------- Interaction features ----------
    if "Area" in df.columns and "distance_to_city_center" in df.columns:
        df["area_x_distance"] = df["Area"] * df["distance_to_city_center"]
        logger.info("Created: area_x_distance")

    logger.info(f"Feature engineering complete. {len(df.columns)} total columns.")
    return df


def get_feature_columns(df: pd.DataFrame, target_col: str = "Price") -> List[str]:
    """
    Get the list of feature columns for modeling.
    Excludes target, IDs, leakage columns, and log_price.
    """
    exclude = {
        target_col, "log_price",
        "Price_Per_Sqft", "Price_Per_Sqft_Outlier", "Price_Tier",
        "Price_vs_Locality_Median_Pct", "Price_Outlier",
        "price_per_sqft", "price_per_bhk", "price_per_bathroom",
        # Locality stats derived from target
        "locality_median_price", "locality_mean_price", "locality_price_std",
    }

    feature_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if df[col].dtype in [np.float64, np.int64, float, int, bool, np.bool_]:
            feature_cols.append(col)
        elif df[col].dtype in [object, "string", "str"]:
            # Will be encoded later
            feature_cols.append(col)

    return feature_cols
