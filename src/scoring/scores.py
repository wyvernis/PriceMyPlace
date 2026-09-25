"""
Scoring module for PriceMyPlace Mumbai.
Implements Fair Price Score, Affordability Index, and Investment Opportunity Score.
"""

import logging
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_scoring_weights() -> dict:
    """Load scoring weights from config."""
    config_path = PROJECT_ROOT / "config" / "feature_mapping.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config.get("scoring", {})


# ============================================================
# 1. FAIR PRICE SCORE
# ============================================================

def compute_fair_price_score(predicted_price: float, listed_price: float) -> dict:
    """
    Fair Price Score = 100 * (Predicted_Price / Listed_Price)

    Interpretation:
    - ~100: Priced close to model estimate
    - >100: Listed below model estimate (potential good deal)
    - <100: Listed above model estimate (potentially overpriced)
    """
    if listed_price <= 0:
        return {"score": None, "interpretation": "Invalid listed price"}

    score = 100 * (predicted_price / listed_price)
    difference = predicted_price - listed_price
    diff_pct = ((predicted_price - listed_price) / listed_price) * 100

    if score > 115:
        interpretation = "Significantly underpriced relative to comparable properties"
        label = "Underpriced"
    elif score > 105:
        interpretation = "Somewhat below market estimate"
        label = "Below Market"
    elif score >= 95:
        interpretation = "Priced close to the model estimate for comparable properties"
        label = "Fairly Priced"
    elif score >= 85:
        interpretation = "Somewhat above market estimate"
        label = "Above Market"
    else:
        interpretation = "Significantly overpriced relative to comparable properties"
        label = "Overpriced"

    return {
        "score": round(score, 1),
        "label": label,
        "interpretation": interpretation,
        "predicted_price": predicted_price,
        "listed_price": listed_price,
        "difference": difference,
        "difference_pct": round(diff_pct, 1),
    }


def compute_fair_price_scores_batch(
    predicted_prices: pd.Series,
    listed_prices: pd.Series
) -> pd.DataFrame:
    """Compute Fair Price Scores for an entire dataset."""
    scores = 100 * (predicted_prices / listed_prices.replace(0, np.nan))

    labels = pd.cut(
        scores,
        bins=[-np.inf, 85, 95, 105, 115, np.inf],
        labels=["Overpriced", "Above Market", "Fairly Priced", "Below Market", "Underpriced"]
    )

    return pd.DataFrame({
        "fair_price_score": scores.round(1),
        "fair_price_label": labels,
        "price_difference": predicted_prices - listed_prices,
        "price_difference_pct": (((predicted_prices - listed_prices) / listed_prices) * 100).round(1),
    })


# ============================================================
# 2. AFFORDABILITY INDEX
# ============================================================

def compute_affordability_index(
    df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None
) -> pd.Series:
    """
    Affordability Index = Amenity_Access_Composite / Price_per_sqft
    Normalized to 0-100 scale.

    Components (inverse distance = closer is better):
    - 1/nearest_hospital, 1/nearest_school, 1/nearest_metro
    - 1/nearest_bus_stop, 1/nearest_park, 1/nearest_mall
    - inverse crime_rate, inverse aqi
    """
    if weights is None:
        scoring_config = load_scoring_weights()
        weights = scoring_config.get("affordability", {})

    # Define component mappings
    component_map = {
        "hospital_weight": "Nearest_Hospital_km",
        "school_weight": "Nearest_School_km",
        "metro_weight": "Nearest_Metro_km",
        "bus_stop_weight": "Nearest_Bus_Stop_km",
        "park_weight": "Nearest_Park_km",
        "mall_weight": "Nearest_Mall_km",
    }

    composite = pd.Series(0.0, index=df.index)
    total_weight = 0.0

    for weight_key, col_name in component_map.items():
        if col_name in df.columns:
            w = weights.get(weight_key, 0.15)
            # Inverse distance (closer = higher score), clamp minimum
            inv_dist = 1.0 / df[col_name].clip(lower=0.1)
            composite += w * inv_dist
            total_weight += w

    # Add crime rate (lower is better)
    if "Crime_Rate" in df.columns:
        w = weights.get("crime_weight", 0.05)
        inv_crime = 1.0 / df["Crime_Rate"].clip(lower=1.0)
        composite += w * inv_crime
        total_weight += w

    # Add AQI (lower is better)
    if "AQI" in df.columns:
        w = weights.get("aqi_weight", 0.05)
        inv_aqi = 1.0 / df["AQI"].clip(lower=1.0)
        composite += w * inv_aqi
        total_weight += w

    if total_weight > 0:
        composite /= total_weight

    # Divide by price per sqft
    if "Price" in df.columns and "Area" in df.columns:
        price_per_sqft = (df["Price"] / df["Area"].replace(0, np.nan)).clip(lower=1)
        affordability = composite / price_per_sqft
    elif "Price_Per_Sqft" in df.columns:
        affordability = composite / df["Price_Per_Sqft"].clip(lower=1)
    else:
        affordability = composite

    # Normalize to 0-100
    min_val = affordability.min()
    max_val = affordability.max()
    if max_val > min_val:
        affordability = ((affordability - min_val) / (max_val - min_val)) * 100
    else:
        affordability = pd.Series(50.0, index=df.index)

    return affordability.round(1)


# ============================================================
# 3. INVESTMENT OPPORTUNITY SCORE
# ============================================================

def compute_investment_score(
    fair_price_scores: pd.Series,
    infra_distances: pd.Series,
    infra_statuses: pd.Series,
    months_since_milestone: pd.Series,
    weights: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """
    Investment Opportunity Score (Exploratory):

    Score = w1 * valuation_component
          + w2 * infrastructure_proximity
          + w3 * infrastructure_horizon

    This is an exploratory analytical indicator, NOT financial advice.
    """
    if weights is None:
        scoring_config = load_scoring_weights()
        weights = scoring_config.get("investment", {})

    w1 = weights.get("valuation_weight", 0.40)
    w2 = weights.get("infrastructure_proximity_weight", 0.35)
    w3 = weights.get("infrastructure_horizon_weight", 0.25)

    # Component 1: Valuation (higher fair_price_score = more underpriced = better investment signal)
    # Normalize to 0-100 scale
    valuation = (fair_price_scores - fair_price_scores.min()) / \
                (fair_price_scores.max() - fair_price_scores.min() + 1e-10) * 100

    # Component 2: Infrastructure proximity (closer = higher score)
    max_dist = infra_distances.max()
    if max_dist > 0:
        proximity = (1 - infra_distances / max_dist) * 100
    else:
        proximity = pd.Series(50.0, index=fair_price_scores.index)

    # Component 3: Infrastructure horizon
    # Higher for "Under Construction" (price hasn't fully absorbed the effect yet)
    status_score_map = {
        "Planned": 80,
        "Under Construction": 100,
        "Completed": 40,
        "Unknown": 20,
    }
    horizon = infra_statuses.map(status_score_map).fillna(20)

    # Adjust by recency (more recent milestones = higher score)
    if months_since_milestone is not None:
        recency_factor = np.clip(1.0 - months_since_milestone.fillna(60) / 120, 0, 1)
        horizon = horizon * (0.5 + 0.5 * recency_factor)

    # Combine
    score = w1 * valuation + w2 * proximity + w3 * horizon

    # Normalize to 0-100
    min_s = score.min()
    max_s = score.max()
    if max_s > min_s:
        score = ((score - min_s) / (max_s - min_s)) * 100
    else:
        score = pd.Series(50.0, index=fair_price_scores.index)

    return score.round(1)
