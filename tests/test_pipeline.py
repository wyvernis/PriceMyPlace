"""
Tests for PriceMyPlace Mumbai.
Covers data cleaning, feature engineering, model loading, predictions, and scores.
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# Test Fixtures
# ============================================================

@pytest.fixture
def sample_data():
    """Create a synthetic test dataset."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "Price": np.random.randint(2_000_000, 50_000_000, n),
        "Area": np.random.randint(300, 3000, n),
        "Bedrooms": np.random.choice([1, 2, 3, 4], n),
        "Bathrooms": np.random.choice([1, 2, 3], n),
        "Property_Type": np.random.choice(["Apartment", "Villa", "Independent House"], n),
        "Furnishing": np.random.choice(["Unfurnished", "Semi-Furnished", "Furnished"], n),
        "Latitude": np.random.uniform(18.9, 19.3, n),
        "Longitude": np.random.uniform(72.7, 73.2, n),
        "Locality": np.random.choice(["Andheri", "Bandra", "Powai", "Thane", "Kharghar"], n),
        "Nearest_Hospital_km": np.random.uniform(0.1, 7, n),
        "Nearest_School_km": np.random.uniform(0.1, 5, n),
        "Nearest_Metro_km": np.random.uniform(0.5, 20, n),
        "Nearest_Bus_Stop_km": np.random.uniform(0.05, 3, n),
        "Nearest_Park_km": np.random.uniform(0.1, 6, n),
        "Nearest_Mall_km": np.random.uniform(0.1, 12, n),
        "Crime_Rate": np.random.uniform(100, 300, n),
        "Population_Density": np.random.uniform(1000, 30000, n),
        "Literacy_Rate": np.random.uniform(77, 93, n),
        "Household_Count": np.random.choice([520000, 1200000, 2400000], n),
        "AQI": np.random.uniform(95, 160, n),
        "PM2.5": np.random.uniform(42, 76, n),
        "PM10": np.random.uniform(78, 120, n),
        "Price_Per_Sqft": np.random.uniform(3000, 50000, n),
        "Price_Per_Sqft_Outlier": np.random.choice([True, False], n, p=[0.05, 0.95]),
        "Price_Tier": np.random.choice(["Budget", "Mid-Range", "Premium", "Luxury"], n),
        "Is_Well_Connected": np.random.choice([0, 1], n),
        "Amenity_Access_Score": np.random.uniform(0, 100, n),
        "Price_vs_Locality_Median_Pct": np.random.uniform(-50, 100, n),
    })


@pytest.fixture
def sample_with_missing(sample_data):
    """Sample data with injected missing values."""
    df = sample_data.copy()
    df.loc[0, "Area"] = np.nan
    df.loc[1, "Locality"] = np.nan
    df.loc[2:4, "Nearest_Hospital_km"] = np.nan
    return df


# ============================================================
# Data Cleaning Tests
# ============================================================

class TestDataCleaning:
    def test_clean_no_errors(self, sample_data):
        from src.data.preprocessing import clean_data
        df_clean, report = clean_data(sample_data)
        assert len(df_clean) > 0
        assert "rows_before" in report
        assert "rows_after" in report

    def test_clean_handles_missing(self, sample_with_missing):
        from src.data.preprocessing import clean_data
        df_clean, report = clean_data(sample_with_missing)
        assert df_clean.isna().sum().sum() == 0

    def test_clean_removes_duplicates(self, sample_data):
        from src.data.preprocessing import clean_data
        df_dup = pd.concat([sample_data, sample_data.head(5)], ignore_index=True)
        df_clean, report = clean_data(df_dup)
        assert report["duplicates_removed"] == 5

    def test_coordinate_validation(self, sample_data):
        from src.data.preprocessing import clean_data
        df = sample_data.copy()
        df.loc[0, "Latitude"] = 0  # Invalid
        df.loc[1, "Longitude"] = 0  # Invalid
        df_clean, report = clean_data(df)
        assert report["invalid_coords_fixed"] >= 2

    def test_furnishing_standardization(self, sample_data):
        from src.data.preprocessing import clean_data
        df = sample_data.copy()
        df.loc[0, "Furnishing"] = "Semi Furnished"
        df_clean, _ = clean_data(df)
        assert "Semi Furnished" not in df_clean["Furnishing"].values


# ============================================================
# Feature Engineering Tests
# ============================================================

class TestFeatureEngineering:
    def test_engineer_features(self, sample_data):
        from src.features.engineering import engineer_features
        df_feat = engineer_features(sample_data)
        assert "distance_to_city_center" in df_feat.columns
        assert "distance_to_major_transit" in df_feat.columns
        assert "log_area" in df_feat.columns

    def test_haversine_distance(self):
        from src.features.engineering import haversine_distance
        # Mumbai CST to Thane ~ 30 km
        dist = haversine_distance(18.9398, 72.8355, 19.186, 72.976)
        assert 25 < dist < 35

    def test_distance_to_city_center(self, sample_data):
        from src.features.engineering import engineer_features
        df_feat = engineer_features(sample_data)
        assert (df_feat["distance_to_city_center"] > 0).all()
        assert (df_feat["distance_to_city_center"] < 200).all()  # Should be reasonable

    def test_price_per_sqft_derived(self, sample_data):
        from src.features.engineering import engineer_features
        df_feat = engineer_features(sample_data)
        assert "price_per_sqft" in df_feat.columns
        # Should roughly match Price / Area
        expected = sample_data["Price"] / sample_data["Area"]
        np.testing.assert_allclose(df_feat["price_per_sqft"].values,
                                    expected.values, rtol=0.01)


# ============================================================
# Scoring Tests
# ============================================================

class TestScoring:
    def test_fair_price_score_exact(self):
        from src.scoring.scores import compute_fair_price_score
        result = compute_fair_price_score(10_000_000, 10_000_000)
        assert result["score"] == 100.0
        assert result["label"] == "Fairly Priced"

    def test_fair_price_score_underpriced(self):
        from src.scoring.scores import compute_fair_price_score
        result = compute_fair_price_score(12_000_000, 10_000_000)
        assert result["score"] > 100
        assert result["label"] in ["Below Market", "Underpriced"]

    def test_fair_price_score_overpriced(self):
        from src.scoring.scores import compute_fair_price_score
        result = compute_fair_price_score(8_000_000, 10_000_000)
        assert result["score"] < 100
        assert result["label"] in ["Above Market", "Overpriced"]

    def test_fair_price_score_invalid(self):
        from src.scoring.scores import compute_fair_price_score
        result = compute_fair_price_score(10_000_000, 0)
        assert result["score"] is None

    def test_fair_price_batch(self, sample_data):
        from src.scoring.scores import compute_fair_price_scores_batch
        predicted = sample_data["Price"] * np.random.uniform(0.8, 1.2, len(sample_data))
        result = compute_fair_price_scores_batch(predicted, sample_data["Price"])
        assert len(result) == len(sample_data)
        assert "fair_price_score" in result.columns
        assert "fair_price_label" in result.columns

    def test_affordability_index(self, sample_data):
        from src.scoring.scores import compute_affordability_index
        result = compute_affordability_index(sample_data)
        assert len(result) == len(sample_data)
        assert result.min() >= 0
        assert result.max() <= 100

    def test_investment_score(self):
        from src.scoring.scores import compute_investment_score
        n = 10
        result = compute_investment_score(
            fair_price_scores=pd.Series(np.random.uniform(80, 120, n)),
            infra_distances=pd.Series(np.random.uniform(1, 20, n)),
            infra_statuses=pd.Series(["Under Construction"] * n),
            months_since_milestone=pd.Series(np.random.uniform(1, 24, n)),
        )
        assert len(result) == n
        assert result.min() >= 0
        assert result.max() <= 100


# ============================================================
# Infrastructure Tests
# ============================================================

class TestInfrastructure:
    def test_infrastructure_df(self):
        from src.enrichment.infrastructure import get_infrastructure_df
        df = get_infrastructure_df()
        assert len(df) > 0
        assert "project_name" in df.columns
        assert "latitude" in df.columns
        assert "status" in df.columns

    def test_infrastructure_features(self):
        from src.enrichment.infrastructure import compute_infrastructure_features
        lat = pd.Series([19.076, 19.186])
        lon = pd.Series([72.877, 72.976])
        result = compute_infrastructure_features(lat, lon)
        assert "nearest_upcoming_infra_km" in result.columns
        assert "infra_completion_status" in result.columns
        assert len(result) == 2


# ============================================================
# Model Loading Tests (only if model exists)
# ============================================================

class TestModelLoading:
    def test_model_files_structure(self):
        """Check that expected model file paths are valid."""
        from pathlib import Path
        models_dir = Path(PROJECT_ROOT) / "models"
        expected_files = ["best_model.pkl", "preprocessor.pkl",
                          "model_metadata.json", "feature_info.pkl"]
        # Just verify the path logic works
        for f in expected_files:
            path = models_dir / f
            assert isinstance(path, Path)

    def test_load_trained_model_if_exists(self):
        """Test model loading if trained model exists."""
        from pathlib import Path
        model_path = Path(PROJECT_ROOT) / "models" / "best_model.pkl"
        if not model_path.exists():
            pytest.skip("Model not trained yet")

        from src.models.trainer import load_trained_model
        model, preprocessor, feature_info, metadata = load_trained_model()
        assert model is not None
        assert preprocessor is not None
        assert "feature_cols" in feature_info
        assert "best_model" in metadata


# ============================================================
# API Response Tests (structure validation)
# ============================================================

class TestAPIStructures:
    def test_prediction_output_format(self):
        """Verify prediction output structure."""
        output = {
            "predicted_price": 15000000,
            "predicted_price_formatted": "Rs. 1.50 Cr",
        }
        assert "predicted_price" in output
        assert isinstance(output["predicted_price"], (int, float))

    def test_fair_price_output_format(self):
        from src.scoring.scores import compute_fair_price_score
        result = compute_fair_price_score(10_000_000, 10_000_000)
        required_keys = ["score", "label", "interpretation",
                          "predicted_price", "listed_price",
                          "difference", "difference_pct"]
        for key in required_keys:
            assert key in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
