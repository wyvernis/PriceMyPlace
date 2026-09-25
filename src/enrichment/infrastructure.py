"""
Mumbai / Navi Mumbai infrastructure dataset and enrichment.
Contains major infrastructure projects for the Investment Opportunity Score.
All data sourced from publicly available government/news sources.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple

logger = logging.getLogger(__name__)

# Major Mumbai / Navi Mumbai Infrastructure Projects
# Sources: MMRDA, CIDCO, Government press releases, major news outlets
INFRASTRUCTURE_PROJECTS = [
    {
        "project_name": "Mumbai Trans Harbour Link (Atal Setu)",
        "latitude": 19.0178,
        "longitude": 72.9920,
        "project_type": "Bridge/Highway",
        "status": "Completed",
        "announcement_date": "2014-01-01",
        "construction_start": "2018-04-01",
        "expected_completion": "2024-01-01",
        "actual_completion": "2024-01-12",
        "last_major_milestone": "2024-01-12",
        "source": "MMRDA / Government of Maharashtra"
    },
    {
        "project_name": "Navi Mumbai International Airport (NMIA)",
        "latitude": 18.9220,
        "longitude": 73.1160,
        "project_type": "Airport",
        "status": "Under Construction",
        "announcement_date": "2007-01-01",
        "construction_start": "2020-02-01",
        "expected_completion": "2025-12-01",
        "actual_completion": None,
        "last_major_milestone": "2024-06-01",
        "source": "CIDCO / Adani Group"
    },
    {
        "project_name": "Mumbai Metro Line 3 (Colaba-Bandra-SEEPZ)",
        "latitude": 19.0630,
        "longitude": 72.8680,
        "project_type": "Metro Rail",
        "status": "Under Construction",
        "announcement_date": "2014-08-01",
        "construction_start": "2016-10-01",
        "expected_completion": "2025-06-01",
        "actual_completion": None,
        "last_major_milestone": "2024-10-07",
        "source": "MMRDA"
    },
    {
        "project_name": "Mumbai Metro Line 2A (Dahisar-DN Nagar)",
        "latitude": 19.1860,
        "longitude": 72.8360,
        "project_type": "Metro Rail",
        "status": "Completed",
        "announcement_date": "2016-01-01",
        "construction_start": "2018-09-01",
        "expected_completion": "2023-06-01",
        "actual_completion": "2024-04-06",
        "last_major_milestone": "2024-04-06",
        "source": "MMRDA"
    },
    {
        "project_name": "Mumbai Metro Line 7 (Dahisar E-Andheri E)",
        "latitude": 19.1530,
        "longitude": 72.8640,
        "project_type": "Metro Rail",
        "status": "Completed",
        "announcement_date": "2016-01-01",
        "construction_start": "2018-09-01",
        "expected_completion": "2023-06-01",
        "actual_completion": "2024-04-06",
        "last_major_milestone": "2024-04-06",
        "source": "MMRDA"
    },
    {
        "project_name": "Mumbai Metro Line 4 (Wadala-Ghatkopar-Mulund-Thane)",
        "latitude": 19.1050,
        "longitude": 72.9150,
        "project_type": "Metro Rail",
        "status": "Under Construction",
        "announcement_date": "2018-01-01",
        "construction_start": "2019-09-01",
        "expected_completion": "2026-12-01",
        "actual_completion": None,
        "last_major_milestone": "2024-06-01",
        "source": "MMRDA"
    },
    {
        "project_name": "Navi Mumbai Metro Line 1 (Belapur-Pendhar)",
        "latitude": 19.0230,
        "longitude": 73.0400,
        "project_type": "Metro Rail",
        "status": "Under Construction",
        "announcement_date": "2011-01-01",
        "construction_start": "2019-11-01",
        "expected_completion": "2025-12-01",
        "actual_completion": None,
        "last_major_milestone": "2024-03-01",
        "source": "CIDCO"
    },
    {
        "project_name": "Mumbai Coastal Road (South)",
        "latitude": 18.9640,
        "longitude": 72.8100,
        "project_type": "Road/Highway",
        "status": "Completed",
        "announcement_date": "2015-01-01",
        "construction_start": "2018-10-01",
        "expected_completion": "2023-12-01",
        "actual_completion": "2024-06-10",
        "last_major_milestone": "2024-06-10",
        "source": "BMC"
    },
    {
        "project_name": "Mumbai Coastal Road (Bandra-Versova Sea Link)",
        "latitude": 19.0800,
        "longitude": 72.8170,
        "project_type": "Road/Highway",
        "status": "Under Construction",
        "announcement_date": "2017-01-01",
        "construction_start": "2021-01-01",
        "expected_completion": "2028-01-01",
        "actual_completion": None,
        "last_major_milestone": "2024-05-01",
        "source": "MSRDC"
    },
    {
        "project_name": "Goregaon-Mulund Link Road (GMLR)",
        "latitude": 19.1640,
        "longitude": 72.9000,
        "project_type": "Road/Highway",
        "status": "Under Construction",
        "announcement_date": "2019-01-01",
        "construction_start": "2022-04-01",
        "expected_completion": "2027-01-01",
        "actual_completion": None,
        "last_major_milestone": "2024-01-01",
        "source": "BMC/MMRDA"
    },
    {
        "project_name": "Virar-Alibaug Multimodal Corridor",
        "latitude": 19.1000,
        "longitude": 72.9500,
        "project_type": "Road/Highway",
        "status": "Under Construction",
        "announcement_date": "2017-01-01",
        "construction_start": "2021-12-01",
        "expected_completion": "2028-06-01",
        "actual_completion": None,
        "last_major_milestone": "2024-02-01",
        "source": "MMRDA"
    },
    {
        "project_name": "Mumbai-Ahmedabad High Speed Rail (Bullet Train)",
        "latitude": 19.0603,
        "longitude": 72.8368,
        "project_type": "Rail",
        "status": "Under Construction",
        "announcement_date": "2015-12-01",
        "construction_start": "2020-06-01",
        "expected_completion": "2028-12-01",
        "actual_completion": None,
        "last_major_milestone": "2024-08-01",
        "source": "NHSRCL"
    },
]


def get_infrastructure_df() -> pd.DataFrame:
    """Return the infrastructure projects as a DataFrame."""
    df = pd.DataFrame(INFRASTRUCTURE_PROJECTS)
    date_cols = ["announcement_date", "construction_start", "expected_completion",
                 "actual_completion", "last_major_milestone"]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def compute_infrastructure_features(
    prop_lat: pd.Series,
    prop_lon: pd.Series,
    reference_date: datetime = None
) -> pd.DataFrame:
    """
    Compute infrastructure-related features for each property.

    Returns DataFrame with:
    - nearest_upcoming_infra_km
    - infra_completion_status
    - months_since_milestone
    - nearest_infra_name
    """
    from src.features.engineering import haversine_distance

    if reference_date is None:
        reference_date = datetime.now()

    infra_df = get_infrastructure_df()

    results = {
        "nearest_upcoming_infra_km": [],
        "infra_completion_status": [],
        "months_since_milestone": [],
        "nearest_infra_name": [],
    }

    for idx in range(len(prop_lat)):
        lat = prop_lat.iloc[idx]
        lon = prop_lon.iloc[idx]

        min_dist = float("inf")
        best_project = None

        for _, proj in infra_df.iterrows():
            dist = haversine_distance(lat, lon, proj["latitude"], proj["longitude"])
            if dist < min_dist:
                min_dist = dist
                best_project = proj

        results["nearest_upcoming_infra_km"].append(round(min_dist, 2))
        results["nearest_infra_name"].append(best_project["project_name"] if best_project is not None else "Unknown")
        results["infra_completion_status"].append(best_project["status"] if best_project is not None else "Unknown")

        if best_project is not None and pd.notna(best_project["last_major_milestone"]):
            delta = reference_date - best_project["last_major_milestone"]
            results["months_since_milestone"].append(round(delta.days / 30.44, 1))
        else:
            results["months_since_milestone"].append(None)

    return pd.DataFrame(results, index=prop_lat.index)
