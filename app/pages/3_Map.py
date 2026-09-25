"""
Interactive Map Page — PriceMyPlace Mumbai
Geospatial visualization of property prices and heatmaps.
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
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
from pathlib import Path

st.set_page_config(page_title="Map | PriceMyPlace", page_icon="🗺️", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

from app.utils import (
    get_project_root,
    load_processed_data,
    format_inr,
)

@st.cache_data
def load_data():
    return load_processed_data()


def create_map(df, map_type="markers"):
    """Create folium map."""
    center_lat = df["Latitude"].median()
    center_lon = df["Longitude"].median()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="cartodbpositron",
    )

    if map_type == "markers":
        marker_cluster = MarkerCluster().add_to(m)
        # Limit to 2000 for performance
        sample = df.head(2000) if len(df) > 2000 else df

        for _, row in sample.iterrows():
            price_sqft = row["Price"] / row["Area"] if row["Area"] > 0 else 0
            fair_score = row.get("fair_price_score", "N/A")
            afford = row.get("affordability_index", "N/A")

            popup_html = f"""
            <div style="font-family: Inter, sans-serif; font-size: 12px; min-width: 200px;">
                <h4 style="margin: 0 0 8px 0; color: #1e293b;">{row['Locality']}</h4>
                <table style="width: 100%;">
                    <tr><td><b>Price:</b></td><td>{format_inr(row['Price'])}</td></tr>
                    <tr><td><b>Area:</b></td><td>{row['Area']:,.0f} sqft</td></tr>
                    <tr><td><b>BHK:</b></td><td>{int(row['Bedrooms'])}</td></tr>
                    <tr><td><b>Rs/sqft:</b></td><td>Rs.{price_sqft:,.0f}</td></tr>
                    <tr><td><b>Type:</b></td><td>{row['Property_Type']}</td></tr>
                    <tr><td><b>Furnishing:</b></td><td>{row['Furnishing']}</td></tr>
                    <tr><td><b>Fair Score:</b></td><td>{fair_score}</td></tr>
                    <tr><td><b>Affordability:</b></td><td>{afford}</td></tr>
                </table>
            </div>
            """

            # Color based on price tier
            price = row["Price"]
            if price < 5_000_000:
                color = "green"
            elif price < 15_000_000:
                color = "blue"
            elif price < 30_000_000:
                color = "orange"
            else:
                color = "red"

            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{row['Locality']} - {format_inr(row['Price'])}",
            ).add_to(marker_cluster)

    elif map_type == "heatmap":
        heat_data = df[["Latitude", "Longitude", "Price"]].dropna()
        # Normalize prices for heatmap
        heat_data["weight"] = (heat_data["Price"] - heat_data["Price"].min()) / \
                               (heat_data["Price"].max() - heat_data["Price"].min())
        HeatMap(
            heat_data[["Latitude", "Longitude", "weight"]].values.tolist(),
            radius=15, blur=20, max_zoom=13,
        ).add_to(m)

    elif map_type == "locality_avg":
        locality_data = df.groupby("Locality").agg({
            "Latitude": "median",
            "Longitude": "median",
            "Price": ["median", "count"],
        })
        locality_data.columns = ["lat", "lon", "median_price", "count"]

        for loc, row in locality_data.iterrows():
            radius = max(5, min(20, row["count"] / 10))
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=radius,
                color="#6366f1",
                fill=True,
                fill_color="#6366f1",
                fill_opacity=0.6,
                popup=f"<b>{loc}</b><br>Median: {format_inr(row['median_price'])}<br>Listings: {int(row['count'])}",
                tooltip=f"{loc}: {format_inr(row['median_price'])}",
            ).add_to(m)

    # Legend
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 999;
                background: white; padding: 12px 16px; border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1); font-size: 12px; font-family: Inter;">
        <b>Price Range</b><br>
        <span style="color: green;">●</span> < 50L<br>
        <span style="color: blue;">●</span> 50L - 1.5Cr<br>
        <span style="color: orange;">●</span> 1.5Cr - 3Cr<br>
        <span style="color: red;">●</span> > 3Cr
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


def main():
    st.markdown("# 🗺️ Interactive Property Map")
    st.markdown("*Explore Mumbai & Navi Mumbai properties geographically*")
    st.markdown("---")

    df = load_data()
    if df is None:
        st.error("No data available.")
        return

    # Filters
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)

    with col_f1:
        map_type = st.selectbox("Map Type", ["markers", "heatmap", "locality_avg"],
                                format_func=lambda x: {"markers": "Property Markers",
                                                        "heatmap": "Price Heatmap",
                                                        "locality_avg": "Locality Averages"}[x])

    with col_f2:
        localities = ["All"] + sorted(df["Locality"].unique().tolist())
        sel_locality = st.selectbox("Locality", localities)

    with col_f3:
        bhk_options = ["All"] + sorted(df["Bedrooms"].unique().tolist())
        sel_bhk = st.selectbox("BHK", bhk_options)

    with col_f4:
        price_range = st.slider(
            "Price Range (Lakhs)",
            min_value=int(df["Price"].min() / 100000),
            max_value=min(int(df["Price"].max() / 100000), 5000),
            value=(int(df["Price"].min() / 100000), min(int(df["Price"].quantile(0.95) / 100000), 5000)),
        )

    # Apply filters
    filtered_df = df.copy()
    if sel_locality != "All":
        filtered_df = filtered_df[filtered_df["Locality"] == sel_locality]
    if sel_bhk != "All":
        filtered_df = filtered_df[filtered_df["Bedrooms"] == int(sel_bhk)]
    filtered_df = filtered_df[
        (filtered_df["Price"] >= price_range[0] * 100000) &
        (filtered_df["Price"] <= price_range[1] * 100000)
    ]

    st.caption(f"Showing {len(filtered_df):,} properties")

    if len(filtered_df) == 0:
        st.warning("No properties match the selected filters.")
        return

    # Create and display map
    m = create_map(filtered_df, map_type)
    st_folium(m, width=None, height=600, use_container_width=True)


if __name__ == "__main__":
    main()
