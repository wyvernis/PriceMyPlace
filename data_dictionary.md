# PriceMyPlace Mumbai — Data Dictionary

## Dataset Overview
- **Source**: Mumbai/Navi Mumbai property listing data
- **Records**: 5,994
- **Features**: 28 columns (22 original + 6 pre-computed derived)
- **Target**: `Price` (property price in INR)
- **Geographic Coverage**: Mumbai, Navi Mumbai, and extended Mumbai Metropolitan Region
- **Missing Values**: 0
- **Duplicate Records**: 0

---

## Column Descriptions

### Property Features (6 columns)

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `Price` | int64 | 11.5L - 42Cr | Property price in INR **(TARGET)** |
| `Area` | int64 | 190 - 14,700 | Property area in square feet |
| `Bedrooms` | int64 | 1 - 10 | Number of bedrooms (BHK) |
| `Bathrooms` | int64 | 1 - 10 | Number of bathrooms |
| `Property_Type` | string | 4 values | Apartment (99.1%), Villa, Independent House, Independent Floor |
| `Furnishing` | string | 3 values | Unfurnished (58.1%), Semi-Furnished (35.3%), Furnished (6.6%) |

### Location Features (3 columns)

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `Latitude` | float64 | 18.87 - 19.83 | Geographic latitude |
| `Longitude` | float64 | 72.72 - 73.53 | Geographic longitude |
| `Locality` | string | 113 values | Neighborhood name (top: Mira Road, Thane, Kandivali, Andheri, Kharghar) |

### Amenity Distances (6 columns)

| Column | Type | Range (km) | Description |
|--------|------|------------|-------------|
| `Nearest_Hospital_km` | float64 | 0.01 - 7.43 | Distance to nearest hospital |
| `Nearest_School_km` | float64 | 0.03 - 4.72 | Distance to nearest school |
| `Nearest_Metro_km` | float64 | 0.45 - 22.21 | Distance to nearest metro station |
| `Nearest_Bus_Stop_km` | float64 | 0.03 - 3.39 | Distance to nearest bus stop |
| `Nearest_Park_km` | float64 | 0.07 - 6.69 | Distance to nearest park |
| `Nearest_Mall_km` | float64 | 0.03 - 12.79 | Distance to nearest mall |

### Demographics & Safety (4 columns)

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `Crime_Rate` | float64 | 106.3 - 306.5 | Locality-level crime rate per 100K |
| `Population_Density` | float64 | 1,020 - 31,028 | Population per sq km |
| `Literacy_Rate` | float64 | 77.0 - 93.0 | Literacy rate (%) |
| `Household_Count` | float64 | 520K - 2.4M | Number of households in locality |

### Air Quality (3 columns)

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `AQI` | float64 | 95.1 - 159.9 | Air Quality Index |
| `PM2.5` | float64 | 42.0 - 76.0 | Fine particulate matter (ug/m3) |
| `PM10` | float64 | 78.0 - 120.0 | Coarse particulate matter (ug/m3) |

### Pre-computed Derived Features (6 columns)

| Column | Type | Description |
|--------|------|-------------|
| `Price_Per_Sqft` | float64 | Price / Area |
| `Price_Per_Sqft_Outlier` | bool | Whether price/sqft is an outlier |
| `Price_Tier` | string | Budget / Mid-Range / Premium / Luxury (25% each) |
| `Is_Well_Connected` | int64 | Binary connectivity flag (21.3% connected) |
| `Amenity_Access_Score` | float64 | Composite amenity access score (0-100) |
| `Price_vs_Locality_Median_Pct` | float64 | % deviation from locality median price |

---

## Important Notes

1. **Price is property sale price**, not monthly rent — despite the original project's rent framing
2. The dataset covers Mumbai, Navi Mumbai, Thane, Kalyan, Mira Road, Vasai/Virar, and other MMR areas
3. 99.2% of coordinates fall within the Mumbai metropolitan region
4. 99.1% of properties are apartments
5. Pre-computed derived features (last 6 columns) are excluded from model training to prevent data leakage
