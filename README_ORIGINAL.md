# PriceMyPlace

PriceMyPlace is a data science project for collecting housing listings, enriching them with location-based public data, and building machine learning models to estimate a property's fair rent. The goal is to make rental prices easier to understand by showing how property characteristics, nearby amenities, safety, population, and air quality affect the predicted price — and to go a step further by telling users whether a specific listing is *fairly priced*, how *affordable* it is relative to what it offers, and whether it looks like a good *investment* given upcoming infrastructure in the area.

## Project Objectives

- Build a reproducible pipeline for collecting housing data from public listing sources.
- Clean and standardize property details such as price, area, bedrooms, bathrooms, furnishing, and locality.
- Enrich each listing with geospatial and socioeconomic features.
- Explore the factors that have the strongest relationship with housing prices.
- Train and compare regression models for rent prediction.
- Provide an interpretable estimate that can help users evaluate whether a listing is reasonably priced.
- **Compute a Fair Price Score** for each listing — how its asking price compares to the model's predicted fair value.
- **Compute an Affordability Index** — how much amenity access (hospitals, schools, transit, parks) a listing offers per rupee, relative to other localities.
- **Compute an Investment Opportunity Score** — a forward-looking signal combining current under/overvaluation with proximity to infrastructure projects (e.g. the Mumbai Trans Harbour Link, Navi Mumbai International Airport, upcoming metro corridors) that are known to move prices in the coming years.

## Dataset

The current version contains 6,037 housing records and 22 columns. The raw dataset is versioned with DVC rather than stored directly in Git.

The features are grouped as follows:

| Category | Features |
| --- | --- |
| Property | `Price`, `Area`, `Bedrooms`, `Bathrooms`, `Property_Type`, `Furnishing` |
| Location | `Latitude`, `Longitude`, `Locality` |
| Nearby amenities | `Nearest_Hospital_km`, `Nearest_School_km`, `Nearest_Metro_km`, `Nearest_Bus_Stop_km`, `Nearest_Park_km`, `Nearest_Mall_km` |
| Demographics and safety | `Crime_Rate`, `Population_Density`, `Literacy_Rate`, `Household_Count` |
| Air quality | `AQI`, `PM2.5`, `PM10` |

`Price` is the prediction target. The remaining columns are candidate model features.

### Planned additions for scoring features

To support the Investment Opportunity Score specifically, the enrichment stage will add a small set of infrastructure-timeline features that go beyond static distance:

| Feature | Description |
| --- | --- |
| `Nearest_Upcoming_Infra_km` | Distance to the nearest major infrastructure project under construction or recently completed (MTHL/Atal Setu, Navi Mumbai International Airport, upcoming metro lines) |
| `Infra_Completion_Status` | Categorical: completed / under construction / planned, per project |
| `Months_Since_Milestone` | Months since the nearest relevant project's last major milestone (e.g. opening date) — captures the "still-catching-up" price effect after a project opens |

These are genuinely harder to source than static amenity distances (no single API provides them — they require compiling known project locations and timelines from government/news sources) and are what differentiate the Investment Opportunity Score from a plain valuation model.

## Derived Insights: Fair Price, Affordability, and Investment Scores

These three scores are computed **after** the base regression model is trained, using its predictions and residuals rather than being separate models trained from scratch.

### 1. Fair Price Score
For a given listing, compare its actual asking price against the model's predicted price for a property with identical characteristics and location:

```
Fair_Price_Score = 100 x (Predicted_Price / Listed_Price)
```

- Score ≈ 100 → priced in line with comparable properties.
- Score > 100 → listing is priced *below* what comparable properties suggest (potentially a good deal).
- Score < 100 → listing is priced *above* comparable properties (potentially overpriced).

This is reported alongside the model's typical error margin (e.g. ±MAPE) so a user doesn't over-read small differences as a genuine "deal."

### 2. Affordability Index
Distinct from Fair Price Score, this measures how much *neighborhood value* a listing offers relative to its price, not just how the price compares to similar listings:

```
Amenity_Access_Composite = weighted combination of
    (1 / Nearest_Hospital_km), (1 / Nearest_School_km), (1 / Nearest_Metro_km),
    (1 / Nearest_Bus_Stop_km), (1 / Nearest_Park_km), (1 / Nearest_Mall_km),
    inverse Crime_Rate, inverse AQI

Affordability_Index = Amenity_Access_Composite / Price_per_sqft
```

Normalized to a 0-100 scale across the dataset. A high Affordability Index flags localities/listings that offer strong amenity access, safety, and air quality relative to their price — useful for surfacing underrated areas rather than just the cheapest listings, which are often cheap *because* they lack access to amenities.

### 3. Investment Opportunity Score
Combines current undervaluation with forward-looking infrastructure exposure:

```
Investment_Opportunity_Score = w1 x (100 - Fair_Price_Score)
                              + w2 x Infra_Proximity_Weight
                              + w3 x Infra_Completion_Horizon_Weight
```

Where `Infra_Proximity_Weight` rewards closeness to a major infrastructure project, and `Infra_Completion_Horizon_Weight` is higher for projects that are under-construction-but-not-yet-open (the period before a completed project's price effect has fully priced in) than for projects completed years ago. Weights (`w1, w2, w3`) will be tuned against any available historical price-trend data by locality; until that calibration step, they start as equal weights and are documented as such.

**This score is exploratory and directional, not a financial recommendation** — see Responsible Use below.

## Proposed Workflow

1. **Data collection**  
   Collect publicly available property listings while respecting website terms, rate limits, and robots.txt rules.

2. **Data validation and cleaning**  
   Remove duplicates, normalize units and categories, handle missing values, and identify implausible prices or coordinates.

3. **Feature enrichment**  
   Use geographic coordinates to calculate distances to essential amenities, join locality-level crime, demographic, and environmental data, and compile the infrastructure-timeline features listed above.

4. **Exploratory data analysis**  
   Study price distributions, geographic patterns, feature correlations, outliers, and differences between property groups.

5. **Model development**  
   Establish a simple baseline and compare suitable regression models such as linear regression, random forest, gradient boosting, and XGBoost.

6. **Evaluation and interpretation**  
   Evaluate models using MAE, RMSE, and R-squared. Use feature importance or SHAP values to explain individual predictions and overall model behavior.

7. **Score generation**  
   Using the trained model's predictions and residuals, compute the Fair Price Score, Affordability Index, and Investment Opportunity Score for every listing, and validate that scores behave sensibly (e.g. spot-check known up-and-coming vs. established localities).

## Data Versioning with DVC

Install DVC and retrieve the versioned dataset:

```bash
pip install dvc
dvc pull
```

The dataset metadata is stored in `final merged datasets.csv.dvc`. A DVC storage remote must be configured before `dvc pull` can download the data on a new machine.

To update the tracked dataset:

```bash
dvc add "final merged datasets.csv"
git add "final merged datasets.csv.dvc" .gitignore
git commit -m "Update raw dataset"
dvc push
```

## Getting Started

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/wyvernis/IsThisRentFr.git
cd IsThisRentFr
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

The analysis, preprocessing, training pipeline, dependency file, and prediction interface will be added as the project develops.

## Responsible Use

- Collect only data that is publicly accessible and permitted for automated use.
- Avoid retaining personal or sensitive information from listings.
- Treat crime and demographic variables carefully because they may encode historical or social bias.
- Present predictions as estimates, not guaranteed market values.
- Monitor model performance across localities and price ranges before deployment.
- **The Fair Price Score, Affordability Index, and Investment Opportunity Score are informational and educational outputs, not financial or investment advice.** They are derived from historical listing data and public infrastructure timelines, which can change, be delayed, or be reported inaccurately; they do not account for a user's individual financial situation, and past amenity/infrastructure patterns are not a guarantee of future price movement. Users making buying, renting, or investment decisions should verify current information independently and consult a qualified professional.
- Because the Investment Opportunity Score weights localities near planned/under-construction infrastructure, monitor it for compounding any existing valuation bias in the crime/demographic inputs (e.g. don't let it systematically favor already-affluent areas simply because more public data exists for them).

## Roadmap

- [x] Create and version the initial enriched dataset
- [ ] Add a reproducible web-scraping pipeline
- [ ] Add automated data validation and preprocessing
- [ ] Complete exploratory data analysis
- [ ] Compile infrastructure-timeline dataset (MTHL, Navi Mumbai Airport, upcoming metro lines)
- [ ] Train and compare regression models
- [ ] Add model explainability
- [ ] Compute Fair Price Score, Affordability Index, and Investment Opportunity Score
- [ ] Build a simple rent prediction application
- [ ] Add tests and continuous integration

## License

This project is licensed under the terms included in the [LICENSE](LICENSE) file.
