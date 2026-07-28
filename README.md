# IsThisRentFr

IsThisRentFr is a data science project for collecting housing listings, enriching them with location-based public data, and building machine learning models to estimate a property's fair rent. The goal is to make rental prices easier to understand by showing how property characteristics, nearby amenities, safety, population, and air quality affect the predicted price.

## Project Objectives

- Build a reproducible pipeline for collecting housing data from public listing sources.
- Clean and standardize property details such as price, area, bedrooms, bathrooms, furnishing, and locality.
- Enrich each listing with geospatial and socioeconomic features.
- Explore the factors that have the strongest relationship with housing prices.
- Train and compare regression models for rent prediction.
- Provide an interpretable estimate that can help users evaluate whether a listing is reasonably priced.

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

## Proposed Workflow

1. **Data collection**  
   Collect publicly available property listings while respecting website terms, rate limits, and robots.txt rules.

2. **Data validation and cleaning**  
   Remove duplicates, normalize units and categories, handle missing values, and identify implausible prices or coordinates.

3. **Feature enrichment**  
   Use geographic coordinates to calculate distances to essential amenities and join locality-level crime, demographic, and environmental data.

4. **Exploratory data analysis**  
   Study price distributions, geographic patterns, feature correlations, outliers, and differences between property groups.

5. **Model development**  
   Establish a simple baseline and compare suitable regression models such as linear regression, random forest, gradient boosting, and XGBoost.

6. **Evaluation and interpretation**  
   Evaluate models using MAE, RMSE, and R-squared. Use feature importance or SHAP values to explain individual predictions and overall model behavior.

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

## Roadmap

- [x] Create and version the initial enriched dataset
- [ ] Add a reproducible web-scraping pipeline
- [ ] Add automated data validation and preprocessing
- [ ] Complete exploratory data analysis
- [ ] Train and compare regression models
- [ ] Add model explainability
- [ ] Build a simple rent prediction application
- [ ] Add tests and continuous integration

## License

This project is licensed under the terms included in the [LICENSE](LICENSE) file.
