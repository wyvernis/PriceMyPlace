# Experiment 1: Case Study Framing and Dataset Preparation

## Aim

To define a real-world housing price prediction problem, study existing solutions, identify opportunities for improvement, acquire and describe a suitable dataset, and establish reproducible data versioning using Git and DVC.

## Objective

The objective of this experiment is to prepare the foundation for **IsThisRentFr**, a data science project that estimates a fair housing price from property, location, amenity, demographic, safety, and environmental attributes. The experiment covers problem formulation, solution benchmarking, success criteria, dataset preparation, and version control.

## 1. Problem Statement

Finding a fairly priced home is difficult because housing prices depend on many interacting factors. Area, number of rooms, furnishing, property type, and locality are obvious influences, but accessibility and neighborhood conditions also matter. A property close to public transport, schools, hospitals, parks, and shopping facilities may be valued differently from a similar property in a less accessible location. Population density, reported crime, and air quality may further affect demand and quality of life.

Most property platforms allow users to search and filter listings, but the displayed price is generally supplied by an owner, broker, or developer. A prospective tenant or buyer may therefore have limited evidence for deciding whether that price is reasonable. Comparisons based only on locality and bedroom count can also overlook differences in floor area, furnishing, transport access, and surrounding conditions.

IsThisRentFr addresses this problem by developing a reproducible machine learning pipeline that predicts a property's expected price from a richer set of attributes. Housing records are combined with geographic coordinates and contextual information, including distances to nearby amenities, crime rate, demographic indicators, and air pollution measurements. Exploratory analysis will be used to understand price patterns and detect data quality problems. Multiple regression algorithms will then be trained and evaluated using the same validation strategy.

The intended users are people comparing housing options, analysts studying local property markets, and developers building decision-support tools. The result should be an interpretable estimate rather than a guaranteed market valuation. Alongside the predicted value, the final system should explain the factors that influenced the estimate and, where possible, communicate uncertainty.

The current dataset requires one important clarification before model training. Its `Price` values range from 1.15 million to 420 million, which appears more consistent with property sale prices than monthly rent. The project must confirm the target definition, currency, and time period. If the values are sale prices, the project should either be reframed as housing price prediction or obtain a verified rental dataset. This validation is necessary to ensure that the model answers the stated real-world question.

The project must also consider responsible data use. Scraping should be performed only where permitted by website terms and robots.txt rules, with conservative request rates. Personal information should not be collected. Crime and demographic variables may encode historical or social bias, so their effect on predictions must be examined and clearly disclosed.

## 2. Benchmark of Existing Solutions

### Property listing platforms

Platforms such as Housing.com, 99acres, Magicbricks, and NoBroker provide searchable listings, locality filters, maps, and property comparisons. Their main strength is access to a large inventory of current listings. However, asking prices may be inconsistent, duplicate listings may exist, and the reasoning behind an estimated or recommended price is not always transparent.

### Automated valuation models

Commercial real-estate products use automated valuation models to estimate property values from historical transactions, listing information, and location. These systems can provide fast estimates at scale, but their training data, feature engineering, and model behavior are usually proprietary. Coverage and accuracy may also vary between localities.

### General machine learning projects

Many open-source housing prediction projects use benchmark datasets and models such as linear regression, random forest, gradient boosting, or XGBoost. They demonstrate model development effectively, but often use static datasets, limited location context, or random train-test splits that can overestimate real-world performance.

### Proposed Improvements

IsThisRentFr will aim to improve on basic listing comparisons and common academic implementations by:

- combining property details with distance-to-amenity features;
- including safety, demographic, and air-quality context;
- detecting duplicate and implausible records before training;
- comparing performance with both a simple baseline and stronger ensemble models;
- using location-aware or time-aware validation where the data permits it;
- reporting feature importance or SHAP explanations;
- versioning the dataset with DVC for reproducibility;
- documenting source provenance, collection date, units, and limitations.

## 3. Success Metrics

Recall is a classification metric and is not suitable as the primary measure for this continuous price prediction task. The project will use regression metrics.

| Metric | Initial success criterion | Purpose |
| --- | ---: | --- |
| Mean Absolute Error (MAE) | At least 20% lower than the median-price baseline | Measures typical error in price units |
| Mean Absolute Percentage Error (MAPE) | 20% or lower | Makes error easier to interpret across price ranges |
| Root Mean Squared Error (RMSE) | Lower than all baseline models | Penalizes large prediction errors |
| R-squared | 0.80 or higher on held-out data | Measures explained variation |

The criteria are provisional and must be revisited after the target definition is confirmed. Performance should also be reported by locality, price range, and property type. A model should not be accepted solely because its overall score is high.

## 4. Data Acquisition

The current project contains a merged raw dataset named `final merged datasets.csv`. It is tracked by DVC through `final merged datasets.csv.dvc`.

For a fully reproducible submission, the following acquisition details still need to be recorded:

- original listing source or open-data portal;
- source URLs or API endpoints;
- date and geographic scope of collection;
- target currency and whether `Price` means sale price or monthly rent;
- scraping or API scripts and their dependencies;
- licensing and permitted use of each source;
- join method used for amenity, crime, demographic, and air-quality data.

A proposed acquisition pipeline is:

1. Send rate-limited requests to a permitted listing source using `requests`.
2. Parse listing pages using BeautifulSoup or consume an official API when available.
3. Store the unmodified extraction as a timestamped raw file.
4. Normalize property fields and remove duplicates in a separate processing stage.
5. Geocode valid locations and calculate distances to nearby amenities.
6. Join locality-level public statistics using documented geographic keys.
7. Validate schemas, units, missing values, coordinates, and price ranges.

No scraper or API collection script is currently present in the repository, so the existing dataset's acquisition cannot yet be independently reproduced.

## 5. Dataset Description

The dataset currently contains **6,044 rows and 22 columns**, covering **113 localities**. No empty cells were detected by a basic CSV scan.

### Property and target fields

- `Price`: target value to be predicted
- `Area`: property area
- `Bedrooms` and `Bathrooms`: room counts
- `Property_Type`: apartment, villa, independent floor, or independent house
- `Furnishing`: unfurnished, semi-furnished, or furnished

### Geographic fields

- `Latitude` and `Longitude`: property coordinates
- `Locality`: locality name

### Distance-to-amenity fields

- `Nearest_Hospital_km`
- `Nearest_School_km`
- `Nearest_Metro_km`
- `Nearest_Bus_Stop_km`
- `Nearest_Park_km`
- `Nearest_Mall_km`

### Neighborhood and environmental fields

- `Crime_Rate`
- `Population_Density`
- `Literacy_Rate`
- `Household_Count`
- `AQI`
- `PM2.5`
- `PM10`

### Initial profile

| Item | Observed value |
| --- | ---: |
| Number of records | 6,044 |
| Number of columns | 22 |
| Number of localities | 113 |
| Minimum price | 1,150,000 |
| Median price | 11,600,000 |
| Mean price | 19,263,210.06 |
| Maximum price | 420,000,000 |
| Median area | 870 |
| Empty cells detected | 0 |

The data is highly concentrated in apartments: 5,989 of 6,044 records are apartments. The remaining records include 28 villas, 11 independent floors, and 16 independent houses. This class imbalance may limit reliable evaluation for less common property types. The large gap between median and maximum price also indicates a right-skewed target with possible luxury-property outliers. A logarithmic target transformation and robust error analysis should be considered.

## 6. Git and DVC Versioning Plan

The project repository is connected to:

`https://github.com/wyvernis/IsThisRentFr.git`

Git tracks source code, documentation, configuration, and the small `.dvc` metadata file. DVC tracks the large raw dataset through its content hash.

The current DVC metadata records:

- file: `final merged datasets.csv`
- size: 1,821,788 bytes
- MD5: `70144d962879adfae8fd374f72ad5b5c`

The initial DVC metadata was committed and pushed to the GitHub `main` branch in commit `16bf68b` (`Track raw dataset with DVC`).

The intended update workflow is:

```bash
dvc add "final merged datasets.csv"
git add "final merged datasets.csv.dvc" .gitignore
git commit -m "Update raw dataset"
dvc push
git push origin main
```

GitHub stores the DVC pointer, but not the dataset contents. A DVC storage remote has not yet been configured. To make `dvc push` and `dvc pull` work across machines, the project must configure a supported storage location such as S3, Azure Blob Storage, Google Cloud Storage, SSH, or another approved remote:

```bash
dvc remote add -d storage <remote-url>
git add .dvc/config
git commit -m "Configure DVC storage remote"
```

Credentials must be kept outside Git, preferably in `.dvc/config.local` or environment variables.

## 7. Deliverables

- **Problem statement:** completed in this report
- **Benchmark and proposed improvements:** completed in this report
- **Success metrics:** defined provisionally for regression
- **Raw dataset:** available locally as `final merged datasets.csv`
- **Dataset description:** completed in this report
- **DVC metadata and configuration:** `.dvc/`, `.dvcignore`, `.gitignore`, and `final merged datasets.csv.dvc`
- **Git remote:** configured and initial DVC metadata pushed to GitHub
- **Acquisition script and source provenance:** not yet available
- **DVC storage remote:** not yet configured

## Conclusion

Experiment 1 establishes the initial case-study framing and reproducibility structure for IsThisRentFr. The raw dataset has been profiled, placed under DVC control, and represented by metadata committed to GitHub. The available features support a richer model than one based only on property size and locality because they include amenity access, neighborhood context, and air quality.

Before modeling begins, the project must confirm whether `Price` represents rent or sale value, document the currency and collection period, and preserve the original data sources and acquisition code. A DVC storage remote must also be configured so collaborators can retrieve the raw data. Once these issues are resolved, the project can proceed to validation, exploratory analysis, feature engineering, and regression model development.
