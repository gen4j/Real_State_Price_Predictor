# California Housing Price Prediction

A machine learning project that predicts median house values in California districts using the classic **California Housing** dataset. This notebook walks through a complete, real-world ML workflow: acquiring and exploring data, building a robust preprocessing pipeline, and training/evaluating regression models.

## 📋 Overview

Given census data for California districts (geography, housing stock, population, and income), the goal is to predict the **median house value** for each district. The project follows a standard supervised machine learning workflow from raw data to a validated baseline model.

## 📊 Dataset

- **Source:** [ageron/handson-ml2](https://github.com/ageron/handson-ml2) housing dataset (California census data)
- **Format:** CSV, ~20,000 district-level records
- **Features:**
  | Feature | Description |
  |---|---|
  | `longitude`, `latitude` | Geographic location of the district |
  | `housing_median_age` | Median age of houses in the district |
  | `total_rooms`, `total_bedrooms` | Aggregate room counts |
  | `population`, `households` | Population and household counts |
  | `median_income` | Median household income (in tens of thousands of USD) |
  | `ocean_proximity` | Categorical: proximity to the ocean |
- **Target:** `median_house_value`

The dataset is downloaded automatically at runtime and cached locally under `datasets/housing/`.

## 🔧 Project Workflow

### 1. Data Acquisition
Automatically fetches and extracts the housing dataset (`fetch_housing_data`) and loads it into a pandas DataFrame (`load_housing_data`).

### 2. Exploratory Data Analysis
- Inspected data types, summary statistics, and category distributions (`ocean_proximity`)
- Visualized feature distributions with histograms
- Plotted geographic data (longitude/latitude) with population size and median house value as color/size encodings
- Computed and ranked feature correlations with `median_house_value`
- Visualized top correlated features with scatter matrices, zooming in on `median_income` vs. `median_house_value` — the strongest predictor

### 3. Stratified Train/Test Split
To ensure the test set is representative of the full income distribution, `median_income` was binned into categories and used for a **stratified shuffle split** (80/20 train/test), rather than a naive random split.

### 4. Data Cleaning & Feature Engineering
- **Missing values:** Imputed using `SimpleImputer` (median strategy), applied to `total_bedrooms` and other numeric columns
- **Categorical encoding:** `ocean_proximity` encoded via `OneHotEncoder`
- **Custom feature engineering:** A custom `CombinedAttributesAdder` transformer (built with `BaseEstimator`/`TransformerMixin`) derives new, more informative features:
  - `rooms_per_household`
  - `population_per_household`
  - `bedrooms_per_room` (optional)

### 5. Preprocessing Pipeline
All preprocessing steps are composed into a single reusable `Pipeline` / `ColumnTransformer`:

```
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])
```

This ensures numeric features are imputed, engineered, and scaled, while categorical features are one-hot encoded — all in one fit/transform call, and safely reusable on the test set.

### 6. Model Training & Evaluation
Two baseline models were trained and evaluated:

| Model | Training RMSE | 10-Fold CV RMSE (mean) |
|---|---|---|
| Linear Regression | High (underfits) | — |
| Decision Tree Regressor | ~0 (overfits) | Much worse than Linear Regression — confirms severe overfitting |

Key takeaway: the Decision Tree's near-zero training error was a red flag for overfitting, confirmed via **10-fold cross-validation** (`cross_val_score` with `neg_mean_squared_error`), which revealed poor generalization performance.

## 🚧 Next Steps

This notebook currently establishes a working baseline and diagnostic framework. Planned next steps:

- [ ] Try additional model families (Random Forest, SVR, etc.)
- [ ] Hyperparameter tuning via **Grid Search** / **Randomized Search**
- [ ] Explore ensemble methods
- [ ] Analyze the best model's errors in depth
- [ ] Prepare the final model for launch, monitoring, and maintenance

## 🛠️ Requirements

```
pandas
numpy
matplotlib
scikit-learn
six
```

## ▶️ Usage

Run the notebook cells sequentially. The dataset will be downloaded automatically on first run. No manual data setup is required.

```bash
jupyter notebook housing.ipynb
```

## 📚 Acknowledgments

This project follows the methodology and dataset popularized by Aurélien Géron's *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*.
