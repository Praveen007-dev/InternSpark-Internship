# House Price Prediction Project

## Overview

This project builds a machine learning regression model to predict house prices using housing-related features. The project focuses on preprocessing, feature engineering, handling missing values, model comparison, and performance evaluation.

The target variable is **house price**, making this a **regression problem**.

---

## Dataset

Dataset used:

**House Price Prediction Dataset (https://www.kaggle.com/datasets/bhanupratapbiswas/house-price-prediction)**

The dataset contains housing-related information such as:

* Number of bedrooms
* Number of bathrooms
* Square footage of living area
* Lot size
* Number of floors
* Waterfront availability
* House condition
* Year built
* Renovation year
* Location details (city, street, state, country)

## Dataset Features & Size

The dataset contains **4,600 housing records (rows)** and **18 features (columns)** used for house price prediction. These features include both numerical and categorical information describing the physical characteristics, condition, and location of houses.

### Features Used

| Feature         | Description                           |
| --------------- | ------------------------------------- |
| `date`          | Date of house sale                    |
| `price`         | House price (target variable)         |
| `bedrooms`      | Number of bedrooms                    |
| `bathrooms`     | Number of bathrooms                   |
| `sqft_living`   | Living area square footage            |
| `sqft_lot`      | Lot size square footage               |
| `floors`        | Number of floors                      |
| `waterfront`    | Waterfront presence (0 = No, 1 = Yes) |
| `view`          | Quality of house view                 |
| `condition`     | Overall house condition               |
| `sqft_above`    | Square footage above ground           |
| `sqft_basement` | Basement square footage               |
| `yr_built`      | Year house was built                  |
| `yr_renovated`  | Year renovated                        |
| `street`        | Street name                           |
| `city`          | City location                         |
| `statezip`      | State and ZIP code                    |
| `country`       | Country name                          |

The target variable for prediction is **`price`**, making this a **supervised regression problem**.

---

## Objective

To build and compare regression models for predicting house prices and select the best-performing model.

---

## Preprocessing & Feature Engineering

The following preprocessing steps were applied:

### 1. Missing Value Handling

* Numerical columns were checked and imputed using **mean imputation** where necessary.

### 2. Feature Transformation

* **Log Transformation** was applied to the target variable (`price`) to reduce skewness and improve regression performance.

### 3. Encoding

Categorical columns were converted into numerical values using **Label Encoding**.

Encoded columns:

* `date`
* `street`
* `city`
* `statezip`
* `country`

### 4. Feature Scaling

**StandardScaler** was used to standardize feature values before training.

---

## Models Implemented

The following regression models were trained and compared:

1. **Linear Regression**
2. **Random Forest Regressor**
3. **Gradient Boosting Regressor**

---

## Evaluation Metrics

The models were evaluated using:

* **RMSE (Root Mean Squared Error)**
* **MAE (Mean Absolute Error)**
* **Residual Analysis**

The best-performing model was automatically selected based on the **lowest RMSE**.

---

## Saved Files

### Saved Model

```text
best_price_model.pkl
```

### Saved Scaler

```text
scaler.pkl
```

---

## Example Prediction Code

```python
import pandas as pd
import numpy as np
import joblib

# Load saved model
model = joblib.load(
    "best_price_model.pkl"
)

# Load scaler
scaler = joblib.load(
    "scaler.pkl"
)

# Example house input
sample = pd.DataFrame(
    [[
        2008,
        3,
        2,
        1800,
        5000,
        2,
        0,
        0,
        3,
        1800,
        0,
        1998,
        0,
        10,
        5,
        20,
        1
    ]],
    columns=X.columns
)

# Scale input
sample_scaled = scaler.transform(
    sample
)

# Predict
prediction = model.predict(
    sample_scaled
)

# Reverse log transform
predicted_price = np.expm1(
    prediction
)

print(
    f"Predicted House Price: "
    f"₹{predicted_price[0]:,.2f}"
)
```

---

## Project Structure

```text
House Price Prediction/
│── House_Price_Prediction.ipynb
│── data.csv
│── best_price_model.pkl
│── scaler.pkl
│── README.md
```

---

## How to Run

1. Download the project files.
2. Install required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

3. Open the notebook:

```text
House_Price_Prediction.ipynb
```

4. Run all notebook cells.

5. Use the prediction cell to estimate house prices.

