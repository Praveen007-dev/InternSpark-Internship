# Titanic Survival Prediction Project

## Overview

This project builds a **Machine Learning Classification Model** to predict whether a passenger survived the Titanic disaster using passenger-related attributes.

The project emphasizes:

* Feature Engineering
* Missing Value Handling
* Data Preprocessing
* Classification Model Comparison
* Model Explainability using **SHAP (SHapley Additive exPlanations)**

The target variable is **`Survived`**, making this a **binary classification problem**.

---

## Dataset

**Dataset Used:**
Titanic Survival Dataset (Kaggle)

The dataset contains passenger-related information such as demographics, travel class, fare, cabin information, and family relationships to predict survival on the Titanic.

---

## Dataset Statistics & Features

The Titanic dataset used in this project contains:

* **418 passenger records (rows)**
* **12 original features (columns)**
* **1 target variable (`Survived`)**
* **3 engineered features created during preprocessing**
* **15 total features after feature engineering**

### Dataset Dimensions

| Metric                           |      Value |
| -------------------------------- | ---------: |
| Number of Rows                   |        418 |
| Number of Original Features      |         12 |
| Engineered Features Added        |          3 |
| Total Features After Engineering |         15 |
| Target Variable                  | `Survived` |

---

### Original Features in Dataset

| Feature       | Data Type | Description                                         |
| ------------- | --------- | --------------------------------------------------- |
| `PassengerId` | Integer   | Unique passenger identifier                         |
| `Survived`    | Integer   | Survival status (0 = Did Not Survive, 1 = Survived) |
| `Pclass`      | Integer   | Passenger class (1st, 2nd, 3rd)                     |
| `Name`        | Text      | Passenger full name                                 |
| `Sex`         | Text      | Passenger gender                                    |
| `Age`         | Numeric   | Passenger age                                       |
| `SibSp`       | Integer   | Number of siblings/spouse travelling                |
| `Parch`       | Integer   | Number of parents/children travelling               |
| `Ticket`      | Text      | Ticket number                                       |
| `Fare`        | Numeric   | Passenger ticket fare                               |
| `Cabin`       | Text      | Cabin information                                   |
| `Embarked`    | Text      | Boarding port                                       |

---

### Engineered Features

The following features were created to improve model performance:

#### 1. Title Extraction

Passenger titles were extracted from the passenger name.

Example:

```text
"Smith, Mr. John" → "Mr"
```

Purpose:

* Captures social status and demographics
* Helps improve classification accuracy

---

#### 2. Family Size

A new feature was created using:

```text
FamilySize = SibSp + Parch + 1
```

Purpose:

* Identifies whether a passenger travelled alone or with family
* Helps understand group-based survival patterns

---

#### 3. Cabin Presence

Instead of imputing cabin values, a new feature called:

```text
CabinPresence
```

was created.

Possible values:

```text
Cabin Available
No Cabin
```

Purpose:

* Indicates whether cabin information exists
* Helps infer socio-economic status

---

## Missing Value Handling

The following missing data strategies were implemented:

| Feature | Missing Values | Strategy Used                          |
| ------- | -------------: | -------------------------------------- |
| `Age`   |             86 | Mean Imputation                        |
| `Fare`  |              1 | Mean Imputation                        |
| `Cabin` |            327 | Converted into `CabinPresence` feature |

After preprocessing, missing values were effectively handled for model training.

---

## Data Preprocessing

### 1. Encoding

Categorical variables were converted into numerical form using **Label Encoding**.

Encoded columns:

* `Sex`
* `Embarked`
* `Title`
* `CabinPresence`

---

### 2. Feature Scaling

**StandardScaler** was used to standardize numerical features before model training.

This improves model consistency and performance.

---

## Models Implemented

The following classification algorithms were trained and compared:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**

---

## Model Evaluation Metrics

The models were evaluated using:

* **Accuracy**
* **Precision**
* **Recall**
* **Confusion Matrix**

The **best-performing model** was selected automatically based on the **highest accuracy score**.

---

## Model Explainability using SHAP

The project uses:

**SHAP (SHapley Additive exPlanations)**

to explain model predictions.

SHAP helps explain:

* Which features most influence prediction
* Positive and negative feature impact
* Overall model decision-making

### SHAP Visualizations Used

1. **SHAP Summary Plot**

   * Shows how each feature affects survival prediction.

2. **SHAP Feature Importance Plot**

   * Ranks features based on contribution strength.

Typical important features include:

* `Sex`
* `Pclass`
* `Fare`
* `Age`
* `Title`
* `FamilySize`

---

## Saved Files

### Saved Model

```text
best_titanic_model.pkl
```

### Saved Scaler

```text
scaler.pkl
```

---

## Example Inference Code

```python
import pandas as pd
import joblib

model = joblib.load(
    "best_titanic_model.pkl"
)

scaler = joblib.load(
    "scaler.pkl"
)

sample = pd.DataFrame(
    [[
        3,      # Passenger Class
        1,      # Gender (Male)
        22,     # Age
        1,      # Siblings/Spouse
        0,      # Parents/Children
        7.25,   # Ticket Fare
        2,      # Boarding Port
        3,      # Passenger Title
        2,      # Family Size
        1       # Cabin Availability
    ]],
    columns=X.columns
)

sample_scaled = scaler.transform(
    sample
)

prediction = model.predict(
    sample_scaled
)

result = (
    "Survived"
    if prediction[0] == 1
    else "Did Not Survive"
)

print(
    "Prediction:",
    result
)
```

---

## Project Structure

```text
Titanic Survival Prediction/
│── Titanic_Survival.ipynb
│── titanic.csv
│── best_titanic_model.pkl
│── scaler.pkl
│── README.md
```

---

## Required Libraries

Install required dependencies using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap joblib
```

---

## How to Run the Project

### Step 1

Download or clone the repository.

### Step 2

Install required libraries.

### Step 3

Open:

```text
Titanic_Survival.ipynb
```

### Step 4

Run all notebook cells sequentially.

### Step 5

Use the inference section to test survival prediction using custom passenger inputs.

---

## Conclusion

This project successfully implements a complete **Titanic Survival Prediction System** using Machine Learning.

The workflow includes:

* Data preprocessing
* Missing value handling
* Feature engineering
* Classification model comparison
* SHAP-based model explainability
* Prediction inference using a saved model

The project demonstrates how Machine Learning can be used to analyze passenger attributes and predict survival outcomes effectively.
