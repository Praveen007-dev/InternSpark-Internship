# Iris Classification Project

## Overview

This project builds a **Machine Learning Classification Model** to predict the species of an iris flower using flower measurements such as sepal length, sepal width, petal length, and petal width.

The model predicts one of the following flower species:

* **Iris-setosa**
* **Iris-versicolor**
* **Iris-virginica**

The project includes:

* Exploratory Data Analysis (EDA)
* Data preprocessing
* Model training and comparison
* Classification evaluation
* Prediction inference using a saved model

Since the target variable is categorical, this is a **classification problem**.

---

## Dataset

**Dataset Used:**
Iris Classification Dataset (Kaggle)

The dataset contains iris flower measurements used for species classification.

---

## Dataset Statistics & Features

The dataset contains:

* **150 flower records (rows)**
* **5 columns**
* **4 input features**
* **1 target variable (`Species`)**
* **3 flower species classes**

### Dataset Dimensions

| Metric             |   Value |
| ------------------ | ------: |
| Number of Records  |     150 |
| Number of Features |       4 |
| Target Variable    | Species |
| Number of Classes  |       3 |

### Features Used

| Feature       | Description                           |
| ------------- | ------------------------------------- |
| `SepalLength` | Length of the sepal                   |
| `SepalWidth`  | Width of the sepal                    |
| `PetalLength` | Length of the petal                   |
| `PetalWidth`  | Width of the petal                    |
| `Species`     | Iris flower species (Target Variable) |

### Target Classes

The model predicts the following species:

* `Iris-setosa`
* `Iris-versicolor`
* `Iris-virginica`

---

## Data Preprocessing

### Missing Value Handling

The dataset was checked for missing values.

No significant missing values were found.

### Feature Scaling

Feature scaling was performed using:

```text
StandardScaler
```

This standardizes numerical features and improves model performance.

---

## Exploratory Data Analysis (EDA)

EDA was performed to understand class separability and feature relationships.

Visualizations included:

* Species Distribution
* Pair Plot
* Correlation Heatmap

These visualizations helped identify patterns among flower species.

---

## Models Implemented

The following classification models were trained and compared:

1. **k-Nearest Neighbors (k-NN)**
2. **Logistic Regression**
3. **Decision Tree Classifier**

The best-performing model was automatically selected based on the **highest accuracy score**.

---

## Model Evaluation Metrics

The models were evaluated using:

* **Accuracy**
* **Precision**
* **Recall**
* **Confusion Matrix**

These metrics were used to compare model performance.

---

## Saved Files

### Saved Model

```text
best_iris_model.pkl
```

### Saved Scaler

```text
scaler.pkl
```

These files are used to classify new flower measurements without retraining the model.

---

## Example Inference Code

```python
import pandas as pd
import joblib

model = joblib.load(
    "best_iris_model.pkl"
)

scaler = joblib.load(
    "scaler.pkl"
)

sample = pd.DataFrame(
    [[
        5.1,
        3.5,
        1.4,
        0.2
    ]],
    columns=[
        'SepalLength',
        'SepalWidth',
        'PetalLength',
        'PetalWidth'
    ]
)

sample_scaled = scaler.transform(
    sample
)

prediction = model.predict(
    sample_scaled
)

print(
    "Predicted Species:",
    prediction[0]
)
```

---

## Project Structure

```text
Iris Classification/
│── Iris_Classification.ipynb
│── IRIS.csv
│── best_iris_model.pkl
│── scaler.pkl
│── README.md
```

---

## Required Libraries

Install dependencies using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

---

## How to Run

### Step 1

Download or clone the repository.

### Step 2

Install required libraries.

### Step 3

Open:

```text
Iris_Classification.ipynb
```

### Step 4

Run all notebook cells sequentially.

### Step 5

Use the prediction section to classify iris flower species using custom flower measurements.

---

## Conclusion

This project successfully demonstrates an end-to-end Machine Learning classification workflow for iris species prediction.

The workflow includes:

* Data preprocessing
* Exploratory Data Analysis (EDA)
* Model training and comparison
* Performance evaluation
* Best model selection
* Prediction inference using saved models

The final system can accurately classify unseen iris flower samples using flower measurements.
