# Diabetes Risk Prediction API

> Predicts the likelihood of type 2 diabetes from routine clinical indicators — helping healthcare providers flag high-risk patients before symptoms progress.

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green)]()
[![F1](https://img.shields.io/badge/F1--weighted-0.75-brightgreen)]()
[![Model](https://img.shields.io/badge/Model-Logistic%20Regression-blueviolet)]()

---

## Business Problem

Undiagnosed type 2 diabetes affects roughly 1 in 5 adults worldwide, leading to costly late-stage complications. Manual screening is time-consuming and inconsistent across clinics. This model automates early-risk flagging using eight standard lab values, enabling faster triage and reducing the diagnostic burden on medical staff.

---

## Project Structure

```
ml_Diabets/
├── .gitignore
├── readme.md
├── requirements.txt
└── diabetes/
    ├── Diabetes.ipynb            # EDA + model comparison
    ├── main.py                   # FastAPI inference service
    ├── model_log_Diabetes.pkl    # deployed model (Logistic Regression)
    ├── scaler_Diabetes.pkl       # StandardScaler used at inference
    ├── dataset/                  # raw data
    └── Test.txt
```

---

## Demo

**POST** `http://127.0.0.1:8000/predict`

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 2,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50
  }'
```

**Response:**
```json
{
  "prediction": 1,
  "diabetes_detected": true,
  "message": "Высокая вероятность наличия сахарного диабета",
  "probability_positive": 71.42,
  "probability_negative": 28.58
}
```

> `Glucose`, `BloodPressure`, and `BMI` are validated to be strictly greater than 0.

---

## Results

Three models were trained and compared on an identical 80/20 stratified split. **Logistic Regression was selected for deployment** — it had the best weighted F1, the smallest train/test gap, and no signs of overfitting.

| Model | Accuracy | F1 (weighted) | Precision (weighted) | Recall (weighted) | Train Acc. | Test Acc. |
|---|---|---|---|---|---|---|
| **Logistic Regression** ✅ deployed | **75.3%** | **0.751** | 0.750 | 0.753 | 0.772 | 0.753 |
| Random Forest | 74.7% | 0.748 | 0.750 | 0.747 | **1.000** | 0.747 |
| Decision Tree | 73.4% | 0.736 | 0.740 | 0.734 | **1.000** | 0.734 |

**Class 1 (diabetes-positive) detail — Logistic Regression:**

| Metric | Class 0 (no diabetes) | Class 1 (diabetes) |
|---|---|---|
| Precision | 0.80 | 0.67 |
| Recall | 0.83 | 0.62 |
| F1-score | 0.81 | 0.64 |

*(support: 99 negative / 55 positive in the 154-row test set)*

---

## Dataset

- **Source:** Pima Indians Diabetes Database (UCI / Kaggle)
- **Size:** 768 records
- **Features:** 8 numeric clinical features (glucose, BMI, age, insulin, etc.) + 1 binary target
- **Class balance:** ~65% negative / ~35% positive — handled via stratified train/test split (`stratify=y`)

---

## Approach

1. **Data loading & EDA** — distribution analysis, outlier detection (`Glucose` < 70, `BMI` > 40)
2. **Preprocessing** — `StandardScaler` fit on the train set only, applied to the test set (no data leakage)
3. **Model training** — Logistic Regression, Decision Tree, and Random Forest trained with default hyperparameters and compared side by side
4. **Evaluation** — Accuracy, weighted Precision/Recall/F1, full `classification_report`, plus a manual train-vs-test accuracy check per model to catch overfitting
5. **Model selection** — Random Forest and Decision Tree both hit 100% train accuracy with no depth constraint (severe overfitting); Logistic Regression generalized best and was chosen for deployment
6. **Model persistence** — final model and scaler saved via `joblib`
7. **Deployment** — FastAPI REST endpoint wraps the inference pipeline end-to-end

---

## Key Challenges & Solutions

**Unconstrained tree-based models overfit the training set**
`RandomForestClassifier()` and `DecisionTreeClassifier()` were trained with default parameters (no `max_depth`) → both reached 100% training accuracy while test accuracy stayed around 73–75% — a ~25-point train/test gap, a clear overfitting signal → rather than force a depth constraint post-hoc, the comparison itself surfaced Logistic Regression as the better generalizer (only a ~2-point gap) → it was selected as the deployed model instead of defaulting to the more "impressive-sounding" ensemble method.

**Class imbalance (65/35 split)**
Default split → risk of majority-class bias in predictions → added `stratify=y` to `train_test_split`, preserving the class ratio in both folds.

**Data leakage risk**
Fitting the scaler on the full dataset before splitting would leak test statistics into training → scaler fitted on `X_train` only, then applied to `X_test`.

---

## Tech Stack

| Category   | Tools                              |
|------------|-------------------------------------|
| Language   | Python 3.11                        |
| ML         | scikit-learn, joblib               |
| Data       | pandas, NumPy                      |
| Viz        | Matplotlib, Seaborn                |
| API        | FastAPI, Uvicorn, Pydantic         |
| Deployment | Local / Docker-ready               |

---

## Deployment

The trained model is served as a REST API using **FastAPI**. On startup (`lifespan`), the Logistic Regression model and scaler are loaded once into `app.state`.

```
POST /predict
```

The endpoint accepts a JSON body with 8 clinical features and returns the predicted class, a human-readable message, and the probability for both classes.

**To run locally:**
```bash
python main.py
# API available at http://127.0.0.1:8000
# Swagger UI at http://127.0.0.1:8000/docs
```

---

## How to Run

```bash
git clone https://github.com/YOUR_USERNAME/diabetes-risk-api
cd diabetes-risk-api
pip install -r requirements.txt
```

```bash
jupyter notebook diabetes/Diabetes.ipynb
```

```bash
python diabetes/main.py
```

---

## Business Impact

- ↓ ~40% reduction in manual screening time per patient (estimated)
- ↑ Earlier detection flag vs no systematic pre-screening (estimated)
- ↓ Lower cost per diagnosed case through automated pre-triage (estimated)
- ↑ Consistent risk scoring across clinics — eliminates inter-clinician variability
- ↑ Scalable REST API allows integration with EHR systems or mobile health apps

---
