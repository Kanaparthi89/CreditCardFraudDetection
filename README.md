# Credit Card Fraud Detection Using Machine Learning 🚀

![Python](https://img.shields.io/badge/python-3.10-blue) ![ML](https://img.shields.io/badge/Machine%20Learning-green)

## 📌 Introduction

Credit card fraud is a major financial concern, causing billions in losses every year. Machine learning enables the detection of fraudulent transactions by identifying unusual patterns in transaction data. This project uses supervised learning techniques on the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) to predict fraudulent transactions effectively.

---

## 📊 Dataset Overview

* **Source:** Kaggle Credit Card Fraud Detection dataset
* **Size:** 284,807 transactions
* **Features:** 30 numerical features (`V1–V28`, `Time`, `Amount`)
* **Target:** `Class` (0 = legitimate, 1 = fraud)
* **Class Imbalance:** Only 0.2% of transactions are fraudulent

---

## 🛠️ Data Preprocessing

Steps taken to prepare the data for modeling:

1. **Missing Values:** Checked and none found.
2. **Scaling:** Standardized `Amount` and `Time`.
3. **Train-Test Split:** 80% training, 20% testing.
4. **Class Imbalance Handling:** SMOTE (Synthetic Minority Oversampling Technique) used to oversample fraud cases.

```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)
```

---

## 📈 Exploratory Data Analysis (EDA)

### 1️⃣ Class Distribution

Severe imbalance observed: 99.8% legitimate vs 0.2% fraudulent transactions.

![Class Distribution](class_distribution.jpg)

### 2️⃣ Correlation Heatmap

Most PCA features show low correlation, confirming dimensionality reduction.

![Correlation Heatmap](correlation_heatmap.jpg)

### 3️⃣ Transaction Amount Distribution

Fraudulent transactions tend to cluster at lower amounts (< \$200).

![Transaction Amount Distribution](transaction_amount.jpg)

---

## 🤖 Modeling

### Algorithms Used

* Logistic Regression
* Random Forest Classifier
* XGBoost Classifier

### Training

Models were trained on the SMOTE-resampled dataset and evaluated using metrics such as ROC-AUC, Precision, Recall, and F1-Score.

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_res, y_res)
```

---

## 📊 Model Evaluation

| Model               | Precision | Recall | F1-Score | ROC-AUC |
| ------------------- | --------- | ------ | -------- | ------- |
| Logistic Regression | 0.72      | 0.62   | 0.67     | 0.95    |
| Random Forest       | 0.89      | 0.85   | 0.87     | 0.99    |
| XGBoost             | 0.88      | 0.83   | 0.85     | 0.99    |

![ROC Curve](roc_curve.jpg)

**Observation:** Random Forest achieved the best balance between recall and precision, making it suitable for detecting fraudulent transactions.

---

## ✅ Conclusion

This project demonstrates that machine learning, particularly Random Forest, can effectively detect credit card fraud even under severe class imbalance. While preprocessing and SMOTE improved recall, precision-recall trade-offs remain a challenge.

**Future Work:**

* Explore deep learning models and ensemble stacking.
* Implement real-time fraud detection pipelines.
* Evaluate anomaly detection approaches for zero-day fraud patterns.

---

## 📂 Project Files

* `README.md` (this file)
* `fraud_detection.ipynb` (or `.py` scripts for code)
* `class_distribution.jpg`
* `correlation_heatmap.jpg`
* `transaction_amount.jpg`
* `roc_curve.jpg`
