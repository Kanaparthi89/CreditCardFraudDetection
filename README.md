# Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using Python. This project handles severe class imbalance and evaluates multiple models for fraud detection.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Dataset](#dataset)  
3. [Exploratory Data Analysis](#exploratory-data-analysis)  
4. [Modeling](#modeling)  
5. [Results](#results)  
6. [Conclusion](#conclusion)  
7. [Folder Structure](#folder-structure)  

---

## Project Overview

Credit card fraud is a significant problem in the financial industry. Detecting fraudulent transactions in real-time can save companies millions of dollars. This project applies supervised machine learning techniques to classify transactions as fraudulent or legitimate.

---

## Dataset

- Source: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
- Features: 28 anonymized PCA features + `Amount` and `Time`  
- Imbalance: 99.8% legitimate vs 0.2% fraudulent transactions

---

## Exploratory Data Analysis

### Class Distribution
![Class Distribution](Plots_hd/class_distribution.png)

### Correlation Heatmap
![Correlation Heatmap](Plots_hd/correlation_heatmap.png)

### Transaction Amount Distribution
![Transaction Amount Distribution](Plots_hd/transaction_amount.png)

---

## Modeling

Multiple machine learning models were evaluated:

- Logistic Regression  
- Random Forest  
- Support Vector Machine  
- K-Nearest Neighbors  
- Decision Tree  

**Example Python snippet:**

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
Results

ROC Curves for different models:

![Decision Tree ROC](Plots_hd/Decision Tree_ROC.png)
![K-Nearest Neighbors ROC](Plots_hd/K-Nearest Neighbors_ROC.png)
![Logistic Regression ROC](Plots_hd/Logistic Regression_ROC.png)
![Support Vector Machine ROC](Plots_hd/Support Vector Machine_ROC.png)

Random Forest achieved the best balance of precision and recall.

SMOTE improved recall but requires careful handling to avoid overfitting.

Conclusion

This project demonstrates that machine learning methods, particularly Random Forest, are effective for credit card fraud detection under severe class imbalance.
Future work could explore deep learning, ensemble stacking, and real-time deployment.
CreditCardFraudDetection/
│
├── Credit Card Fraud Detection.ipynb
├── README.md
├── Plots_hd/
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── transaction_amount.png
│   ├── Decision Tree_ROC.png
│   ├── K-Nearest Neighbors_ROC.png
│   ├── Logistic Regression_ROC.png
│   └── Support Vector Machine_ROC.png
Requirements
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn