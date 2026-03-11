# Insurance Claim Risk Prediction

## Overview

This project builds a machine learning model to predict the probability that a customer will file an insurance claim.

The dataset contains over **595,000 records with anonymized features**. Due to severe class imbalance (~3.6% claim cases), evaluation focuses on **ROC-AUC and Recall rather than accuracy**.

The goal is to support:

- Risk-based pricing
- Underwriting decisions
- Loss mitigation
- Portfolio risk monitoring

---

## Dataset

Dataset contains:

- 595k observations
- 59 anonymized features
- Binary target variable (claim / no claim)

Due to privacy restrictions, feature names are anonymized.

---

## Exploratory Data Analysis

EDA includes:

- Target distribution analysis
- Feature distribution visualization
- Outlier detection using boxplots
- Correlation heatmap

class_distribution.png, correlation_heatmap.png, roc_curve.png (Visuals in Images folder)

The dataset shows **severe class imbalance**, which significantly influences model behavior.

---

## Modeling Approach

Three models were trained and evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost

Key techniques used:

- Stratified train-test split
- Memory optimization
- StandardScaler for Logistic Regression
- RandomizedSearchCV for hyperparameter tuning
- Cross-validation
- Threshold optimization

---

## Model Performance

| Model | ROC-AUC | Recall |
|------|------|------|
| Logistic Regression | ~0.62 | **53%** |
| Random Forest | ~0.62 | 41% |
| XGBoost | ~0.63 | ~0% |

Although XGBoost achieved slightly higher ROC-AUC, Logistic Regression provided significantly better recall and interpretability.

---

## Final Model Selection

Logistic Regression was selected because:

- Comparable ROC-AUC to XGBoost
- Higher recall for detecting claims
- Greater interpretability
- Regulatory suitability in financial risk modeling

---

## Business Impact Simulation

The model estimates financial exposure from missed claims by calculating potential losses based on false negatives.

This allows insurers to evaluate the cost of prediction errors.

---

## Key Challenges

- Severe class imbalance
- Anonymous features limiting domain interpretation
- Moderate ROC-AUC due to limited feature information

---

## Future Improvements

Potential improvements include:

- SMOTE oversampling
- Cost-sensitive learning
- Feature engineering
- Threshold calibration

---

## Tech Stack

Python  
Pandas  
Scikit-learn  
XGBoost  
Seaborn  
Matplotlib

---

## Author

Viknesh
