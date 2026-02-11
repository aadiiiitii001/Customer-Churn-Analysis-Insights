# Customer Churn Prediction

## Overview
This project predicts customer churn using machine learning based on historical customer behavior and account data. The objective is to identify high-risk customers early so retention strategies can be applied before revenue is lost.

Churn prediction is treated as a **binary classification problem**, with emphasis on recall and F1-score rather than meaningless accuracy.

---

## Problem Statement
Customer churn directly impacts revenue and growth. Acquiring new customers is significantly more expensive than retaining existing ones.  
The challenge is to accurately predict which customers are likely to leave based on behavioral, contractual, and payment-related features.

---

## Dataset
- Historical customer records
- Includes:
  - Demographics
  - Account and contract details
  - Usage behavior
  - Payment information
- Target variable:
  - `Churn` (Yes / No)

> Example dataset: Telco Customer Churn Dataset

---

## Workflow
1. **Data Cleaning**
   - Removed irrelevant columns
   - Handled missing and inconsistent values
   - Converted data types

2. **Exploratory Data Analysis**
   - Analyzed churn distribution
   - Identified churn patterns across tenure, contract type, and charges

3. **Feature Engineering**
   - Encoded categorical variables
   - Scaled numerical features
   - Selected features with business relevance

4. **Model Training**
   - Logistic Regression (baseline)
   - Random Forest
   - XGBoost

5. **Evaluation Metrics**
   - Precision
   - Recall
   - F1-score
   - ROC-AUC
   - Confusion Matrix

6. **Optimization**
   - Hyperparameter tuning
   - Cross-validation
   - Threshold tuning to reduce false negatives

---

## Results
- Tree-based models outperformed linear models
- High recall achieved for churn class
- Key churn indicators:
  - Short tenure
  - Month-to-month contracts
  - Higher monthly charges
  - Electronic check payment method

---

## Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Models:** Logistic Regression, Random Forest, XGBoost

---

## Project Structure
```text
├── data/
│   └── churn_data.csv
├── notebooks/
│   ├── eda.ipynb
│   └── model_training.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
├── requirements.txt
└── README.md
```
## How to Run
- pip install -r requirements.txt
- python src/train.py

## Screenshot
<img width="1920" height="1080" alt="Screenshot (26)" src="https://github.com/user-attachments/assets/a1894c2a-b51f-4839-a4ef-e51e415b601d" />
