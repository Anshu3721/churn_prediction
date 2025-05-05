
---

## 🎯 Objective

To develop a robust predictive model that can:
- Accurately identify telecom customers who are likely to churn.
- Help businesses take proactive measures to reduce churn rate.
- Provide a clean UI via Streamlit for business users to input customer details and get real-time predictions.

---

## 🧠 Models Used

We trained and evaluated the following classifiers:
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- Gradient Boosting (**Selected**)
- XGBoost
- LightGBM

**Gradient Boosting** was chosen as the best-performing model based on:
- Highest F1-Score
- Best ROC AUC
- Strong performance on imbalanced data

---

## 🗂️ Dataset Info

Dataset: [`WA_Fn-UseC_-Telco-Customer-Churn.csv`](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

- `customerID`: Unique customer identifier
- `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- `tenure`, `PhoneService`, `MultipleLines`
- `InternetService`, `OnlineSecurity`, `OnlineBackup`, etc.
- `Contract`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
- `Churn`: Target variable (Yes/No)

---

## 🧼 Data Preprocessing

- Dropped `customerID`
- Converted `TotalCharges` to numeric (handling NaNs)
- Encoded binary variables (`Yes/No`, `Male/Female`)
- One-hot encoded categorical columns (InternetService, Contract, etc.)
- Scaled numerical columns using `StandardScaler`
- Created tenure bins (e.g., 0–12 months, 13–24 months)

---

## ⚖️ Handling Class Imbalance

Used **SMOTE** to oversample the minority class (`Churn = 1`) to balance the dataset.

---

## 🔍 Model Training & Evaluation

- Performed train-test split with stratification
- Applied SMOTE on training data
- Tuned Gradient Boosting with `GridSearchCV`
- Evaluated using:
  - **F1-Score** (for imbalanced data)
  - **ROC AUC Score**
  - **Confusion Matrix**
  - **Classification Report**

---

## 🚀 Streamlit App

The app allows users to:
- Input customer attributes via dropdowns and sliders
- View predicted churn result (`Likely to stay` or `Likely to churn`)
- See churn probability in percentage
