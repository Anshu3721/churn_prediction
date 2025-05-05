### Loading Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import joblib
### Loading Data
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.head()
df.shape
### Basic Information about data
df.describe().T
df.info()
df.isnull().sum()
from pandas_profiling import ProfileReport

# Create a ProfileReport object
profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)

profile.to_notebook_iframe()
### Cleaning Data
# Drop the unique ID column
df.drop("customerID", axis=1, inplace=True)
# Convert TotalCharges to numeric, coerce errors to NaN
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
# Fill missing TotalCharges with the median
median_tc = df["TotalCharges"].median()
df["TotalCharges"].fillna(median_tc, inplace=True)
df.sample()
### Data Visualisation
sns.set_style("whitegrid")
# Churn distribution
plt.figure(figsize=(6,4))
sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.show()
# Tenure by churn
plt.figure(figsize=(6,4))
sns.histplot(data=df, x="tenure", hue="Churn", bins=30, alpha=0.6)
plt.title("Tenure by Churn")
plt.show()


# Monthly charges by churn
plt.figure(figsize=(6,4))
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges by Churn")
plt.show()


# Contract Type VS Churn
plt.figure(figsize=(6,4))
sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Churn Rate by Contract Type")
plt.show()

# Internet service type vs churn
plt.figure(figsize=(6,4))
sns.countplot(x="InternetService", hue="Churn", data=df)
plt.title("Churn by Internet Service Type")
plt.show()


# Creting diffrent tenure groups
df["tenure_group"] = pd.cut(df["tenure"], bins=[0, 12, 24, 48, 60, 72], labels=["0–12","13–24","25–48","49–60","61–72"])

# tenure group vs churn
plt.figure(figsize=(6,4))
sns.countplot(x="tenure_group", hue="Churn", data=df)
plt.title("Churn Rate by Tenure Group")
plt.show()

### Data Preprocessing
# Map binary columns
binary_map = {"Yes": 1, "No": 0, "Female": 1, "Male": 0}
df["gender"]        = df["gender"].map(binary_map)
df["Partner"]       = df["Partner"].map(binary_map)
df["Dependents"]    = df["Dependents"].map(binary_map)
df["PhoneService"]  = df["PhoneService"].map(binary_map)
df["PaperlessBilling"] = df["PaperlessBilling"].map(binary_map)
df["Churn"]         = df["Churn"].map(binary_map)
df.sample()
df["MultipleLines"].unique()
# Handle "No phone service" in MultipleLines
df["MultipleLines"] = df["MultipleLines"].replace({"No phone service": "No"}).map(binary_map)
# One-hot encode remaining categorical features
to_encode = [
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"
]
df = pd.get_dummies(df, columns=to_encode, drop_first=True)
df.head()
# Scale numerical features
scaler = StandardScaler()
for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
    df[col] = scaler.fit_transform(df[[col]])
df.head()
### Model Training
# Trai-test split

X = df.drop(["Churn", "tenure_group"], axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", pd.Series(y_resampled).value_counts())
# rf = RandomForestClassifier(class_weight="balanced", random_state=42)
# param_grid = {
#     "n_estimators": [100, 200],
#     "max_depth": [None, 10, 20],
#     "min_samples_split": [2, 5],
#     "min_samples_leaf": [1, 2],
#     "bootstrap": [True, False],
# }
# grid = GridSearchCV(estimator=rf,param_grid=param_grid,cv=5,scoring="roc_auc",
#     n_jobs=-1,verbose=1)
# grid.fit(X_resampled, y_resampled)
# best_rf = grid.best_estimator_
# print("Best hyperparameters:", grid.best_params_)
# y_pred = best_rf.predict(X_test)
# y_proba = best_rf.predict_proba(X_test)[:, 1]

# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Decision Tree": DecisionTreeClassifier(class_weight='balanced', random_state=42),
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
    "KNN": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42)
}
results = []

for name, model in models.items():
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "F1-Score": f1,
        "ROC AUC": roc
    })
results_df = pd.DataFrame(results).sort_values("F1-Score", ascending=False)
plt.figure(figsize=(12,6))
sns.barplot(x="Model", y="Accuracy", data=results_df, label="Accuracy")
sns.barplot(x="Model", y="F1-Score", data=results_df, label="F1-Score", color="orange")
plt.title("Model Accuracy vs F1 Score")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
print("\n📊 Model Comparison:")
print(results_df.to_string(index=False))

##### Hyperparameter tuning 
gb = GradientBoostingClassifier(random_state=42)

param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [3, 5],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid = GridSearchCV(gb,param_grid,cv=5,scoring='roc_auc',n_jobs=-1, verbose=1)
grid.fit(X_resampled, y_resampled)
best_model = grid.best_estimator_
print("Best Hyperparameters:", grid.best_params_)
# # Confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
# # ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

joblib.dump(best_model, "best_churn_model.pkl")
print("Gradient Boosting model saved!")