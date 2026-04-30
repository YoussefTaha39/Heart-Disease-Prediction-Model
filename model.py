# ==============================
# 1. Import Libraries
# ==============================
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.utils import class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
)


# ==============================
# 2. Load Data
# ==============================
train_df = pd.read_csv("/Users/kevinharvey/Desktop/Projects/ML Project/Heart-Disease-Prediction-Model/data/train.csv")
test_df  = pd.read_csv("/Users/kevinharvey/Desktop/Projects/ML Project/Heart-Disease-Prediction-Model/data/test.csv")

print("Shape of training data:", train_df.shape)
print("Shape of test data    :", test_df.shape)
print("\nTraining Columns:\n", train_df.columns.tolist())

# ==============================
# 3. Data Cleaning
# ==============================
for df in [train_df, test_df]:
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)

print("\nMissing Values — Training:\n", train_df.isnull().sum())
print("\nMissing Values — Test:\n",     test_df.isnull().sum())

num_cols = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']
cat_cols = ['Sex', 'Chest pain type', 'FBS over 120', 'EKG results',
            'Exercise angina', 'Slope of ST', 'Number of vessels fluro', 'Thallium']

for col in num_cols:
    median_val = train_df[col].median()
    train_df[col].fillna(median_val, inplace=True)
    test_df[col].fillna(median_val, inplace=True)

for col in cat_cols:
    mode_val = train_df[col].mode()[0]
    train_df[col].fillna(mode_val, inplace=True)
    test_df[col].fillna(mode_val, inplace=True)

# ==============================
# 4. Target Encoding
# ==============================
train_df['Heart Disease'] = train_df['Heart Disease'].map({'Absence': 0, 'Presence': 1})

print("\nTarget Distribution:\n", train_df['Heart Disease'].value_counts())

# ==============================
# 5. Class Imbalance Check
# ==============================
counts = train_df['Heart Disease'].value_counts()
ratio  = counts.min() / counts.max()
print(f"\nClass balance ratio: {ratio:.2f}  ({'balanced ✅' if ratio > 0.7 else 'imbalanced ⚠️  — applying class_weight=balanced'})")

use_class_weight = 'balanced' if ratio <= 0.7 else None

# ==============================
# 6. Feature Scaling
# ==============================
X = train_df.drop('Heart Disease', axis=1)
y = train_df['Heart Disease']

scaler  = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# ==============================
# 7. Correlation & Top Features
# ==============================
corr_df = X_scaled.copy()
corr_df['Heart Disease'] = y.values
corr = corr_df.corr()

top_features = corr['Heart Disease'].abs().sort_values(ascending=False)[1:6]
print("\nTop 5 Correlated Features:\n", top_features)

# ==============================
# 8. Train / Validation Split
# ==============================
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}  |  Val size: {X_val.shape[0]}")


# ==============================
# 10. Logistic Regression
# ==============================
lr_params = {
    'C'      : [0.01, 0.1, 1, 10, 100],
    'solver' : ['lbfgs', 'liblinear'],
    'penalty': ['l2'],
}
lr_grid = GridSearchCV(
    LogisticRegression(max_iter=1000, class_weight=use_class_weight, random_state=42),
    lr_params, cv=5, scoring='accuracy', n_jobs=-1
)
lr_grid.fit(X_train, y_train)
print(f"\nBest LR params : {lr_grid.best_params_}")
# ==============================
# 11. Decision Tree 
# ==============================
dt_params = {
    'max_depth'       : [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf' : [1, 3, 5, 10],
    'criterion'        : ['gini', 'entropy'],
}
dt_grid = GridSearchCV(
    DecisionTreeClassifier(class_weight=use_class_weight, random_state=42),
    dt_params, cv=5, scoring='accuracy', n_jobs=-1
)
dt_grid.fit(X_train, y_train)
print(f"\nBest DT params : {dt_grid.best_params_}")
dt_model, dt_metrics = evaluate(
    "Decision Tree (Tuned)", dt_grid.best_estimator_,
    X_train, y_train, X_val, y_val
)

# ==============================
# 12. Random Forest 
# ==============================
rf_params = {
    'n_estimators'    : [100, 200, 300],
    'max_depth'       : [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'max_features'    : ['sqrt', 'log2'],
}
rf_grid = GridSearchCV(
    RandomForestClassifier(class_weight=use_class_weight, random_state=42),
    rf_params, cv=5, scoring='accuracy', n_jobs=-1
)
rf_grid.fit(X_train, y_train)
print(f"\nBest RF params : {rf_grid.best_params_}")


# ==============================
# 13. XGBoost 
# ==============================

xgb_params = {
    'n_estimators'  : [100, 200, 300],
    'learning_rate' : [0.01, 0.05, 0.1],
    'max_depth'     : [3, 5, 7],
    'subsample'     : [0.8, 1.0],
}
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
xgb_grid = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss',
                  scale_pos_weight=scale_pos if use_class_weight else 1),
    xgb_params, cv=5, scoring='accuracy', n_jobs=-1
)
xgb_grid.fit(X_train, y_train)
print(f"\nBest XGB params: {xgb_grid.best_params_}")
