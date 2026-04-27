# 🫀 Heart Disease Prediction Model — Documentation

> A machine learning pipeline for binary classification of heart disease presence or absence, trained on clinical patient data using Logistic Regression, Decision Tree, Random Forest, and XGBoost.

---

## 📁 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Project Structure](#3-project-structure)
4. [Pipeline Overview](#4-pipeline-overview)
5. [Data Preprocessing](#5-data-preprocessing)
6. [Feature Engineering](#6-feature-engineering)
7. [Models](#7-models)
   - [Logistic Regression](#71-logistic-regression)
   - [Decision Tree](#72-decision-tree)
   - [Random Forest](#73-random-forest)
   - [XGBoost](#74-xgboost)
8. [Hyperparameter Tuning](#8-hyperparameter-tuning)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Model Comparison](#10-model-comparison)
11. [Saved Artifacts](#11-saved-artifacts)
12. [How to Run](#12-how-to-run)
13. [Dependencies](#13-dependencies)
14. [Future Improvements](#14-future-improvements)

---

## 1. Project Overview

This project builds, tunes, and evaluates multiple machine learning classifiers to predict whether a patient has heart disease based on clinical features. The target variable is binary:

| Label | Meaning |
|---|---|
| `0` | Absence of heart disease |
| `1` | Presence of heart disease |

The pipeline includes data cleaning, class imbalance detection, feature scaling, hyperparameter tuning via `GridSearchCV`, 10-fold cross-validation, and confusion matrix visualisation for every model.

---

## 2. Dataset

| Property | Detail |
|---|---|
| Format | CSV (train/test split) |
| Train path | `data/train.csv` |
| Test path | `data/test.csv` |
| Target column | `Heart Disease` (`Absence` / `Presence`) |

### Features

**Numerical Features**

| Feature | Description |
|---|---|
| `Age` | Patient age in years |
| `BP` | Resting blood pressure (mm Hg) |
| `Cholesterol` | Serum cholesterol (mg/dl) |
| `Max HR` | Maximum heart rate achieved |
| `ST depression` | ST depression induced by exercise relative to rest |

**Categorical Features**

| Feature | Description |
|---|---|
| `Sex` | Patient sex (0 = female, 1 = male) |
| `Chest pain type` | Type of chest pain (1–4) |
| `FBS over 120` | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| `EKG results` | Resting electrocardiographic results (0, 1, 2) |
| `Exercise angina` | Exercise-induced angina (1 = yes, 0 = no) |
| `Slope of ST` | Slope of the peak exercise ST segment |
| `Number of vessels fluro` | Number of major vessels coloured by fluoroscopy (0–3) |
| `Thallium` | Thallium stress test result (3, 6, 7) |

---

## 3. Project Structure

```
Heart-Disease-Prediction-Model/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── heart_disease_refined.py       # Main training script
├── heart_disease_model_docs.md    # This file
│
├── clean_dataset.csv              # Cleaned & scaled dataset (generated)
│
├── logistic_regression_model.pkl  # Saved model (generated)
├── decision_tree_model.pkl        # Saved model (generated)
├── random_forest_model.pkl        # Saved model (generated)
├── xgboost_model.pkl              # Saved model (generated, if XGBoost installed)
└── scaler.pkl                     # Saved StandardScaler (generated)
```

---

## 4. Pipeline Overview

```
Raw CSV Data
     │
     ▼
Drop ID Column
     │
     ▼
Impute Missing Values
(Median → Numerical | Mode → Categorical)
     │
     ▼
Encode Target: Absence→0, Presence→1
     │
     ▼
Class Imbalance Check
(Auto-apply class_weight='balanced' if ratio ≤ 0.7)
     │
     ▼
StandardScaler (fit on train, transform both)
     │
     ▼
Correlation Analysis → Top 5 Features
     │
     ▼
Stratified Train/Validation Split (80/20)
     │
     ▼
GridSearchCV (5-fold) per Model
     │
     ▼
10-Fold Cross-Validation
     │
     ▼
Confusion Matrix + Classification Report
     │
     ▼
Model Comparison & Best Model Selection
```

---

## 5. Data Preprocessing

### 5.1 Dropping Irrelevant Columns

The `id` column is dropped from both train and test sets if present, as it carries no predictive signal.

### 5.2 Missing Value Imputation

Missing values are imputed using statistics derived **from the training set only** to prevent data leakage into the test set.

| Column Type | Strategy |
|---|---|
| Numerical | Replaced with **median** of training column |
| Categorical | Replaced with **mode** of training column |

### 5.3 Target Encoding

```python
train_df['Heart Disease'] = train_df['Heart Disease'].map({
    'Absence': 0,
    'Presence': 1
})
```

### 5.4 Class Imbalance Detection

The pipeline automatically checks whether the target is imbalanced:

$\begin{equation}
\text{balance\_ratio} = \frac{\text{min\_class\_count}}{\text{max\_class\_count}}
\end{equation}$

If $\text{ratio} \leq 0.70$, `class_weight='balanced'` is applied to all applicable models, and `scale_pos_weight` is set for XGBoost. This prevents the model from developing a lazy habit of always predicting the majority class.

### 5.5 Feature Scaling

All features are standardised using `StandardScaler` (zero mean, unit variance):

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
```

> ⚠️ The scaler is **fit on training data only** and then applied to validation/test data — no leakage here.

### 5.6 Train/Validation Split

```python
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
```

`stratify=y` ensures both splits have the same class distribution — especially important for imbalanced datasets.

---

## 6. Feature Engineering

### Correlation Analysis

After scaling, Pearson correlation is computed between all features and the target. The **top 5 most correlated features** (by absolute correlation) are identified and used for a parallel reduced-feature experiment.

```python
top_features = corr['Heart Disease'].abs().sort_values(ascending=False)[1:6]
```

### Feature Importance

After training, Random Forest and XGBoost both expose `feature_importances_`, which are plotted as bar charts to explain which clinical variables drive predictions most.

---

## 7. Models

All models are trained using the `evaluate()` helper function which handles fitting, prediction, 10-fold cross-validation, metric computation, and confusion matrix display.

---

### 7.1 Logistic Regression

**What it is:** A linear probabilistic classifier that estimates the log-odds of heart disease presence.

**Why use it:** Highly interpretable, fast to train, and a great baseline. Works especially well when the decision boundary is roughly linear.

**Key parameters tuned:**

| Parameter | Values Searched | Description |
|---|---|---|
| `C` | 0.01, 0.1, 1, 10, 100 | Inverse of regularisation strength — smaller = stronger regularisation |
| `solver` | lbfgs, liblinear | Optimisation algorithm |
| `penalty` | l2 | Regularisation type |

---

### 7.2 Decision Tree

**What it is:** A tree-based model that splits the feature space using a series of if/else rules learned from the training data.

**Why use it:** Visually interpretable and captures non-linear relationships. However, prone to overfitting without depth constraints — which is why tuning `max_depth` is critical.

**Key parameters tuned:**

| Parameter | Values Searched | Description |
|---|---|---|
| `max_depth` | 3, 5, 7, 10, None | Maximum depth of the tree — prevents overfitting |
| `min_samples_split` | 2, 5, 10, 20 | Minimum samples required to split a node |
| `min_samples_leaf` | 1, 3, 5, 10 | Minimum samples required at a leaf node |
| `criterion` | gini, entropy | Splitting quality measure |

> ⚠️ Without `max_depth`, a Decision Tree will memorise the training set perfectly and generalise poorly. Always constrain it.

---

### 7.3 Random Forest

**What it is:** An ensemble of Decision Trees trained on random subsets of data and features, with predictions made by majority vote.

**Why use it:** More robust than a single tree, handles non-linearity well, and naturally provides feature importance. Usually your safest bet for tabular data.

**Key parameters tuned:**

| Parameter | Values Searched | Description |
|---|---|---|
| `n_estimators` | 100, 200, 300 | Number of trees in the forest |
| `max_depth` | None, 5, 10, 15 | Maximum depth per tree |
| `min_samples_split` | 2, 5, 10 | Minimum samples to split a node |
| `max_features` | sqrt, log2 | Number of features considered at each split |


---

### 7.4 XGBoost

**What it is:** A gradient-boosted ensemble that builds trees sequentially, with each tree correcting the errors of the previous one.

**Why use it:** State-of-the-art on tabular data. Regularisation is built in, it handles missing values natively, and it typically achieves the highest accuracy of the four models.

**Key parameters tuned:**

| Parameter | Values Searched | Description |
|---|---|---|
| `n_estimators` | 100, 200, 300 | Number of boosting rounds |
| `learning_rate` | 0.01, 0.05, 0.1 | Step size — lower = more conservative, less overfitting |
| `max_depth` | 3, 5, 7 | Maximum depth per tree |
| `subsample` | 0.8, 1.0 | Fraction of training samples per tree |

**Class imbalance handling:** `scale_pos_weight = count(negative) / count(positive)`

**Saved as:** `xgboost_model.pkl`

---

## 8. Hyperparameter Tuning

All models are tuned using `GridSearchCV` with 5-fold cross-validation:

```python
grid = GridSearchCV(
    estimator,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1          # Uses all CPU cores — goes brrr 🚀
)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
```

After tuning, the best estimator is passed to `evaluate()` for final assessment on the validation set.

> ⏱️ **Runtime note:** GridSearchCV across all four models can take 5–15 minutes depending on dataset size and machine specs. Consider using `RandomizedSearchCV` to speed things up if needed.

---

## 9. Evaluation Metrics

Each model is evaluated on the following metrics:

| Metric | Formula | What It Tells You |
|---|---|---|
| **Accuracy** | $\frac{(TP + TN)}{Total}$ | Overall correctness — can be misleading if classes are imbalanced |
| **Precision** | $\frac{TP}{(TP + FP)}$ | Of all predicted positives, how many were actually positive? |
| **Recall** | $\frac{TP}{(TP + FN)}$ | Of all actual positives, how many did we catch? Critical in medical settings |
| **F1-Score** | $2 \times \frac{(P \times R)}{(P + R)}$ | Harmonic mean of precision and recall — best single metric when classes are imbalanced |
| **CV Mean** | Mean accuracy across 10 folds | Reliable estimate of generalisation performance |
| **CV Std** | Std dev across 10 folds | Stability — lower is better |

> 💡 **In medical diagnosis, Recall (Sensitivity) matters most.** A false negative (telling a sick patient they're healthy) is far more dangerous than a false positive.

### Confusion Matrix

```
                 Predicted
Actual         Absence      Presence
Absence  [      TN       |      FP      ]
Presence [      FN       |      TP      ]
```
---

## 10. Dependencies


- pandas 🐼
- numpy 🔢
- matplotlib 📊
- scikit-learn 🦏
- xgboost ⚡
- pickle 🔐


Install all at once:

```bash
pip install pandas numpy matplotlib scikit-learn xgboost
```

---