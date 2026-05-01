# ==============================
# 1. IMPORT LIBRARIES
# ==============================
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

# ==============================
# 2. LOAD DATA
# ==============================
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

print("Training Shape:", train_df.shape)
print("Test Shape:", test_df.shape)
print(train_df.head())
print(train_df.info())

# ==============================
# 3. DATA CLEANING
# ==============================
for df in [train_df, test_df]:
    # Drop ID column if exists
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

# Define columns
num_cols = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']
cat_cols = [
    'Sex', 'Chest pain type', 'FBS over 120', 'EKG results',
    'Exercise angina', 'Slope of ST',
    'Number of vessels fluro', 'Thallium'
]

# Fill missing numeric values with median
for col in num_cols:
    median_val = train_df[col].median()
    train_df[col].fillna(median_val, inplace=True)
    test_df[col].fillna(median_val, inplace=True)

# Fill missing categorical values with mode
for col in cat_cols:
    mode_val = train_df[col].mode()[0]
    train_df[col].fillna(mode_val, inplace=True)
    test_df[col].fillna(mode_val, inplace=True)

# ==============================
# 4. TARGET ENCODING
# ==============================
train_df['Heart Disease'] = train_df['Heart Disease'].map({
    'Absence': 0,
    'Presence': 1
})

print("\nTarget Distribution:\n", train_df['Heart Disease'].value_counts())

# ==============================
# 5. CLASS IMBALANCE CHECK
# ==============================
counts = train_df['Heart Disease'].value_counts()
ratio = counts.min() / counts.max()

print(f"Class balance ratio: {ratio:.2f}")

use_class_weight = 'balanced' if ratio <= 0.7 else None

# ==============================
# 6. EXPLORATORY DATA ANALYSIS (EDA)
# ==============================
print("\nStatistical Summary:\n", train_df.describe())

train_df.hist(figsize=(14,10))
plt.tight_layout()
plt.show()

plt.figure(figsize=(14,6))
sns.boxplot(data=train_df)
plt.xticks(rotation=90)
plt.title("Feature Boxplots")
plt.show()

plt.figure(figsize=(14,10))
sns.heatmap(train_df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ==============================
# 7. FEATURE / TARGET SPLIT
# ==============================
X = train_df.drop('Heart Disease', axis=1)
y = train_df['Heart Disease']

# ==============================
# 8. FEATURE SCALING
# ==============================
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns
)

# ==============================
# 9. TOP FEATURE CORRELATION
# ==============================
corr_df = X_scaled.copy()
corr_df['Heart Disease'] = y.values

corr = corr_df.corr()
top_features = corr['Heart Disease'].abs().sort_values(ascending=False)[1:11]

print("\nTop Correlated Features:\n", top_features)

# ==============================
# 10. TRAIN / VALIDATION SPLIT
# ==============================
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Train Size: {X_train.shape[0]}")
print(f"Validation Size: {X_val.shape[0]}")

# ==============================
# 11. MODEL DEFINITIONS + GRIDSEARCH
# ==============================
models = {
    "Logistic Regression": GridSearchCV(
        LogisticRegression(
            max_iter=1000,
            class_weight=use_class_weight,
            random_state=42
        ),
        {
            "C": [0.01, 0.1, 1, 10, 100],
            "solver": ["liblinear", "lbfgs"],
            "penalty": ["l2"]
        },
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    ),

    "Decision Tree": GridSearchCV(
        DecisionTreeClassifier(
            class_weight=use_class_weight,
            random_state=42
        ),
        {
            "max_depth": [3, 5, 7, 10, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 3, 5],
            "criterion": ["gini", "entropy"]
        },
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    ),

    "Random Forest": GridSearchCV(
        RandomForestClassifier(
            class_weight=use_class_weight,
            random_state=42
        ),
        {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "max_features": ["sqrt", "log2"]
        },
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    ),

    "XGBoost": GridSearchCV(
        XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()
            if use_class_weight else 1
        ),
        {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
            "subsample": [0.8, 1.0]
        },
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
}

# ==============================
# 12. EVALUATION FUNCTION
# ==============================
def evaluate_model(name, model):
    print(f"\n{'='*50}")
    print(f"🚀 Training {name}")
    print(f"{'='*50}")

    model.fit(X_train, y_train)
    best_model = model.best_estimator_

    print("Best Parameters:", model.best_params_)

    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)

    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)

    print(f"Train Accuracy : {train_acc:.4f}")
    print(f"Val Accuracy   : {val_acc:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1 Score       : {f1:.4f}")
    print(f"CV Mean        : {cv_scores.mean():.4f}")
    print(f"CV Std         : {cv_scores.std():.4f}")

    print("\nClassification Report:\n")
    print(classification_report(y_val, y_val_pred))

    ConfusionMatrixDisplay.from_predictions(y_val, y_val_pred)
    plt.title(f"{name} Confusion Matrix")
    plt.show()

    return {
        "Model": name,
        "Train Acc": train_acc,
        "Val Acc": val_acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "CV Mean": cv_scores.mean(),
        "CV Std": cv_scores.std(),
        "Model_Object": best_model
    }

# ==============================
# 13. TRAIN ALL MODELS
# ==============================
results_list = []

for name, model in models.items():
    results_list.append(evaluate_model(name, model))

results = pd.DataFrame(results_list)

print("\nFINAL MODEL RESULTS:")
print(results[[
    "Model", "Val Acc", "Precision",
    "Recall", "F1", "CV Mean", "CV Std"
]])

# ==============================
# 14. VISUAL MODEL COMPARISON
# ==============================
metrics = ['Val Acc', 'Precision', 'Recall', 'F1']
results.set_index('Model')[metrics].plot(
    kind='bar',
    figsize=(12,8)
)
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# ==============================
# 15. BEST MODEL SELECTION
# ==============================
best_row = results.sort_values(by='Recall', ascending=False).iloc[0]
final_model = best_row['Model_Object']

print(f"\n🏆 Best Model Selected: {best_row['Model']}")

# ==============================
# 16. FEATURE IMPORTANCE
# ==============================
if hasattr(final_model, 'feature_importances_'):
    importances = final_model.feature_importances_
elif hasattr(final_model, 'coef_'):
    importances = np.abs(final_model.coef_[0])
else:
    importances = None

if importances is not None:
    importances = importances / np.sum(importances)

    feat_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print("\nTop Feature Importances:\n", feat_df.head(10))

    plt.figure(figsize=(10,6))
    plt.barh(feat_df['Feature'][:10], feat_df['Importance'][:10])
    plt.gca().invert_yaxis()
    plt.title("Top 10 Important Features")
    plt.show()

# ==============================
# 17. ROC CURVE + AUC
# ==============================
y_probs = final_model.predict_proba(X_val)[:, 1]

fpr, tpr, _ = roc_curve(y_val, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], '--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

print(f"\n📊 Final AUC Score: {roc_auc:.4f}")

# ==============================
# 18. SAMPLE PREDICTION
# ==============================
sample = X_val.iloc[0:1]
prediction = final_model.predict(sample)

print("\n🔮 Sample Prediction:", prediction)

# ==============================
# 19. SAVE MODELS + SCALER
# ==============================
joblib.dump(final_model, "best_heart_model.pkl")
joblib.dump(scaler, "scaler.pkl")

all_models = {
    'logistic_regression_model.pkl': results_list[0]['Model_Object'],
    'decision_tree_model.pkl': results_list[1]['Model_Object'],
    'random_forest_model.pkl': results_list[2]['Model_Object'],
    'xgboost_model.pkl': results_list[3]['Model_Object']
}

for filename, model_obj in all_models.items():
    with open(filename, 'wb') as f:
        pickle.dump(model_obj, f)
    print(f"✅ Saved: {filename}")

print("✅ Saved: scaler.pkl")
print("\n🎉 ALL MODELS TRAINED, EVALUATED, AND SAVED SUCCESSFULLY!")