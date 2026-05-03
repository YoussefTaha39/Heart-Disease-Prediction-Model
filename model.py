import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.model_selection import StratifiedKFold, cross_validate

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("heart_cleveland_upload.csv")
# =========================
# 2. TARGET
# =========================
df["target"] = df["condition"].apply(lambda x: 1 if x > 0 else 0)
df = df.drop(columns=["condition"])

X = df.drop("target", axis=1)
y = df["target"]


# =========================
# 3. FEATURES
# =========================
numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]

categorical_features = [
    "sex", "cp", "fbs", "restecg",
    "exang", "slope", "ca", "thal"
]


# =========================
# 4. PREPROCESSING
# =========================
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", categorical_pipe, categorical_features)
])

# =========================
# 5. MODELS
# =========================
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=5, ccp_alpha=0.01, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM (RBF)": SVC(probability=True)
}

# =========================
# 6. CROSS VALIDATION
# =========================
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

results = []

print("\n=== TRAINING MODELS ===\n")

best_model_name = None
best_score = -1
best_pipeline = None

for name, model in models.items():
    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    scores = cross_validate(
        pipe,
        X,
        y,
        cv=cv,
        scoring=["accuracy", "roc_auc", "f1", "recall"]
    )

    mean_auc = np.mean(scores["test_roc_auc"])

    results.append({
        "Model": name,
        "Accuracy": np.mean(scores["test_accuracy"]),
        "ROC-AUC": mean_auc,
        "F1": np.mean(scores["test_f1"]),
        "Recall": np.mean(scores["test_recall"])
    })

    print(f"{name} → ROC-AUC: {mean_auc:.4f}")

    # Track best model
    if mean_auc > best_score:
        best_score = mean_auc
        best_model_name = name
        best_pipeline = pipe


# =========================
# 7. RESULTS TABLE
# =========================
results_df = pd.DataFrame(results)

print("\n=== MODEL COMPARISON ===")
print(results_df)
# =========================
# 8. TRAIN BEST MODEL ON FULL DATA
# =========================
print(f"\n✅ Best Model: {best_model_name}")

best_pipeline.fit(X, y)

# =========================
# 9. SAVE MODEL (FOR FLASK)
# =========================
joblib.dump(best_pipeline, "heart_model.pkl")

print("\n💾 Model saved as heart_model.pkl")