# ============================================================
# HEART DISEASE PREDICTION PROJECT
# Full Production-Style ML Workflow
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import os

warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_validate
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)

from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree,
    export_text
)

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# ============================================================
# 2. CREATE OUTPUT FOLDER
# ============================================================

os.makedirs("outputs", exist_ok=True)

# ============================================================
# 3. LOAD DATA
# ============================================================

df = pd.read_csv("data/heart_statlog_cleveland_hungary_final.csv")

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicates:")
print(df.duplicated().sum())

print("\nTarget Distribution:")
print(df["target"].value_counts())

# ============================================================
# 4. EDA (EXPLORATORY DATA ANALYSIS)
# ============================================================

# ------------------------------------------------------------
# Target Distribution
# ------------------------------------------------------------

plt.figure(figsize=(6,4))

sns.countplot(x="target", data=df)

plt.title("Target Distribution")
plt.xlabel("Heart Disease")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    "outputs/target_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# Correlation Heatmap
# ------------------------------------------------------------

plt.figure(figsize=(12,8))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "outputs/correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# Histograms
# ------------------------------------------------------------

numeric_cols = [
    "age",
    "resting bp s",
    "cholesterol",
    "max heart rate",
    "oldpeak"
]

for col in numeric_cols:

    plt.figure(figsize=(6,4))

    sns.histplot(df[col], kde=True)

    plt.title(f"{col} Distribution")

    plt.tight_layout()

    plt.savefig(
        f"outputs/{col}_histogram.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ------------------------------------------------------------
# Boxplots
# ------------------------------------------------------------

for col in numeric_cols:

    plt.figure(figsize=(6,4))

    sns.boxplot(x=df[col])

    plt.title(f"{col} Boxplot")

    plt.tight_layout()

    plt.savefig(
        f"outputs/{col}_boxplot.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ------------------------------------------------------------
# Chest Pain vs Target
# ------------------------------------------------------------

plt.figure(figsize=(7,5))

sns.countplot(
    x="chest pain type",
    hue="target",
    data=df
)

plt.title("Chest Pain Type vs Heart Disease")

plt.tight_layout()

plt.savefig(
    "outputs/chest_pain_vs_target.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# Sex vs Target
# ------------------------------------------------------------

plt.figure(figsize=(6,4))

sns.countplot(
    x="sex",
    hue="target",
    data=df
)

plt.title("Sex vs Heart Disease")

plt.tight_layout()

plt.savefig(
    "outputs/sex_vs_target.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# 5. FEATURES & TARGET
# ============================================================

X = df.drop("target", axis=1)
y = df["target"]

# ============================================================
# 6. FEATURE GROUPS
# ============================================================

numeric_features = [
    "age",
    "resting bp s",
    "cholesterol",
    "max heart rate",
    "oldpeak"
]

categorical_features = [
    "chest pain type",
    "resting ecg",
    "ST slope"
]

binary_features = [
    "sex",
    "fasting blood sugar",
    "exercise angina"
]

# ============================================================
# 7. PREPROCESSING PIPELINES
# ============================================================

def build_preprocessor(scale=True):

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        *([("scaler", StandardScaler())] if scale else [])
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, numeric_features),
        ("cat", cat_pipe, categorical_features),
        ("bin", "passthrough", binary_features)
    ])

    return preprocessor

# ============================================================
# 8. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ============================================================
# 9. CROSS VALIDATION SETUP
# ============================================================

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

SCORING = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc"
]

# ============================================================
# 10. DECISION TREE (PRIMARY MODEL)
# ============================================================

print("\n" + "=" * 60)
print("GRIDSEARCHCV - DECISION TREE")
print("=" * 60)

dt_pipe = Pipeline([
    ("prep", build_preprocessor(scale=False)),
    ("model", DecisionTreeClassifier(random_state=42))
])

dt_param_grid = {

    "model__max_depth": [3, 5, 7, None],

    "model__min_samples_split": [2, 10, 20],

    "model__criterion": ["gini", "entropy"],

    "model__ccp_alpha": [0.0, 0.01, 0.05]
}

grid_search = GridSearchCV(
    estimator=dt_pipe,
    param_grid=dt_param_grid,
    cv=cv,
    scoring="recall",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_dt = grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Recall:")
print(grid_search.best_score_)

# ============================================================
# 11. MODELS
# ============================================================

models = {

    "Decision Tree": best_dt,

    "Random Forest": Pipeline([
        ("prep", build_preprocessor(scale=False)),
        ("model", RandomForestClassifier(
            n_estimators=300,
            random_state=42
        ))
    ]),

    "Gradient Boosting": Pipeline([
        ("prep", build_preprocessor()),
        ("model", GradientBoostingClassifier(
            random_state=42
        ))
    ]),

    "Logistic Regression": Pipeline([
        ("prep", build_preprocessor()),
        ("model", LogisticRegression(
            max_iter=1000
        ))
    ]),

    "SVM (RBF)": Pipeline([
        ("prep", build_preprocessor()),
        ("model", SVC(
            probability=True,
            kernel="rbf",
            random_state=42
        ))
    ])
}

# ============================================================
# 12. CROSS VALIDATION COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

results = []

for name, model in models.items():

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=SCORING
    )

    results.append({

        "Model": name,

        "Accuracy": np.mean(scores["test_accuracy"]),

        "Precision": np.mean(scores["test_precision"]),

        "Recall": np.mean(scores["test_recall"]),

        "F1 Score": np.mean(scores["test_f1"]),

        "ROC-AUC": np.mean(scores["test_roc_auc"])
    })

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Recall",
    ascending=False
)

print("\nFinal Results:\n")

print(results_df)

# ============================================================
# 13. TRAIN ALL MODELS
# ============================================================

fitted_models = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    fitted_models[name] = model

# ============================================================
# 14. CLASSIFICATION REPORTS
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORTS")
print("=" * 60)

for name, model in fitted_models.items():

    y_pred = model.predict(X_test)

    print(f"\n{name}")
    print("-" * 40)

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["No Disease", "Disease"]
        )
    )

# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

for name, model in fitted_models.items():

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5,4))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Disease", "Disease"]
    )

    disp.plot(ax=ax)

    plt.title(f"{name} - Confusion Matrix")

    plt.tight_layout()

    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").lower()

    plt.savefig(
        f"outputs/confusion_matrix_{safe_name}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# 16. ROC CURVES
# ============================================================

plt.figure(figsize=(10,7))

for name, model in fitted_models.items():

    y_prob = model.predict_proba(X_test)[:,1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    auc = roc_auc_score(y_test, y_prob)

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {auc:.3f})"
    )

plt.plot([0,1], [0,1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison")

plt.legend()

plt.tight_layout()

plt.savefig(
    "outputs/roc_curve_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# 17. FEATURE IMPORTANCE
# ============================================================

rf_model = fitted_models["Random Forest"]

rf_classifier = rf_model.named_steps["model"]

prep = rf_model.named_steps["prep"]

cat_names = prep.named_transformers_["cat"] \
    .named_steps["onehot"] \
    .get_feature_names_out(categorical_features)

feature_names = (
    numeric_features
    + list(cat_names)
    + binary_features
)

importance_df = pd.DataFrame({

    "Feature": feature_names,

    "Importance": rf_classifier.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop Important Features:\n")

print(importance_df.head(10))

plt.figure(figsize=(10,6))

sns.barplot(
    data=importance_df.head(10),
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Important Features")

plt.tight_layout()

plt.savefig(
    "outputs/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# 18. DECISION TREE VISUALIZATION
# ============================================================

dt_model = best_dt.named_steps["model"]

prep_dt = best_dt.named_steps["prep"]

cat_names_dt = prep_dt.named_transformers_["cat"] \
    .named_steps["onehot"] \
    .get_feature_names_out(categorical_features)

feature_names_dt = (
    numeric_features
    + list(cat_names_dt)
    + binary_features
)

plt.figure(figsize=(20,10))

plot_tree(
    dt_model,
    feature_names=feature_names_dt,
    class_names=["No Disease", "Disease"],
    filled=True,
    rounded=True,
    fontsize=8,
    max_depth=4
)

plt.title("Decision Tree Visualization")

plt.tight_layout()

plt.savefig(
    "outputs/decision_tree_visualization.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# 19. EXPORT DECISION TREE RULES
# ============================================================

rules = export_text(
    dt_model,
    feature_names=feature_names_dt,
    max_depth=5
)

with open("outputs/decision_tree_rules.txt", "w") as f:
    f.write(rules)

print("\nDecision Tree Rules Saved!")

# ============================================================
# 20. MODEL COMPARISON BAR CHART
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]

results_plot = results_df.set_index("Model")

results_plot[metrics].plot(
    kind="bar",
    figsize=(12,6)
)

plt.title("Model Comparison")

plt.ylabel("Score")

plt.ylim(0.5, 1.0)

plt.xticks(rotation=0)

plt.grid(axis="y")

plt.tight_layout()

plt.savefig(
    "outputs/model_comparison_chart.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# 21. SAVE BEST MODEL (RANDOM FOREST)
# ============================================================

joblib.dump(
    fitted_models["Random Forest"],
    "outputs/heart_disease_model.pkl"
)

print("\nRandom Forest model saved successfully!")

# ============================================================
# 22. SAVE FEATURE NAMES
# ============================================================

joblib.dump(
    feature_names,
    "outputs/feature_names.pkl"
)

print("Feature names saved successfully!")

# ============================================================
# 23. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(results_df.to_string(index=False))

print("\nAll outputs saved inside outputs/ folder")