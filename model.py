import pandas as pd
import numpy as np
import joblib
<<<<<<< Updated upstream
=======
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")
>>>>>>> Stashed changes

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

<<<<<<< Updated upstream
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
=======
from sklearn.model_selection import (
    StratifiedKFold, cross_validate, GridSearchCV, train_test_split
)
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    ConfusionMatrixDisplay, confusion_matrix
)

from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
>>>>>>> Stashed changes

import os
os.makedirs("outputs", exist_ok=True)

<<<<<<< Updated upstream
# =========================
# 3. FEATURES
# =========================
numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]

categorical_features = [
    "sex", "cp", "fbs", "restecg",
    "exang", "slope", "ca", "thal"
]
=======
# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv("data/heart_statlog_cleveland_hungary_final.csv")
print(f"Dataset shape: {df.shape}")
print(f"Target distribution:\n{df['target'].value_counts()}\n")
>>>>>>> Stashed changes

X = df.drop("target", axis=1)
y = df["target"]

<<<<<<< Updated upstream
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
=======
# ============================================================
# 2. FEATURES
# ============================================================
numeric_features = [
    "age",
    "resting bp s",
    "cholesterol",
    "max heart rate",
    "oldpeak"
]

categorical_features = [
    "sex",
    "chest pain type",
    "fasting blood sugar",
    "resting ecg",
    "exercise angina",
    "ST slope"
]

# ============================================================
# 3. PREPROCESSORS — Trees don't need scaling, others do
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
    return ColumnTransformer([
        ("num", num_pipe, numeric_features),
        ("cat", cat_pipe, categorical_features)
    ])

# ============================================================
# 4. TRAIN / TEST SPLIT (for final eval + ROC curves)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ============================================================
# 5. CV SETUP  — optimise for RECALL (medical priority)
# ============================================================
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
SCORING = ["accuracy", "roc_auc", "f1", "recall", "precision"]

# ============================================================
# 6. DECISION TREE — GridSearchCV (PRIMARY MODEL)
# ============================================================
print("=" * 60)
print("  GridSearchCV — Decision Tree (Primary Model)")
print("=" * 60)

dt_param_grid = {
    "model__max_depth":        [3, 5, 7, None],
    "model__min_samples_split": [2, 10, 20],
    "model__criterion":        ["gini", "entropy"],
    "model__ccp_alpha":        [0.0, 0.01, 0.05],
}

dt_pipe = Pipeline([
    ("prep",  build_preprocessor(scale=False)),   # trees don't need scaling
    ("model", DecisionTreeClassifier(random_state=42))
])

grid_search = GridSearchCV(
    dt_pipe,
    dt_param_grid,
    cv=cv,
    scoring="recall",          # medical priority → recall first
    refit=True,
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train, y_train)

best_dt = grid_search.best_estimator_
print(f"\n✅ Best DT params: {grid_search.best_params_}")
print(f"   CV Recall (best): {grid_search.best_score_:.4f}\n")

# ============================================================
# 7. BASELINE MODELS
# ============================================================
baselines = {
    "Random Forest (n=100)":  Pipeline([("prep", build_preprocessor()), ("model", RandomForestClassifier(n_estimators=100, random_state=42))]),
    "Random Forest (n=300)":  Pipeline([("prep", build_preprocessor()), ("model", RandomForestClassifier(n_estimators=300, random_state=42))]),
    "Gradient Boosting":      Pipeline([("prep", build_preprocessor()), ("model", GradientBoostingClassifier(random_state=42))]),
    "Logistic Regression":    Pipeline([("prep", build_preprocessor()), ("model", LogisticRegression(max_iter=1000))]),
    "SVM (RBF)":              Pipeline([("prep", build_preprocessor()), ("model", SVC(probability=True, random_state=42))]),
}

# ============================================================
# 8. CROSS-VALIDATION COMPARISON
# ============================================================
print("=" * 60)
print("  10-Fold CV — Model Comparison")
print("=" * 60)

results = []

# Decision Tree (best from grid)
dt_scores = cross_validate(best_dt, X, y, cv=cv, scoring=SCORING)
results.append({
    "Model":     "Decision Tree ★",
    "Recall":    np.mean(dt_scores["test_recall"]),
    "F1":        np.mean(dt_scores["test_f1"]),
    "Precision": np.mean(dt_scores["test_precision"]),
    "Accuracy":  np.mean(dt_scores["test_accuracy"]),
    "ROC-AUC":   np.mean(dt_scores["test_roc_auc"]),
})

for name, pipe in baselines.items():
    scores = cross_validate(pipe, X, y, cv=cv, scoring=SCORING)
    results.append({
        "Model":     name,
        "Recall":    np.mean(scores["test_recall"]),
        "F1":        np.mean(scores["test_f1"]),
        "Precision": np.mean(scores["test_precision"]),
        "Accuracy":  np.mean(scores["test_accuracy"]),
        "ROC-AUC":   np.mean(scores["test_roc_auc"]),
    })
>>>>>>> Stashed changes

results_df = pd.DataFrame(results).sort_values("Recall", ascending=False)
print("\n" + results_df.to_string(index=False, float_format="{:.4f}".format))

<<<<<<< Updated upstream
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
=======
# ============================================================
# 9. FIT ALL MODELS ON TRAIN SET FOR TEST-SET EVALUATION
# ============================================================
all_models = {"Decision Tree ★": best_dt}
all_models.update(baselines)

fitted = {}
for name, pipe in all_models.items():
    pipe.fit(X_train, y_train)
    fitted[name] = pipe

# ============================================================
# 10. CLASSIFICATION REPORTS
# ============================================================
print("\n" + "=" * 60)
print("  Classification Reports (Test Set)")
print("=" * 60)

for name, pipe in fitted.items():
    y_pred = pipe.predict(X_test)
    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

# ============================================================
# 11. ROC-AUC CURVES — ALL MODELS
# ============================================================
print("\nGenerating ROC-AUC curves...")

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor("#0f172a")
ax.set_facecolor("#1e293b")

colors = ["#f43f5e", "#3b82f6", "#22c55e", "#f59e0b", "#a855f7", "#06b6d4"]

for (name, pipe), color in zip(fitted.items(), colors):
    y_prob = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    lw = 2.5 if "★" in name else 1.5
    ls = "-" if "★" in name else "--"
    ax.plot(fpr, tpr, color=color, lw=lw, ls=ls,
            label=f"{name}  (AUC = {auc:.3f})")

ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.4, label="Random Classifier")

ax.set_xlabel("False Positive Rate", color="white", fontsize=12)
ax.set_ylabel("True Positive Rate (Recall)", color="white", fontsize=12)
ax.set_title("ROC-AUC Curves — All Models\n(Optimised for Recall · Medical Use)",
             color="white", fontsize=14, fontweight="bold", pad=15)
ax.tick_params(colors="white")
ax.spines["bottom"].set_color("#334155")
ax.spines["left"].set_color("#334155")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="lower right", framealpha=0.2, labelcolor="white",
          facecolor="#0f172a", edgecolor="#334155", fontsize=9)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
ax.grid(True, color="#334155", alpha=0.4, linewidth=0.5)

plt.tight_layout()
plt.savefig("outputs/roc_auc_curves.png", dpi=150, bbox_inches="tight",
            facecolor="#0f172a")
plt.close()
print("  ✅ Saved: outputs/roc_auc_curves.png")

# ============================================================
# 12. CONFUSION MATRIX — DECISION TREE (PRIMARY)
# ============================================================
y_pred_dt = best_dt.predict(X_test)
cm = confusion_matrix(y_test, y_pred_dt)

fig, ax = plt.subplots(figsize=(6, 5))
fig.patch.set_facecolor("#0f172a")
ax.set_facecolor("#1e293b")

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["No Disease", "Disease"])
disp.plot(ax=ax, colorbar=False, cmap="RdYlGn")
ax.set_title("Confusion Matrix — Decision Tree ★",
             color="white", fontsize=13, fontweight="bold", pad=12)
ax.tick_params(colors="white")
ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")
for text in disp.text_.ravel():
    text.set_color("black")
    text.set_fontsize(14)
    text.set_fontweight("bold")

plt.tight_layout()
plt.savefig("outputs/confusion_matrix_dt.png", dpi=150, bbox_inches="tight",
            facecolor="#0f172a")
plt.close()
print("  ✅ Saved: outputs/confusion_matrix_dt.png")

# ============================================================
# 13. DECISION TREE VISUALISATION
# ============================================================
print("\nGenerating Decision Tree visualisation...")

# Get feature names after preprocessing
prep_fitted = best_dt.named_steps["prep"]
num_names = numeric_features
cat_names = prep_fitted.named_transformers_["cat"] \
                       .named_steps["onehot"] \
                       .get_feature_names_out(categorical_features).tolist()
feature_names = num_names + cat_names

dt_model = best_dt.named_steps["model"]

fig, ax = plt.subplots(figsize=(24, 12))
fig.patch.set_facecolor("#0f172a")
ax.set_facecolor("#0f172a")

plot_tree(
    dt_model,
    feature_names=feature_names,
    class_names=["No Disease", "Disease"],
    filled=True,
    rounded=True,
    max_depth=4,           # cap depth for readability
    fontsize=8,
    ax=ax,
    impurity=True,
    proportion=False,
)
ax.set_title("Pruned Decision Tree — CardioCare\n(Displayed up to depth 4 for readability)",
             color="white", fontsize=16, fontweight="bold", pad=20)

plt.tight_layout()
plt.savefig("outputs/decision_tree_viz.png", dpi=120, bbox_inches="tight",
            facecolor="#0f172a")
plt.close()
print("  ✅ Saved: outputs/decision_tree_viz.png")

# ============================================================
# 14. IF-ELSE RULES EXPORT
# ============================================================
rules = export_text(dt_model, feature_names=feature_names, max_depth=5)
with open("outputs/decision_tree_rules.txt", "w") as f:
    f.write("=" * 60 + "\n")
    f.write("  CardioCare — Decision Tree If-Else Rules\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Best Params: {grid_search.best_params_}\n\n")
    f.write(rules)

print("  ✅ Saved: outputs/decision_tree_rules.txt")

# ============================================================
# 15. METRICS COMPARISON BAR CHART
# ============================================================
metrics_to_plot = ["Recall", "F1", "Precision", "Accuracy", "ROC-AUC"]
x = np.arange(len(metrics_to_plot))
n_models = len(results_df)
width = 0.13

fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor("#0f172a")
ax.set_facecolor("#1e293b")

bar_colors = ["#f43f5e", "#3b82f6", "#22c55e", "#f59e0b", "#a855f7", "#06b6d4"]

for i, (_, row) in enumerate(results_df.iterrows()):
    offset = (i - n_models / 2) * width + width / 2
    vals = [row[m] for m in metrics_to_plot]
    bars = ax.bar(x + offset, vals, width, label=row["Model"],
                  color=bar_colors[i % len(bar_colors)], alpha=0.85,
                  edgecolor="none")

ax.set_xticks(x)
ax.set_xticklabels(metrics_to_plot, color="white", fontsize=11)
ax.tick_params(axis="y", colors="white")
ax.set_ylim(0.5, 1.05)
ax.set_ylabel("Score", color="white", fontsize=12)
ax.set_title("Model Comparison — Recall › F1 › Precision › Accuracy › ROC-AUC",
             color="white", fontsize=13, fontweight="bold", pad=15)
ax.legend(loc="lower right", framealpha=0.2, labelcolor="white",
          facecolor="#0f172a", edgecolor="#334155", fontsize=8)
ax.grid(True, axis="y", color="#334155", alpha=0.4, linewidth=0.5)
for spine in ax.spines.values():
    spine.set_color("#334155")

plt.tight_layout()
plt.savefig("outputs/model_comparison.png", dpi=150, bbox_inches="tight",
            facecolor="#0f172a")
plt.close()
print("  ✅ Saved: outputs/model_comparison.png")

# ============================================================
# 16. SAVE BEST MODEL
# ============================================================
joblib.dump(best_dt, "heart_model.pkl")
print("\n💾 Model saved as heart_model.pkl")

# ============================================================
# 17. FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("  FINAL SUMMARY")
print("=" * 60)
print(results_df[["Model", "Recall", "F1", "Precision", "Accuracy", "ROC-AUC"]]
      .to_string(index=False, float_format="{:.4f}".format))
print("\n📁 Outputs saved in: outputs/")
print("   • roc_auc_curves.png")
print("   • confusion_matrix_dt.png")
print("   • decision_tree_viz.png")
print("   • decision_tree_rules.txt")
print("   • model_comparison.png")
print("   • heart_model.pkl")
>>>>>>> Stashed changes
