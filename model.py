# =============================================================================
# Heart Disease Classifier Pipeline
# =============================================================================

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


RANDOM_STATE = 42
TARGET = "Heart Disease"
ARTIFACT_DIR = Path("artifacts")


def build_preprocessor():
    numeric_features = ["Age", "BP", "Cholesterol", "Max HR", "ST depression"]
    categorical_features = [
        "Sex",
        "Chest pain type",
        "FBS over 120",
        "EKG results",
        "Exercise angina",
        "Slope of ST",
        "Number of vessels fluro",
        "Thallium",
    ]

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ]
    )


def load_training_data():
    train_df = pd.read_csv("data/train.csv")

    # Keep the existing project data flow. The file is large, so train on a
    # reproducible sample while preserving the current dataset instead of the
    # separate Cleveland CSV.
    train_df = train_df.sample(frac=0.1, random_state=RANDOM_STATE)
    train_df[TARGET] = train_df[TARGET].map({"Absence": 0, "Presence": 1})

    feature_cols = [col for col in train_df.columns if col not in [TARGET, "id", "Id"]]
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=train_df[TARGET],
    )

    return (
        train_df[feature_cols],
        train_df[TARGET],
        val_df[feature_cols],
        val_df[TARGET],
    )


def tune_decision_tree(preprocessor, cv, X_train, y_train):
    tree_pipeline = Pipeline(
        [
            ("prep", preprocessor),
            ("model", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]
    )

    param_grid = {
        "model__max_depth": [3, 5, 7, None],
        "model__min_samples_split": [2, 10, 20],
        "model__criterion": ["gini", "entropy"],
        "model__ccp_alpha": [0.0, 0.01, 0.05],
    }

    grid = GridSearchCV(
        tree_pipeline,
        param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)

    print("\n=== DECISION TREE GRID SEARCH ===")
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV ROC-AUC: {grid.best_score_:.4f}")

    return grid.best_estimator_, grid.best_params_, grid.best_score_


def evaluate_models(preprocessor, tuned_tree, cv, X_train, y_train):
    models = {
        "Decision Tree (tuned/pruned)": tuned_tree,
        "Random Forest (n=100)": Pipeline(
            [
                ("prep", preprocessor),
                ("model", RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest (n=300)": Pipeline(
            [
                ("prep", preprocessor),
                ("model", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE)),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("prep", preprocessor),
                ("model", GradientBoostingClassifier(random_state=RANDOM_STATE)),
            ]
        ),
        "Logistic Regression": Pipeline(
            [
                ("prep", preprocessor),
                ("model", LogisticRegression(max_iter=1000, n_jobs=-1)),
            ]
        ),
        "SVM (RBF)": Pipeline(
            [
                ("prep", preprocessor),
                ("model", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
            ]
        ),
    }

    results = []
    best_name = None
    best_score = -1
    best_pipeline = None

    print("\n=== TRAINING MODELS (10-Fold Stratified CV) ===\n")

    for name, pipe in models.items():
        scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring=["accuracy", "roc_auc", "f1", "recall"],
            n_jobs=-1,
        )

        mean_acc = np.mean(scores["test_accuracy"])
        mean_auc = np.mean(scores["test_roc_auc"])
        mean_f1 = np.mean(scores["test_f1"])
        mean_rec = np.mean(scores["test_recall"])

        results.append(
            {
                "Model": name,
                "Accuracy": round(mean_acc, 4),
                "ROC-AUC": round(mean_auc, 4),
                "F1": round(mean_f1, 4),
                "Recall": round(mean_rec, 4),
            }
        )

        print(f"{name:<30} | AUC: {mean_auc:.4f} | F1: {mean_f1:.4f} | Recall: {mean_rec:.4f}")

        if mean_auc > best_score:
            best_score = mean_auc
            best_name = name
            best_pipeline = pipe

    results_df = (
        pd.DataFrame(results)
        .sort_values("ROC-AUC", ascending=False)
        .reset_index(drop=True)
    )

    return results_df, best_name, best_score, best_pipeline


def save_tree_deliverables(tree_pipeline, X_train, y_train):
    ARTIFACT_DIR.mkdir(exist_ok=True)
    tree_pipeline.fit(X_train, y_train)

    feature_names = tree_pipeline.named_steps["prep"].get_feature_names_out()
    tree_model = tree_pipeline.named_steps["model"]

    rules = export_text(tree_model, feature_names=list(feature_names))
    rules_path = ARTIFACT_DIR / "decision_tree_rules.txt"
    rules_path.write_text(rules, encoding="utf-8")

    plt.figure(figsize=(24, 14))
    plot_tree(
        tree_model,
        feature_names=feature_names,
        class_names=["No Disease", "Disease"],
        filled=True,
        rounded=True,
        impurity=False,
        max_depth=4,
        fontsize=8,
    )
    plt.tight_layout()
    tree_plot_path = ARTIFACT_DIR / "pruned_decision_tree.png"
    plt.savefig(tree_plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    return rules_path, tree_plot_path


def save_interpretability_comparison(results_df):
    ARTIFACT_DIR.mkdir(exist_ok=True)

    comparison = results_df[
        results_df["Model"].isin(
            [
                "Decision Tree (tuned/pruned)",
                "Random Forest (n=100)",
                "Random Forest (n=300)",
            ]
        )
    ].copy()

    comparison["Interpretability"] = comparison["Model"].map(
        {
            "Decision Tree (tuned/pruned)": "High: visual tree and if-else rules exported",
            "Random Forest (n=100)": "Lower: stronger ensemble performance, harder to explain",
            "Random Forest (n=300)": "Lower: stronger ensemble performance, harder to explain",
        }
    )

    path = ARTIFACT_DIR / "interpretability_vs_random_forest.csv"
    comparison.to_csv(path, index=False)
    return path


def main():
    X_train, y_train, X_test, y_test = load_training_data()
    preprocessor = build_preprocessor()
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    tuned_tree, tree_params, tree_auc = tune_decision_tree(preprocessor, cv, X_train, y_train)
    results_df, best_name, best_score, best_pipeline = evaluate_models(
        preprocessor,
        tuned_tree,
        cv,
        X_train,
        y_train,
    )

    ARTIFACT_DIR.mkdir(exist_ok=True)
    results_path = ARTIFACT_DIR / "model_comparison.csv"
    results_df.to_csv(results_path, index=False)

    print("\n=== MODEL COMPARISON (sorted by ROC-AUC) ===")
    print(results_df.to_string(index=False))

    print(f"\nBest Model: {best_name} (CV ROC-AUC = {best_score:.4f})")
    print("Retraining on the full training split...\n")
    best_pipeline.fit(X_train, y_train)

    y_pred = best_pipeline.predict(X_test)

    print("=== VALIDATION SET RESULTS ===\n")
    print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual: No Disease", "Actual: Disease"],
        columns=["Pred: No Disease", "Pred: Disease"],
    )
    print("Confusion Matrix:")
    print(cm_df)

    rules_path, tree_plot_path = save_tree_deliverables(tuned_tree, X_train, y_train)
    interpretability_path = save_interpretability_comparison(results_df)

    joblib.dump(best_pipeline, "best_model.joblib")
    joblib.dump(best_pipeline, "heart_model.pkl")

    print("\n=== SAVED ARTIFACTS ===")
    print(f"Model comparison: {results_path}")
    print(f"Decision tree rules: {rules_path}")
    print(f"Pruned decision tree plot: {tree_plot_path}")
    print(f"Interpretability comparison: {interpretability_path}")
    print("Best model: best_model.joblib")
    print("Flask app model copy: heart_model.pkl")


if __name__ == "__main__":
    main()
