# Import Libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load Dataset
df = pd.read_csv('/kaggle/input/competitions/playground-series-s6e2/train.csv')

# Display first rows
print(df.head())

# Check dataset information
print(df.info())
print(df.isnull().sum())

# Remove unnecessary column
df = df.drop(columns=['id'])

# Encode target column
df['Heart Disease'] = df['Heart Disease'].map({
    'Absence': 0,
    'Presence': 1
})

# Separate features and target
X = df.drop('Heart Disease', axis=1)
y = df['Heart Disease']

# Apply Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert scaled data to DataFrame
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Combine features with target
clean_df = X_scaled
clean_df['Heart Disease'] = y

# Save cleaned dataset
clean_df.to_csv("clean_dataset.csv", index=False)

# Display final dataset
print(clean_df.head())

# ==============================
# 14. Function Evaluation
# ==============================

def evaluate_model(name, model, X_train, y_train, X_val, y_val):
    # Train
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)

    print(f"\n{name}")
    print("Train Accuracy:", train_acc)
    print("Validation Accuracy:", val_acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # Overfitting Check
    print("Overfitting Difference:", abs(train_acc - val_acc))

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_val_pred)
    print("Confusion Matrix:\n", cm)

    # Visual Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_val, y_val_pred)
    plt.title(name)
    plt.show()

    # Classification Report
    print("\nClassification Report:\n", classification_report(y_val, y_val_pred))

    return {
        "Model": name,
        "Train Acc": train_acc,
        "Val Acc": val_acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

# ==============================
# Evaluate Logistic Regression 
# ==============================

lr_metrics = evaluate_model(
    "Logistic Regression",
    lr_grid.best_estimator_,
    X_train, y_train, X_val, y_val
)

# ==============================
# Evaluate Decision Tree
# ==============================

dt_metrics = evaluate_model(
    "Decision Tree",
    dt_grid.best_estimator_,
    X_train, y_train, X_val, y_val
)

# ==============================
# Evaluate Random Forest
# ==============================

rf_metrics = evaluate_model(
    "Random Forest",
    rf_grid.best_estimator_,
    X_train, y_train, X_val, y_val
)

# ==============================
# Evaluate XGBoost
# ==============================

xgb_metrics = evaluate_model(
    "XGBoost",
    xgb_grid.best_estimator_,
    X_train, y_train, X_val, y_val
)

# ==============================
# 15. Compare Models
# ==============================

results = pd.DataFrame([lr_metrics, dt_metrics, rf_metrics, xgb_metrics])
print("\nModel Comparison:\n")
print(results)

# ==============================
# 16. Choose Best Model
# ==============================

best_model = results.sort_values(by="Recall", ascending=False).iloc[0]
print("\nBest Model:\n", best_model)

# ==============================
# 17. Test on New Data
# ==============================

sample = X_val.iloc[0:1]

best_model_name = best_model["Model"]

if best_model_name == "Logistic Regression":
    final_model = lr_grid.best_estimator_
elif best_model_name == "Decision Tree":
    final_model = dt_grid.best_estimator_
elif best_model_name == "Random Forest":
    final_model = rf_grid.best_estimator_
else:
    final_model = xgb_grid.best_estimator_

prediction = final_model.predict(sample)

print("\nTest Sample Prediction:", prediction) 
