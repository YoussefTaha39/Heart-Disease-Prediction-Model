# Heart Disease Prediction Model - Final Report

## 1. Project Overview
The objective of this project is to develop a predictive machine learning pipeline to estimate the likelihood of heart disease in patients based on clinical and demographic data. This report covers the methodology, data processing, model training, and evaluation results.

## 2. Dataset Information
The dataset used is `heart_statlog_cleveland_hungary_final.csv`, containing **1190 patient records**. 
- **Target Distribution**: 
  - Positive cases (Disease): **629**
  - Negative cases (No Disease): **561**
- The dataset is relatively balanced, providing a strong foundation for supervised learning.

### Feature Categorization
- **Numeric Features**: Age, Resting BP, Cholesterol, Max Heart Rate, Oldpeak.
- **Categorical Features**: Chest Pain Type, Resting ECG, ST Slope.
- **Binary Features**: Sex, Fasting Blood Sugar, Exercise Angina.

## 3. Data Preprocessing Pipeline
To prepare the data for modeling and prevent data leakage, a robust preprocessing pipeline was implemented using `scikit-learn`'s `Pipeline` and `ColumnTransformer`:
- **Numeric Data**: Missing values were imputed using the median. Standard scaling (`StandardScaler`) was applied to normalize the features (except for tree-based models, where scaling is disabled).
- **Categorical Data**: Missing values were imputed using the most frequent category, followed by one-hot encoding (`OneHotEncoder` with `handle_unknown='ignore'`).
- **Binary Data**: Passed through the pipeline without modification.

## 4. Modeling Strategy
The data was split into an 80/20 train-test split using stratification (`stratify=y`) to maintain the class distribution. The models were evaluated extensively using **10-fold Stratified Cross-Validation**.

The following algorithms were trained and evaluated:
1. **Decision Tree** *(Baseline model tuned with GridSearchCV)*
2. **Random Forest** *(Ensemble model with 300 estimators)*
3. **Gradient Boosting**
4. **Support Vector Machine** *(SVM with RBF kernel)*
5. **Logistic Regression**

## 5. Final Model Evaluation
Below is the cross-validation performance comparison of all models, sorted by **Recall**. In medical diagnostics, Recall (Sensitivity) is prioritized to minimize false negatives (failing to identify a diseased patient).

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** | **94.70%** | **93.83%** | **96.33%** | **95.05%** | **97.30%** |
| SVM (RBF) | 87.98% | 87.47% | 90.30% | 88.81% | 93.46% |
| Gradient Boosting | 88.99% | 89.64% | 89.66% | 89.60% | 94.94% |
| Decision Tree | 89.49% | 91.29% | 88.71% | 89.91% | 89.54% |
| Logistic Regression | 84.53% | 84.41% | 86.95% | 85.60% | 91.71% |

*The **Random Forest** model was definitively selected as the primary production model due to its superior performance across all metrics, particularly achieving a 96.3% recall and 97.3% AUC.*

## 6. Feature Importance
Using the trained Random Forest model, we extracted the feature importances to understand what drives the model's predictions. The top 10 most influential predictors are:

1. **ST slope_1** (12.75%)
2. **Oldpeak** (10.72%)
3. **Max Heart Rate** (10.57%)
4. **Cholesterol** (10.43%)
5. **Chest Pain Type 4** (9.93%)
6. **Age** (8.94%)
7. **Resting BP (s)** (7.98%)
8. **ST slope_2** (6.89%)
9. **Exercise Angina** (6.46%)
10. **Sex** (4.13%)

## 7. Model Artifacts
The full end-to-end ML pipeline (including the preprocessor and the classifier) has been serialized to ensure seamless predictions on unseen data. The following files have been saved to the `outputs/` directory:
- `heart_disease_model.pkl`: The fully trained Random Forest pipeline object.
- `feature_names.pkl`: The extracted feature names post-preprocessing.
- **Visual Artifacts**: Several visualization graphs (ROC Curves, Confusion Matrices, Correlation Heatmap, Feature Importances, Decision Tree rules) were successfully generated.

## 8. Conclusion
The Random Forest model serves as a highly accurate (~94.7%) and highly sensitive (~96.3% recall) diagnostic tool for identifying the presence of heart disease. Integrating the preprocessing logic directly inside the serialized pipeline guarantees that any incoming raw data from future inference tasks will be processed perfectly consistently with the training data.
