# ==============================
# 1. Import Libraries
# ==============================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ==============================
# 2. Load Data
# ==============================
df = pd.read_csv("train (2) machine project.csv")

# ==============================
# 3. Basic Info
# ==============================
print("Shape of data:", df.shape)
print("\nColumns:\n", df.columns)

print("\nData Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# ==============================
# 4. Data Cleaning
# ==============================
# Drop id column if exists
if 'id' in df.columns:
    df.drop(columns=['id'], inplace=True)

# Check missing values
print("\nMissing Values:\n", df.isnull().sum())

# ==============================
# 5. Target Encoding
# ==============================
df['Heart Disease'] = df['Heart Disease'].map({
    'Absence': 0,
    'Presence': 1
})

# ==============================
# 6. Target Analysis
# ==============================
print("\nTarget Distribution:\n", df['Heart Disease'].value_counts())

plt.figure()
df['Heart Disease'].value_counts().plot(kind='bar')
plt.title("Heart Disease Distribution")
plt.xlabel("Class (0 = Absence, 1 = Presence)")
plt.ylabel("Count")
plt.show()

# ==============================
# 7. Feature Categories
# ==============================
num_cols = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']

cat_cols = ['Sex', 'Chest pain type', 'FBS over 120', 'EKG results',
            'Exercise angina', 'Slope of ST', 'Number of vessels fluro', 'Thallium']

# ==============================
# 8. Numerical Features Analysis
# ==============================
for col in num_cols:
    plt.figure()
    df[col].hist()
    plt.title(f"{col} Distribution")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# ==============================
# 9. Categorical Features Analysis
# ==============================
for col in cat_cols:
    plt.figure()
    df[col].value_counts().plot(kind='bar')
    plt.title(f"{col} Distribution")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()

# ==============================
# 10. Feature vs Target Analysis
# ==============================
for col in num_cols:
    plt.figure()
    df.groupby('Heart Disease')[col].mean().plot(kind='bar')
    plt.title(f"{col} vs Heart Disease")
    plt.ylabel("Average Value")
    plt.show()

for col in cat_cols:
    plt.figure()
    pd.crosstab(df[col], df['Heart Disease']).plot(kind='bar')
    plt.title(f"{col} vs Heart Disease")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()

# ==============================
# 11. Correlation Matrix
# ==============================
corr = df.corr()

plt.figure()
plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Matrix")
plt.show()

# ==============================
# 12. Top Correlated Features
# ==============================
print("\nCorrelation with Target:")
print(corr['Heart Disease'].sort_values(ascending=False))

top_features = corr['Heart Disease'].abs().sort_values(ascending=False)[1:6]
print("\nTop 5 Important Features:\n", top_features)

# ==============================
# 13. Feature Scaling (NEW from Code 2)
# ==============================
X = df.drop('Heart Disease', axis=1)
y = df['Heart Disease']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# ==============================
# 14. Final Clean Dataset
# ==============================
clean_df = X_scaled
clean_df['Heart Disease'] = y

# Save dataset
clean_df.to_csv("clean_dataset.csv", index=False)

print("\nFinal Clean Dataset Preview:")
print(clean_df.head())
