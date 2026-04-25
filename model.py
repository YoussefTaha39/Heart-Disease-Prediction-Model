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