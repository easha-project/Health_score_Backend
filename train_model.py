import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("healthcare_dataset.csv")

# Clean strings and normalize casing
df['Test Results'] = df['Test Results'].str.strip().str.title()
df['Medication'] = df['Medication'].str.strip().str.title()
df['Medical Condition'] = df['Medical Condition'].str.strip().str.title()

# Encode categorical fields
label_cols = ['Medical Condition', 'Medication', 'Test Results']
label_encoders = {}
for col in label_cols:
    le = LabelEncoder()
    df[col + '_Encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le

# Compute health score (simulated logic)
def compute_health_score(row):
    score = 100
    score -= row['Age'] * 0.4
    score -= row['Medical Condition_Encoded'] * 4
    score -= row['Medication_Encoded'] * 2
    if row['Test Results'] == 'Abnormal':
        score -= 20
    elif row['Test Results'] == 'Inconclusive':
        score -= 10
    return max(0, min(100, score))

df['Health_Score'] = df.apply(compute_health_score, axis=1)

# Features and labels
X = df[['Age', 'Medical Condition_Encoded', 'Medication_Encoded', 'Test Results_Encoded']]
y = df['Health_Score']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'health_score_model.pkl')

# Optionally, save encoders if needed
joblib.dump(label_encoders, 'encoders.pkl')

print("✅ Model saved as health_score_model.pkl")
