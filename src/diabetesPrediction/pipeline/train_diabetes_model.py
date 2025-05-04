import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
df = pd.read_csv("diabetes_prediction_dataset.csv")

# Encode categorical columns
le = LabelEncoder()
categorical_columns = ['gender', 'smoking_history']

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Split features and target
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model/diabetes_prediction_model.pkl')

print("Model saved as diabetes_prediction_model.pkl")
