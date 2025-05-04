from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle
import pandas as pd
import os

cancer_df = pd.read_csv('Cancer_Data.csv')
# Drop unnecessary columns
cancer_df = cancer_df.drop(['id', 'Unnamed: 32'], axis=1)

# Encode the diagnosis column
cancer_df['diagnosis'] = LabelEncoder().fit_transform(cancer_df['diagnosis'])  # M=1, B=0

# Separate features and target
X = cancer_df.drop('diagnosis', axis=1)
y = cancer_df['diagnosis']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
cancer_model = RandomForestClassifier(random_state=42)
cancer_model.fit(X_train, y_train)

# Evaluate the model
y_pred = cancer_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save the model
model_path = "model/cancer_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(cancer_model, f)

model_path, accuracy
