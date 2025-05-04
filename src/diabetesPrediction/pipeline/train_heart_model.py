import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load the uploaded dataset
heart_df = pd.read_csv("heart.csv")

# Separate features and target
X_heart = heart_df.drop("target", axis=1)
y_heart = heart_df["target"]

# Train-test split
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_heart, y_heart, test_size=0.2, random_state=42)

# Train the model
heart_model = RandomForestClassifier(random_state=42)
heart_model.fit(X_train_h, y_train_h)

# Evaluate the model
y_pred_h = heart_model.predict(X_test_h)
heart_accuracy = accuracy_score(y_test_h, y_pred_h)

# Save the model to a .pkl file
heart_model_path = "model/heart_model.pkl"
with open(heart_model_path, "wb") as f:
    pickle.dump(heart_model, f)

heart_accuracy, heart_model_path
