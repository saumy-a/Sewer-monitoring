import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
data = pd.read_csv("firebase_data.csv")

# Features & Target
X = data[['temp','humidity','air','methane','distance','flow']]
y = data['status']

# Convert labels to numbers
le = LabelEncoder()
y = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Save model
joblib.dump(model, "model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("Model saved!")