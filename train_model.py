"""
train_model.py
──────────────
Trains a Random Forest classifier on the merged dataset (training_data.csv).
Falls back to firebase_data.csv if training_data.csv doesn't exist.

Output:
  model.pkl          — trained RandomForestClassifier
  label_encoder.pkl  — LabelEncoder for status classes
"""

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# ── Load dataset ──────────────────────────────────────────────
DATA_FILE = "training_data.csv" if os.path.exists("training_data.csv") else "firebase_data.csv"
print(f"Loading: {DATA_FILE}")
data = pd.read_csv(DATA_FILE)
data = data.dropna()

print(f"Dataset shape: {data.shape}")
print("Class distribution:")
print(data['status'].value_counts())

# ── Features & Target ─────────────────────────────────────────
FEATURES = ['temp', 'humidity', 'air', 'methane', 'distance', 'flow']
X = data[FEATURES]
y = data['status']

le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"\nClasses: {list(le.classes_)}")

# ── Train / Test split ────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# ── Model ─────────────────────────────────────────────────────
# Random Forest with tuned hyperparameters for IoT sensor data
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight='balanced',   # handles imbalanced status distribution
    random_state=42,
    n_jobs=-1,
)

print("\nTraining model...")
model.fit(X_train, y_train)

# ── Evaluation ────────────────────────────────────────────────
y_pred = model.predict(X_test)
acc = model.score(X_test, y_test)

print(f"\nTest Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nFeature Importances:")
for feat, imp in importances.items():
    bar = '█' * int(imp * 40)
    print(f"  {feat:12s} {bar}  {imp:.4f}")

# ── Save ──────────────────────────────────────────────────────
joblib.dump(model, "model.pkl")
joblib.dump(le,    "label_encoder.pkl")
print("\n✅ model.pkl and label_encoder.pkl saved!")
print(f"   Trained on {len(X_train)} samples, tested on {len(X_test)} samples")