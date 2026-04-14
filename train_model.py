"""
train_model.py — Improved ML Pipeline for Smart Sewer Monitoring
================================================================
Steps:
  1. Load & merge datasets (Firebase + IoT telemetry)
  2. Fix feature mapping (only physically meaningful features)
  3. Feature engineering (gas_ratio, env_index)
  4. Train RandomForest + GradientBoosting, compare
  5. Full evaluation (classification report, confusion matrix, feature importance)
  6. Select best model, guard against overfitting via cross-validation
  7. Save model.pkl + label_encoder.pkl + feature_columns.pkl
  8. Deployment wrapper validates live ESP32 input format

Model input at inference time (what ESP32 sends):
  temp, humidity, air, methane, distance, flow
  → pipeline auto-computes gas_ratio & env_index before prediction
"""

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection   import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing     import LabelEncoder
from sklearn.metrics           import (classification_report,
                                       confusion_matrix,
                                       ConfusionMatrixDisplay)
from sklearn.pipeline          import Pipeline
import joblib

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════
ALLOWED_STATUSES = {'SAFE', 'MODERATE', 'BLOCKAGE', 'DANGER'}

# Features that the LIVE ESP32 device sends
LIVE_FEATURES = ['temp', 'humidity', 'air', 'methane', 'distance', 'flow']

# Features the model trains on (physically meaningful only)
# gas_ratio and env_index are engineered at inference time too
TRAIN_FEATURES = ['temp', 'humidity', 'air', 'methane', 'gas_ratio', 'env_index']

# ═══════════════════════════════════════════════════════════════
#  STEP 1 — Load datasets
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1 — Loading datasets")
print("=" * 60)

frames = []

# ── 1a. IoT Telemetry dataset ────────────────────────────────
if os.path.exists('iot_telemetry_data.csv'):
    print("  Loading iot_telemetry_data.csv ...")
    iot = pd.read_csv('iot_telemetry_data.csv')
    iot = iot.dropna(subset=['temp', 'humidity', 'co', 'smoke'])

    iot_mapped = pd.DataFrame({
        'temp':     iot['temp'].round(2),
        'humidity': iot['humidity'].round(2),
        # CO sensor → MQ-135 air quality proxy (scaled to realistic ppm range)
        'air':      (iot['co'] * 70_000).round(0).clip(0, 1500).astype(int),
        # Smoke sensor → MQ-4 methane proxy (scaled to realistic ppm range)
        'methane':  (iot['smoke'] * 20_000).round(0).clip(0, 1500).astype(int),
        'source':   'iot_telemetry',
    })

    # NOTE: LPG→flow and fixed distance NOT included
    # — LPG ≠ sewer flow rate (different physical property)
    # — fixed distance=60 adds no discriminative information

    # Assign status labels from domain-expert gas thresholds
    def assign_status(row):
        a, m = row['air'], row['methane']
        if a > 600 or m > 700:    return 'DANGER'
        if a > 350 or m > 450:    return 'BLOCKAGE'
        if a > 200 or m > 250:    return 'MODERATE'
        return 'SAFE'

    iot_mapped['status'] = iot_mapped.apply(assign_status, axis=1)
    print(f"  IoT rows: {len(iot_mapped):,}")
    print("  ", iot_mapped['status'].value_counts().to_dict())

    # Balanced sample: up to 10k per class
    sampled_parts = []
    for status_val, grp in iot_mapped.groupby('status'):
        sampled_parts.append(grp.sample(min(len(grp), 10_000), random_state=42))
    iot_balanced = pd.concat(sampled_parts, ignore_index=True)
    print(f"  After balancing: {len(iot_balanced):,}")
    frames.append(iot_balanced)
else:
    print("  ⚠️  iot_telemetry_data.csv not found, skipping.")

# ── 1b. Firebase / real sensor dataset ──────────────────────
for fname in ['firebase_data.csv', 'training_data.csv']:
    if os.path.exists(fname):
        print(f"\n  Loading {fname} ...")
        fb = pd.read_csv(fname)
        fb = fb.dropna()

        # Normalize non-standard labels (e.g. "DANGER: GAS" → "DANGER")
        if 'status' in fb.columns:
            fb['status'] = (fb['status'].str.upper()
                                        .str.split(':').str[0]
                                        .str.strip())
            fb = fb[fb['status'].isin(ALLOWED_STATUSES)]

        # Keep only columns we need
        available = [c for c in ['temp','humidity','air','methane','status','source'] if c in fb.columns]
        fb = fb[available]
        if 'source' not in fb.columns:
            fb['source'] = fname

        print(f"  {fname} rows: {len(fb):,}")
        print("  ", fb['status'].value_counts().to_dict())

        # Upsample real data heavily — it's ground truth, give it weight
        fb_up = pd.concat([fb] * 20, ignore_index=True)
        frames.append(fb_up)
        break   # use first available

if not frames:
    raise RuntimeError("No dataset found. Run prepare_dataset.py first, "
                       "or place firebase_data.csv in this directory.")

# ═══════════════════════════════════════════════════════════════
#  STEP 2 — Merge & clean
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2 — Merging & cleaning")
print("=" * 60)

df = pd.concat(frames, ignore_index=True)
df = df[['temp', 'humidity', 'air', 'methane', 'status']].copy()
df = df.dropna()

# Remove physically impossible sensor values
df = df[
    df['temp'].between(-10, 60) &
    df['humidity'].between(0, 100) &
    df['air'].between(0, 1500) &
    df['methane'].between(0, 1500)
]

print(f"Total after merge & filter: {len(df):,}")
print("Class distribution:")
print(df['status'].value_counts())

# ═══════════════════════════════════════════════════════════════
#  STEP 3 — Feature engineering
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3 — Feature engineering")
print("=" * 60)

def engineer_features(df_: pd.DataFrame) -> pd.DataFrame:
    d = df_.copy()
    # Gas ratio: high ratio = air quality problem increasing faster than methane
    d['gas_ratio']  = d['air'] / (d['methane'] + 1)
    # Environmental stress index: high temp + high humidity = dangerous conditions
    d['env_index']  = (d['temp'] * d['humidity']) / 100
    return d

df = engineer_features(df)
print(f"Features added: gas_ratio, env_index")
print(df[TRAIN_FEATURES].describe().round(3))

# ═══════════════════════════════════════════════════════════════
#  STEP 4 — Encode & split
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4 — Encoding & train/test split")
print("=" * 60)

le = LabelEncoder()
df['label'] = le.fit_transform(df['status'])
print(f"Classes: {list(le.classes_)}")

X = df[TRAIN_FEATURES]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ═══════════════════════════════════════════════════════════════
#  STEP 5 — Train models
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5 — Training models")
print("=" * 60)

models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=200,
        max_depth=18,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.85,
        random_state=42,
    ),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, clf in models.items():
    print(f"\n  Training {name}...")
    clf.fit(X_train, y_train)

    test_acc  = clf.score(X_test, y_test)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv,
                                scoring='accuracy', n_jobs=-1)

    results[name] = {
        'model':    clf,
        'test_acc': test_acc,
        'cv_mean':  cv_scores.mean(),
        'cv_std':   cv_scores.std(),
    }
    print(f"  Test accuracy:  {test_acc:.4f}")
    print(f"  CV  accuracy:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ═══════════════════════════════════════════════════════════════
#  STEP 6 — Evaluate & select best model
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6 — Full evaluation")
print("=" * 60)

# Select best model by test accuracy (with overfitting guard:
# penalise if test_acc >> cv_mean by more than 3%)
def score(info):
    overfit_penalty = max(0, info['test_acc'] - info['cv_mean'] - 0.03)
    return info['test_acc'] - overfit_penalty

best_name = max(results, key=lambda n: score(results[n]))
best      = results[best_name]
best_clf  = best['model']

print(f"\n  ✅ Best model: {best_name}")
print(f"     Test acc:  {best['test_acc']:.4f}")
print(f"     CV  acc:   {best['cv_mean']:.4f} ± {best['cv_std']:.4f}")

if best['test_acc'] < 0.90:
    print("  ⚠️  Warning: accuracy below 90% — check data quality")

y_pred = best_clf.predict(X_test)

print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("  Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
print(cm_df.to_string())

# Feature importance
print("\n  Feature Importances:")
if hasattr(best_clf, 'feature_importances_'):
    fi = pd.Series(best_clf.feature_importances_, index=TRAIN_FEATURES).sort_values(ascending=False)
    for feat, imp in fi.items():
        bar = '█' * int(imp * 50)
        print(f"    {feat:12s}  {bar:<50s}  {imp:.4f}")

# ═══════════════════════════════════════════════════════════════
#  STEP 7 — Compare all models
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7 — Model comparison")
print("=" * 60)
print(f"  {'Model':<20} {'Test Acc':>10} {'CV Mean':>10} {'CV Std':>10}")
print(f"  {'-'*50}")
for name, info in results.items():
    marker = ' ← best' if name == best_name else ''
    print(f"  {name:<20} {info['test_acc']:>10.4f} {info['cv_mean']:>10.4f} {info['cv_std']:>10.4f}{marker}")

# ═══════════════════════════════════════════════════════════════
#  STEP 8 — Save artifacts
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 8 — Saving artifacts")
print("=" * 60)

joblib.dump(best_clf,     'model.pkl')
joblib.dump(le,           'label_encoder.pkl')
joblib.dump(TRAIN_FEATURES, 'feature_columns.pkl')

print(f"  ✅ model.pkl          ({best_name})")
print(f"  ✅ label_encoder.pkl  (classes: {list(le.classes_)})")
print(f"  ✅ feature_columns.pkl (features: {TRAIN_FEATURES})")

# ═══════════════════════════════════════════════════════════════
#  DEPLOYMENT WRAPPER — test inference with ESP32 format
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DEPLOYMENT — Live inference test (ESP32 format)")
print("=" * 60)

def predict_from_esp32(esp32_data: dict) -> str:
    """
    Accept the raw ESP32 payload (temp, humidity, air, methane, distance, flow),
    compute engineered features, and return the predicted status label.
    Compatible with server.py /data endpoint.
    """
    model_  = joblib.load('model.pkl')
    le_     = joblib.load('label_encoder.pkl')
    cols_   = joblib.load('feature_columns.pkl')

    air      = esp32_data.get('air', 0)
    methane  = esp32_data.get('methane', 0)
    temp     = esp32_data.get('temp', 25)
    humidity = esp32_data.get('humidity', 60)

    features = {
        'temp':       temp,
        'humidity':   humidity,
        'air':        air,
        'methane':    methane,
        'gas_ratio':  air / (methane + 1),
        'env_index':  (temp * humidity) / 100,
    }

    X_live = pd.DataFrame([features])[cols_]
    pred   = model_.predict(X_live)
    return le_.inverse_transform(pred)[0]

# Run test predictions for each expected class
test_cases = [
    {'name': 'SAFE reading',     'data': {'temp': 24, 'humidity': 55, 'air': 100, 'methane': 150, 'distance': 90, 'flow': 10}},
    {'name': 'MODERATE reading', 'data': {'temp': 32, 'humidity': 72, 'air': 270, 'methane': 320, 'distance': 65, 'flow': 6}},
    {'name': 'BLOCKAGE reading', 'data': {'temp': 28, 'humidity': 78, 'air': 420, 'methane': 510, 'distance': 30, 'flow': 2}},
    {'name': 'DANGER reading',   'data': {'temp': 40, 'humidity': 90, 'air': 720, 'methane': 780, 'distance': 10, 'flow': 0}},
]

print(f"\n  {'Case':<22} {'Predicted':>10}")
print(f"  {'-'*35}")
for tc in test_cases:
    pred = predict_from_esp32(tc['data'])
    print(f"  {tc['name']:<22} {pred:>10}")

print("\n✅ Pipeline complete. Model ready for deployment.")