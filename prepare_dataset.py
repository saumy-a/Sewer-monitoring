"""
prepare_dataset.py
──────────────────
Merges the real IoT telemetry dataset (iot_telemetry_data.csv) with our
local Firebase readings (firebase_data.csv) to create a larger, more
representative training dataset for the sewer monitor ML model.

Columns in our schema:
  temp, humidity, air, methane, distance, flow, status

iot_telemetry_data.csv mapping:
  temp       → temp  (direct)
  humidity   → humidity (direct)
  co*70000   → air  (CO sensor proxy for MQ-135 air quality, scaled to ppm)
  smoke*20000→ methane (smoke density proxy for MQ-4, scaled to ppm)
  lpg*2000   → flow  (LPG flow proxy, scaled to L/min)
  60         → distance (fixed reasonable sewer water depth in cm, no sensor in dataset)

Status labels assigned by gas threshold rules matching our ESP32 logic:
  air>500 or methane>1000  → DANGER
  air>300 or methane>200  → BLOCKAGE
  air>150 or methane>50  → MODERATE
  else                    → SAFE
"""

import pandas as pd
import numpy as np

print("Loading datasets...")

# ── 1. IoT Telemetry (405k rows) ─────────────────────────────
iot = pd.read_csv("iot_telemetry_data.csv")
iot = iot.dropna(subset=['temp','humidity','co','smoke','lpg'])

iot_mapped = pd.DataFrame({
    'temp':     iot['temp'].round(2),
    'humidity': iot['humidity'].round(2),
    'air':      ((iot['co'] - 0.0011) / 0.0134 * 2100 + 400).round(0).clip(400, 3000).astype(int),
    'methane':  ((iot['smoke'] - 0.0066) / 0.0400 * 1400).round(0).clip(0, 2000).astype(int),
    'distance': 60.0,                           # fixed proxy
    'flow':     (iot['lpg']   * 2000).round(2), # L/min proxy
})

def assign_status(row):
    if row['air'] > 2000 or row['methane'] > 1000:
        return 'DANGER'
    elif row['air'] > 1500 or row['methane'] > 200:
        return 'BLOCKAGE'
    elif row['air'] > 800 or row['methane'] > 50:
        return 'MODERATE'
    return 'SAFE'

iot_mapped['status'] = iot_mapped.apply(assign_status, axis=1)

print(f"IoT dataset: {len(iot_mapped)} rows")
print(iot_mapped['status'].value_counts())

# ── 2. Our Firebase data ──────────────────────────────────────
try:
    fb = pd.read_csv("firebase_data.csv")
    fb = fb.dropna()
    fb = fb[['temp','humidity','air','methane','distance','flow','status']]
    # Normalize non-standard labels from ESP32 (e.g. 'DANGER: GAS' → 'DANGER')
    fb['status'] = fb['status'].str.upper().str.split(':').str[0].str.strip()
    allowed = {'SAFE', 'MODERATE', 'BLOCKAGE', 'DANGER'}
    fb = fb[fb['status'].isin(allowed)]
    # Upsample firebase data (it's small but real) 10x to give it weight
    fb_upsampled = pd.concat([fb] * 10, ignore_index=True)
    print(f"Firebase dataset: {len(fb)} rows → upsampled to {len(fb_upsampled)}")
    print(fb['status'].value_counts())
except FileNotFoundError:
    print("firebase_data.csv not found, skipping.")
    fb_upsampled = pd.DataFrame()

# ── 3. Merge ──────────────────────────────────────────────────
# Sample IoT data to keep training manageable (50k rows representative)
iot_sample = iot_mapped.groupby('status', group_keys=False).apply(
    lambda g: g.sample(min(len(g), 12500), random_state=42)
)

all_data = pd.concat([iot_sample, fb_upsampled], ignore_index=True)

# Shuffle
all_data = all_data.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nFinal merged dataset: {len(all_data)} rows")
print(all_data['status'].value_counts())
print("\nColumn stats:")
print(all_data[['temp','humidity','air','methane','distance','flow']].describe().round(2))

# ── 4. Save ───────────────────────────────────────────────────
all_data.to_csv("training_data.csv", index=False)
print("\n✅ Saved to training_data.csv")
