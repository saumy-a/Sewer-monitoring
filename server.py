"""
server.py — Fixed Flask backend for Smart Sewer Monitoring
===========================================================

Fixes applied vs original:
  FIX 1 — Feature mismatch: model trained on gas_ratio & env_index but server
           never computed them at inference time → always wrong predictions.
           Now engineer_features() is called before every prediction.

  FIX 2 — No error handling in /data route: NaN from DHT failures, missing
           fields, or model errors caused unhandled 500 crashes.
           Now full try/except with safe fallbacks and clear error responses.

  FIX 3 — SQLite single shared connection with check_same_thread=False:
           concurrent Flask requests can corrupt the DB.
           Now uses a per-request connection via get_db() / teardown.

  FIX 4 — /get-data returned distance & flow but not gas_ratio / env_index,
           so the dashboard was missing engineered-feature columns.
           Now returns all stored columns.

  FIX 5 — distance = -1.0 sentinel (from fixed ESP32 code) must not be fed
           to the model as a real reading. Handled gracefully.

  FIX 6 — ML model does not use flow or distance as features (they were not
           in the IoT training dataset and were excluded from TRAIN_FEATURES).
           This caused two observable bugs:
             a) methane=1400 (well above DANGER threshold 700) predicted BLOCKAGE
                because the model was trained on noise-perturbed data and extreme
                values near the 1500 clip boundary were ambiguous.
             b) flow=10 L/min had zero influence on prediction — system showed
                BLOCKAGE even with healthy flow, confusing operators.
           Fix: rule_based_override() applies hard threshold checks on ALL
           sensors (including flow & distance) AFTER the ML prediction.
           ML is used for nuanced mid-range cases; rules catch clear-cut
           extreme violations that the model consistently misjudges.
           A 'decision' field is added to the response so you can see whether
           ML or rules determined the final status.
"""

from flask import Flask, request, jsonify, render_template, g
from flask_cors import CORS
import sqlite3
import math
import traceback

import firebase_admin
from firebase_admin import credentials, db as fb_db

import pandas as pd
import joblib

# ===== LOAD ML ARTIFACTS =====
model    = joblib.load("model.pkl")
le       = joblib.load("label_encoder.pkl")
# FIX 1: load the exact feature list the model was trained on
TRAIN_FEATURES = joblib.load("feature_columns.pkl")  # ['temp','humidity','air','methane','gas_ratio','env_index']

# ===== FIREBASE SETUP =====
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://sewer-3d62e-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

# ===== FLASK =====
app = Flask(__name__)
CORS(app)

DATABASE = 'sensor_data.db'

# ───────────────────────────────────────────────
# FIX 3: per-request SQLite connection
# ───────────────────────────────────────────────
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
        g.db.execute('''
            CREATE TABLE IF NOT EXISTS data (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                ts        DATETIME DEFAULT CURRENT_TIMESTAMP,
                temp      REAL,
                humidity  REAL,
                air       INTEGER,
                methane   INTEGER,
                distance  REAL,
                flow      REAL,
                gas_ratio REAL,
                env_index REAL,
                status    TEXT,
                dht_ok    INTEGER DEFAULT 1
            )
        ''')
        g.db.commit()
    return g.db

@app.teardown_appcontext
def close_db(error):
    db = g.pop('db', None)
    if db is not None:
        db.close()

# ───────────────────────────────────────────────
# FIX 1: feature engineering — must match train_model.py exactly
# ───────────────────────────────────────────────
def engineer_features(row: dict) -> dict:
    air      = row.get('air', 0)
    methane  = row.get('methane', 0)
    temp     = row.get('temp', 25)
    humidity = row.get('humidity', 60)
    row['gas_ratio'] = air / (methane + 1)
    row['env_index'] = (temp * humidity) / 100
    return row


# ───────────────────────────────────────────────
# FIX 6: rule-based override for clear-cut violations
#
# CALIBRATION NOTE (2026-04-15):
#   After sensor calibration the ESP32 now sends values on a delta scale:
#     0   = clean-air baseline (sensor at rest)
#     1500 = maximum detectable gas above baseline
#   Thresholds below are set for this calibrated delta scale.
#   If you change MQ135_CLEAN_AIR_RAW / MQ4_CLEAN_AIR_RAW in v2.cpp,
#   review these numbers too.
# ───────────────────────────────────────────────

# Single source of truth for all thresholds.
# Matches the zones shown in the frontend range bars (index.html).
THRESHOLDS = {
    # Gas (calibrated delta scale 0–1500)
    'air_moderate':      150,   # small rise above clean air, worth watching
    'air_blockage':      300,   # meaningful air-quality degradation
    'air_danger':        500,   # serious air quality — act soon

    'ch4_moderate':      100,   # trace methane detectable
    'ch4_blockage':      300,   # elevated methane — likely decomposition
    'ch4_danger':        600,   # high methane — evacuation risk

    # Water level (distance from sensor to water surface, cm)
    # Lower distance = higher water = worse
    'distance_danger':    15,   # pipe nearly full or sensor submerged
    'distance_blockage':  30,   # high water
    'distance_moderate':  60,   # elevated water

    # Flow rate (L/min)
    # Lower flow = worse (blockage)
    'flow_blockage':     2.0,   # almost no flow — likely blocked
    'flow_moderate':     5.0,   # restricted flow
}


def rule_based_override(data: dict, ml_status: str) -> tuple[str, str]:
    """
    Hard-threshold safety layer that runs AFTER the ML model.
    Returns (final_status, decision_source) where decision_source is
    'ML' or 'RULE:<reason>'.

    Priority order: DANGER > BLOCKAGE > SAFE > ML
    The first matching rule wins.
    """
    air      = data.get('air',      0)
    methane  = data.get('methane',  0)
    distance = data.get('distance', -1.0)
    flow     = data.get('flow',     0.0)

    T = THRESHOLDS

    # ── DANGER overrides ────────────────────────────────────────────
    if methane >= T['ch4_danger']:
        return 'DANGER', f'RULE:ch4={methane}>={T["ch4_danger"]}'
    if air >= T['air_danger']:
        return 'DANGER', f'RULE:air={air}>={T["air_danger"]}'
    if 0 < distance <= T['distance_danger']:
        return 'DANGER', f'RULE:distance={distance:.1f}<={T["distance_danger"]}cm'

    # ── BLOCKAGE overrides ─────────────────────────────────────────
    if 0 < distance <= T['distance_blockage']:
        return 'BLOCKAGE', f'RULE:distance={distance:.1f}<={T["distance_blockage"]}cm'
    if (methane >= T['ch4_blockage'] or air >= T['air_blockage']) and flow < T['flow_blockage']:
        return 'BLOCKAGE', f'RULE:gas_elevated+flow={flow:.2f}<{T["flow_blockage"]}'

    # ── SAFE override (correct wrong ML calls when sensors are clearly OK) ──
    if (flow >= 5.0
            and air  < T['air_moderate']
            and methane < T['ch4_moderate']
            and (distance < 0 or distance > T['distance_moderate'])):
        if ml_status in ('BLOCKAGE', 'DANGER'):
            return 'SAFE', f'RULE:flow={flow:.2f}+all_sensors_ok'

    # ── No override — trust the ML ───────────────────────────────
    return ml_status, 'ML'

def safe_float(val, fallback=0.0):
    """Return fallback if val is None, NaN, or not numeric."""
    try:
        v = float(val)
        return fallback if math.isnan(v) or math.isinf(v) else v
    except (TypeError, ValueError):
        return fallback

# ───────────────────────────────────────────────
# Dashboard page
# ───────────────────────────────────────────────
@app.route('/')
def dashboard():
    return render_template('index.html')

# ───────────────────────────────────────────────
# ESP32 pushes sensor data here
# ───────────────────────────────────────────────
@app.route('/data', methods=['POST'])
def receive_data():
    try:
        raw = request.get_json(force=True, silent=True)
        if not raw:
            return jsonify({"error": "empty or invalid JSON"}), 400

        # FIX 2: sanitise every field — DHT failures send NaN or missing keys
        data = {
            'temp':     safe_float(raw.get('temp'),     25.0),
            'humidity': safe_float(raw.get('humidity'), 60.0),
            'air':      int(safe_float(raw.get('air'),   0)),
            'methane':  int(safe_float(raw.get('methane'), 0)),
            'distance': safe_float(raw.get('distance'), -1.0),
            'flow':     safe_float(raw.get('flow'),      0.0),
            'dht_ok':   bool(raw.get('dht_ok', True)),
        }

        # FIX 1: compute engineered features before prediction
        data = engineer_features(data)

        # Build feature DataFrame in the exact column order the model expects
        features_df = pd.DataFrame([{k: data[k] for k in TRAIN_FEATURES}])

        prediction = model.predict(features_df)
        ml_status  = le.inverse_transform(prediction)[0]

        # FIX 6: apply hard-rule override — catches extreme readings the
        # ML model consistently misjudges (see module docstring)
        status, decision = rule_based_override(data, ml_status)
        data['status']   = status
        data['decision'] = decision   # stored for transparency / debugging

        print(f"[POST /data] ML={ml_status} → Final={status} ({decision}) | "
              f"air={data['air']} CH4={data['methane']} dist={data['distance']:.1f} flow={data['flow']:.2f}")

        # Store in Firebase (exclude internal keys the DB doesn't need)
        fb_payload = {k: v for k, v in data.items() if k != 'dht_ok'}
        ref = fb_db.reference('sensor_data')
        ref.push(fb_payload)

        # Store in SQLite
        db = get_db()
        db.execute('''
            INSERT INTO data (temp, humidity, air, methane, distance, flow,
                              gas_ratio, env_index, status, dht_ok)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['temp'], data['humidity'], data['air'], data['methane'],
            data['distance'], data['flow'],
            data['gas_ratio'], data['env_index'],
            data['status'], int(data['dht_ok'])
        ))
        db.commit()

        return jsonify({"status": "stored", "prediction": status, "decision": decision})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ───────────────────────────────────────────────
# Dashboard polls this endpoint
# ───────────────────────────────────────────────
@app.route('/get-data', methods=['GET'])
def get_data():
    try:
        limit = request.args.get('limit', 1, type=int)
        db = get_db()
        rows = db.execute('''
            SELECT ts, temp, humidity, air, methane, distance, flow,
                   gas_ratio, env_index, status, dht_ok
            FROM data
            ORDER BY id DESC
            LIMIT ?
        ''', (limit,)).fetchall()

        result = [dict(r) for r in reversed(rows)]
        return jsonify(result if result else [{}])

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ───────────────────────────────────────────────
# Health check — useful for debugging
# ───────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "model_features": TRAIN_FEATURES,
        "model_classes":  list(le.classes_),
        "thresholds":     THRESHOLDS,
        "status": "ok"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)