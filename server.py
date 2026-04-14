from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sqlite3
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
import joblib

# ===== LOAD ML MODEL =====
model = joblib.load("model.pkl")
le    = joblib.load("label_encoder.pkl")

# ===== FIREBASE SETUP =====
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://sewer-3d62e-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

# ===== FLASK =====
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests (ESP32, dashboard on different ports)

# ===== SQLITE SETUP =====
conn   = sqlite3.connect('sensor_data.db', check_same_thread=False)
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS data (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    temp     REAL,
    humidity REAL,
    air      INTEGER,
    methane  INTEGER,
    distance REAL,
    flow     REAL,
    status   TEXT
)
''')
conn.commit()


# ──────────────────────────────────────────────
#  Dashboard page
# ──────────────────────────────────────────────
@app.route('/')
def dashboard():
    return render_template('index.html')


# ──────────────────────────────────────────────
#  ESP32 / IoT device pushes data here
# ──────────────────────────────────────────────
@app.route('/data', methods=['POST'])
def receive_data():
    data = request.json

    # ML prediction
    features_df = pd.DataFrame([{
        'temp':     data.get('temp'),
        'humidity': data.get('humidity'),
        'air':      data.get('air'),
        'methane':  data.get('methane'),
        'distance': data.get('distance'),
        'flow':     data.get('flow')
    }])
    prediction = model.predict(features_df)
    data['status'] = le.inverse_transform(prediction)[0]

    print(f"[POST /data] Received & Predicted: {data}")

    # Store in Firebase
    ref = db.reference('sensor_data')
    ref.push(data)

    # Store in SQLite
    cursor.execute('''
        INSERT INTO data (temp, humidity, air, methane, distance, flow, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        data.get('temp'),
        data.get('humidity'),
        data.get('air'),
        data.get('methane'),
        data.get('distance'),
        data.get('flow'),
        data.get('status')
    ))
    conn.commit()

    return jsonify({"status": "stored", "prediction": data['status']})


# ──────────────────────────────────────────────
#  Dashboard polls this endpoint
# ──────────────────────────────────────────────
@app.route('/get-data', methods=['GET'])
def get_data():
    limit = request.args.get('limit', 1, type=int)

    cursor.execute('''
        SELECT temp, humidity, air, methane, distance, flow, status
        FROM data
        ORDER BY id DESC
        LIMIT ?
    ''', (limit,))

    rows = cursor.fetchall()
    keys = ['temp', 'humidity', 'air', 'methane', 'distance', 'flow', 'status']
    result = [dict(zip(keys, row)) for row in reversed(rows)]

    return jsonify(result if result else [{}])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)