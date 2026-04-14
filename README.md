# 🚰 Smart Sewer Monitoring System

A real-time IoT-based sewer monitoring system using an **ESP32**, Firebase, machine learning, and a Flask web dashboard.

---

## Architecture

```
ESP32 Sensors → Flask Server (Python) → Firebase Realtime DB
                       ↓                         ↓
               SQLite Local DB          ML Prediction (Random Forest)
                       ↓
             Web Dashboard (HTML/JS)
```

---

## Features

- 📡 **ESP32** reads: Temperature, Humidity, Air Quality (MQ-135), Methane (MQ-4), Water Level (Ultrasonic), Flow Rate
- 🤖 **AI Prediction** – Random Forest model classifies system status: `SAFE`, `MODERATE`, `BLOCKAGE`, `DANGER`
- 🔥 **Firebase Realtime Database** – stores all sensor readings
- 🗃️ **SQLite** – local backup storage
- 🌐 **Web Dashboard** – futuristic dark-mode UI with live charts (Chart.js)

---

## Project Structure

```
IOT_MAIN/
├── espcode.cpp          # ESP32 Arduino code
├── server.py            # Flask backend (REST API + dashboard)
├── train_model.py       # Train ML model on exported data
├── dataex.py            # Export Firebase data to CSV/JSON
├── templates/
│   └── index.html       # Web dashboard (HTML/CSS/JS)
└── requirements.txt     # Python dependencies
```

---

## Setup

### 1. Install Python dependencies
```bash
pip install flask flask-cors firebase-admin pandas scikit-learn joblib
```

### 2. Add your Firebase credentials
Place your `serviceAccountKey.json` in the root directory (do NOT commit this).

### 3. Export and train
```bash
python dataex.py        # Export Firebase data to CSV
python train_model.py   # Train the ML model
```

### 4. Run the server
```bash
python server.py
```

Open **http://localhost:5001** to see the dashboard.

### 5. Flash the ESP32
Open `espcode.cpp` in Arduino IDE, update your WiFi credentials and server IP, then flash to your ESP32.

---

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Web dashboard |
| POST | `/data` | ESP32 pushes sensor data |
| GET | `/get-data` | Dashboard polls latest reading |

---

## Status Meanings

| Status | Meaning |
|--------|---------|
| ✅ SAFE | All parameters normal |
| ⚠️ MODERATE | Elevated readings, monitor closely |
| 🚧 BLOCKAGE | Flow blocked, inspect sewer |
| 🚨 DANGER | Critical gas/overflow — take immediate action |
