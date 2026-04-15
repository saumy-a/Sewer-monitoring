#include <WiFi.h>
#include <HTTPClient.h>
#include <DHT.h>

// ===== WIFI =====
const char* ssid     = "SVB";
const char* password = "123456789";

// ===== SERVER =====
const char* serverName = "http://10.205.23.59:5001/data";

// ===== PINS =====
#define MQ135_PIN  34
#define MQ4_PIN    35
#define TRIG_PIN   23
#define ECHO_PIN   22
#define FLOW_PIN   27
#define DHTPIN      4
#define DHTTYPE  DHT22

// ===== SENSOR CALIBRATION =====
// HOW TO FIND YOUR BASELINE:
//   1. Power on the ESP32 in clean open air (outdoors or well-ventilated room)
//   2. Wait at least 5 minutes for MQ sensors to warm up
//   3. Enable CALIBRATION_MODE below (set to 1), flash, open Serial Monitor
//   4. Watch the "RAW" values for 2-3 minutes until stable
//   5. Set the values you see as MQ135_CLEAN_AIR_RAW and MQ4_CLEAN_AIR_RAW
//   6. Set CALIBRATION_MODE back to 0, re-flash
//
// Your sensor logs showed methane mapped ~1200-1400 in open air.
// That means MQ4 raw ADC is approximately 3270-3815 in clean air.
// Adjust after running calibration mode.
#define CALIBRATION_MODE      0       // set 1 to print raw ADC only

#define MQ135_CLEAN_AIR_RAW   500     // replace after calibration
#define MQ4_CLEAN_AIR_RAW     3200    // estimated from your logs (mapped ~1170 = raw ~3200)

// Maps sensor reading to 0-1500 scale where 0 = clean air baseline
// Values below baseline clamp to 0 (sensor noise / temperature drift)
int mapSensor(int rawValue, int cleanAirBaseline) {
  int delta = rawValue - cleanAirBaseline;
  if (delta <= 0) return 0;  // at or below baseline → clean air
  // Scale remaining range (baseline→4095) onto 0→1500
  return (int)((long)delta * 1500 / (4095 - cleanAirBaseline));
}

DHT dht(DHTPIN, DHTTYPE);

// ===== FLOW SENSOR =====
// FIX: pulseCount must be volatile so the ISR write is visible to loop()
volatile uint32_t pulseCount = 0;

void IRAM_ATTR pulseCounter() {
  pulseCount++;
}

// ===== LOOP TIMING =====
// FIX: measure flow over the full loop interval, not a buried delay(1000)
#define LOOP_INTERVAL_MS 5000UL
unsigned long lastLoopTime = 0;

void setup() {
  Serial.begin(115200);

  // WiFi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected!");
  Serial.println(WiFi.localIP());

  // Sensors
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(FLOW_PIN, INPUT_PULLUP);

  // FIX: use RISING for YF-S201 flow sensor (pulse on rising edge)
  attachInterrupt(digitalPinToInterrupt(FLOW_PIN), pulseCounter, RISING);

  dht.begin();

  lastLoopTime = millis();
}

void loop() {
  // ===== NON-BLOCKING TIMING =====
  // FIX: replaced delay(5000) with millis()-based timer so flow pulses
  //      accumulate over exactly LOOP_INTERVAL_MS without any blind spot.
  unsigned long now = millis();
  if (now - lastLoopTime < LOOP_INTERVAL_MS) return;

  // ===== FLOW RATE =====
  // FIX: snapshot and reset atomically by disabling the interrupt briefly.
  //      Previously: pulseCount was reset, then delay(1000) ran inside the
  //      loop that also had delay(5000) at the end → 6-second cycle, 1-second
  //      count window, and pulses from the 5-second sleep were thrown away.
  noInterrupts();
  uint32_t pulses = pulseCount;
  pulseCount = 0;
  interrupts();

  float elapsedSec = (now - lastLoopTime) / 1000.0f;
  lastLoopTime = now;

  // YF-S201: 7.5 pulses per second per L/min
  float flowRate = (pulses / 7.5f) / elapsedSec;   // L/min

  // ===== READ DHT =====
  float temp     = dht.readTemperature();
  float humidity = dht.readHumidity();

  // FIX: DHT22 returns NaN on failure — sending NaN in JSON crashes the
  //      Flask server (pd.DataFrame can't handle it). Substitute safe defaults
  //      and log the error so you know the sensor needs attention.
  bool dhtOk = !(isnan(temp) || isnan(humidity));
  if (!dhtOk) {
    Serial.println("[WARN] DHT read failed — using fallback values");
    temp     = 0.0;
    humidity = 0.0;
  }

  // ===== READ GAS =====
  // CALIBRATED mapping: subtract clean-air baseline first so that 0 = clean
  // air and 1500 = maximum detected gas above baseline. This prevents the
  // sensor's resting resistance in fresh air from being misread as gas.
  // Previously: map(0,4095,0,1500) meant clean-air ADC ~3200 mapped to
  // ~1170 ppm — far above the DANGER threshold of 700, causing false alerts.
  int rawAir     = analogRead(MQ135_PIN);
  int rawMethane = analogRead(MQ4_PIN);

#if CALIBRATION_MODE
  // ── CALIBRATION MODE: just print raw values, skip server POST ──────────
  Serial.printf("[CAL] RAW MQ135: %d | RAW MQ4: %d\n", rawAir, rawMethane);
  return;  // skip the rest of loop()
#endif

  int airValue     = mapSensor(rawAir,     MQ135_CLEAN_AIR_RAW);
  int methaneValue = mapSensor(rawMethane, MQ4_CLEAN_AIR_RAW);
  Serial.printf("RAW  MQ135: %d → mapped air: %d\n",     rawAir,     airValue);
  Serial.printf("RAW  MQ4:   %d → mapped methane: %d\n", rawMethane, methaneValue);

  // ===== ULTRASONIC =====
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  // 30 ms timeout ≈ max range ~500 cm — sufficient for sewer monitoring
  long duration = pulseIn(ECHO_PIN, HIGH, 30000UL);

  // FIX: pulseIn returns 0 on timeout (no echo / sensor error).
  //      distance=0 previously triggered false OVERFLOW alert.
  //      Use -1.0 as a sentinel so the server can ignore bad readings.
  float distance = -1.0f;
  if (duration > 0) {
    distance = duration * 0.034f / 2.0f;   // cm
  } else {
    Serial.println("[WARN] Ultrasonic timeout — distance set to -1 (invalid)");
  }

  // ===== SERIAL DEBUG =====
  Serial.println("----- DATA -----");
  Serial.printf("Temp: %.1f C  |  Humidity: %.1f %%\n", temp, humidity);
  Serial.printf("Air   — raw:%d baseline:%d mapped:%d\n", rawAir,     MQ135_CLEAN_AIR_RAW, airValue);
  Serial.printf("CH4   — raw:%d baseline:%d mapped:%d\n", rawMethane, MQ4_CLEAN_AIR_RAW,   methaneValue);
  Serial.printf("Distance: %.1f cm  |  Flow: %.2f L/min\n", distance, flowRate);
  Serial.printf("DHT OK: %s\n", dhtOk ? "yes" : "NO (fallback)");

  // ===== SEND TO SERVER =====
  // FIX: removed local status computation — the server's Random Forest ML
  //      model classifies status. Local rules were redundant, used different
  //      label strings ("DANGER: GAS" vs "DANGER"), and sent a stale
  //      field the server overwrites anyway.
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(serverName);
    http.addHeader("Content-Type", "application/json");
    http.setTimeout(8000);   // 8 s timeout — avoid hanging the loop

    // Build JSON manually (ArduinoJson not required for this simple payload)
    String jsonData = "{";
    jsonData += "\"temp\":"     + String(temp, 2)     + ",";
    jsonData += "\"humidity\":" + String(humidity, 2) + ",";
    jsonData += "\"air\":"      + String(airValue)    + ",";
    jsonData += "\"methane\":"  + String(methaneValue) + ",";
    jsonData += "\"distance\":" + String(distance, 1) + ",";
    jsonData += "\"flow\":"     + String(flowRate, 3) + ",";
    jsonData += "\"dht_ok\":"   + String(dhtOk ? "true" : "false");
    jsonData += "}";

    int httpResponseCode = http.POST(jsonData);
    Serial.printf("HTTP Response: %d\n", httpResponseCode);

    if (httpResponseCode > 0) {
      String response = http.getString();
      Serial.println("Server reply: " + response);
    } else {
      Serial.println("[ERROR] POST failed: " + String(http.errorToString(httpResponseCode)));
    }

    http.end();
  } else {
    Serial.println("[WARN] WiFi disconnected — attempting reconnect...");
    WiFi.reconnect();
  }
}