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
  // ADC readings (0–4095 on ESP32 12-bit ADC).
  // The server ML model was trained on scaled ppm-range values (~0–1500).
  // FIX: map 12-bit ADC range to 0–1500 to match training data distribution.
  int rawAir     = analogRead(MQ135_PIN);
  int rawMethane = analogRead(MQ4_PIN);
  int airValue     = map(rawAir,     0, 4095, 0, 1500);
  int methaneValue = map(rawMethane, 0, 4095, 0, 1500);

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
  Serial.printf("Air (mapped): %d  |  Methane (mapped): %d\n", airValue, methaneValue);
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