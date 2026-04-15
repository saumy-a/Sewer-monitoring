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

// ===================================================================
// STEP 1 — SET THIS TO 1, FLASH, OPEN SERIAL MONITOR, WAIT 2 MINUTES
//          IN CLEAN OPEN AIR. NOTE THE STABLE RAW VALUES YOU SEE.
//          THEN SET BACK TO 0 AND FILL IN THE BASELINES BELOW.
// ===================================================================
#define CALIBRATION_MODE 0

// ===================================================================
// STEP 2 — Fill in your observed clean-air raw ADC values here.
//          From your logs your MQ-4 baseline is around 2800-3100.
//          Run CALIBRATION_MODE=1 to get your exact numbers.
// ===================================================================
#define MQ135_CLEAN_AIR_RAW  600    // replace with YOUR observed value
#define MQ4_CLEAN_AIR_RAW   2800    // replace with YOUR observed value

DHT dht(DHTPIN, DHTTYPE);

// ===== FLOW SENSOR =====
volatile uint32_t pulseCount = 0;

void IRAM_ATTR pulseCounter() {
  pulseCount++;
}

// ===== LOOP TIMING =====
#define LOOP_INTERVAL_MS 5000UL
unsigned long lastLoopTime = 0;

// ===================================================================
// mapSensor() — converts raw ADC to 0-2000 scale relative to
// clean-air baseline. Values at or below baseline map to 0.
// Values above baseline scale linearly up to 2000.
// ===================================================================
int mapSensor(int rawValue, int cleanAirBaseline) {
  int delta = rawValue - cleanAirBaseline;
  if (delta <= 0) return 0;
  int maxDelta = 4095 - cleanAirBaseline;
  return (int)((float)delta / maxDelta * 2000.0f);
}

void setup() {
  Serial.begin(115200);

  #if CALIBRATION_MODE
    Serial.println("\n===== CALIBRATION MODE =====");
    Serial.println("Let sensor warm up 2+ minutes in clean open air.");
    Serial.println("Note the stable RAW values — put them in the #defines above.");
    Serial.println("============================\n");
  #else
    WiFi.begin(ssid, password);
    Serial.print("Connecting to WiFi");
    while (WiFi.status() != WL_CONNECTED) {
      delay(500);
      Serial.print(".");
    }
    Serial.println("\nConnected!");
    Serial.println(WiFi.localIP());
  #endif

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(FLOW_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(FLOW_PIN), pulseCounter, RISING);

  dht.begin();
  lastLoopTime = millis();
}

void loop() {
  unsigned long now = millis();
  if (now - lastLoopTime < LOOP_INTERVAL_MS) return;
  lastLoopTime = now;

  // ===== READ GAS =====
  int rawAir     = analogRead(MQ135_PIN);
  int rawMethane = analogRead(MQ4_PIN);

  #if CALIBRATION_MODE
    Serial.printf("[CAL] RAW MQ135: %d  |  RAW MQ4: %d\n", rawAir, rawMethane);
    return;
  #endif

  int airValue     = mapSensor(rawAir,     MQ135_CLEAN_AIR_RAW);
  int methaneValue = mapSensor(rawMethane, MQ4_CLEAN_AIR_RAW);

  // ===== FLOW RATE =====
  noInterrupts();
  uint32_t pulses = pulseCount;
  pulseCount = 0;
  interrupts();

  float elapsedSec = LOOP_INTERVAL_MS / 1000.0f;  // fixed 5-second window
  float flowRate = (pulses / 7.5f) / elapsedSec;

  // ===== READ DHT =====
  float temp     = dht.readTemperature();
  float humidity = dht.readHumidity();

  bool dhtOk = !(isnan(temp) || isnan(humidity));
  if (!dhtOk) {
    Serial.println("[WARN] DHT read failed — using fallback values");
    temp     = 0.0;
    humidity = 0.0;
  }

  // ===== ULTRASONIC =====
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000UL);
  float distance = -1.0f;
  if (duration > 0) {
    distance = duration * 0.034f / 2.0f;
  } else {
    Serial.println("[WARN] Ultrasonic timeout — distance set to -1 (invalid)");
  }

  // ===== SERIAL DEBUG =====
  Serial.println("----- DATA -----");
  Serial.printf("Temp: %.1f C  |  Humidity: %.1f %%\n", temp, humidity);
  Serial.printf("Air   — raw:%d  baseline:%d  mapped:%d\n", rawAir,     MQ135_CLEAN_AIR_RAW, airValue);
  Serial.printf("CH4   — raw:%d  baseline:%d  mapped:%d\n", rawMethane, MQ4_CLEAN_AIR_RAW,   methaneValue);
  Serial.printf("Distance: %.1f cm  |  Flow: %.2f L/min\n", distance, flowRate);
  Serial.printf("DHT OK: %s\n", dhtOk ? "yes" : "NO (fallback)");

  // ===== SEND TO SERVER =====
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(serverName);
    http.addHeader("Content-Type", "application/json");
    http.setTimeout(8000);

    String jsonData = "{";
    jsonData += "\"temp\":"      + String(temp, 2)      + ",";
    jsonData += "\"humidity\":"  + String(humidity, 2)  + ",";
    jsonData += "\"air\":"       + String(airValue)     + ",";
    jsonData += "\"methane\":"   + String(methaneValue) + ",";
    jsonData += "\"distance\":"  + String(distance, 1)  + ",";
    jsonData += "\"flow\":"      + String(flowRate, 3)  + ",";
    jsonData += "\"dht_ok\":"    + String(dhtOk ? "true" : "false");
    jsonData += "}";

    int httpResponseCode = http.POST(jsonData);
    Serial.printf("HTTP Response: %d\n", httpResponseCode);

    if (httpResponseCode > 0) {
      Serial.println("Server reply: " + http.getString());
    } else {
      Serial.println("[ERROR] POST failed: " + String(http.errorToString(httpResponseCode)));
    }
    http.end();

  } else {
    Serial.println("[WARN] WiFi disconnected — attempting reconnect...");
    WiFi.reconnect();
  }
}