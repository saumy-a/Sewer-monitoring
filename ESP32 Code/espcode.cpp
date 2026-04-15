// For Esp code
#include <WiFi.h>
#include <HTTPClient.h>
#include <DHT.h>

// ===== WIFI =====
const char* ssid = "SVB"; // Wifi name
const char* password = "123456789"; // Wifi Password 

// ===== SERVER =====
const char* serverName = "http://10.205.23.59:5001/data"; // from FastAPI 

// ===== PINS =====
#define MQ135_PIN 34
#define MQ4_PIN 35
#define TRIG_PIN 23
#define ECHO_PIN 22
#define FLOW_PIN 27
#define DHTPIN 4
#define DHTTYPE DHT22

DHT dht(DHTPIN, DHTTYPE);

// ===== FLOW SENSOR =====
volatile int pulseCount = 0;
float flowRate = 0;

void IRAM_ATTR pulseCounter() {
  pulseCount++;
}

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

  attachInterrupt(digitalPinToInterrupt(FLOW_PIN), pulseCounter, FALLING);

  dht.begin();
}

void loop() {

  // ===== READ DHT =====
  float temp = dht.readTemperature();
  float humidity = dht.readHumidity();

  // ===== READ GAS =====
  int airValue = analogRead(MQ135_PIN);
  int methaneValue = analogRead(MQ4_PIN);

  // ===== ULTRASONIC =====
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000);
  float distance = duration * 0.034 / 2;

  // ===== FLOW =====
  pulseCount = 0;
  delay(1000);
  flowRate = pulseCount / 7.5;

  // ===== AI LOGIC =====
  String status = "SAFE";

  if (airValue > 2500 || methaneValue > 2500) {
    status = "DANGER: GAS";
  }
  else if (flowRate < 1 && distance < 15) {
    status = "BLOCKAGE";
  }
  else if (distance < 10) {
    status = "OVERFLOW";
  }

  // ===== SERIAL OUTPUT =====
  Serial.println("----- DATA -----");
  Serial.print("Temp: "); Serial.print(temp);
  Serial.print(" | Humidity: "); Serial.println(humidity);

  Serial.print("Air: "); Serial.print(airValue);
  Serial.print(" | Methane: "); Serial.println(methaneValue);

  Serial.print("Distance: "); Serial.print(distance);
  Serial.print(" cm | Flow: "); Serial.println(flowRate);

  Serial.print("STATUS: ");
  Serial.println(status);

  // ===== SEND TO SERVER =====
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;

    http.begin(serverName);
    http.addHeader("Content-Type", "application/json");

    String jsonData = "{";
    jsonData += "\"temp\":" + String(temp) + ",";
    jsonData += "\"humidity\":" + String(humidity) + ",";
    jsonData += "\"air\":" + String(airValue) + ",";
    jsonData += "\"methane\":" + String(methaneValue) + ",";
    jsonData += "\"distance\":" + String(distance) + ",";
    jsonData += "\"flow\":" + String(flowRate) + ",";
    jsonData += "\"status\":\"" + status + "\"}";

    int httpResponseCode = http.POST(jsonData);

    Serial.print("HTTP Response: ");
    Serial.println(httpResponseCode);

    http.end();
  }

  delay(5000);
}