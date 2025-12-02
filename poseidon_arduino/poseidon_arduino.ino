#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>  // ADD THIS
#include <OneWire.h>
#include <DallasTemperature.h>
#include <time.h>

// ----------------------------
// WIFI DETAILS
// ----------------------------
const char* ssid     = "ALU_Staff";
const char* password = "@fr1canLU_staff";

// ----------------------------
// API ENDPOINT
// ----------------------------
String apiURL = "https://fierceee-poseidon-aquaculture.hf.space/ingest_reading/";

// ----------------------------
// PIN DEFINITIONS
// ----------------------------
#define TURBIDITY_PIN 13
#define PH_PIN 15
#define TEMP_PIN 4   // DS18B20

// ----------------------------
// TEMP SENSOR SETUP
// ----------------------------
OneWire oneWire(TEMP_PIN);
DallasTemperature sensors(&oneWire);

// ----------------------------
// GET TIMESTAMP IN ISO FORMAT
// ----------------------------
String getTimestamp() {
  time_t now;
  time(&now);

  struct tm * timeinfo = gmtime(&now);
  char buffer[30];
  strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", timeinfo);

  return String(buffer);
}

// ----------------------------
// WIFI CONNECT FUNCTION
// ----------------------------
void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;

  Serial.print("Connecting to WiFi");
  WiFi.begin(ssid, password);

  int retries = 0;
  while (WiFi.status() != WL_CONNECTED && retries < 20) {
    Serial.print(".");
    delay(500);
    retries++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n✔ WiFi Connected");
  } else {
    Serial.println("\n✖ WiFi Failed");
  }
}

// ----------------------------
// SEND JSON TO API
// ----------------------------
bool sendToAPI(String jsonData) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("✖ Cannot send. WiFi disconnected.");
    return false;
  }

  WiFiClientSecure client;      // CREATE SECURE CLIENT
  client.setInsecure();          // ALLOW HTTPS WITHOUT CERTIFICATE VALIDATION

  HTTPClient http;
  http.begin(client, apiURL);    // PASS THE SECURE CLIENT TO HTTPClient
  http.addHeader("Content-Type", "application/json");

  int httpCode = http.POST(jsonData);

  Serial.print("✔ Server Response Code: ");
  Serial.println(httpCode);
  Serial.println(http.getString());

  http.end();

  return (httpCode == 200 || httpCode == 201);
}

// ----------------------------
// SETUP
// ----------------------------
void setup() {
  Serial.begin(115200);
  sensors.begin();
  pinMode(TURBIDITY_PIN, INPUT);

  connectWiFi();

  // Setup NTP time
  configTime(0, 0, "pool.ntp.org", "time.nist.gov");
  Serial.println("Syncing time…");
  delay(2000);
}

// ----------------------------
// MAIN LOOP
// ----------------------------
void loop() {

  // ---- READ TURBIDITY ----
  int turbRaw = analogRead(TURBIDITY_PIN);
  float turbVoltage = turbRaw * (3.3 / 4095.0);

  // ---- READ PH ----
  int phRaw = analogRead(PH_PIN);
  float phVoltage = phRaw * (3.3 / 4095.0);
  float pH = 7 + ((2.5 - phVoltage) / 0.18);

  // ---- READ TEMPERATURE ----
  sensors.requestTemperatures();
  float tempC = sensors.getTempCByIndex(0);

  // ---- CREATE JSON ----
  String jsonData = "{";
  jsonData += "\"pond_id\": \"pond\",";
  jsonData += "\"reading\": {";
  jsonData += "\"timestamp\": \"" + getTimestamp() + "\",";
  jsonData += "\"pH\": " + String(pH) + ",";
  jsonData += "\"turbidity_proxy\": " + String(turbVoltage) + ",";
  jsonData += "\"temperature\": " + String(tempC);
  jsonData += "}";
  jsonData += "}";

  Serial.println("\n--- JSON SENT ---");
  Serial.println(jsonData);

  // ---- SEND TO SERVER ----
  connectWiFi();
  bool sent = sendToAPI(jsonData);

  if (!sent) {
    Serial.println("⚠ Data NOT sent.");
  }

  delay(15000);  // send every 15 seconds
}