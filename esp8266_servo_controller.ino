/*
ESP8266 MQTT Servo Controller for Face-Locked Servo System
Receives movement commands via MQTT and controls servo motor
*/

#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <Servo.h>

// =========================
// 🔹 Configuration
// =========================

// WiFi Credentials - CHANGE THESE
const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// MQTT Broker Configuration
const char* MQTT_SERVER = "157.173.101.159";  // VPS IP
const int MQTT_PORT = 1883;
const char* MQTT_CLIENT_ID = "esp8266_servo";

// Team Configuration - CHANGE THIS TO YOUR TEAM ID
const char* TEAM_ID = "team_benite";

// MQTT Topics
char MOVEMENT_TOPIC[64];
char HEARTBEAT_TOPIC[64];

// Servo Configuration
#define SERVO_PIN D2  // GPIO2 on ESP8266
Servo myservo;

// Movement Configuration
#define SERVO_MIN 0    // Minimum servo angle
#define SERVO_MAX 180  // Maximum servo angle
#define SERVO_CENTER 90 // Center position
#define SERVO_STEP 5   // Degrees per movement
#define MOVEMENT_DELAY 50 // Delay between movements (ms)

// State Variables
int current_servo_pos = SERVO_CENTER;
int target_servo_pos = SERVO_CENTER;
unsigned long last_movement_time = 0;
unsigned long last_heartbeat_time = 0;
const unsigned long HEARTBEAT_INTERVAL = 30000; // 30 seconds

// WiFi and MQTT Clients
WiFiClient espClient;
PubSubClient client(espClient);

// =========================
// 🔹 WiFi Connection
// =========================

void setup_wifi() {
  delay(10);
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(WIFI_SSID);

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

// =========================
// 🔹 MQTT Callbacks
// =========================

void callback(char* topic, byte* payload, unsigned int length) {
  Serial.print("Message arrived [");
  Serial.print(topic);
  Serial.print("] ");
  
  // Convert payload to string
  String message = "";
  for (int i = 0; i < length; i++) {
    message += (char)payload[i];
  }
  Serial.println(message);

  // Parse JSON payload
  StaticJsonDocument<200> doc;
  DeserializationError error = deserializeJson(doc, message);
  
  if (error) {
    Serial.print("JSON parsing failed: ");
    Serial.println(error.c_str());
    return;
  }

  // Extract movement status
  String status = doc["status"] | "";
  float confidence = doc["confidence"] | 0.0;
  
  Serial.print("Movement status: ");
  Serial.print(status);
  Serial.print(" (confidence: ");
  Serial.print(confidence, 2);
  Serial.println(")");

  // Process movement command
  process_movement_command(status, confidence);
}

void process_movement_command(String status, float confidence) {
  // Only process if confidence is above threshold
  if (confidence < 0.5) {
    Serial.println("Confidence too low, ignoring command");
    return;
  }

  if (status == "MOVE_LEFT") {
    target_servo_pos = max(SERVO_MIN, current_servo_pos - SERVO_STEP);
    Serial.print("Moving LEFT to: ");
    Serial.println(target_servo_pos);
  } 
  else if (status == "MOVE_RIGHT") {
    target_servo_pos = min(SERVO_MAX, current_servo_pos + SERVO_STEP);
    Serial.print("Moving RIGHT to: ");
    Serial.println(target_servo_pos);
  } 
  else if (status == "CENTERED") {
    target_servo_pos = SERVO_CENTER;
    Serial.print("Centering to: ");
    Serial.println(target_servo_pos);
  }
  else if (status == "NO_FACE") {
    // Optional: Return to center when no face detected
    target_servo_pos = SERVO_CENTER;
    Serial.print("No face - centering to: ");
    Serial.println(target_servo_pos);
  }
}

// =========================
// 🔹 MQTT Functions
// =========================

void reconnect_mqtt() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    
    if (client.connect(MQTT_CLIENT_ID)) {
      Serial.println("connected");
      
      // Subscribe to movement topic
      client.subscribe(MOVEMENT_TOPIC);
      Serial.print("Subscribed to: ");
      Serial.println(MOVEMENT_TOPIC);
      
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      delay(5000);
    }
  }
}

void publish_heartbeat() {
  StaticJsonDocument<200> doc;
  doc["node"] = "esp8266_servo";
  doc["status"] = "active";
  doc["servo_position"] = current_servo_pos;
  doc["timestamp"] = millis();
  
  String payload;
  serializeJson(doc, payload);
  
  client.publish(HEARTBEAT_TOPIC, payload.c_str());
  Serial.println("Heartbeat sent");
}

// =========================
// 🔹 Servo Control
// =========================

void update_servo_position() {
  if (current_servo_pos != target_servo_pos) {
    // Smooth movement
    if (current_servo_pos < target_servo_pos) {
      current_servo_pos = min(target_servo_pos, current_servo_pos + 1);
    } else {
      current_servo_pos = max(target_servo_pos, current_servo_pos - 1);
    }
    
    myservo.write(current_servo_pos);
    delay(MOVEMENT_DELAY);
  }
}

// =========================
// 🔹 Setup and Loop
// =========================

void setup() {
  Serial.begin(115200);
  Serial.println("ESP8266 MQTT Servo Controller Starting...");
  
  // Build MQTT topics
  sprintf(MOVEMENT_TOPIC, "vision/%s/movement", TEAM_ID);
  sprintf(HEARTBEAT_TOPIC, "vision/%s/heartbeat", TEAM_ID);
  
  // Initialize servo
  myservo.attach(SERVO_PIN);
  myservo.write(SERVO_CENTER);
  current_servo_pos = SERVO_CENTER;
  target_servo_pos = SERVO_CENTER;
  
  Serial.println("Servo initialized at center position");
  
  // Connect to WiFi
  setup_wifi();
  
  // Configure MQTT
  client.setServer(MQTT_SERVER, MQTT_PORT);
  client.setCallback(callback);
  
  Serial.println("Setup complete");
  Serial.print("Movement topic: ");
  Serial.println(MOVEMENT_TOPIC);
}

void loop() {
  // Check MQTT connection
  if (!client.connected()) {
    reconnect_mqtt();
  }
  client.loop();
  
  // Update servo position smoothly
  update_servo_position();
  
  // Send heartbeat periodically
  unsigned long current_time = millis();
  if (current_time - last_heartbeat_time > HEARTBEAT_INTERVAL) {
    publish_heartbeat();
    last_heartbeat_time = current_time;
  }
  
  // Small delay to prevent overwhelming
  delay(10);
}
