# Face-Locked Servo System

A distributed vision-control system that detects and tracks faces, publishes movement commands via MQTT, and controls servo motors in real-time. The system includes a PC-based vision engine, ESP8266 edge controller, cloud backend service, and real-time web dashboard.

## 🏗️ System Architecture

```
┌─────────────┐    MQTT     ┌─────────────┐    WebSocket    ┌─────────────┐
│   PC Vision │ ─────────► │   VPS MQTT  │ ─────────────► │  Dashboard  │
│   Engine    │             │   Broker    │                │   Browser   │
└─────────────┘             └─────────────┘                └─────────────┘
                                      │
                                      ▼ MQTT
                              ┌─────────────┐
                              │  ESP8266    │
                              │  Servo      │
                              │  Controller │
                              └─────────────┘
```

## 📋 Features

- **Face Detection & Recognition**: Multi-face recognition with target locking
- **Real-time Tracking**: Smooth face movement detection and servo control
- **MQTT Communication**: Reliable message publishing/subscribing
- **WebSocket Dashboard**: Live monitoring with real-time updates
- **Team Isolation**: Support for multiple teams without interference
- **Heartbeat Monitoring**: System health monitoring for all nodes

## 🛠️ Components

### 1. PC Vision Engine (`mqtt_vision_publisher.py`)
- Face detection and recognition using OpenCV and MediaPipe
- Movement command generation (MOVE_LEFT, MOVE_RIGHT, CENTERED, NO_FACE)
- MQTT publishing to `vision/<team_id>/movement`
- Real-time visualization

### 2. ESP8266 Servo Controller (`esp8266_servo_controller.ino`)
- MQTT subscription to movement commands
- Servo motor control with smooth movement
- Heartbeat publishing to `vision/<team_id>/heartbeat`
- WiFi connectivity

### 3. WebSocket Backend (`websocket_backend.py`)
- MQTT message subscription
- Real-time WebSocket broadcasting
- Client connection management
- System statistics tracking

### 4. Web Dashboard (`dashboard/`)
- Real-time movement status display
- System monitoring interface
- Visual servo position indicator
- Live log console

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Arduino IDE with ESP8266 support
- MQTT Broker (Mosquitto)
- VPS with Ubuntu 20.04+

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd face-recognition-5pt
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure your team ID**
   - Update `TEAM_ID` in all components
   - Default: `"team_benite"`

4. **Deploy to VPS**
```bash
chmod +x deploy_vps.sh
./deploy_vps.sh
```

5. **Configure Arduino**
   - Open `esp8266_servo_controller.ino` in Arduino IDE
   - Update WiFi credentials and team ID
   - Upload to ESP8266

6. **Run the system**
   - Start PC vision publisher: `python mqtt_vision_publisher.py`
   - Access dashboard: `http://<your-vps-ip>`

## 📁 Project Structure

```
face-recognition-5pt/
├── src/                          # Face recognition modules
│   ├── main_lock.py             # Main face locking system
│   ├── haar_5pt.py              # Face detection
│   └── ...
├── mqtt_vision_publisher.py     # PC vision + MQTT publisher
├── websocket_backend.py         # WebSocket backend service
├── esp8266_servo_controller.ino # Arduino servo controller
├── dashboard/                   # Web dashboard
│   ├── index.html              # Main dashboard page
│   ├── style.css               # Dashboard styles
│   └── script.js               # Dashboard JavaScript
├── deploy_vps.sh               # VPS deployment script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🔧 Configuration

### MQTT Topics
- **Movement Commands**: `vision/<team_id>/movement`
- **Heartbeat**: `vision/<team_id>/heartbeat`

### Movement Message Format
```json
{
  "status": "MOVE_LEFT|MOVE_RIGHT|CENTERED|NO_FACE",
  "confidence": 0.85,
  "timestamp": 1640995200.0,
  "bbox_center": [320, 240]
}
```

### Heartbeat Message Format
```json
{
  "node": "vision_pc|esp8266_servo",
  "status": "active",
  "timestamp": 1640995200.0,
  "servo_position": 90
}
```

## 🌐 Dashboard Features

- **Real-time Movement Status**: Shows current face movement direction
- **Confidence Display**: Face recognition confidence percentage
- **Visual Servo Indicator**: Animated servo position display
- **System Statistics**: Connected clients, message counts
- **Heartbeat Monitoring**: Node status and health
- **Live Log Console**: Real-time system events
- **Movement History**: Recent movement commands

## 🔌 Hardware Requirements

### ESP8266 Setup
- ESP8266 NodeMCU or similar
- Servo motor (SG90 or similar)
- Jumper wires
- Power supply for servo

### Wiring
```
ESP8266 D2 → Servo Signal (Orange)
ESP8266 3V3 → Servo Power (Red)
ESP8266 GND → Servo Ground (Brown)
```

## 📊 Monitoring & Debugging

### VPS Management
```bash
# View backend logs
sudo journalctl -u face-servo-backend -f

# Restart backend service
sudo systemctl restart face-servo-backend

# View MQTT logs
sudo tail -f /var/log/mosquitto/mosquitto.log

# Test MQTT connection
mosquitto_sub -h localhost -t "vision/<team_id>/movement"
```

### Arduino Debugging
- Open Serial Monitor at 115200 baud
- Check WiFi connection status
- Verify MQTT connection messages
- Monitor servo position updates

### PC Vision Debugging
- Face detection confidence thresholds
- Movement sensitivity settings
- MQTT connection status
- Camera feed verification

## 🏆 Evaluation Criteria

### System Performance
- **Face Recognition Accuracy**: >90% recognition rate
- **Movement Detection**: Responsive and accurate tracking
- **Servo Control**: Smooth and stable movement
- **Real-time Updates**: <100ms latency

### Architecture Compliance
- **MQTT Topic Isolation**: Proper team-based topics
- **WebSocket Communication**: No polling, real-time updates
- **Distributed Design**: Proper separation of concerns
- **Error Handling**: Robust connection management

## 🔒 Security Considerations

- MQTT broker configured for team isolation
- WebSocket connections with proper validation
- Input sanitization in all components
- Network firewall configuration

## 🚨 Troubleshooting

### Common Issues

1. **MQTT Connection Failed**
   - Check VPS firewall settings
   - Verify MQTT broker is running
   - Confirm team ID configuration

2. **Servo Not Moving**
   - Check servo wiring
   - Verify power supply
   - Monitor Arduino serial output

3. **Dashboard Not Updating**
   - Check WebSocket backend service
   - Verify browser console for errors
   - Test MQTT message flow

4. **Face Recognition Issues**
   - Check camera connection
   - Verify face database exists
   - Adjust confidence thresholds
