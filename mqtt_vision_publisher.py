"""
MQTT Vision Publisher for Distributed Face-Locked Servo System
Publishes face movement commands to MQTT broker for ESP8266 servo control
"""

import json
import time
import socket
import paho.mqtt.client as mqtt
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np

# Import your existing face recognition system
from face_recognition_5pt.src.main_lock import FaceLockingSystem

@dataclass
class MovementCommand:
    """Movement command structure"""
    status: str  # MOVE_LEFT, MOVE_RIGHT, CENTERED, NO_FACE
    confidence: float
    timestamp: float
    bbox_center: Optional[tuple] = None
    
class MQTTVisionPublisher:
    def __init__(self, team_id: str, vps_ip: str = "157.173.101.159", mqtt_port: int = 1883):
        self.team_id = team_id
        self.vps_ip = vps_ip
        self.mqtt_port = mqtt_port
        
        # MQTT topics
        self.movement_topic = f"vision/{team_id}/movement"
        self.heartbeat_topic = f"vision/{team_id}/heartbeat"
        
        # MQTT client
        self.client = mqtt.Client(client_id=f"vision_pc_{team_id}")
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        
        # Face tracking
        self.face_system = FaceLockingSystem()
        self.last_center_x = None
        self.movement_threshold = 50  # pixels
        
        # Connection status
        self.connected = False
        
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            self.connected = True
            print(f"[MQTT] Connected to broker at {self.vps_ip}:{self.mqtt_port}")
            print(f"[MQTT] Publishing to: {self.movement_topic}")
        else:
            print(f"[MQTT] Failed to connect, return code {rc}")
            self.connected = False
    
    def _on_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback"""
        self.connected = False
        print(f"[MQTT] Disconnected from broker")
    
    def connect(self):
        """Connect to MQTT broker"""
        try:
            self.client.connect(self.vps_ip, self.mqtt_port, 60)
            self.client.loop_start()
            
            # Wait for connection
            timeout = 10
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            return self.connected
        except Exception as e:
            print(f"[MQTT] Connection error: {e}")
            return False
    
    def determine_movement(self, face_results):
        """Determine movement status from face detection results"""
        if not face_results:
            return MovementCommand("NO_FACE", 0.0, time.time())
        
        # Find the locked face or the largest face
        target_face = None
        for result in face_results:
            if result.is_locked:
                target_face = result
                break
        
        if not target_face:
            # Use the largest face if no locked face
            target_face = max(face_results, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        
        # Calculate face center
        x1, y1, x2, y2 = target_face.bbox
        center_x = (x1 + x2) / 2
        bbox_center = (center_x, (y1 + y2) / 2)
        
        # Determine movement
        if self.last_center_x is None:
            self.last_center_x = center_x
            return MovementCommand("CENTERED", target_face.similarity, time.time(), bbox_center)
        
        movement = center_x - self.last_center_x
        self.last_center_x = center_x
        
        if abs(movement) > self.movement_threshold:
            if movement > 0:
                status = "MOVE_RIGHT"
            else:
                status = "MOVE_LEFT"
        else:
            status = "CENTERED"
        
        return MovementCommand(status, target_face.similarity, time.time(), bbox_center)
    
    def publish_movement(self, command: MovementCommand):
        """Publish movement command to MQTT"""
        if not self.connected:
            print("[MQTT] Not connected, skipping publish")
            return
        
        payload = {
            "status": command.status,
            "confidence": command.confidence,
            "timestamp": command.timestamp
        }
        
        if command.bbox_center:
            payload["bbox_center"] = command.bbox_center
        
        try:
            self.client.publish(self.movement_topic, json.dumps(payload), qos=1)
            print(f"[MQTT] Published: {command.status} (confidence: {command.confidence:.2f})")
        except Exception as e:
            print(f"[MQTT] Publish error: {e}")
    
    def publish_heartbeat(self):
        """Publish heartbeat message"""
        if not self.connected:
            return
        
        payload = {
            "node": "vision_pc",
            "status": "active",
            "timestamp": time.time()
        }
        
        try:
            self.client.publish(self.heartbeat_topic, json.dumps(payload), qos=0)
        except Exception as e:
            print(f"[MQTT] Heartbeat error: {e}")
    
    def run_vision_loop(self):
        """Main vision processing loop with MQTT publishing"""
        if not self.connect():
            print("[ERROR] Failed to connect to MQTT broker")
            return
        
        # Initialize camera
        self.face_system.initialize_camera()
        
        print("[SYSTEM] Vision-MQTT system started")
        print("Press 'q' to quit")
        
        last_heartbeat = time.time()
        heartbeat_interval = 30  # seconds
        
        try:
            while True:
                ret, frame = self.face_system.cap.read()
                if not ret:
                    break
                
                # Process frame with face recognition
                processed_frame = self.face_system.process_frame(frame)
                
                # Get face results and determine movement
                face_results = self.face_system.recognize_faces(frame)
                movement_cmd = self.determine_movement(face_results)
                
                # Publish movement command
                self.publish_movement(movement_cmd)
                
                # Send heartbeat periodically
                if time.time() - last_heartbeat > heartbeat_interval:
                    self.publish_heartbeat()
                    last_heartbeat = time.time()
                
                # Show frame
                cv2.imshow("Vision-MQTT Publisher", processed_frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.client.loop_stop()
        self.client.disconnect()
        
        if hasattr(self.face_system, 'cap') and self.face_system.cap:
            self.face_system.cap.release()
        cv2.destroyAllWindows()
        
        print("[SYSTEM] Vision-MQTT system shutdown complete")

def main():
    """Main entry point"""
    # Configuration - CHANGE THIS TO YOUR TEAM ID
    TEAM_ID = "team_benite"  # Replace with your actual team ID
    
    # Create and run the publisher
    publisher = MQTTVisionPublisher(team_id=TEAM_ID)
    publisher.run_vision_loop()

if __name__ == "__main__":
    main()
